"""Create publications page from BibTeX database file."""

import argparse
import calendar
from collections import defaultdict
from pathlib import Path
from warnings import warn

import pybtex.database
import requests
from pybtex.backends.html import Backend as HTMLBackend
from pybtex.style.formatting import toplevel
from pybtex.style.formatting.unsrt import Style as UnsrtStyle
from pybtex.style.formatting.unsrt import date as publication_date
from pybtex.style.names import BaseNameStyle, name_part
from pybtex.style.sorting import BaseSortingStyle
from pybtex.style.template import (
    FieldIsMissing,
    field,
    first_of,
    href,
    join,
    node,
    optional,
    sentence,
    tag,
    words,
)


class InlineHTMLBackend(HTMLBackend):
    """Backend for bibliography output as plain list suitable for inclusion in a HTML document."""

    def write_prologue(self):
        self.output("<ul style='list-style-type: none; padding-left: 0;'>\n")

    def write_epilogue(self):
        self.output("</ul>\n")

    def write_entry(self, _key, _label, text):
        self.output(f"<li style='list-style: none;'>{text}</li>\n")


class DateSortingStyle(BaseSortingStyle):
    """Sorting style for bibliography in reverse (newest first) publication date order."""

    def sorting_key(self, entry):
        months = list(calendar.month_name)
        return (
            -int(entry.fields.get("year")),
            -months.index(entry.fields.get("month", "")),
            entry.fields.get("title", ""),
        )


class LastOnlyNameStyle(BaseNameStyle):
    """Name style showing only last names and associated name particles."""

    def format(self, person, _abbr=False):
        return join[
            name_part(tie=True)[person.rich_prelast_names],
            name_part[person.rich_last_names],
            name_part(before=", ")[person.rich_lineage_names],
        ]


@node
def summarized_names(children, context, role, summarize_limit=3, **kwargs):
    """Return formatted names with et al. summarization when number exceeds specified limit."""

    assert not children

    try:
        persons = context["entry"].persons[role]
    except KeyError:
        raise FieldIsMissing(role, context["entry"])

    name_style = LastOnlyNameStyle()
    if len(persons) > summarize_limit:
        return words[name_style.format(persons[0]), "et al."].format_data(context)
    else:
        formatted_names = [name_style.format(person) for person in persons]
        return join(**kwargs)[formatted_names].format_data(context)


class SummarizedStyle(UnsrtStyle):
    """
    Bibliography style showing summarized names, year, title and journal with expandable details.

    Not suitable for use with LaTeX backend due to use of details tags.
    """

    default_sorting_style = DateSortingStyle

    def _format_summarized_names(self, role):
        return summarized_names(role, sep=", ", sep2=" and ", last_sep=", and ")

    def _format_label(self, label):
        return tag("em")[f"{label}: "]

    def _format_details_as_table(self, details):
        return tag("table")[
            toplevel[
                *(
                    tag("tr")[toplevel[tag("td")[tag("em")[key]], tag("td")[value]]]
                    for key, value in details.items()
                )
            ]
        ]

    def _get_summary_template(self, e, type_):
        venue_field = "journal" if type_ == "article" else "publisher"
        url = first_of[
            optional[join["https://doi.org/", field("doi", raw=True)]],
            optional[field("url", raw=True)],
            "#",
        ]
        return href[
            url,
            sentence(sep=". ")[
                words[
                    self._format_summarized_names("author"),
                    optional["(", field("year"), ")"],
                ],
                self.format_title(e, "title", as_sentence=False),
                tag("em")[field(venue_field)],
            ],
        ]

    def _get_details_template(self, type_):
        bibtex_type_to_label = {"article": "Journal article", "misc": "Pre-print"}
        return self._format_details_as_table(
            {
                "Type": bibtex_type_to_label[type_],
                "DOI": optional[field("doi")],
                "Date": publication_date,
                "Authors": self.format_names("author"),
                "Abstract": field("abstract"),
            }
        )

    def _get_summarized_template(self, e, type_):
        summary_template = self._get_summary_template(e, type_)
        details_template = self._get_details_template(type_)
        return tag("details")[tag("summary")[summary_template], details_template]

    def get_article_template(self, e):
        return self._get_summarized_template(e, "article")

    def get_misc_template(self, e):
        return self._get_summarized_template(e, "misc")


def write_publications_list(stream, bibliography_data, section_names, backend, style):
    """Write bibliography data with given backend and style to a stream splitting in to sections."""
    keys_by_section = defaultdict(list)
    for key, entry in bibliography_data.entries.items():
        keywords = set(k.strip() for k in entry.fields.get("keywords", "").split(","))
        section_names_in_keywords = keywords & set(section_names)
        if len(section_names_in_keywords) == 1:
            keys_by_section[section_names_in_keywords.pop()].append(key)
        elif len(section_names_in_keywords) == 0:
            msg = (
                f"BibTeX entry with key {key} does not have a keyword / tag corresponding to "
                f"one of section names {section_names} and so will not be included in output."
            )
            warn(msg, stacklevel=2)
        else:
            msg = (
                f"BibTeX entry with key {key} has multiple keywords / tags corresponding to "
                f"section names {section_names} and so will not be included in output."
            )
            warn(msg, stacklevel=2)
    for section_name in section_names:
        stream.write(f"<h2>{section_name}</h2>\n")
        formatted_bibliography = style.format_bibliography(
            bibliography_data, keys_by_section[section_name]
        )
        backend.write_to_stream(formatted_bibliography, stream)
        stream.write("\n")


if __name__ == "__main__":
    docs_directory = Path(__file__).parent
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--bib-file",
        type=Path,
        default=docs_directory / "publications.bib",
        help="BibTeX file containing publication details",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=docs_directory / "_publications_list.html",
        help="File to write publication list to in HTML format",
    )
    parser.add_argument(
        "--update-from-zotero",
        action="store_true",
        help="Update BibTeX file at path specified by --bib-file from Zotero group library",
    )
    parser.add_argument(
        "--zotero-group-id",
        default="5746396",
        help="Integer identifier for Zotero group library",
    )
    args = parser.parse_args()
    if args.update_from_zotero:
        endpoint_url = f"https://api.zotero.org/groups/{args.zotero_group_id}/items"
        # Zotero API requires maximum number of results to return (limit parameter)
        # to be explicitly specified for export formats such as bibtex and allows a
        # maximum value of 100 - if we exceed this number of publications will need
        # to switch to making multiple requests with different start indices
        response = requests.get(
            endpoint_url, params={"format": "bibtex", "limit": "100"}
        )
        if response.ok:
            with open(args.bib_file, "w") as bib_file:
                bib_file.write(response.text)
        else:
            msg = (
                f"Request to {endpoint_url} failed with status code "
                f"{response.status_code} ({response.reason})"
            )
            raise RuntimeError(msg)
    with open(args.output_file, "w") as output_file:
        write_publications_list(
            stream=output_file,
            bibliography_data=pybtex.database.parse_file(args.bib_file),
            section_names=[
                "Overview of the model",
                "Analyses using the model",
                "Healthcare seeking behaviour",
            ],
            backend=InlineHTMLBackend(),
            style=SummarizedStyle(),
        )
