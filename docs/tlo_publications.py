"""Create publications page from BibTeX database file."""

import argparse
import calendar
from collections import defaultdict
from pathlib import Path
from warnings import warn

import pybtex.database
from pybtex.backends.html import Backend as HTMLBackend
from pybtex.style.formatting.unsrt import Style as UnsrtStyle
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
    """Backend for bibliography output as unordered list suitable for inclusion in a HTML document."""

    def write_prologue(self):
        self.output("<ul>\n")

    def write_epilogue(self):
        self.output("</ul>\n")

    def write_entry(self, _key, _label, text):
        self.output(f"<li>{text}</li>\n")


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

    def format(self, person, abbr=False):
        return join[
            name_part(tie=True)[person.rich_prelast_names],
            name_part[person.rich_last_names],
            name_part(before=", ")[person.rich_lineage_names],
        ]


@node
def abbreviated_names(children, context, role, summarize_limit=3, **kwargs):
    """Return formatted names with et al. summarization when number exceeds specified limit."""

    assert not children

    try:
        persons = context["entry"].persons[role]
    except KeyError:
        raise FieldIsMissing(role, context["entry"])

    style = context["style"]
    if len(persons) > summarize_limit:
        return words[
            style.format_name(persons[0], style.abbreviate_names), "et al."
        ].format_data(context)
    else:
        formatted_names = [
            style.format_name(person, style.abbreviate_names) for person in persons
        ]
        return join(**kwargs)[formatted_names].format_data(context)


class AbbreviatedStyle(UnsrtStyle):
    """Abbreviated bibliography style showing summarized names, year, title and journal / publisher."""

    default_name_style = LastOnlyNameStyle
    default_sorting_style = DateSortingStyle

    def format_abbreviated_names(self, role):
        return abbreviated_names(role, sep=", ", sep2=" and ", last_sep=", and ")

    def _get_summarized_template(self, e, venue_field):
        url = first_of[
            optional[join["https://doi.org/", field("doi", raw=True)]],
            optional[field("url", raw=True)],
            "#",
        ]
        template = href[
            url,
            sentence(sep=". ")[
                words[
                    self.format_abbreviated_names("author"),
                    optional["(", field("year"), ")"],
                ],
                self.format_title(e, "title", as_sentence=False),
                tag("em")[field(venue_field)],
            ],
        ]
        return template

    def get_article_template(self, e):
        return self._get_summarized_template(e, "journal")

    def get_misc_template(self, e):
        return self._get_summarized_template(e, "publisher")


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bib_file",
        type=Path,
        default=docs_directory / "publications.bib",
        help="BibTeX file containing publication details",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=docs_directory / "_publications_list.html",
        help="File to write publication list to in HTML format",
    )
    args = parser.parse_args()
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
            style=AbbreviatedStyle(),
        )
