"""Create publications page from BibTeX database file."""

import argparse
import calendar
from collections import defaultdict
from pathlib import Path

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

    def write_prologue(self):
        self.output("<ul>\n")

    def write_epilogue(self):
        self.output("</ul>\n")
        
    def write_entry(self, _key, _label, text):
        self.output(f"<li>{text}</li>\n")


class DateSortingStyle(BaseSortingStyle):

    def sorting_key(self, entry):
        months = list(calendar.month_name)
        return -int(entry.fields.get("year")), -months.index(
            entry.fields.get("month", "")
        )


class LastOnlyNameStyle(BaseNameStyle):

    def format(self, person, abbr=False):
        return join[
            name_part(tie=True)[person.rich_prelast_names],
            name_part[person.rich_last_names],
            name_part(before=", ")[person.rich_lineage_names],
        ]


@node
def abbreviated_names(children, context, role, summarize_limit=3, **kwargs):
    """Return formatted names."""

    assert not children

    try:
        persons = context["entry"].persons[role]
    except KeyError:
        raise FieldIsMissing(role, context["entry"])

    style = context["style"]
    if len(persons) > summarize_limit:
        return words[
            style.format_name(persons[0], style.abbreviate_names), "et al"
        ].format_data(context)
    else:
        formatted_names = [
            style.format_name(person, style.abbreviate_names) for person in persons
        ]
        return join(**kwargs)[formatted_names].format_data(context)


class AbbreviatedStyle(UnsrtStyle):

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


def write_publications_list(stream, bibliography_data, section_names, style):
    keys_by_section = defaultdict(list)
    for key, entry in bibliography_data.entries.items():
        note = entry.fields.get("note")
        if note in section_names:
            keys_by_section[note].append(key)
        else:
            keys_by_section["Other"].append(key)
    backend = InlineHTMLBackend()
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
    parser.add_argument("--bib_file", type=Path, default=docs_directory / "publications.bib")
    parser.add_argument("--output_file", type=Path, default=docs_directory / "_publications_list.html")
    args = parser.parse_args()
    bibliography_data = pybtex.database.parse_file(args.bib_file)
    style = AbbreviatedStyle()
    with open(args.output_file, "w") as output_file:
        write_publications_list(
            output_file,
            bibliography_data,
            [
                "Overview of the model",
                "Analyses using the model",
                "Healthcare seeking behaviour",
            ],
            style,
        )
