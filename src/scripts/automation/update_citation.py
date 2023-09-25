"""Update citation file from contributors list or check if fill will be updated."""

import argparse
import difflib
from pathlib import Path

import yaml

CFF_PERSON_ALLOWED_KEYS = (
    "address",
    "affiliation",
    "alias",
    "city",
    "country",
    "email",
    "family-names",
    "fax",
    "given-names",
    "name-particle",
    "name-suffix",
    "orcid",
    "post-code",
    "region",
    "tel",
    "website",
)


def strip_non_cff_fields(person):
    return {
        key: value for key, value in person.items() if key in CFF_PERSON_ALLOWED_KEYS
    }


def strip_and_filter_contributors(contributors):
    return [
        strip_non_cff_fields(contributor)
        for contributor in contributors
        if "Software development" in contributor.get("contributions", ())
    ]


if __name__ == "__main__":
    root_directory = Path(__file__).parent.parent.parent.parent
    parser = argparse.ArgumentParser(
        description="Update author list in citation file format (CFF) file"
    )
    parser.add_argument(
        "--contributors-file-path",
        type=Path,
        default=root_directory / "contributors.yaml",
        help="Path to YAML file defining contributors data",
    )
    parser.add_argument(
        "--citation-file-path",
        type=Path,
        default=root_directory / "CITATION.cff",
        help="Path to existing citation file to update",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Whether to check if citation file will be updated without modifying file "
            "and raise an error if so."
        ),
    )
    args = parser.parse_args()
    assert args.contributors_file_path.exists()
    with open(args.contributors_file_path, "r") as f:
        contributors = yaml.safe_load(f)
    assert args.citation_file_path.exists()
    with open(args.citation_file_path, "r") as f:
        citation = yaml.safe_load(f)
        f.seek(0)
        original_citation_lines = f.readlines()
    citation["authors"] = strip_and_filter_contributors(contributors)
    comment = (
        "# Author list automatically generated from contributors.yaml by running\n"
        "#     python src/scripts/automation/update_citation.py\n"
        "# Any manual updates to author list here may be overwritten\n"
    )
    yaml_string = yaml.dump(citation, sort_keys=False, allow_unicode=True, indent=2)
    updated_citation_lines = (comment + yaml_string).splitlines(keepends=True)
    if args.check:
        citation_diff = "".join(
            difflib.unified_diff(
                original_citation_lines,
                updated_citation_lines,
                fromfile=f"original ({args.citation_file_path})",
                tofile="updated",
            )
        )
        if citation_diff != "":
            raise RuntimeError(
                f"Citation file at {args.citation_file_path} would be updated:\n\n"
                + citation_diff
                + f"\nRe-run script {__file__} to update."
            )
    else:
        with open(args.citation_file_path, "w") as f:
            f.writelines(updated_citation_lines)
