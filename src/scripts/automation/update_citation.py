import argparse
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
    args = parser.parse_args()
    assert args.contributors_file_path.exists()
    with open(args.contributors_file_path, "r") as f:
        contributors = yaml.safe_load(f)
    assert args.citation_file_path.exists()
    with open(args.citation_file_path, "r") as f:
        citation = yaml.safe_load(f)
    citation["authors"] = strip_and_filter_contributors(contributors)
    with open(args.citation_file_path, "w") as f:
        comment = (
            "# Author list automatically generated from contributors.yaml by running\n"
            "#     python src/scripts/automation/update_citation.py\n"
            "# Any manual updates to author list here may be overwritten\n"
        )
        yaml_string = yaml.dump(citation, sort_keys=False, allow_unicode=True, indent=2)
        f.write(comment + yaml_string)
