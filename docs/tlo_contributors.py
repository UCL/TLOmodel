import argparse
from collections import defaultdict
from pathlib import Path

import yaml


def homepage_link_html(homepage_url):
    homepage_icon_url = (
        "https://raw.githubusercontent.com/primer/octicons/main/icons/home-16.svg"
    )
    return (
        f"<a href='{homepage_url}' style='padding: 2px'>"
        f"<img src='{homepage_icon_url}' alt='Homepage' width='16' height='16' />"
        "</a>"
    )


def orcid_link_html(orcid):
    orcid_icon_url = "https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png"
    return (
        f"<a href='{orcid}' style='padding: 2px'>"
        f"<img src='{orcid_icon_url}' alt='ORCID logo' width='16' height='16' />"
        "</a>"
    )


def github_link_html(github_username):
    github_icon_url = "https://raw.githubusercontent.com/primer/octicons/main/icons/mark-github-16.svg"
    return (
        f"<a href='https://github.com/{github_username}' style='padding: 2px'>"
        f"<img src='{github_icon_url}' alt='GitHub mark' width='16' height='16'/>"
        "</a>"
    )


def contributor_html(contributor):
    name_string = f"{contributor['given-names']} {contributor['family-names']}"
    if "website" in contributor:
        html_string = f"<a href='{contributor['website']}'>{name_string}</a> "
    else:
        html_string = f"{name_string} "
    if "role" in contributor:
        html_string += f" ({contributor['role']}) "
    if "orcid" in contributor:
        html_string += orcid_link_html(contributor["orcid"])
    if "github-username" in contributor:
        html_string += github_link_html(contributor["github-username"])
    return html_string


def contributor_list_html(contributors, sort_key="family-names"):
    sorted_contributors = sorted(contributors, key=lambda c: c[sort_key])
    list_items = [
        f"<li>{contributor_html(contributor)}</li>"
        for contributor in sorted_contributors
    ]
    return "<ul>\n  " + "\n  ".join(list_items) + "\n</ul>"


def categorized_contributor_lists_html(
    contributors, category_predicates, sort_key="family-names"
):
    categorized_contributors = defaultdict(list)
    for contributor in contributors:
        for category, predicate in category_predicates.items():
            if predicate(contributor):
                categorized_contributors[category].append(contributor)
                break
    assert len(contributors) == sum(len(c) for c in categorized_contributors.values())
    html_string = ""
    for category in category_predicates.keys():
        html_string += f"<h2>{category}</h2>\n"
        html_string += f"{contributor_list_html(categorized_contributors[category])}\n"
    return html_string


if __name__ == "__main__":
    docs_directory = Path(__file__).parent
    root_directory = docs_directory.parent
    parser = argparse.ArgumentParser(
        description="Generate contributors list as HTML file"
    )
    parser.add_argument(
        "--contributors-file-path",
        type=Path,
        default=root_directory / "contributors.yaml",
        help="Path to YAML file defining contributors data",
    )
    parser.add_argument(
        "--output-file-path",
        type=Path,
        default=docs_directory / "_contributors_list.html",
        help="Path to write output file to",
    )
    args = parser.parse_args()
    assert args.contributors_file_path.exists()
    with open(args.contributors_file_path, "r") as f:
        contributors = yaml.safe_load(f)
    contribution_categories = (
        "Policy translation",
        "Epidemiology and modelling",
        "Health economics",
        "Software development",
        "Clinical consultant",
        "Project management",
    )
    category_predicates = {
        "Scientific leads": lambda c: "lead" in c.get("role", "").lower(),
        **{
            category: lambda c, d=category: d in c.get("contributions", ())
            for category in contribution_categories
        },
        "Acknowledgements": lambda c: True,
    }
    with open(args.output_file_path, "w") as f:
        f.write(categorized_contributor_lists_html(contributors, category_predicates))
