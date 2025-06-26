import os
from pathlib import Path
from typing import List
import pandas
from tabulate import tabulate


def github_download_url(resource_file_path: Path) -> str:
    path_parts = resource_file_path.resolve().parts
    relative_url = "/".join(path_parts[path_parts.index("TLOmodel") + 1 :])
    return f"https://github.com/UCL/TLOmodel/raw/master/{relative_url}"


def rst_download_link(resource_file_path: Path, file_type: str) -> str:
    return (
        f":download:`Download original {file_type} file "
        f"from GitHub <{github_download_url(resource_file_path)}>`"
    )


def rst_header(title: str, level: int = 0) -> str:
    separator_character = '=-^"'[level]
    return title + "\n" + (separator_character * len(title)) + "\n\n"


def _remove_prefix(text, prefix):
    """Remove prefix if present from text.

    Equivalent to str.removeprefix method present in Python 3.9+.
    """
    return text[len(prefix) :] if text.startswith(prefix) else text


def rst_resource_file_header(resource_file_path: Path) -> str:
    resource_file_name = (
        _remove_prefix(resource_file_path.stem, "ResourceFile_").replace("_", " ")
        + f" ({resource_file_path.suffix})"
    )
    return (
        rst_header(resource_file_name)
        + rst_download_link(resource_file_path, resource_file_path.suffix)
        + "\n\n"
    )


def rst_toc(entries: List[str], max_depth: int = 1) -> str:
    return (
        ".. toctree::\n    "
        + f":maxdepth: {max_depth}\n\n    "
        + "\n    ".join(entries)
        + "\n\n"
    )


def rst_file_index_toc(
    filenames: List[str], subdirectories: List[str], max_depth: int = 1
) -> str:
    index_files = [
        str(Path(subdirectory) / "index.rst") for subdirectory in subdirectories
    ]
    return rst_toc(sorted(index_files) + sorted(filenames), max_depth)


def write_placeholder(input_path: Path, output_path: Path) -> None:
    with open(output_path, "w") as output_file:
        output_file.write(rst_resource_file_header(input_path))


def escape_rst_markup_characters(dataframe: pandas.DataFrame) -> pandas.DataFrame:
    dataframe.rename(
        columns=lambda name: str(name).replace("_", "\_").replace("*", "\*"),
        inplace=True,
    )
    dataframe.replace(regex={"_": "\_", "\*": "\\*"}, inplace=True)


def csv_to_rst_table(input_path: Path, output_path: Path) -> None:
    dataframe = pandas.read_csv(input_path, na_filter=False)
    escape_rst_markup_characters(dataframe)
    table = tabulate(dataframe, headers="keys", tablefmt="rst")
    with open(output_path, "w") as output_file:
        output_file.write(rst_resource_file_header(input_path))
        output_file.write(table)


def excel_to_rst_table(input_path: Path, output_path: Path) -> None:
    sheet_dataframes = pandas.read_excel(input_path, na_filter=False, sheet_name=None)
    with open(output_path, "w") as output_file:
        output_file.write(rst_resource_file_header(input_path))
        output_file.write(".. contents::\n\n")
        for sheet_name, dataframe in sheet_dataframes.items():
            escape_rst_markup_characters(dataframe)
            table = tabulate(dataframe, headers="keys", tablefmt="rst")
            output_file.write(rst_header(sheet_name, level=1))
            output_file.write(table)
            output_file.write("\n\n")


def generate_docs_pages_from_resource_files(
    resources_directory: Path,
    docs_directory: Path,
    max_file_size_bytes: int = 2**15,
) -> None:
    root_output_directory = docs_directory / "resources"
    root_output_directory.mkdir(exist_ok=True)
    for current_path, subdirectories, resource_file_names in os.walk(
        resources_directory
    ):
        current_path = Path(current_path)
        output_directory = root_output_directory / current_path.relative_to(
            resources_directory
        )
        if not output_directory.exists():
            output_directory.mkdir(parents=True)
        index_file_path = output_directory / "index.rst"
        with open(index_file_path, "w") as index_file:
            title = (
                index_file_path.parent.stem
                if current_path != resources_directory
                else "Resource files"
            )
            index_file.write(rst_header(title))
            if current_path == resources_directory:
                index_file.write("Resource  files used in ``TLOmodel`` simulations\n\n")
            index_file.write(rst_file_index_toc(resource_file_names, subdirectories))
        for resource_file_name in resource_file_names:
            resource_file_path = current_path / resource_file_name
            output_path = output_directory / (resource_file_name + ".rst")
            if resource_file_path.stat().st_size > max_file_size_bytes:
                write_placeholder(resource_file_path, output_path)
                print(f"Wrote placeholder only for large file {resource_file_path}")
            elif resource_file_path.suffix in {".csv", ".xlsx"}:
                try:
                    if resource_file_path.suffix == ".csv":
                        csv_to_rst_table(resource_file_path, output_path)
                    else:
                        excel_to_rst_table(resource_file_path, output_path)
                    print(f"Converted {resource_file_path} to table")
                except UnicodeDecodeError:
                    write_placeholder(resource_file_path, output_path)
                    print(
                        f"Wrote placeholder only for {resource_file_path} as not UTF-8 encoded"
                    )

            else:
                write_placeholder(resource_file_path, output_path)
                print(
                    f"Wrote placeholder only for {resource_file_path} with unknown file extension"
                )


if __name__ == "__main__":
    docs_directory = Path(__file__).parent
    resources_directory = docs_directory.parent / "resources"
    generate_docs_pages_from_resource_files(resources_directory, docs_directory)
