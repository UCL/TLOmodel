"""
A script to parse the data sources spreadsheet and generate an .rst file
"""
import textwrap
from pathlib import Path

import pandas as pd
from pandas import CategoricalDtype

dagger = "\\ :sup:`†`\\ "
data_file = "data_sources.csv"
data_out = "_data_sources.rst"


def heading(string, level):
    length = len(string)
    if level == 1:
        return textwrap.dedent(f"""\
        {'*' * length}
        {string}
        {'*' * length}
        """)
    if level == 2:
        return textwrap.dedent(f"""\
        {string}
        {'#' * length}
        """)
    if level == 3:
        return textwrap.dedent(f"""\
        {string}
        {'*' * length}
        """)

    raise Exception("Can't handle level '{level}' heading")


def generate_content(_data):
    current_category = None
    module_name = None

    lines = list()

    for index, row in _data.iterrows():
        if row.Category is not current_category:
            current_category = row.Category
            lines.append(heading(current_category, 2))

        if row["Module Name"] is not module_name:
            module_name = row["Module Name"]
            lines.append(heading(module_name, 3))

        lines.append(
            f"#. {dagger if row['Malawi-specific'] else ''}"
            f"{row.Citation} "
            f"(for {row['Year Relevant to Data']}{'; ' + row.URL if not pd.isna(row.URL) else ''})\n"
        )

    return lines


def sort_by_category_and_module_name(_data):
    """Sort the imported list of data_sources by Category (specific order) and module name (alphabetical)."""

    # define ordered category
    _data['Category'] = _data['Category'].astype(
        CategoricalDtype(
            ['Core Functions',
             'Healthcare System',
             'Contraception, Pregnancy and Labour',
             'Conditions of Early Childhood',
             'Infectious Diseases',
             'Cardiometabolic Disorders',
             'Cancers',
             'Other Non-Communicable Conditions',
             'Injuries'],
            ordered=True)
    )
    assert not _data['Category'].isna().any()

    return _data.sort_values(['Category', 'Module Name'])


if __name__ == "__main__":
    directory = Path(__file__).resolve().parents[0]
    data = pd.read_csv(directory / data_file)
    print(f"Found {len(data)} lines in {directory / data_file}.")
    data = sort_by_category_and_module_name(data)
    content = generate_content(data)
    print(f"Writing data sources to {directory / data_out}.")
    with open(directory / data_out, 'w') as outfile:
        outfile.write('\n'.join(content))
