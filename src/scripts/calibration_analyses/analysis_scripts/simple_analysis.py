"""This is the mock-up of a file that does some analysis on a folder has resulted from running a Scenario."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tlo.analysis.utils import extract_results, summarize


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Produce some plots based on the results."""
    print(f"Using results in: {results_folder}\nOutputting results to: {output_folder}")

    res = summarize(
        extract_results(
            results_folder,
            module='tlo.methods.demography',
            key='death',
            custom_generate_series=lambda _df: _df.groupby(by=['sex']).size(),
            do_scaling=True,
        )
    )

    data = pd.read_csv(resourcefilepath / "demography" / "ResourceFile_Deaths_2018Census.csv")
    target_number_of_deaths_per_year = int(
        pd.to_numeric(data['Count'], errors='coerce').dropna().sum()
    )

    fig, ax = plt.subplots()
    res.plot.bar(ax=ax)
    ax.axhline(target_number_of_deaths_per_year, color='r')
    ax.set_title("Number of deaths")
    fig.tight_layout()
    fig.savefig(output_folder / "my_plot.png")
    plt.show()


if __name__ == "__main__":
    """Use files locally on Tim's machine for purpose of testing that `apply` works as expected. This are the results
    of running `long_run_no_diseases.py`."""

    LOCAL_RESULTS_FOLDER = Path(
        "/Users/tbh03/GitHub/TLOmodel/outputs/mini_run/results"
    )
    LOCAL_OUTPUT_FOLDER = Path(
        "/Users/tbh03/GitHub/TLOmodel/outputs/mini_run/output"
    )
    LOCAL_RESOURCEFILEPATH = Path(
        "/Users/tbh03/GitHub/TLOmodel/resources"
    )

    apply(
        results_folder=LOCAL_RESULTS_FOLDER,
        output_folder=LOCAL_OUTPUT_FOLDER,
        resourcefilepath=LOCAL_RESOURCEFILEPATH
    )
