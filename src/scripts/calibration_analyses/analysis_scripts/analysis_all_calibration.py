import argparse
import glob
import os.path
import zipfile
from pathlib import Path

import analysis_cause_of_death_and_disability_calibrations
import analysis_demography_calibrations
import analysis_hsi_descriptions
import plot_legends

from scripts.calibration_analyses.analysis_scripts import (
    analysis_compare_appt_usage_real_and_simulation,
    plot_appt_use_by_hsi,
)


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Run all the calibration analyses (demographic, epidemiological & descriptions of healthcare system usage) on
    the results of running `long_run_all_diseases.py`."""

    analysis_demography_calibrations.apply(
        results_folder=results_folder, output_folder=output_folder, resourcefilepath=resourcefilepath)

    analysis_cause_of_death_and_disability_calibrations.apply(
        results_folder=results_folder, output_folder=output_folder, resourcefilepath=resourcefilepath)

    analysis_hsi_descriptions.apply(
        results_folder=results_folder, output_folder=output_folder, resourcefilepath=resourcefilepath)

    analysis_compare_appt_usage_real_and_simulation.apply(
        results_folder=results_folder, output_folder=output_folder, resourcefilepath=resourcefilepath)

    plot_appt_use_by_hsi.apply(
        results_folder=results_folder, output_folder=output_folder, resourcefilepath=resourcefilepath)

    # Plot the legends
    plot_legends.apply(
        results_folder=None, output_folder=output_folder, resourcefilepath=resourcefilepath)

    # make html page to present results
    html = "<html><body>"

    for filename in sorted(glob.glob(str(output_folder / "*.png"))):
        basename = os.path.basename(filename)
        html += f"<p style='text-align: center; font-size: 130%'><a href='{basename}'>{basename}</a></p>"
        html += f"<img style='max-width:100%; display:block; margin-left:auto; margin-right:auto' src='{basename}'/>"
        html += "<br><br>"

    with zipfile.ZipFile(output_folder / "images.zip", mode="w") as archive:
        for filename in sorted(glob.glob(str(output_folder / "*.png"))):
            archive.write(filename, os.path.basename(filename))

    html += """<hr><p><a href='images.zip'>images.zip</a> <a href='stdout.txt'>stdout.txt</a>
            <a href='stderr.txt'>stderr.txt</a> <a href='task.txt'>task.txt</a></p>"""
    html += "</body></html>"

    with open(output_folder / "index.html", "w") as output_file:
        output_file.write(html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources')
    )
