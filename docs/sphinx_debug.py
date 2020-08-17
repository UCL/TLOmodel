# Stef Piatek 11th August 2020
# Enable Sphinx stuff to be run under PyCharm debugger.
from pathlib import Path
from typing import List

from sphinx.cmd import build
from sphinx.ext import apidoc

root_dir = Path(__file__).resolve().parents[1]


def run_sphinx_apidoc(input_dir, output_dir):
    """
    Generate sphinx sources from module
    :param input_dir: module path
    :param output_dir: output sources
    """
    apidoc.main(["-e", "-f", "-o", output_dir, input_dir])


def run_sphinx_build(goal, input_dir, output_dir, args: List[str] = None):
    """
    Run sphinx build for TLO, setting the configuration directory

    :param goal: sphinx build goal
    :param input_dir: input directory
    :param output_dir: output directory
    :param args: optional arguments to be passed to sphinx build
    """
    config_dir_settings = ["-c", f"{root_dir}/docs"]
    if args is None:
        args = []
    build.main([*config_dir_settings, "-b", goal, *args, input_dir, output_dir])


if __name__ == '__main__':
    module_path = f"{root_dir}/src/tlo"
    output_sources = f"{root_dir}/docs/reference"
    sphinx_sources = f"{root_dir}/docs"
    output_html = f"{root_dir}/dist/docs"

    run_sphinx_apidoc(input_dir=module_path, output_dir=output_sources)
    run_sphinx_build(goal="html", input_dir=sphinx_sources, output_dir=output_html, args=["-E"])
    run_sphinx_build(goal="linkcheck", input_dir=sphinx_sources, output_dir=output_html)

