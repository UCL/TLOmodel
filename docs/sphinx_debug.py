# Stef Piatek 11th August 2020
# Enable Sphinx stuff to be run under PyCharm debugger.
from pathlib import Path
from typing import List

from sphinx.cmd import build

root_dir = Path(__file__).resolve().parents[1]


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
    input_dir = f"{root_dir}/docs"
    output_dir = f"{root_dir}/dist/docs"

    run_sphinx_build(goal="html", input_dir=input_dir, output_dir=output_dir, args=["-E"])
    run_sphinx_build(goal="linkcheck", input_dir=input_dir, output_dir=output_dir)

