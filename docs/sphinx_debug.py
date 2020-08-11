# Stef Piatek 11th August 2020
# Enable Sphinx stuff to be run under PyCharm debugger.
from pathlib import Path

from sphinx.cmd import build

root_dir = Path(__file__).resolve().parents[1]


def run_sphinx_build(goal, args):
    """Runs sphinx build for TLO, setting the configuration directory"""
    config_dir_settings = ["-c", f"{root_dir}/docs"]
    build.main([*config_dir_settings, "-b", goal, *args])


if __name__ == '__main__':
    run_sphinx_build(goal="html", args=["-E", f"{root_dir}/docs", f"{root_dir}/dist/docs"])
    run_sphinx_build(goal="linkcheck", args=[f"{root_dir}/docs", f"{root_dir}/dist/docs"])


