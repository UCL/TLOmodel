""" This module is used to generate nice documentation.

A typical invocation, as is done in tox.ini, would be:
 > python docs/tlo_methods_rst.py

Within the src/tlo/methods directory, and any directory
structure within that, it parses the Python source files
and uses a mixture of Sphinx auto methods and bespoke
methods to generate the module-specific .rst files (which are
subsequently converted by Sphinx into the HTMl files desired).
e.g. the PARAMETERS and PROPERTIES dictionaries are displayed
as nice tables rather than the raw Python representation of
dictionaries, using bespoke methods defined here."""

import importlib
from pathlib import Path

from tlo.docs import (
    generate_module_dict,
    get_fully_qualified_name,
    get_package_name,
    write_rst_file,
)

if __name__ == '__main__':

    # Add command-line processing here
    # Ideally make these defaults but have command-line options.
    root_dir = Path(__file__).resolve().parents[1]  # e.g. TLOmodel directory.
    # NB If module_directory is set to .../src/tlo/, the trailing slash
    # after tlo is required.
    module_directory = f"{root_dir}/src/tlo/methods"
    rst_directory = f"{root_dir}/docs/reference"

    # Need the trailing slash after tlo - it needs "/tlo/":
    # mydata = generate_module_dict("./src/tlo/")
    mydata = generate_module_dict(module_directory)

    for directory in mydata:  # e.g. .../src/tlo/logging/sublog
        package = get_package_name(directory)  # e.g. "tlo.logging.sublog"
        files = mydata[directory]  # e.g. ["fileA.py", "fileB.py", ...]
        print(f"In directory [{directory}]: files are {files}")
        for f in files:
            # e.g. "tlo.logging.sublog.fileA":
            fqn = get_fully_qualified_name(f, package)
            # print(f"DEBUG: dir: {dir}, package:{package}, f:{f}, fqn:{fqn}")
            # Object creation from string:
            module_obj = importlib.import_module(fqn)
            # print(f"module_obj is {module_obj}")
            write_rst_file(rst_directory, fqn, module_obj)
    print(f"\n\nroot dir is {root_dir}")
