"""Script to automatically mark slow running tests with `pytest.mark.slow` decorator."""


import argparse
import difflib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Set, Tuple, Union

import redbaron


SLOW_MARK_DECORATOR = "pytest.mark.slow"


class TestFunction(NamedTuple):
    module_path: Path
    name: str


class TestMethod(NamedTuple):
    module_path: Path
    class_name: str
    method_name: str


TestNode = Union[TestFunction, TestMethod]


def parse_nodeid_last_part(last_part: str) -> Tuple[str, Optional[str]]:
    match = re.match(r"(.+)\[(.+)\]", last_part)
    if match is not None:
        return match[1], match[2]
    else:
        return last_part, None


def parse_nodeid(nodeid: str) -> TestNode:
    parts = nodeid.split("::")
    if len(parts) == 2:
        module_path, last_part = parts
        name, _ = parse_nodeid_last_part(last_part)
        return TestFunction(Path(module_path), name)
    elif len(parts) == 3:
        module_path, class_name, last_part = parts
        method_name, _ = parse_nodeid_last_part(last_part)
        return TestMethod(Path(module_path), class_name, method_name)
    else:
        msg = f"Test nodeid has unexpected format: {nodeid}"
        raise ValueError(msg)


def parse_test_report(
    json_test_report_path: Path,
    remove_slow_threshold: float,
    add_slow_threshold: float,
) -> Dict[Path, Dict[str, Set[TestNode]]]:
    with open(json_test_report_path, "r") as f:
        test_report = json.load(f)
    tests_to_change_slow_mark_by_module: defaultdict = defaultdict(
        lambda: {"add": set(), "remove": set()}
    )
    for test in test_report["tests"]:
        if test["outcome"] != "passed":
            continue
        test_node = parse_nodeid(test["nodeid"])
        marked_slow = "slow" in test["keywords"]
        call_duration = test["call"]["duration"]
        if marked_slow and call_duration < remove_slow_threshold:
            tests_to_change_slow_mark_by_module[test_node.module_path]["remove"].add(
                test_node
            )
        elif not marked_slow and call_duration > add_slow_threshold:
            tests_to_change_slow_mark_by_module[test_node.module_path]["add"].add(
                test_node
            )
    return dict(tests_to_change_slow_mark_by_module)


def find_function(
    module_fst: redbaron.RedBaron, function_name: str
) -> redbaron.DefNode:
    return module_fst.find("def", lambda node: node.name == function_name)


def find_class_method(
    module_fst: redbaron.RedBaron, class_name: str, method_name: str
) -> redbaron.DefNode:
    class_fst = module_fst.find("class", lambda node: node.name == class_name)
    return class_fst.fund("def", lambda node: node.name == method_name)


def find_decorator(
    function_fst: redbaron.DefNode, decorator_code: str
) -> redbaron.DecoratorNode:
    return function_fst.find(
        "decorator", lambda node: str(node.value) == decorator_code
    )


def add_decorator(function_fst: redbaron.DefNode, decorator_code: str):
    if len(function_fst.decorators) == 0:
        function_fst.decorators = f"@{decorator_code}"
    else:
        function_fst.decorators.append(f"@{decorator_code}")


def remove_mark_from_tests(
    module_fst: redbaron.RedBaron,
    tests_to_remove_mark: Set[TestNode],
    mark_decorator: str,
):
    for test_node in tests_to_remove_mark:
        if isinstance(test_node, TestFunction):
            function_fst = find_function(module_fst, test_node.name)
            decorator_fst = find_decorator(function_fst, mark_decorator)
            function_fst.decorators.remove(decorator_fst)
        else:
            method_fst = find_class_method(
                module_fst, test_node.class_name, test_node.method_name
            )
            decorator_fst = find_decorator(function_fst, mark_decorator)
            method_fst.decorators.remove(decorator_fst)


def add_mark_to_tests(
    module_fst: redbaron.RedBaron, tests_to_add_mark: Set[TestNode], mark_decorator: str
):
    for test_node in tests_to_add_mark:
        if isinstance(test_node, TestFunction):
            function_fst = find_function(module_fst, test_node.name)
            add_decorator(function_fst, mark_decorator)
        else:
            method_fst = find_class_method(
                module_fst, test_node.class_name, test_node.method_name
            )
            add_decorator(method_fst, mark_decorator)


def add_import(module_fst: redbaron.RedBaron, import_statement: str):
    last_top_level_import = module_fst.find_all(
        "import", lambda node: node.parent is module_fst
    )[-1]
    if last_top_level_import is not None:
        last_top_level_import.insert_after(import_statement)
    else:
        if isinstance(module_fst[0], redbaron.Nodes.StringNode):
            module_fst[0].insert_after(import_statement)
        else:
            module_fst[0].insert_before(import_statement)


def update_test_slow_marks(
    tests_to_change_slow_mark_by_module: Dict[Path, Dict[str, Set[TestNode]]],
    show_diff: bool,
):
    for (
        module_path,
        test_nodes_to_change,
    ) in tests_to_change_slow_mark_by_module.items():
        with open(module_path, "r") as source_code:
            module_fst = redbaron.RedBaron(source_code.read())
            original_module_fst = module_fst.copy()
        remove_mark_from_tests(
            module_fst, test_nodes_to_change["remove"], SLOW_MARK_DECORATOR
        )
        add_mark_to_tests(module_fst, test_nodes_to_change["add"], SLOW_MARK_DECORATOR)
        any_marked = (
            module_fst.find(
                "decorator", lambda node: str(node.value) == SLOW_MARK_DECORATOR
            )
            is not None
        )
        pytest_imported = (
            module_fst.find("import", lambda node: "pytest" in node.modules())
            is not None
        )
        if any_marked and not pytest_imported:
            add_import(module_fst, "import pytest")
        if show_diff:
            diff_lines = difflib.unified_diff(
                original_module_fst.dumps().split("\n"),
                module_fst.dumps().split("\n"),
                fromfile=str(module_path),
                tofile=f"Updated {module_path}",
            )
            print("\n".join(diff_lines), end="")
        else:
            with open(module_path, "w") as source_code:
                source_code.write(module_fst.dumps())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Mark slow running tests with pytest.mark.slow")
    parser.add_argument(
        "--json-test-report-path",
        type=Path,
        help="JSON report output from pytest-json-report plugin listing test durations",
    )
    parser.add_argument(
        "--remove-slow-threshold",
        type=float,
        default=9.0,
        help="Threshold in seconds for test duration below which to remove slow marker",
    )
    parser.add_argument(
        "--add-slow-threshold",
        type=float,
        default=11.0,
        help="Threshold in seconds for test duration above which to add slow marker",
    )
    parser.add_argument(
        "--show-diff",
        action="store_true",
        help="Print line-by-line diff of changes to stdout without changing files",
    )
    args = parser.parse_args()
    if not args.json_test_report_path.exists():
        msg = f"No file found at --json-test-report-path={args.json_test_report_path}"
        raise FileNotFoundError(msg)
    # We want a hysteris effect by having remove_slow_threshold < add_slow_threshold
    # so a test with duration close to the thresholds doesn't keep getting marks added
    # and removed due to noise in durations
    if args.remove_slow_threshold > args.add_slow_threshold:
        msg = (
            "Argument --remove-slow-threshold should be less than or equal to "
            "--add-slow-threshold"
        )
        raise ValueError(msg)
    tests_to_change_slow_mark_by_module = parse_test_report(
        args.json_test_report_path, args.remove_slow_threshold, args.add_slow_threshold
    )
    update_test_slow_marks(tests_to_change_slow_mark_by_module, args.show_diff)
