"""Script to automatically mark slow running tests with `pytest.mark.slow` decorator."""


import argparse
import difflib
import json
import re
import warnings
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
    tests_to_keep_slow_mark_by_module: defaultdict = defaultdict(set)
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
        elif marked_slow:
            tests_to_keep_slow_mark_by_module[test_node.module_path].add(test_node)
    # Parameterized tests may have different call durations for different parameters
    # however slow mark applies to all parameters, therefore if any tests appear in
    # both set of tests to keep slow mark and test to remove slow mark (corresponding
    # to runs of same test with different parameters) we remove them from the set of
    # tests to remove slow mark
    for (
        module_path,
        test_nodes_to_change,
    ) in tests_to_change_slow_mark_by_module.items():
        test_nodes_to_change["remove"].difference_update(
            tests_to_keep_slow_mark_by_module[module_path]
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


def remove_decorator(
    function_fst: redbaron.DefNode, decorator_fst: redbaron.DecoratorNode
):
    # Need to remove both decorator and associated end line node so we find index of
    # decorator and pop it and next node (which should be end line node) rather than
    # use remove method of decorators proxy list directly
    decorator_index = function_fst.decorators.node_list.index(decorator_fst)
    popped_decorator_fst = function_fst.decorators.node_list.pop(decorator_index)
    endline_fst = function_fst.decorators.node_list.pop(decorator_index)
    if popped_decorator_fst is not decorator_fst or not isinstance(
        endline_fst, redbaron.EndlNode
    ):
        msg = (
            f"Removed {popped_decorator_fst} and {endline_fst} when expecting "
            f"{decorator_fst} and end line node."
        )
        raise RuntimeError(msg)


def remove_mark_from_tests(
    module_fst: redbaron.RedBaron,
    tests_to_remove_mark: Set[TestNode],
    mark_decorator: str,
):
    for test_node in tests_to_remove_mark:
        if isinstance(test_node, TestFunction):
            function_fst = find_function(module_fst, test_node.name)
        else:
            function_fst = find_class_method(
                module_fst, test_node.class_name, test_node.method_name
            )
        decorator_fst = find_decorator(function_fst, mark_decorator)
        if decorator_fst is None:
            msg = (
                f"Test {test_node} unexpectedly does not have a decorator "
                f"{mark_decorator} - this suggests you may be using a JSON test report "
                "generated using a different version of tests code."
            )
            warnings.warn(msg, stacklevel=2)
        else:
            remove_decorator(function_fst, decorator_fst)


def add_mark_to_tests(
    module_fst: redbaron.RedBaron, tests_to_add_mark: Set[TestNode], mark_decorator: str
):
    for test_node in tests_to_add_mark:
        if isinstance(test_node, TestFunction):
            function_fst = find_function(module_fst, test_node.name)
        else:
            function_fst = find_class_method(
                module_fst, test_node.class_name, test_node.method_name
            )
        if find_decorator(function_fst, mark_decorator) is not None:
            msg = (
                f"Test {test_node} unexpectedly already has a decorator "
                f"{mark_decorator} - this suggests you may be using a JSON test report "
                "generated using a different version of tests code."
            )
            warnings.warn(msg, stacklevel=2)
        else:
            add_decorator(function_fst, mark_decorator)


def add_import(module_fst: redbaron.RedBaron, module_name: str):
    last_top_level_import = module_fst.find_all(
        "import", lambda node: node.parent is module_fst
    )[-1]
    import_statement = f"import {module_name}"
    if last_top_level_import is not None:
        last_top_level_import.insert_after(import_statement)
    else:
        if isinstance(module_fst[0], redbaron.Nodes.StringNode):
            module_fst[0].insert_after(import_statement)
        else:
            module_fst[0].insert_before(import_statement)


def remove_import(module_fst: redbaron.RedBaron, module_name: str):
    import_fst = module_fst.find("import", lambda node: module_name in node.modules())
    if len(import_fst.modules()) > 1:
        import_fst.remove(module_name)
    else:
        module_fst.remove(import_fst)


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
            add_import(module_fst, "pytest")
        elif not any_marked and pytest_imported:
            pytest_references = module_fst.find_all("name", "pytest")
            if (
                len(pytest_references) == 1
                and pytest_references[0].parent_find("import") is not None
            ):
                remove_import(module_fst, "pytest")
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
    # We want a hysteresis effect by having remove_slow_threshold < add_slow_threshold
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
