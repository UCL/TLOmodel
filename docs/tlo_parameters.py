"""Create listings of model parameters in tabular format"""

import argparse
from collections import defaultdict
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import TypeAlias, get_args
import numpy
import pandas

import tlo
from tlo import Date, Module, Simulation
from tlo.methods import fullmodel
from tlo.analysis.utils import get_parameters_for_status_quo


_TYPE_TO_DESCRIPTION = {
    bool: "Boolean",
    pandas.Categorical: "Categorical",
    pandas.DataFrame: "Dataframe",
    pandas.Timestamp: "Date",
    defaultdict: "Dictionary",
    dict: "Dictionary",
    int: "Integer",
    numpy.int64: "Integer",
    list: "List",
    float: "Real",
    numpy.float64: "Real",
    pandas.Series: "Series",
    set: "Set",
    str: "String",
}


ScalarParameterValue: TypeAlias = float | int | bool | str | numpy.generic | Date
StructuredParameterValue: TypeAlias = (
    dict | list | tuple | set | pandas.Series | pandas.DataFrame
)
ParameterValue: TypeAlias = (
    ScalarParameterValue | pandas.Categorical | StructuredParameterValue
)

_SCALAR_TYPES = get_args(ScalarParameterValue)


ModuleParameterTablesDict: TypeAlias = dict[str, dict[str, pandas.DataFrame]]
ModuleStructuredParametersDict: TypeAlias = dict[
    str, dict[str, pandas.DataFrame | dict[str, pandas.DataFrame]]
]


def structured_value_to_dataframe(
    value: StructuredParameterValue,
) -> pandas.DataFrame | dict[str, pandas.DataFrame]:
    if isinstance(value, (list, tuple, set)):
        return pandas.DataFrame.from_records([value], index=["Value"])
    elif isinstance(value, pandas.Series):
        return pandas.DataFrame(value)
    elif isinstance(value, pandas.DataFrame):
        return value
    elif isinstance(value, dict):
        if all(isinstance(v, _SCALAR_TYPES) for v in value.values()):
            return pandas.DataFrame(value, index=["Value"])
        else:
            return {k: structured_value_to_dataframe(v) for k, v in value.items()}
    else:
        raise ValueError(
            f"Unrecognized structured value type {type(value)} for value {value}"
        )


def get_parameter_tables(
    modules: Iterable[Module],
    overriden_parameters: dict[str, dict[str, ParameterValue]],
    excluded_modules: set[str],
    excluded_parameters: dict[str, set[str]],
    escape_characters: callable,
    format_internal_link: callable,
    max_inline_parameter_length: int = 10,
) -> tuple[ModuleParameterTablesDict, ModuleStructuredParametersDict]:
    module_parameter_tables = {}
    module_structured_parameters = {}
    for module in sorted(modules, key=lambda m: m.name):
        if module.name in excluded_modules:
            continue
        parameter_records = []
        module_structured_parameters[module.name] = {}
        module_excluded_parameters = excluded_parameters.get(module.name, set())
        for parameter_name, parameter in module.PARAMETERS.items():
            if parameter_name in module_excluded_parameters:
                continue
            if (
                module.name in overriden_parameters
                and parameter_name in overriden_parameters[module.name]
            ):
                value = overriden_parameters[module.name][parameter_name]
            else:
                value = module.parameters.get(parameter_name)
            if value is None:
                continue
            record = {
                "Name": escape_characters(parameter_name),
                "Description": escape_characters(parameter.description),
                "Type": _TYPE_TO_DESCRIPTION[type(value)],
            }
            if (
                isinstance(value, _SCALAR_TYPES)
                or isinstance(value, (list, set, tuple))
                and len(value) < max_inline_parameter_length
            ):
                record["Value"] = str(value)
            elif isinstance(value, pandas.Categorical):
                assert len(value) == 1
                record["Value"] = str(value[0])
            else:
                record["Value"] = format_internal_link(
                    "...", parameter_id(module.name, parameter_name)
                )
                module_structured_parameters[module.name][parameter_name] = (
                    structured_value_to_dataframe(value)
                )
            parameter_records.append(record)
        module_parameter_tables[module.name] = pandas.DataFrame.from_records(
            parameter_records,
        )
    return module_parameter_tables, module_structured_parameters


def parameter_id(module_name, parameter_name):
    return f"{module_name}-{parameter_name}"


def dataframe_as_table(dataframe, rows_threshold=None, tablefmt="pipe"):
    summarize = rows_threshold is not None and len(dataframe) > rows_threshold
    if summarize:
        original_rows = len(dataframe)
        dataframe = dataframe[1:rows_threshold]
    table_string = dataframe.to_markdown(index=False, tablefmt=tablefmt)
    if summarize:
        table_string += (
            f"\n\n*Only first {rows_threshold} rows of {original_rows} are shown.*\n"
        )
    return table_string


def md_anchor_tag(id: str) -> str:
    return f"<a id='{id}'></a>"


def md_list_item(text: str, bullet: str = "-", indent_level: int = 0) -> str:
    return "  " * indent_level + f"{bullet} {text}\n"


def md_hyperlink(link_text: str, url: str) -> str:
    return f"[{link_text}]({url})"


def md_internal_link_with_backlink_anchor(
    link_text: str, id: str, suffix: str = "backlink"
):
    return md_anchor_tag(f"{id}-{suffix}") + md_hyperlink(link_text, f"#{id}")


def rst_internal_link(link_text: str, id: str):
    return f":ref:`{link_text}<{id}>`"


def escape_rst_markup_characters(text: str):
    return text.replace("_", "\_").replace("*", "\*")


def md_anchor_and_backlink(id: str, suffix: str = "backlink"):
    return md_anchor_tag(id) + md_hyperlink("↩", f"#{id}-{suffix}")


def md_table_of_contents(module_names):
    return "\n".join(
        [
            md_list_item(
                md_internal_link_with_backlink_anchor(module_name, module_name.lower())
            )
            for module_name in module_names
        ]
    )


def rst_table_of_contents(_module_names):
    return ".. contents::\n   :local:\n   :depth: 1\n   :backlinks: entry\n\n"


def md_header(text: str, level: int) -> str:
    return ("#" * level if level > 0 else "%") + " " + text + "\n\n"


def rst_header(title: str, level: int = 0) -> str:
    separator_character = '*=-^"'[level]
    line = separator_character * len(title)
    return (line + "\n" if level == 0 else "") + title + "\n" + line + "\n\n"


def md_module_header(module_name):
    return md_header(f"{module_name} " + md_anchor_and_backlink(module_name.lower()), 1)


def rst_module_header(module_name):
    return rst_header(module_name, 1)


def md_structured_parameter_header(parameter_name, module_name):
    return md_header(
        f"{parameter_name} "
        + md_anchor_and_backlink(parameter_id(module_name, parameter_name)),
        2,
    )


def rst_structured_parameter_header(parameter_name, module_name):
    return f".. _{parameter_id(module_name, parameter_name)}:\n\n" + rst_header(
        parameter_name, 2
    )


_formatters = {
    ".md": {
        "header": md_header,
        "table_of_contents": md_table_of_contents,
        "module_header": md_module_header,
        "structured_parameter_header": md_structured_parameter_header,
        "dataframe_as_table": partial(dataframe_as_table, tablefmt="pipe"),
        "internal_link": md_internal_link_with_backlink_anchor,
        "character_escaper": lambda x: x,
    },
    ".rst": {
        "header": rst_header,
        "table_of_contents": rst_table_of_contents,
        "module_header": rst_module_header,
        "structured_parameter_header": rst_structured_parameter_header,
        "dataframe_as_table": partial(dataframe_as_table, tablefmt="grid"),
        "internal_link": rst_internal_link,
        "character_escaper": escape_rst_markup_characters,
    },
}


def write_parameters_file(
    output_file_path: Path,
    module_parameter_tables: ModuleParameterTablesDict,
    module_structured_parameters: ModuleStructuredParametersDict,
    summarization_rows_threshold: int = 10,
) -> None:
    formatter = _formatters[output_file_path.suffix]
    with output_file_path.open("w") as output_file:
        output_file.write(formatter["header"]("Parameters", 0))
        output_file.write("Default parameter values used in simulations.\n\n")
        output_file.write(
            formatter["table_of_contents"](module_parameter_tables.keys())
        )
        output_file.write("\n")
        for module_name, parameter_table in module_parameter_tables.items():
            output_file.write(formatter["module_header"](module_name))
            output_file.write(formatter["dataframe_as_table"](parameter_table))
            output_file.write("\n\n")
            for (
                parameter_name,
                structured_parameter,
            ) in module_structured_parameters[module_name].items():
                output_file.write(
                    formatter["structured_parameter_header"](
                        parameter_name, module_name
                    )
                )
                if isinstance(structured_parameter, dict):
                    for key, dataframe in structured_parameter.items():
                        output_file.write(formatter["header"](key, 3))
                        output_file.write(
                            formatter["dataframe_as_table"](
                                dataframe, summarization_rows_threshold
                            )
                        )
                        output_file.write("\n\n")
                else:
                    output_file.write(
                        formatter["dataframe_as_table"](
                            structured_parameter, summarization_rows_threshold
                        )
                    )
                    output_file.write("\n")
                output_file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "resource_file_path",
        type=Path,
        default=Path(tlo.__file__).parent.parent.parent / "resources",
        help="Path to resource directory",
    )
    parser.add_argument(
        "output_file_path", type=Path, help="Path to file to write tables to"
    )
    args = parser.parse_args()
    simulation = Simulation(
        start_date=Date(2010, 1, 1), seed=1234, log_config={"suppress_stdout": True}
    )
    status_quo_parameters = get_parameters_for_status_quo()
    simulation.register(*fullmodel.fullmodel(args.resource_file_path))
    internal_link_formatter = _formatters[args.output_file_path.suffix]["internal_link"]
    character_escaper = _formatters[args.output_file_path.suffix]["character_escaper"]
    module_parameter_tables, module_structured_parameters = get_parameter_tables(
        simulation.modules.values(),
        status_quo_parameters,
        {"HealthBurden", "Wasting"},
        {"Demography": {"gbd_causes_of_death_data"}, "Tb": {"who_incidence_estimates"}},
        character_escaper,
        internal_link_formatter,
    )
    write_parameters_file(
        args.output_file_path, module_parameter_tables, module_structured_parameters
    )
