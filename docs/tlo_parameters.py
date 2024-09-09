"""Create listings of model parameters in tabular format"""

import argparse
from collections import defaultdict
from collections.abc import Iterable
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
                "Name": parameter_name,
                "Description": parameter.description,
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
                record["Value"] = internal_link_with_backlink_anchor(
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


def markdown_table_or_html_summary(dataframe, rows_threshold=10):
    if len(dataframe) > rows_threshold:
        return dataframe._repr_html_()
    else:
        return dataframe.to_markdown(index=False)


def anchor_tag(id: str) -> str:
    return f"<a id='{id}'></a>"


def title(text: str) -> str:
    return f"% {text}\n\n"


def heading(text: str, level: int) -> str:
    return "#" * level + " " + text + "\n\n"


def list_item(text: str, bullet: str = "-", indent_level: int = 0) -> str:
    return "  " * indent_level + f"{bullet} {text}\n"


def hyperlink(link_text: str, url: str) -> str:
    return f"[{link_text}]({url})"


def internal_link_with_backlink_anchor(
    link_text: str, id: str, suffix: str = "backlink"
):
    return anchor_tag(f"{id}-{suffix}") + hyperlink(link_text, f"#{id}")


def anchor_and_backlink(id: str, suffix: str = "backlink"):
    return anchor_tag(id) + hyperlink("↩", f"#{id}-{suffix}")


def write_parameters_markdown_file(
    output_file_path: Path,
    module_parameter_tables: ModuleParameterTablesDict,
    module_structured_parameters: ModuleStructuredParametersDict,
) -> None:
    with output_file_path.open("w") as output_file:
        output_file.write(title("Parameters"))
        for module_name in module_parameter_tables.keys():
            output_file.write(
                list_item(
                    internal_link_with_backlink_anchor(module_name, module_name.lower())
                )
            )
        output_file.write("\n")
        for module_name, parameter_table in module_parameter_tables.items():
            output_file.write(
                heading(f"{module_name} " + anchor_and_backlink(module_name.lower()), 2)
            )
            output_file.write(parameter_table.to_markdown(index=False))
            output_file.write("\n\n")
            for (
                parameter_name,
                structured_parameter,
            ) in module_structured_parameters[module_name].items():
                output_file.write(
                    heading(
                        f"{parameter_name} "
                        + anchor_and_backlink(
                            parameter_id(module_name, parameter_name)
                        ),
                        3,
                    )
                )
                if isinstance(structured_parameter, dict):
                    for key, dataframe in structured_parameter.items():
                        output_file.write(heading(key, 4))
                        output_file.write(markdown_table_or_html_summary(dataframe))
                        output_file.write("\n")
                else:
                    output_file.write(
                        markdown_table_or_html_summary(structured_parameter)
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
    module_parameter_tables, module_structured_parameters = get_parameter_tables(
        simulation.modules.values(),
        status_quo_parameters,
        {"HealthBurden", "Wasting"},
        {"Demography": {"gbd_causes_of_death_data"}, "Tb": {"who_incidence_estimates"}}
    )
    write_parameters_markdown_file(
        args.output_file_path, module_parameter_tables, module_structured_parameters
    )
