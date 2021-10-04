"""Generate formatted description of health system interaction event details."""

import argparse
import importlib
import inspect
import json
import os.path
import pkgutil
from typing import Any, Iterable, List, Mapping, Set
import warnings
from pathlib import Path

import tlo.methods
from tlo import Date, Module, Simulation
from tlo.dependencies import (
    get_dependencies_and_initialise, get_init_dependencies, get_module_class_map, is_valid_tlo_module_subclass
)
from tlo.methods import alri, contraception, diarrhoea, healthseekingbehaviour, hiv
from tlo.methods.healthsystem import HSI_Event, HSIEventDetails


def is_valid_hsi_event_subclass(obj: Any) -> bool:
    """Whether an object is a *strict* subclass of HSI_Event"""
    return inspect.isclass(obj) and issubclass(obj, HSI_Event) and obj is not HSI_Event


def get_hsi_event_classes_per_module(
    excluded_modules: Set[str],
    zero_module_class_map: Mapping[str, Module],
    multiple_module_class_map: Mapping[str, Module],
) -> Mapping[Module, List[HSI_Event]]:
    """Get details of HSI event classes for each (non-excluded) module in tlo.methods"""
    methods_package_path = os.path.dirname(inspect.getfile(tlo.methods))
    hsi_event_classes_per_module = {}
    for _, module_name, _ in pkgutil.iter_modules([methods_package_path]):
        if module_name in excluded_modules:
            # Module does not need processing therefore skip
            continue
        module = importlib.import_module(f'tlo.methods.{module_name}')
        tlo_module_classes = [
            obj for _, obj in inspect.getmembers(module)
            if is_valid_tlo_module_subclass(obj, excluded_modules)
        ]
        hsi_event_classes = [
            obj for _, obj in inspect.getmembers(module)
            if is_valid_hsi_event_subclass(obj) and inspect.getmodule(obj) is module
        ]
        if len(hsi_event_classes) == 0:
            # No HSI events defined so skip
            continue
        if len(tlo_module_classes) == 1:
            tlo_module_class = tlo_module_classes[0]
        elif len(tlo_module_classes) == 0 and module_name in zero_module_class_map:
            tlo_module_class = zero_module_class_map[module_name]
        elif len(tlo_module_classes) > 1 and module_name in multiple_module_class_map:
            tlo_module_class = multiple_module_class_map[module_name]
        else:
            raise RuntimeError(
                f'Module {module_name} defines HSI events but contains '
                f'{len(tlo_module_classes)} TLO Module classes and no specific '
                'exceptions have been defined in `zero_module_class_map` or '
                '`multiple_module_class_map`.'
            )
        hsi_event_classes_per_module[tlo_module_class] = hsi_event_classes
    return hsi_event_classes_per_module


def get_details_of_defined_hsi_events(
    excluded_modules: Set[str],
    zero_module_class_map: Mapping[str, Module],
    multiple_module_class_map: Mapping[str, Module],
    init_population: int,
    resource_file_path: str,
) -> Set[HSIEventDetails]:
    """Get details of all HSI events defined in `tlo.methods`.

    :param excluded_modules: Set of tlo.methods module names to not search for HSI
        events in.
    :param zero_module_class_map: Map from ``tlo.methods`` module name to ``Module ``
        subclass to use for HSI events in module for modules with no ``Module``
        subclasses defined in module.
    :param multiple_module_class_map: Map from tlo.methods module name to ``Module``
        subclass to use for HSI events in module for modules with multiple ``Module``
        subclasses defined.
    """
    hsi_event_classes_per_module = get_hsi_event_classes_per_module(
        excluded_modules, zero_module_class_map, multiple_module_class_map
    )
    # Setting show_progress_bar=True hacky way to disable all log output to stdout
    sim = Simulation(start_date=Date(2010, 1, 1), seed=1, show_progress_bar=True)
    # Register modules and their dependencies
    sim.register(
        *get_dependencies_and_initialise(
            *hsi_event_classes_per_module.keys(),
            module_class_map=get_module_class_map(set()),
            # Only select initialisation dependencies as we will not actually run
            # simulation
            get_dependencies=get_init_dependencies,
            resourcefilepath=resource_file_path
        ),
        # As we only select initialisation dependencies, disable check that additional
        # dependencies are present
        check_all_dependencies=False
    )
    # Initialise a small population for events that access population dataframe
    sim.make_initial_population(n=init_population)
    details_of_defined_hsi_events = set()
    for tlo_module_class, hsi_event_classes in hsi_event_classes_per_module.items():
        module = sim.modules[tlo_module_class.__name__]
        for hsi_event_class in hsi_event_classes:
            signature = inspect.signature(hsi_event_class)
            dummy_kwargs = {
                param_name:
                # Use default value if specified
                param.default if param.default is not param.empty
                # If target person_id set to 0 as should always be present in population
                else 0 if param_name == 'person_id'
                # Otherwise use None value to indicate only known at runtime
                # We could replace this with a unittest.Mock instance if the constructor
                # raises an exception on trying to use a None value for the argument
                else None
                for param_name, param in signature.parameters.items()
                if param_name != 'module'
            }
            arguments = signature.bind(module=module, **dummy_kwargs)
            try:
                hsi_event = hsi_event_class(*arguments.args, **arguments.kwargs)
            except NotImplementedError:
                # If method called in HSI event constructor is not implemented assume
                # this is an abstract base class and so does not need documenting
                pass
            else:
                details_of_defined_hsi_events.add(
                    HSIEventDetails(
                        event_name=type(hsi_event).__name__,
                        module_name=tlo_module_class.__name__,
                        treatment_id=hsi_event.TREATMENT_ID,
                        facility_level=hsi_event.ACCEPTED_FACILITY_LEVEL,
                        appt_footprint=tuple(hsi_event.EXPECTED_APPT_FOOTPRINT)
                    )
                )
    return details_of_defined_hsi_events


def sort_hsi_event_details(
    set_of_hsi_event_details: Iterable[HSIEventDetails]
) -> List[HSIEventDetails]:
    """Hierarchically sort set of HSI event details."""
    return sorted(
        set_of_hsi_event_details,
        key=lambda event_details: (
            event_details.module_name,
            event_details.treatment_id,
            event_details.facility_level,
            event_details.appt_footprint,
        )
    )


def _rst_table_row(column_values):
    return '   * - ' + '\n     - '.join(column_values) + '\n'


def _md_table_row(column_values):
    return '| ' + ' | '.join(column_values) + ' |\n'


def _rst_table_header(column_names, title=''):
    header = (
        f'.. list-table:: {title}\n'
        '   :widths: auto\n'
        '   :header-rows: 1\n\n'
    )
    header += _rst_table_row(column_names)
    return header


def _md_table_header(column_names, title=''):
    header = f'*{title}*\n\n' if title != '' else ''
    header += _md_table_row(column_names)
    header += _md_table_row('-' * len(name) for name in column_names)
    return header


_formatters = {
    'rst': {
        'heading': lambda text, level: f'{text}\n{len(text) * "#*=-^"[level - 1]}\n\n',
        'inline_code': lambda code: f'``{code}``',
        'table_header': _rst_table_header,
        'table_row': _rst_table_row,
        'list_item': lambda item_text: f'* {item_text}\n',
    },
    'md': {
        'heading': lambda text, level: f'{level * "#"} {text}\n\n',
        'inline_code': lambda code: f'`{code}`',
        'table_header': _md_table_header,
        'table_row': _md_table_row,
        'list_item': lambda item_text: f'  * {item_text}\n',
    }
}


def _format_facility_level(facility_level):
    return '?' if facility_level is None else str(facility_level)


def _format_appt_footprint(appt_footprint, inline_code_formatter):
    return ', '.join(f'{inline_code_formatter(a)}' for a in appt_footprint)


def _format_treatment_id(treatment_id, module_name, inline_code_formatter):
    prefixes = [
        f"{module_name}_",
        f"HSI_{module_name}_",
        f"{module_name[0].lower()}{module_name[1:]}_"
    ]
    for prefix in prefixes:
        if treatment_id.startswith(prefix):
            treatment_id = treatment_id[len(prefix):]
            break
    return inline_code_formatter(treatment_id)


def format_hsi_event_details_as_table(
    hsi_event_details: Iterable[HSIEventDetails],
    text_format: str = 'rst'
) -> str:
    """Format HSI event details into a table."""
    formatters = _formatters[text_format]
    table_string = formatters['table_header'](
        ['Module', 'Event', 'Treatment', 'Facility level', 'Appointment footprint'],
        'Health system interaction events'
    )
    for event_details in hsi_event_details:
        table_string += formatters['table_row'](
            [
                formatters['inline_code'](event_details.module_name),
                formatters['inline_code'](event_details.event_name),
                _format_treatment_id(
                    event_details.treatment_id,
                    event_details.module_name,
                    formatters['inline_code']
                ),
                _format_facility_level(event_details.facility_level),
                _format_appt_footprint(
                    event_details.appt_footprint, formatters['inline_code']
                )
            ]
        )
    return table_string


def format_hsi_event_details_as_list(
    hsi_event_details: Iterable[HSIEventDetails],
    text_format: str = 'rst'
) -> str:
    """Format HSI event details into per module lists."""
    formatters = _formatters[text_format]
    event_details_by_module = {}
    for event_details in hsi_event_details:
        if event_details.module_name not in event_details_by_module:
            event_details_by_module[event_details.module_name] = []
        event_details_by_module[event_details.module_name].append(event_details)
    list_string = ""
    for module_name, module_event_details in event_details_by_module.items():
        list_string += formatters['heading'](module_name, 2)
        for event_details in module_event_details:
            appt_footprint_string = _format_appt_footprint(
                event_details.appt_footprint, formatters['inline_code']
            )
            treatment_id_string = _format_treatment_id(
                event_details.treatment_id,
                event_details.module_name,
                formatters['inline_code']
            )
            list_string += formatters['list_item'](
                f'{treatment_id_string} at '
                f'facility level {_format_facility_level(event_details.facility_level)}'
                f' with appointment footprint: {appt_footprint_string}.'
            )
        list_string += '\n'
    return list_string


def merge_hsi_event_details(
    inspect_hsi_event_details: Iterable[HSIEventDetails],
    run_hsi_event_details: Iterable[HSIEventDetails],
) -> Set[HSIEventDetails]:
    """Merge HSI event details collected using `inspect` and from simulation run."""
    # Create set of event details from simulation run, excluding facility level from
    # entries to allow matching event details from inspect of `tlo.methods` for which
    # facility level is not known (indicated by value of None) to be matched
    def without_facility_level(event_details):
        return (
            event_details.event_name,
            event_details.module_name,
            event_details.treatment_id,
            event_details.appt_footprint
        )
    run_hsi_event_details_without_facility_level = {
        without_facility_level(event_details) for event_details in run_hsi_event_details
    }
    # Create merged set of HSI event details by forming union of run_hsi_event_details
    # and inspect_hsi_event_details minus those details in inspect_hsi_event_details
    # with facility level unknown (set to None) for which there is a corresponding
    # entry in run_hsi_event_details with known facility level
    merged_hsi_event_details = set(run_hsi_event_details)
    for event_details in inspect_hsi_event_details:
        if not (
            event_details.facility_level is None
            and without_facility_level(event_details)
            in run_hsi_event_details_without_facility_level
        ):
            merged_hsi_event_details.add(event_details)
    return merged_hsi_event_details


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate formatted description of HSI events."
    )
    parser.add_argument(
        '--json-file',
        type=Path,
        help=(
            "JSON file containing previously saved event details (e.g. recorded in "
            "a simulation run). If omitted then a set of HSI events is extracted by "
            "inspecting `tlo.methods`. If `--merge-json-details` flag is set then "
            "the set of details in JSON file and from inspecting `tlo.methods` are "
            "merged to produce the formatted output."
        ),
        default=None
    )
    parser.add_argument(
        '--merge-json-details',
        action='store_true',
        help=(
            "If set then the event details in the JSON file specified by `--json-file` "
            "(which must be specified if this flag is set) are merged with the event "
            "details extracted by inspecting `tlo.methods`. Otherwise only the event "
            "details in the JSON file are used to generate the formatted output."
        )

    )
    parser.add_argument(
        '--output-file',
        type=Path,
        help=(
            "File to write generated formatted description of HSI events to. If "
            "omitted then description is printed to standard output."
        ),
        default=None
    )
    parser.add_argument(
        '--output-format',
        choices=('rst-list', 'rst-table', 'md-list', 'md-table', 'json'),
        help="Format to use for output.",
        default='rst-list'
    )
    args = parser.parse_args()
    if args.json_file is None and args.merge_json_details:
        parser.error("--json-file must be specified if --merge-json-files used.")
    warnings.simplefilter('ignore')
    if args.json_file is not None:
        with open(args.json_file, 'r') as f:
            hsi_event_details = json.load(f)
        json_hsi_event_details = set(
            # JSON serializes tuples to lists therefore need to reformat
            HSIEventDetails(*event_details[:-1], tuple(event_details[-1]))
            for event_details in hsi_event_details
        )
        print(f'HSI events loaded from JSON file {args.json_file}.')
    if args.json_file is None or args.merge_json_details:
        # Set of tlo.methods module names to not search for HSI events in
        excluded_modules = {'mockitis', 'chronicsyndrome', 'skeleton'}
        # Map from tlo.methods module name to Module subclass to use for HSI events in
        # module for modules with no Module subclasses defined in module
        zero_module_class_map = {
            # Assume generic first appointments generated by HealthSeekingBehaviour
            # In reality HSI_GenericEmergencyFirstApptAtFacilityLevel1 may also be
            # generate by Labour or PregnancySupervisor currently
            'hsi_generic_first_appts': healthseekingbehaviour.HealthSeekingBehaviour
        }
        # Map from tlo.methods module name to Module subclass to use for HSI events in
        # module for modules with multiple Module subclasses defined
        multiple_module_class_map = {
            'hiv': hiv.Hiv,
            'alri': alri.Alri,
            'diarrhoea': diarrhoea.Diarrhoea,
            'contraception': contraception.Contraception,
        }
        print('Getting details of defined HSI events by inspecting tlo.methods...')
        inspect_hsi_event_details = get_details_of_defined_hsi_events(
            excluded_modules=excluded_modules,
            zero_module_class_map=zero_module_class_map,
            multiple_module_class_map=multiple_module_class_map,
            init_population=10,
            resource_file_path='resources'
        )
        print('...done.\n')
    if args.merge_json_details:
        hsi_event_details = merge_hsi_event_details(
            inspect_hsi_event_details, json_hsi_event_details
        )
    elif args.json_file is not None:
        hsi_event_details = json_hsi_event_details
    else:
        hsi_event_details = inspect_hsi_event_details
    sorted_hsi_event_details = sort_hsi_event_details(hsi_event_details)
    formatters = {
        'rst-list': lambda details: format_hsi_event_details_as_list(details, 'rst'),
        'rst-table': lambda details: format_hsi_event_details_as_table(details, 'rst'),
        'md-list': lambda details: format_hsi_event_details_as_list(details, 'md'),
        'md-table': lambda details: format_hsi_event_details_as_table(details, 'md'),
        'json': lambda details: json.dumps(details, indent=4)
    }
    formatted_details = formatters[args.output_format](sorted_hsi_event_details)
    if args.output_file is not None:
        with open(args.output_file, 'w') as f:
            f.write(formatted_details)
        print(f'Output written to {args.output_file}.')
    else:
        print('Output:\n\n' + formatted_details)
