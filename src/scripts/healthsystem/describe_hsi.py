"""Generate formatted description of health system interaction event details."""

import argparse
import importlib
import inspect
import json
import os.path
import pkgutil
import warnings
from pathlib import Path

import tlo.methods
from tlo import Date, Module, Simulation
from tlo.methods import (
    bladder_cancer,
    breast_cancer,
    cardio_metabolic_disorders,
    care_of_women_during_pregnancy,
    contraception,
    demography,
    depression,
    diarrhoea,
    dx_algorithm_adult,
    dx_algorithm_child,
    enhanced_lifestyle,
    epi,
    epilepsy,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    malaria,
    measles,
    newborn_outcomes,
    oesophagealcancer,
    other_adult_cancers,
    postnatal_supervisor,
    pregnancy_supervisor,
    prostate_cancer,
    symptommanager,
)
from tlo.methods.healthsystem import HSI_Event, HSIEventDetails

# Dependencies of each Module subclass on other Module subclasses
module_dependencies = {
    bladder_cancer.BladderCancer: (
        demography.Demography,
        enhanced_lifestyle.Lifestyle,
        healthsystem.HealthSystem,
        symptommanager.SymptomManager,
    ),
    breast_cancer.BreastCancer: (
        demography.Demography,
        healthsystem.HealthSystem,
        symptommanager.SymptomManager,
    ),
    cardio_metabolic_disorders.CardioMetabolicDisorders: (
        demography.Demography,
        enhanced_lifestyle.Lifestyle,
        healthsystem.HealthSystem,
        symptommanager.SymptomManager,
    ),
    care_of_women_during_pregnancy.CareOfWomenDuringPregnancy: (
        demography.Demography,
        enhanced_lifestyle.Lifestyle,
        healthsystem.HealthSystem,
        contraception.Contraception,
        labour.Labour,
        pregnancy_supervisor.PregnancySupervisor,
    ),
    contraception.Contraception: (
        demography.Demography,
        labour.Labour,
    ),
    demography.Demography: (),
    depression.Depression: (
        demography.Demography,
        enhanced_lifestyle.Lifestyle,
        contraception.Contraception,
        healthsystem.HealthSystem,
        symptommanager.SymptomManager,
    ),
    diarrhoea.Diarrhoea: (
        demography.Demography,
        enhanced_lifestyle.Lifestyle,
        healthsystem.HealthSystem,
        symptommanager.SymptomManager,
    ),
    dx_algorithm_adult.DxAlgorithmAdult: (
        healthsystem.HealthSystem,
        symptommanager.SymptomManager,
    ),
    dx_algorithm_child.DxAlgorithmChild: (
        healthsystem.HealthSystem,
        symptommanager.SymptomManager,
    ),
    enhanced_lifestyle.Lifestyle: {
        demography.Demography,
    },
    epi.Epi: (
        demography.Demography,
        healthsystem.HealthSystem,
    ),
    epilepsy.Epilepsy: (
        demography.Demography,
        healthsystem.HealthSystem,
        symptommanager.SymptomManager,
    ),
    healthburden.HealthBurden: (
        demography.Demography,
    ),
    healthsystem.HealthSystem: (
        demography.Demography,
    ),
    healthseekingbehaviour.HealthSeekingBehaviour: (
        demography.Demography,
        enhanced_lifestyle.Lifestyle,
        healthsystem.HealthSystem,
        symptommanager.SymptomManager,
    ),
    hiv.Hiv: (
        demography.Demography,
        enhanced_lifestyle.Lifestyle,
        healthsystem.HealthSystem,
        symptommanager.SymptomManager,
    ),
    labour.Labour: (
        demography.Demography,
        enhanced_lifestyle.Lifestyle,
        healthsystem.HealthSystem,
        contraception.Contraception,
        care_of_women_during_pregnancy.CareOfWomenDuringPregnancy,
        pregnancy_supervisor.PregnancySupervisor,
        postnatal_supervisor.PostnatalSupervisor,
    ),
    malaria.Malaria: (
        demography.Demography,
        contraception.Contraception,
        healthsystem.HealthSystem,
        symptommanager.SymptomManager,
    ),
    measles.Measles: (
        demography.Demography,
        healthsystem.HealthSystem,
        symptommanager.SymptomManager,
    ),
    newborn_outcomes.NewbornOutcomes: (
        demography.Demography,
        enhanced_lifestyle.Lifestyle,
        healthsystem.HealthSystem,
        symptommanager.SymptomManager,
        labour.Labour,
        pregnancy_supervisor.PregnancySupervisor,
    ),
    oesophagealcancer.OesophagealCancer: (
        demography.Demography,
        enhanced_lifestyle.Lifestyle,
        healthsystem.HealthSystem,
        symptommanager.SymptomManager,
    ),
    other_adult_cancers.OtherAdultCancer: (
        demography.Demography,
        healthsystem.HealthSystem,
        symptommanager.SymptomManager,
    ),
    pregnancy_supervisor.PregnancySupervisor: (
        demography.Demography,
        enhanced_lifestyle.Lifestyle,
        contraception.Contraception,
        labour.Labour,
        care_of_women_during_pregnancy.CareOfWomenDuringPregnancy,
    ),
    postnatal_supervisor.PostnatalSupervisor: (
        demography.Demography,
        enhanced_lifestyle.Lifestyle,
        healthsystem.HealthSystem,
        labour.Labour,
        care_of_women_during_pregnancy.CareOfWomenDuringPregnancy,
    ),
    prostate_cancer.ProstateCancer: (
        demography.Demography,
        healthsystem.HealthSystem,
        symptommanager.SymptomManager,
    ),
    symptommanager.SymptomManager: (
        demography.Demography,
    ),
}


def init_simulation_and_module(module_class, resourcefilepath='resources'):
    """Register module and dependences in simulation and initialise population."""
    # Setting show_progress_bar=True hacky way to disable all log output to stdout
    sim = Simulation(start_date=Date(2010, 1, 1), seed=1, show_progress_bar=True)
    module = module_class(resourcefilepath=resourcefilepath)
    sim.register(
        *(
            dependency_class(resourcefilepath=resourcefilepath)
            for dependency_class in module_dependencies[module_class]
        ),
        module
    )
    # Initialise a small population for events that access population dataframe
    sim.make_initial_population(n=1000)
    return sim, module


def is_valid_tlo_module_class(obj):
    """Whether an object is a *strict* subclass of Module"""
    return inspect.isclass(obj) and issubclass(obj, Module) and obj is not Module


def is_valid_hsi_event_class(obj):
    """Whether an object is a *strict* subclass of HSI_Event"""
    return inspect.isclass(obj) and issubclass(obj, HSI_Event) and obj is not HSI_Event


def get_details_of_defined_hsi_events():
    """Get details of all HSI events defined in `tlo.methods`."""
    methods_package_path = os.path.dirname(inspect.getfile(tlo.methods))
    excluded_modules = {'mockitis', 'chronicsyndrome', 'skeleton'}
    details_of_defined_hsi_events = set()
    for _, module_name, _ in pkgutil.iter_modules([methods_package_path]):
        if module_name in excluded_modules:
            # Module does not need processing therefore skip
            continue
        module = importlib.import_module(f'tlo.methods.{module_name}')
        tlo_module_classes = [
            obj for _, obj in inspect.getmembers(module)
            if is_valid_tlo_module_class(obj)
        ]
        hsi_event_classes = [
            obj for _, obj in inspect.getmembers(module)
            if is_valid_hsi_event_class(obj) and inspect.getmodule(obj) is module
        ]
        if len(hsi_event_classes) == 0:
            # No HSI events defined so skip
            continue
        if len(tlo_module_classes) == 1:
            tlo_module_class = tlo_module_classes[0]
        elif len(tlo_module_classes) == 0 and module_name == 'hsi_generic_first_appts':
            # Assume generic first appointments generated by HealthSeekingBehaviour
            # In reality HSI_GenericEmergencyFirstApptAtFacilityLevel1 may also be
            # generate by Labour or PregnancySupervisor currently
            tlo_module_class = healthseekingbehaviour.HealthSeekingBehaviour
        elif len(tlo_module_classes) == 2 and module_name == 'hiv':
            # Use full HIV module rather than dummy module
            tlo_module_class = hiv.Hiv
        else:
            raise RuntimeError(
                f'Module {module_name} defines HSI events but contains '
                f'{len(tlo_module_classes)} TLO Module classes and no specific '
                f'exception rule has been defined.'
            )
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
            sim, module = init_simulation_and_module(tlo_module_class)
            arguments = signature.bind(module=module, **dummy_kwargs)
            try:
                hsi_event = hsi_event_class(*arguments.args, **arguments.kwargs)
                details_of_defined_hsi_events.add(
                    HSIEventDetails(
                        event_name=type(hsi_event).__name__,
                        module_name=tlo_module_class.__name__,
                        treatment_id=hsi_event.TREATMENT_ID,
                        facility_level=hsi_event.ACCEPTED_FACILITY_LEVEL,
                        appt_footprint=tuple(hsi_event.EXPECTED_APPT_FOOTPRINT)
                    )
                )
            except NotImplementedError:
                # If method called in HSI event constructor is not implemented assume
                # this is an abstract base class and so does not need documenting
                pass
    return details_of_defined_hsi_events


def sort_hsi_event_details(set_of_hsi_event_details):
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


def format_hsi_event_details_as_table(hsi_event_details, text_format='rst'):
    """Format HSI event details into a table."""
    formatters = _formatters[text_format]
    table_string = formatters['table_header'](
        ['Module', 'Treatment ID', 'Facility level', 'Appointment footprint'],
        'Health system interaction events'
    )
    for event_details in hsi_event_details:
        table_string += formatters['table_row'](
            [
                formatters['inline_code'](event_details.module_name),
                formatters['inline_code'](event_details.treatment_id),
                _format_facility_level(event_details.facility_level),
                _format_appt_footprint(
                    event_details.appt_footprint, formatters['inline_code']
                )
            ]
        )
    return table_string


def format_hsi_event_details_as_list(hsi_event_details, text_format='rst'):
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
            list_string += formatters['list_item'](
                f'{formatters["inline_code"](event_details.treatment_id)}: '
                f'facility level {_format_facility_level(event_details.facility_level)}'
                f' with appointment footprint {appt_footprint_string}.'
            )
        list_string += '\n'
    return list_string


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate formatted description of HSI events."
    )
    parser.add_argument(
        '--json-file',
        type=Path,
        help=(
            "JSON file containing event details recorded in simulation run. If omitted "
            "then a list of all HSI events defined in `tlo.methods` is used instead."
        ),
        default=None
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
    warnings.simplefilter('ignore')
    if args.json_file is not None:
        with open(args.json_file, 'r') as f:
            hsi_event_details = json.load(f)
        hsi_event_details = [
            HSIEventDetails(*event_details) for event_details in hsi_event_details
        ]
        print(f'HSI events loaded from JSON file {args.json_file}.')
    else:
        print('Getting details of all defined HSI events...')
        hsi_event_details = get_details_of_defined_hsi_events()
        print('...done.\n')
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
