"""
Disease numbers Module

This module collects and logs the number of individuals in each age group for each disease
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from tlo import DateOffset, Module, Parameter, Types, logging
from tlo.analysis.utils import (
    flatten_multi_index_series_into_dict_for_logging,
    get_counts_by_sex_and_age_group,
)
from tlo.events import PopulationScopeEventMixin, Priority, RegularEvent
from tlo.methods import Metadata

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DiseaseNumbers(Module):
    """
    Module to collect the number of individuals with each disease.
    The dict can contain:
    - Simple counts: {'statistic_name': value}
    - Age/sex stratified: {'statistic_name': pd.Series with multi-index (sex, age_range)}
    """

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # Instance variables
        self.registered_modules = []
        self._months_written_to_log = []

    INIT_DEPENDENCIES = {'Demography'}

    METADATA = {
        Metadata.USES_HEALTHSYSTEM,
    }

    PARAMETERS = {
        'logging_frequency': Parameter(
            Types.STRING,
            'Frequency at which to collect numbers: "day", "month", or "year"'
        ),
    }

    PROPERTIES = {}

    def read_parameters(self, resourcefilepath: Optional[Path] = None):
        """Read parameters for the module."""
        # Set default logging frequency
        self.parameters['logging_frequency'] = 'month'

    def initialise_population(self, population):
        """Nothing needed here."""
        pass

    def initialise_simulation(self, sim):
        """Do before simulation starts:
        1) Collect modules that will report disease numbers
        2) Schedule the logging event based on configured frequency
        """

        # 1) Collect modules that will report disease numbers
        self.registered_modules = [
            module for module in self.sim.modules.values()
            if (Metadata.REPORTS_DISEASE_NUMBERS in module.METADATA and
                hasattr(module, 'report_disease_numbers') and
                callable(module.report_disease_numbers))
        ]

        if not self.registered_modules:
            logger.warning(
                key='warning',
                data='DiseasenumbersLogger registered but no disease modules report numbers'
            )
            return

        # Check that all registered disease modules have the report_disease_numbers() function
        for module in self.registered_modules:
            assert getattr(module, 'report_disease_numbers', None) and \
                   callable(module.report_disease_numbers), \
                f'Module {module.name} declares REPORTS_DISEASE_NUMBERS but does not have ' \
                'a callable function "report_disease_numbers"'

        # 2) Schedule the logging event based on configured frequency
        freq_map = {
            'day': DateOffset(days=1),
            'month': DateOffset(months=1),
            'year': DateOffset(years=1),
        }

        frequency = freq_map.get(
            self.parameters['logging_frequency'],
            DateOffset(months=1)  # default to monthly
        )

        # Schedule first event at start of simulation
        sim.schedule_event(
            DiseasenumbersLoggingEvent(self, frequency=frequency),
            sim.date
        )

    def on_birth(self, mother_id, child_id):
        """Nothing needed here."""
        pass

    def on_simulation_end(self):
        """Write to the log anything that has not already been logged (i.e., if simulation terminating mid-way
        through a logging period when the WriteToLog event has not run)."""
        self.write_to_log()

    def _validate_stat_value(self, module_name: str, key: str, value):
        """
        Validate that a statistic value is in an acceptable format for logging.

        Acceptable formats:
        - Scalar values (int, float, str, bool)
        - pd.Series with a MultiIndex (for age/sex stratified data)
        - dict with scalar values or nested dicts with scalar values

        Returns the validated value, or None if invalid (with a warning logged).
        """
        # Scalar values are fine
        if isinstance(value, (int, float, str, bool, type(None))):
            return value

        # numpy scalar types
        if hasattr(value, 'item') and callable(value.item):
            try:
                return value.item()
            except (ValueError, AttributeError):
                pass

        # pd.Series with MultiIndex is acceptable (will be flattened later)
        if isinstance(value, pd.Series):
            if isinstance(value.index, pd.MultiIndex):
                return value
            else:
                # Single-level index Series - convert to dict
                return value.to_dict()

        # Dict is acceptable if all values are scalars or nested dicts with scalars
        if isinstance(value, dict):
            validated_dict = {}
            for k, v in value.items():
                if isinstance(v, (int, float, str, bool, type(None))):
                    validated_dict[k] = v
                elif hasattr(v, 'item') and callable(v.item):
                    # numpy scalar
                    try:
                        validated_dict[k] = v.item()
                    except (ValueError, AttributeError):
                        validated_dict[k] = v
                elif isinstance(v, dict):
                    # Nested dict - validate recursively
                    nested_valid = self._validate_stat_value(module_name, f"{key}.{k}", v)
                    if nested_valid is not None:
                        validated_dict[k] = nested_valid
                else:
                    logger.warning(
                        key='warning',
                        data=f'Module {module_name} returned invalid nested value type '
                             f'{type(v).__name__} for key {key}.{k}'
                    )
            return validated_dict

        # DataFrame is NOT acceptable - this is likely the source of the error
        if isinstance(value, pd.DataFrame):
            logger.warning(
                key='warning',
                data=f'Module {module_name} returned a DataFrame for key "{key}". '
                     f'DataFrames are not supported. Please return a dict or pd.Series instead.'
            )
            return None

        # Unknown type - warn and skip
        logger.warning(
            key='warning',
            data=f'Module {module_name} returned unsupported type {type(value).__name__} '
                 f'for key "{key}". Expected scalar, dict, or pd.Series with MultiIndex.'
        )
        return None

    def write_to_log(self):
        """Write disease numbers to the log.
        N.B. This is called at the end of the simulation as well as at regular intervals, so we need to check that
        the current period is not being written to the log more than once."""

        # Create a unique identifier for this logging period
        if self.parameters['logging_frequency'] == 'day':
            period_id = (self.sim.date.year, self.sim.date.month, self.sim.date.day)
        elif self.parameters['logging_frequency'] == 'month':
            period_id = (self.sim.date.year, self.sim.date.month)
        else:  # year
            period_id = (self.sim.date.year,)

        if period_id in self._months_written_to_log:
            return  # Skip if this period has already been logged

        # Get population dataframe
        df = self.sim.population.props

        # Dictionary to collect all numbers
        all_numbers = {}

        # Add basic population numbers
        all_numbers[('Population', 'total')] = sum(df.is_alive)
        all_numbers[('Population', 'by_age_and_sex')] = \
            get_counts_by_sex_and_age_group(df, 'is_alive')

        # Collect numbers from each registered disease module
        for module in self.registered_modules:
            try:
                stats = module.report_disease_numbers()

                if not isinstance(stats, dict):
                    logger.warning(
                        key='warning',
                        data=f'Module {module.name} returned non-dict from report_disease_numbers(). '
                             f'Got {type(stats).__name__} instead.'
                    )
                    continue

                # Add module name as prefix to all numbers
                for key, value in stats.items():
                    # Validate that the value is in an acceptable format
                    validated_value = self._validate_stat_value(module.name, key, value)
                    if validated_value is not None:
                        all_numbers[(module.name, key)] = validated_value

            except Exception as e:
                logger.warning(
                    key='error',
                    data=f'Error collecting numbers from {module.name}: {str(e)}'
                )

        # Convert to multi-index series and log
        multi_index_series = pd.Series(
            all_numbers,
            index=pd.MultiIndex.from_tuples(
                all_numbers.keys(),
                names=['Module', 'Statistic']
            )
        )

        logger.info(
            key='disease_numbers',
            description='Disease prevalence and incidence numbers from all registered modules',
            data=flatten_multi_index_series_into_dict_for_logging(multi_index_series)
        )

        # Mark this period as logged
        self._months_written_to_log.append(period_id)


class DiseasenumbersLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    Event that collects disease numbers from registered modules and logs them.
    Runs at regular intervals (daily, monthly, or yearly) based on module configuration.
    """

    def __init__(self, module, frequency: DateOffset):
        super().__init__(module, frequency=frequency, priority=Priority.END_OF_DAY)

    def apply(self, population):
        """Collect numbers from all registered modules and log them."""
        # Do nothing if no disease modules are registered
        if not self.module.registered_modules:
            return

        # Call the write_to_log method to handle the logging
        self.module.write_to_log()
