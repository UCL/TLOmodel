"""
Record Summary Statistics Module

This module collects and logs the number of individuals in each age group for each disease
"""
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

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


class RecordSummaryStats(Module):
    """
    Module to collect summary statistics from each disease module.
    """

    def __init__(self, name=None):
        super().__init__(name)
        self._registered_modules = []

    INIT_DEPENDENCIES = {'Demography'}

    METADATA = {}

    PARAMETERS = {
        'logging_frequency': Parameter(
            Types.STRING,
            'Frequency at which to collect numbers: "day", "month", or "year"'
        ),
    }

    PROPERTIES = {}

    def read_parameters(self, resourcefilepath: Optional[Path] = None):
        """Read parameters for the module."""
        self.load_parameters_from_dataframe(pd.read_csv(resourcefilepath / 'ResourceFile_RecordSummaryStats.csv'))

    def initialise_population(self, population):
        pass

    def initialise_simulation(self, sim):
        """Do before simulation starts:
        1) Collect modules that will report disease numbers
        2) Schedule the logging event based on the configured frequency (if any models will do reporting)
        """

        # 1) Collect modules that will report disease numbers
        self._registered_modules = [
            module for module in self.sim.modules.values()
            if Metadata.REPORTS_DISEASE_NUMBERS in module.METADATA
        ]

        if not self._registered_modules:
            logger.warning(
                key='warning',
                data='DiseasenumbersLogger registered but no disease modules report numbers'
            )
            return

        # Check that all registered disease modules have the report_summary_stats() function
        for module in self._registered_modules:
            assert getattr(module, 'report_summary_stats', None) and \
                   callable(module.report_summary_stats), \
                f'Module {module.name} declares REPORTS_DISEASE_NUMBERS but does not have ' \
                'a callable function "report_summary_stats"'

        # 2) Schedule the logging event based on configured frequency
        freq_map = {
            'day': DateOffset(days=1),
            'month': DateOffset(months=1),
            'year': DateOffset(years=1),
        }
        if self.parameters["logging_frequency"] in freq_map:
            frequency = freq_map[self.parameters['logging_frequency']]
        else:
            raise ValueError(f"Invalid logging frequency: {self.parameters['logging_frequency']}")

        # Schedule first event at start of simulation
        sim.schedule_event(
            DiseasenumbersLoggingEvent(self, frequency=frequency),
            sim.date
        )

    def on_birth(self, mother_id, child_id):
        pass

    def _validate_stat_value(self, module_name: str, key: str, value: Any):
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

    def collect_stats_and_write_to_log(self):
        """Write disease numbers to the log."""
        # Get population dataframe
        df = self.sim.population.props

        # Dictionary to collect all stats, of the form {module: {stat: content}}
        all_stats = defaultdict(dict)

        # Add basic population numbers
        all_stats['Population'] = {
            'total': sum(df.is_alive),
            'by_age_and_sex': get_counts_by_sex_and_age_group(df, 'is_alive')
        }

        # Collect numbers from each registered disease module
        for module in self._registered_modules:
            try:
                stats: Dict = module.report_summary_stats()

                # Add module name as prefix to all numbers
                for key, value in stats.items():
                    # Validate that the value is in an acceptable format
                    validated_value = self._validate_stat_value(module.name, key, value)
                    if validated_value is not None:
                        all_stats[module.name].update({key: validated_value})

            except Exception as e:
                logger.warning(
                    key='error',
                    data=f'Error collecting numbers from {module.name}: {str(e)}'
                )

        # Write to log: each module is assigned it's own key
        for module_name, stats_from_module in all_stats.items():
            logger.info(
                key=module_name,
                description=f'Summary statistics from module {module_name}',
                data=stats_from_module
            )


class DiseasenumbersLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    Event that collects disease numbers from registered modules and logs them.
    Runs at regular intervals (daily, monthly, or yearly) based on module configuration.
    """

    def __init__(self, module, frequency: DateOffset):
        super().__init__(module, frequency=frequency, priority=Priority.END_OF_DAY)

    def apply(self, population):
        """Collect numbers from all registered modules and log them."""

        # Call the method to collect the statistics and do the logging
        self.module.collect_stats_and_write_to_log()
