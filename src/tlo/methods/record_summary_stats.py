"""
Record Summary Statistics Module

This module collects and logs the number of individuals in each age group for each disease
"""
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Types, logging
from tlo.analysis.utils import get_counts_by_sex_and_age_group
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
        'do_checks': Parameter(
            Types.BOOL,
            'Whether to check that the collected statistics are valid'
        )
    }

    PROPERTIES = {}

    def read_parameters(self, resourcefilepath: Optional[Path] = None):
        """Read parameters for the module."""
        self.load_parameters_from_dataframe(pd.read_csv(resourcefilepath / 'ResourceFile_RecordSummaryStats.csv'))

        if not isinstance(self.parameters['do_checks'], bool):
            raise ValueError(f"Invalid value for do_checks - it must be a bool: {self.parameters['do_checks']}")

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

    def _check_stats(self, d: Dict) -> None:
        """
        Validate that a statistic value is in an acceptable format for logging.

        This should be a dict of the form {statistic: value}, where statistic is a string
        and value can be a numerical value (including numpy types) or a dict (or nested dict).

        Returns nothing, but raises Error if any problem
        """

        # Check that input is a dict
        if not isinstance(d, dict):
            raise TypeError(f"Expected dict, got {type(d).__name__}")

        # Check that all keys are strings
        for key in d.keys():
            if not isinstance(key, str):
                raise TypeError(f"All keys must be strings, found key of type {type(key).__name__}")

        # Check that all values are valid types (recursively for nested dicts)
        def _check_value(value, path=""):
            # Acceptable scalar types
            if isinstance(value, (int, float, str, bool)):
                return

            # Numpy scalar types
            if isinstance(value, np.generic):
                return

            # Check if it has an item() method (numpy scalars)
            if hasattr(value, 'item') and callable(value.item):
                try:
                    value.item()
                    return
                except (ValueError, AttributeError):
                    pass

            # Dict types - recursively validate
            if isinstance(value, dict):
                for k, v in value.items():
                    if not isinstance(k, str):
                        raise TypeError(f"All dict keys must be strings at path '{path}', found {type(k).__name__}")
                    _check_value(v, path=f"{path}.{k}" if path else k)
                return

            # If we get here, the type is not acceptable
            raise TypeError(
                f"Invalid value type {type(value).__name__} at path '{path}'. "
                f"Expected int, float, str, bool, numpy scalar, or dict."
            )

        # Validate all values in the dict
        for key, value in d.items():
            _check_value(value, path=key)

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

                if self.parameters['do_checks']:
                    self._check_stats(stats)

                all_stats[module.name].update(stats)

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
