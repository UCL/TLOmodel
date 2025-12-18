import os
from pathlib import Path

import pandas as pd

from tlo import (
    DAYS_IN_YEAR,
    Date,
    DateOffset,
    Module,
    Parameter,
    Property,
    Simulation,
    Types,
    logging,
)
from tlo.analysis.utils import parse_log_file, unflatten_flattened_multi_index_in_logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import Metadata, demography, enhanced_lifestyle, healthburden
from tlo.methods.causes import Cause
from tlo.methods.demography import InstantaneousDeath
from tlo.methods.fullmodel import fullmodel
from tlo.methods.hsi_event import HSI_Event
from tlo.methods.hsi_generic_first_appts import GenericFirstAppointmentsMixin
from tlo.methods.symptommanager import Symptom

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
outputpath = Path("./outputs/")

start_date = Date(2010, 1, 1)
end_date = Date(2011, 1, 12)

popsize = 1_000


# ---------------------------------------------------
# Even simpler Dummy Disease - only infection. For checking logging
# ---------------------------------------------------
class DummyDisease(Module):
    """This is a dummy disease"""

    INIT_DEPENDENCIES = {'Demography'}

    OPTIONAL_INIT_DEPENDENCIES = {'HealthBurden'}

    CAUSES_OF_DISABILITY = {
        'Dummy': Cause(label='Dummy')
    }

    # Declare Metadata
    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_HEALTHBURDEN
    }

    PROPERTIES = {
        'dm_is_infected': Property(
            Types.BOOL, 'Current status of DummyDisease'),
    }

    def __init__(self, name=None):
        super().__init__(name)
        self.stored_prevalence = {}

    def read_parameters(self, data_folder):
        """Read in parameters and do the registration of this module and its symptoms"""
        p = self.parameters
        p['average_prevalence'] = 0.5

    def initialise_population(self, population):
        # randomly selected some individuals as infected
        df = population.props
        df.loc[df.is_alive, 'dm_is_infected'] = self.rng.random_sample(size=df.is_alive.sum()) < self.parameters[
            'average_prevalence']

    def initialise_simulation(self, sim):
        event = DummyDiseaseEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=1))
        sim.schedule_event(DummyDiseaseLoggingEvent(self), sim.date + DateOffset(months=1))

        self.store_and_record_current_prevalence()


    def on_birth(self, mother_id, child_id):
        df = self.sim.population.props
        df.at[child_id, 'dm_is_infected'] = False

    def report_prevalence(self):
        df = self.sim.population.props
        infected_total = df.loc[df.is_alive, 'dm_is_infected'].sum()
        proportion_infected = infected_total / sum(df.is_alive)
        return {'DummyDisease_proportion_infected': proportion_infected}

    def report_daly_values(self):
        df = self.sim.population.props
        dw = pd.Series(index=df.index[df.is_alive], data=0.0)
        dw.loc[df.dm_is_infected] = 0.8
        return dw

    def store_and_record_current_prevalence(self):
        df = self.sim.population.props
        infected_total = df.loc[df.is_alive, 'dm_is_infected'].sum()
        proportion_infected = infected_total / df.is_alive.sum()
        self.stored_prevalence[self.sim.date] = proportion_infected

class DummyDiseaseEvent(RegularEvent, PopulationScopeEventMixin):
    """
    This event is occurring regularly at one monthly intervals and controls the infection process
    and onset of symptoms of DummyDisease.
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        assert isinstance(module, DummyDisease)

    def apply(self, population):
        # randomly select some individuals to be infected
        df = population.props
        df.loc[df.is_alive, 'dm_is_infected'] = self.module.rng.random_sample(size=df.is_alive.sum()) < self.module.parameters['average_prevalence']

class DummyDiseaseLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """Produce a summmary of the numbers of people with respect to their 'DummyDisease status'
        """
        super().__init__(module, frequency=DateOffset(months=1))
        assert isinstance(module, DummyDisease)

    def apply(self, population):
        # get some summary statistics
        self.module.store_and_record_current_prevalence()



def test_basic_mechanics_with_dummy_disease(tmpdir, seed):
    start_date = Date(2010, 1, 1)
    end_date = Date(2011, 1, 1)

    sim = Simulation(start_date=start_date,
                     seed=0,
                     resourcefilepath=resourcefilepath,
                     log_config={'filename': 'tmp', 'directory': tmpdir})
    sim.register(
        demography.Demography(),
        healthburden.HealthBurden(),
        dummydisease:= DummyDisease(),
        enhanced_lifestyle.Lifestyle(),
        sort_modules=False,
        check_all_dependencies=False
    )

    sim.make_initial_population(n=popsize)
    sim.modules['HealthBurden'].parameters['logging_frequency_prevalence'] = 'month'
    sim.simulate(end_date=end_date)
    output = parse_log_file(sim.log_filepath)

    # Get results from the 'prevalence' summary logger
    healthburden_logger = unflatten_flattened_multi_index_in_logging(output['tlo.methods.healthburden']['prevalence_of_diseases'].set_index('date'))
    prev_in_healthburden_logger = healthburden_logger[('DummyDisease', 'DummyDisease_proportion_infected')]

    # Get results from the actual module
    prevalence_from_actual_module = pd.Series(dummydisease.stored_prevalence)

    # Confirm they give the same result
    pd.testing.assert_series_equal(prev_in_healthburden_logger, prevalence_from_actual_module, check_names=False)


def test_run_with_real_diseases(tmpdir, seed):
    """Check that everything runs when using the full model and daily logging cadence."""

    sim = Simulation(start_date=start_date,
                     seed=seed,
                     resourcefilepath=resourcefilepath,
                     log_config={'filename': 'test_log', 'directory': outputpath}
                     )
    sim.register(*fullmodel(use_simplified_births=False))
    sim.make_initial_population(n=popsize)
    sim.modules['HealthBurden'].parameters['logging_frequency_prevalence'] = 'day'
    sim.simulate(end_date=end_date)
    output = parse_log_file(sim.log_filepath)

    healthburden_logger = unflatten_flattened_multi_index_in_logging(
        output['tlo.methods.healthburden']['prevalence_of_diseases'].set_index('date'))

    assert isinstance(healthburden_logger, pd.DataFrame)
