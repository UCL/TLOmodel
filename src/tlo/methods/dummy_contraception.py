
import pandas as pd
from tlo import DateOffset, Module, Property, Types, logging
from tlo.events import PopulationScopeEventMixin, RegularEvent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DummyContraceptionModule(Module):
    """Dummy Contraception Module - replaces the contraception module to generate very high levels of pregnancy in the
    population for calibration of maternal/newborn outcomes."""

    INIT_DEPENDENCIES = {"Demography"}
    ALTERNATIVE_TO = {"Contraception"}
    ADDITIONAL_DEPENDENCIES = {
        'Labour', 'PregnancySupervisor', 'HealthSystem'}

    PROPERTIES = {'is_pregnant': Property(Types.BOOL, ""),
                  'date_of_last_pregnancy': Property(Types.DATE, "")
                  }

    def __init__(self, name=None):
        super().__init__(name)

    def read_parameters(self, data_folder):
        pass

    def initialise_population(self, population):
        df = population.props

        reproductive_age_women = df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)
        for woman in reproductive_age_women.loc[reproductive_age_women].index:
            df.at[woman, 'is_pregnant'] = True
            df.at[woman, 'date_of_last_pregnancy'] = self.sim.date
            self.sim.modules['Labour'].set_date_of_labour(woman)
            logger.info(key='pregnancy', data={'mother': woman, 'age': df.at[woman, 'age_years']})

        df.loc[reproductive_age_women.loc[~reproductive_age_women].index, 'is_pregnant'] = False
        df.loc[reproductive_age_women.loc[~reproductive_age_women].index, 'date_of_last_pregnancy'] = pd.NaT

    def initialise_simulation(self, sim):
        sim.schedule_event(DummyPregnancyPoll(self),
                           sim.date + DateOffset(days=0))

    def on_birth(self, mother, child):
        df = self.sim.population.props
        df.at[child, 'is_pregnant'] = False
        df.at[child, 'date_of_last_pregnancy'] = pd.NaT

        df.at[mother, 'is_pregnant'] = False


class DummyPregnancyPoll(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module, ):
        super().__init__(module, frequency=DateOffset(days=1))

    def apply(self, population):
        df = self.sim.population.props

        possible_pregnancy = ((df.sex == 'F') & df.is_alive & ~df.is_pregnant & ~df.la_currently_in_labour &
                              ~df.la_has_had_hysterectomy & (df.age_years > 14) & (df.age_years < 50) &
                              ~df.la_is_postpartum & (df.ps_ectopic_pregnancy == 'none') & ~df.hs_is_inpatient)

        for woman in possible_pregnancy.loc[possible_pregnancy].index:
            df.at[woman, 'is_pregnant'] = True
            df.at[woman, 'date_of_last_pregnancy'] = self.sim.date
            self.sim.modules['Labour'].set_date_of_labour(woman)
            logger.info(key='pregnancy', data={'mother': woman, 'age': df.at[woman, 'age_years']})
