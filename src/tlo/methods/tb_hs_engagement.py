"""
A skeleton template for disease methods.
"""
import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, PopulationScopeEventMixin, RegularEvent, IndividualScopeEventMixin


class health_system_tb(Module):
    """ routinely tests proportion of the population and
    determines availability of ART for HIV+ dependent on UNAIDS coverage estimates
    """

    def __init__(self, name=None):
        super().__init__(name)
        self.store = {'Time': [], 'Number_tested_tb': []}

    PARAMETERS = {
        'tb_testing_coverage': Parameter(Types.REAL, 'proportion of population tested'),
        'prop_smear_positive': Parameter(Types.REAL, 'proportion of TB cases smear positive'),
        'prop_smear_positive_hiv': Parameter(Types.REAL, 'proportion of HIV/TB cases smear positive'),
        'testing_prob_xpert': Parameter(Types.REAL, 'probability of individual receiving xpert test')
        'prop_xpert_positive': Parameter(Types.REAL, 'proportion active tb cases tested with xpert with positive results')
    }

    PROPERTIES = {
        'tb_ever_tested': Property(Types.BOOL, 'ever had a tb test'),
        'tb_smear_test': Property(Types.BOOL, 'ever had a tb smear test'),
        'tb_result_smear_test': Property(Types.BOOL, 'result from tb smear test'),
        'tb_date_smear_test': Property(Types.DATE, 'date of tb smear test'),
        'tb_xpert_test': Property(Types.BOOL, 'ever had a tb Xpert test'),
        'tb_result_xpert_test': Property(Types.BOOL, 'result from tb Xpert test'),
        'tb_date_xpert_test': Property(Types.DATE, 'date of tb Xpert test'),
        'tb_diagnosed': Property(Types.BOOL, 'active tb and tested')
    }

    def read_parameters(self, data_folder):
        params = self.parameters
        params['tb_testing_coverage'] = 0.1  # dummy value
        params['prop_smear_positive'] = 0.8
        params['prop_smear_positive_hiv'] = 0.5
        params['testing_prob_xpert'] = 0.7
        params['prop_xpert_positive'] = 0.5

    def initialise_population(self, population):
        df = population.props

        df['tb_ever_tested'] = False  # default: no individuals tested
        df['tb_smear_test'] = False
        df['tb_result_smear_test'] = False
        df['tb_date_smear_test'] = pd.NaT
        df['tb_xpert_test'] = False
        df['tb_result_xpert_test'] = False
        df['tb_date_xpert_test'] = pd.NaT
        df['tb_diagnosed'] = False

    def initialise_simulation(self, sim):
        sim.schedule_event(TbTestingEvent(self), sim.date + DateOffset(months=12))

        # add an event to log to screen
        sim.schedule_event(TbHealthSystemLoggingEvent(self), sim.date + DateOffset(months=1))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.
        """
        df = self.sim.population.props

        df.at[child_id, 'tb_ever_tested'] = False  # default: no individuals tested
        df.at[child_id, 'tb_smear_test'] = False
        df.at[child_id, 'tb_result_smear_test'] = False
        df.at[child_id, 'tb_date_smear_test'] = pd.NaT
        df.at[child_id, 'tb_xpert_test'] = False
        df.at[child_id, 'tb_result_xpert_test'] = False
        df.at[child_id, 'tb_date_xpert_test'] = pd.NaT
        df.at[child_id, 'tb_diagnosed'] = False


class TbTestingEvent(RegularEvent, PopulationScopeEventMixin):
    """ Testing for TB
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=12))  # every 12 months
        # make sure any rates are annual if frequency of event is annual

    def apply(self, population):
        params = self.module.parameters
        now = self.sim.date
        df = population.props

        # get a list of random numbers between 0 and 1 for the whole population
        random_draw = self.sim.rng.random_sample(size=len(df))

        # probability of TB smear test
        # can be repeat tested
        # delay of at least 2 months after Xpert test before repeat smear test
        testing_index = df.index[(random_draw < params['tb_testing_coverage']) & df.is_alive & (
                ((now - df.tb_date_xpert_test) / np.timedelta64(1, 'M')) > 2)]
        df.loc[testing_index, 'tb_ever_tested'] = True
        df.loc[testing_index, 'tb_smear_test'] = True
        df.loc[testing_index, 'tb_date_smear_test'] = now

        # 80% of smear tested active cases will be diagnosed
        # this is lower for HIV+ (higher prop of extrapulmonary tb
        diagnosed_idx = df[(df.tb_date_smear_test == now) & df.is_alive & (df.has_tb == 'Active') & ~df.has_hiv].sample(frac=params['prop_smear_positive']).index
        diagnosed_idx_hiv = df[(df.tb_date_smear_test == now) & df.is_alive & (df.has_tb == 'Active') & df.has_hiv].sample(frac=params['prop_smear_positive_hiv']).index

        df.loc[diagnosed_idx | diagnosed_idx_hiv, 'result_smear_test'] = True
        df.loc[diagnosed_idx | diagnosed_idx_hiv, 'tb_diagnosed'] = True

        # remaining 20% of active cases referred for xpert testing with some delay
        # also some true negatives may have follow-up testing
        # schedule xpert testing at future date
        # random draw approx 2 months?
        undiagnosed_idx = df.index[(df.tb_date_smear_test == now) & df.is_alive & ~df.tb_diagnosed]

        for person in undiagnosed_idx:
            refer_xpert = tbXpertTest(self.module, individual_id=person)
            referral_time = np.random.normal(loc=(2/12), scale=(1/12), size=1)
            referral_time_yrs = pd.to_timedelta(referral_time, unit='y')
            future_referral_time = now + referral_time_yrs
            self.sim.schedule_event(refer_xpert, future_referral_time)


class tbXpertTest(Event, IndividualScopeEventMixin):
    """ Xpert test for people with negative smear result
    """

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        params = self.module.parameters
        df = self.sim.population.props
        now = self.sim.date

        if df.at[individual_id.is_alive & ~individual_id.tb_diagnosed]:
            # probability of receiving xpert testing
            df.at[individual_id, 'tb_xpert_test'] = np.random.choice([True, False], size=1,
                                                                   p=[params['testing_prob_xpert'],
                                                                      1 - params['testing_prob_xpert']])

            df.at[individual_id, 'tb_xpert_test'] = True
            df.at[individual_id, 'tb_date_xpert_test'] = self.sim.date

            # around 50% of active cases will still be negative, different for HIV+?
            if df.at[individual_id, (df.has_tb == 'Active') & df.tb_xpert_test]:
                df.at[individual_id, 'tb_result_xpert_test'] = np.random.choice([True, False], size=1,
                                                                       p=[params['prop_xpert_positive'],
                                                                          1 - params['prop_xpert_positive']])
                df.at[individual_id, 'tb_result_xpert_test'] = True
                df.at[individual_id, 'tb_diagnosed'] = True


class TbHealthSystemLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """ produce some outputs to check
        """
        # run this event every 12 months (every year)
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props

        mask = (df['tb_date_tested'] > self.sim.date - DateOffset(months=self.repeat))
        recently_tested = mask.sum()

        self.module.store['Time'].append(self.sim.date)
        self.module.store['Number_tested_tb'].append(recently_tested)
