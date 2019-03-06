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
        'testing_prob_xpert': Parameter(Types.REAL, 'probability of individual receiving xpert test'),
        'prop_xpert_positive': Parameter(Types.REAL,
                                         'proportion active tb cases tested with xpert with positive results'),
        'prob_tb_treatment': Parameter(Types.REAL, 'probability of individual starting treatment'),
        'prob_mdr': Parameter(Types.REAL, 'probability tb case is mdr'),
        'prob_tb_mdr_treatment': Parameter(Types.REAL, 'probability of individual starting mdr treatment'),

    }

    PROPERTIES = {
        'tb_ever_tested': Property(Types.BOOL, 'ever had a tb test'),
        'tb_smear_test': Property(Types.BOOL, 'ever had a tb smear test'),
        'tb_result_smear_test': Property(Types.BOOL, 'result from tb smear test'),
        'tb_date_smear_test': Property(Types.DATE, 'date of tb smear test'),
        'tb_xpert_test': Property(Types.BOOL, 'ever had a tb Xpert test'),
        'tb_result_xpert_test': Property(Types.BOOL, 'result from tb Xpert test'),
        'tb_date_xpert_test': Property(Types.DATE, 'date of tb Xpert test'),
        'tb_diagnosed': Property(Types.BOOL, 'active tb and tested'),
        'tb_treated': Property(Types.BOOL, 'on tb treatment regimen'),
        'tb_date_treated': Property(Types.DATE, 'date tb treatment started'),
        'tb_treatment_failure': Property(Types.BOOL, 'failed first line tb treatment'),
        'tb_treatedMDR': Property(Types.BOOL, 'on tb treatment MDR regimen'),
        'tb_date_treatedMDR': Property(Types.DATE, 'date tb MDR treatment started'),
        'request_mdr_regimen': Property(Types.BOOL, 'request for mdr treatment'),
    }

    def read_parameters(self, data_folder):
        # TODO: if events run every 3months, probabilities need to be 3-monthly
        params = self.parameters
        params['tb_testing_coverage'] = 0.1  # dummy value
        params['prop_smear_positive'] = 0.8
        params['prop_smear_positive_hiv'] = 0.5
        params['testing_prob_xpert'] = 0.7
        params['prop_xpert_positive'] = 0.5
        params['prob_tb_treatment'] = 0.75
        params['prob_mdr'] = 0.05
        params['prob_tb_mdr_treatment'] = 0.8

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
        df['tb_treated'] = False
        df['tb_date_treated'] = pd.NaT
        df['tb_treatment_failure'] = False
        df['tb_treatedMDR'] = False
        df['tb_date_treatedMDR'] = pd.NaT
        df['request_mdr_regimen'] = False

    def initialise_simulation(self, sim):
        sim.schedule_event(TbTestingEvent(self), sim.date + DateOffset(months=12))
        sim.schedule_event(tbTreatmentEvent(self), sim.date + DateOffset(months=12))
        sim.schedule_event(tbTreatmentMDREvent(self), sim.date + DateOffset(months=12))

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
        df.at[child_id, 'tb_treated'] = False
        df.at[child_id, 'tb_date_treated'] = pd.NaT
        df.at[child_id, 'tb_treatment_failure'] = False
        df.at[child_id, 'tb_treatedMDR'] = False
        df.at[child_id, 'tb_date_treatedMDR'] = pd.NaT
        df.at[child_id, 'request_mdr_regimen'] = False


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
        testing_index = df.index[(random_draw < params['tb_testing_coverage']) & df.is_alive]
        # print('testing_index', testing_index)
        df.loc[testing_index, 'tb_ever_tested'] = True
        df.loc[testing_index, 'tb_smear_test'] = True
        df.loc[testing_index, 'tb_date_smear_test'] = now

        # 80% of smear tested active cases will be diagnosed
        # this is lower for HIV+ (higher prop of extrapulmonary tb
        tested_idx = df.index[(df.tb_date_smear_test == now) & df.is_alive & (df.has_tb == 'Active') & ~df.has_hiv]
        diagnosed_idx = pd.Series(np.random.choice([True, False], size=len(tested_idx),
                                                   p=[params['prop_smear_positive'],
                                                      (1 - params['prop_smear_positive'])]),
                                  index=tested_idx)
        idx = tested_idx[diagnosed_idx]

        tested_idx_hiv = df.index[
            (df.tb_date_smear_test == now) & df.is_alive & (df.has_tb == 'Active') & df.has_hiv]

        diagnosed_idx_hiv = pd.Series(np.random.choice([True, False], size=len(tested_idx_hiv),
                                                       p=[params['prop_smear_positive_hiv'],
                                                          (1 - params['prop_smear_positive_hiv'])]),
                                      index=tested_idx_hiv)
        idx_hiv = tested_idx_hiv[diagnosed_idx_hiv]

        if len(idx):
            df.loc[idx, 'result_smear_test'] = True
            df.loc[idx, 'tb_diagnosed'] = True

        if len(idx_hiv):
            df.loc[idx_hiv, 'result_smear_test'] = True
            df.loc[idx_hiv, 'tb_diagnosed'] = True

        # print('test date', now)

        # remaining 20% of active cases referred for xpert testing with some delay
        # also some true negatives may have follow-up testing
        # schedule xpert testing at future date
        # random draw approx 2 months?
        undiagnosed_idx = df.index[(df.tb_date_smear_test == now) & df.is_alive & ~df.tb_diagnosed]

        for person in undiagnosed_idx:
            refer_xpert = tbXpertTest(self.module, individual_id=person)
            # TODO: take absolute value so no negatives
            referral_time = np.random.normal(loc=(2/12), scale=(0.5/12), size=1)  # in years
            referral_time_yrs = pd.to_timedelta(referral_time[0] * 365.25, unit='d')
            future_referral_time = now + referral_time_yrs
            print('future_referral_time', now, referral_time_yrs, future_referral_time)
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
        print('xpert date now', now)

        # prob of receiving xpert test
        if df.at[individual_id, 'is_alive'] and not df.at[individual_id, 'tb_diagnosed'] and (
            np.random.choice([True, False], size=1,
                             p=[params['testing_prob_xpert'],
                                1 - params[
                                    'testing_prob_xpert']])):
            # print('xpert test happening')

            df.at[individual_id, 'tb_xpert_test'] = True
            df.at[individual_id, 'tb_date_xpert_test'] = now

            diagnosed = np.random.choice([True, False], size=1,
                                         p=[params['prop_xpert_positive'],
                                            (1 - params['prop_xpert_positive'])])

            if len(diagnosed):
                df.at[individual_id, 'tb_result_xpert_test'] = True
                df.at[individual_id, 'tb_diagnosed'] = True


class tbTreatmentEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        params = self.module.parameters
        now = self.sim.date
        df = population.props

        # get a list of random numbers between 0 and 1 for the whole population
        random_draw = self.sim.rng.random_sample(size=len(df))

        # probability of treatment
        treat_idx = df.index[
            (random_draw < params['prob_tb_treatment']) & ~df.tb_diagnosed & df.is_alive]

        df.loc[treat_idx, 'tb_treated'] = True
        df.loc[treat_idx, 'date_tb_treated'] = now

        # if on treatment for 6 months, take off and change to cured (95%)
        random_draw2 = self.sim.rng.random_sample(size=len(df))
        cure_idx = df.index[
            df.tb_treated & (((now - df.date_tb_treated) / np.timedelta64(1, 'M')) >= 6) & (random_draw2 < (
                1 - params['prob_mdr']))]
        df.loc[cure_idx, 'tb_treated'] = False
        df.loc[cure_idx, 'has_tb'] = 'Latent'

        # if on treatment for 6 months, 5% will not be cured and request MDR regimen
        random_draw3 = self.sim.rng.random_sample(size=len(df))
        mdr_idx = df.index[
            df.tb_treated & (((now - df.date_tb_treated) / np.timedelta64(1, 'M')) >= 6) & (
                    random_draw3 < params['prob_mdr'])]
        df.loc[mdr_idx, 'tb_treated'] = False
        df.loc[mdr_idx, 'request_mdr_regimen'] = True


class tbTreatmentMDREvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        params = self.module.parameters
        now = self.sim.date
        df = population.props

        # get a list of random numbers between 0 and 1 for the whole population
        random_draw = self.sim.rng.random_sample(size=len(df))

        # probability of mdr treatment
        mdr_treated_idx = df.index[
            (random_draw < params['prob_tb_mdr_treatment']) & ~df.request_mdr_regimen & df.is_alive]

        df.loc[mdr_treated_idx, 'request_mdr_regimen'] = False  # switch off this flag
        df.loc[mdr_treated_idx, 'tb_treatedMDR'] = True
        df.loc[mdr_treated_idx, 'tb_date_treatedMDR'] = now

        # if on treatment for 6 months, take off and change to cured (100%)
        random_draw = self.sim.rng.random_sample(size=len(df))
        cure_idx = df.index[
            df.tb_treatedMDR & (((now - df.tb_date_treatedMDR) / np.timedelta64(1, 'M')) >= 6)]
        df.loc[cure_idx, 'tb_treated'] = False
        df.loc[cure_idx, 'has_tb'] = 'Latent'



class TbIPTEvent(RegularEvent, PopulationScopeEventMixin):
    """ IPT to all paediatric contacts of a TB case - randomly select 5 children <5 yrs old
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        params = self.module.parameters
        now = self.sim.date
        df = population.props

        #  sum number of active TB cases * 5
        ipt_needed = len(df.index[df.has_tb & df.is_alive & ~df.tb_treated]) * 5

        # randomly sample from <5 yr olds
        ipt_sample = df[(df.age_years <= 5) & (~df.has_tb == 'Active')].sample(
            n=ipt_needed, replace=False).index

        df.loc[ipt_sample, 'on_ipt'] = True
        df.loc[ipt_sample, 'date_ipt'] = now

        # TODO: ending ipt



class TbExpandedIPTEvent(RegularEvent, PopulationScopeEventMixin):
    """ IPT to all adults and adolescents with HIV
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        params = self.module.parameters
        now = self.sim.date
        df = population.props

        # randomly sample from >=15 yrs with HIV
        ipt_sample = df[(df.age_years >= 15) & (~df.has_hiv)].sample(
            frac=0.5, replace=False).index

        df.loc[ipt_sample, 'on_ipt'] = True
        df.loc[ipt_sample, 'date_ipt'] = now

        # TODO: ending ipt





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

        mask = (df['tb_date_smear_test'] > self.sim.date - DateOffset(months=self.repeat))
        recently_tested = mask.sum()

        self.module.store['Time'].append(self.sim.date)
        self.module.store['Number_tested_tb'].append(recently_tested)
