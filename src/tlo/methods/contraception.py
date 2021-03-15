"""
Contraception module covering baseline fertility, contraception methods use, failure (pregnancy),
Switching contraceptive methods, and discontinuation rates by age
please see Dropbox/Thanzi la Onse/05 - Resources/Model design/Contraception-Pregnancy.pdf
for conceptual diagram
"""
from pathlib import Path

import pandas as pd

from tlo import Date, DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import PopulationScopeEventMixin, RegularEvent
from tlo.util import transition_states
from tlo.methods.healthsystem import HSI_Event
from tlo.methods import Metadata

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Contraception(Module):
    """
    Contraception module covering baseline contraception methods use, failure (pregnancy),
    Switching contraceptive methods, and discontinuation rates by age
    please see Dropbox/Thanzi la Onse/05 - Resources/Programming Notes/Contraception-Pregnancy.pdf
    for conceptual diagram (lucid chart)
    """

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    # Declare Metadata
    METADATA = {Metadata.DISEASE_MODULE,
                Metadata.USES_HEALTHSYSTEM}

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    PARAMETERS = {
        'fertility_schedule': Parameter(Types.DATA_FRAME, 'Age_spec fertility'),
        'contraception_initiation1': Parameter(Types.DATA_FRAME, 'irate1_'),  # 2011-2016 rate
        'contraception_initiation2': Parameter(Types.DATA_FRAME, 'irate2_'),  # 2011-2016 rate
        'contraception_switching': Parameter(Types.DATA_FRAME, 'Switching'),
        'contraception_switching_matrix': Parameter(Types.DATA_FRAME, 'switching_matrix'),
        'contraception_discontinuation': Parameter(Types.DATA_FRAME, 'Discontinuation'),
        'contraception_failure': Parameter(Types.DATA_FRAME, 'Failure'),
        # from Fracpoly regression:
        'r_init1_age': Parameter(Types.REAL, 'proportioniate change in irate1 for each age in years'),
        # from Fracpoly regression:
        'r_discont_age': Parameter(Types.REAL, 'proportioniate change in drate for each age in years'),
        'rr_fail_under25': Parameter(Types.REAL, 'Increase in Failure rate for under-25s'),
        'r_init_year': Parameter(Types.REAL,
                                 'proportional change in contraception initiation rates for each year, 2010 to 2100'),
        'r_discont_year': Parameter(Types.REAL,
                                    'proportional change in contraception_discontinuation rate for each year,\
                                     2010 to 2100'),
        # TODO: add relative fertility rates for HIV+ compared to HIV- by age group from Marston et al 2017
        'contraception_consumables': Parameter(Types.DATA_FRAME, 'consumables'),
        'contraception_interventions': Parameter(Types.DATA_FRAME, 'interventions'),
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'co_contraception': Property(Types.CATEGORICAL, 'Current contraceptive method',
                                     categories=['not_using', 'pill', 'IUD', 'injections', 'implant', 'male_condom',
                                                 'female_sterilization', 'other_modern', 'periodic_abstinence',
                                                 'withdrawal', 'other_traditional']),
        # These are the 11 categories of contraception ('not using' + 10 methods) from the DHS analysis of initiation,
        # discontinuation, failure and switching rates
        # 'other modern' includes Male sterilization, Female Condom, Emergency contraception
        # 'other traditional' includes
        #   lactational amenohroea (LAM),
        #   standard days method (SDM),
        #   'other traditional method'
        # Have replaced Age-spec fertility sheet in ResourceFile_DemographicData.xlsx (in this branch)
        # with the one in ResourceFile_Contraception.xlsx
        # (has 11 categories and one row for each age with baseline contraception prevalence for
        # each of the 11 categories)
        'co_date_of_childbirth': Property(Types.DATE, 'Due date of child for those who become pregnant'),
        'is_pregnant': Property(Types.BOOL, 'Whether this individual is currently pregnant'),
        'date_of_last_pregnancy': Property(Types.DATE,
                                           'Date of the last pregnancy of this individual'),
        'co_unintended_preg': Property(Types.BOOL, 'Unintended pregnancies following contraception failure'),
        #TODO: add link to unintended preg from not using
    }

    def read_parameters(self, data_folder):
        """
        Please see Contraception-Pregnancy.pdf for lucid chart explaining the relationships between
        baseline fertility rate, intitiation rates, discontinuation, failure and switching rates, and being on
        contraception or not, and being pregnant
        """
        workbook = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_Contraception.xlsx', sheet_name=None)

        self.parameters['fertility_schedule'] = workbook['Age_spec fertility']

        self.parameters['contraception_initiation2'] = workbook['irate2_'].loc[0]
        # this Excel sheet is irate2_all.csv outputs from 'initiation rates_age_stcox.do'
        # Stata analysis of DHS contraception calendar data

        self.parameters['contraception_failure'] = workbook['Failure'].loc[0]
        # this Excel sheet is from contraception_failure_discontinuation_switching.csv outputs from
        # 'failure discontinuation switching rates.do' Stata analysis of DHS contraception calendar data

        self.parameters['rr_fail_under25'] = 2.2
        # From Guttmacher analysis - do not apply to female steriliztion or male sterilization
        # - note that as these are already 0 (see 'Failure' Excel sheet) the rate will remain 0

        self.parameters['r_init1_age'] = workbook['r_init1_age']

        self.parameters['r_discont_age'] = workbook['r_discont_age']
        # from Stata analysis Step 3.5 of discontinuation & switching rates_age.do: fracpoly: regress drate_allmeth age:
        # - see 'Discontinuation by age' worksheet, results are in 'r_discont_age' sheet

        self.parameters['r_init_year'] = workbook['r_init_year'].set_index('year')

        self.parameters['r_discont_year'] = workbook['r_discont_year'].set_index('year')

        self.parameters['contraception_consumables'] = workbook['consumables']
        # this has data on all the contraception consumables used - currently not used as link to main consumables
        # worksheet in health system instead

        self.parameters['contraception_interventions'] = workbook['interventions']
        # this has multipliers of initiation rates to model effect of interventions to increase contraception uptake
        # multiplier: of r_init1 by contraception type e.g. 1.2 for pill means a 20% increase in initiation of pill
        # PPFP_multiplier: "Post Partum Family Planning" multiplier of r_init2 by contraception type e.g. 1.8 for IUD
        #   means 80% increase in IUD initiation after pregnancy/birth.

        # =================== ARRANGE INPUTS FOR USE BY REGULAR EVENTS =============================

        # For ContraceptionSwitchingPoll.init1 -----------------------------------------------------

        # from Stata analysis line 250 of initiation rates_age_stcox_2005_2016_5yrPeriods.do:
        # fracpoly: regress _d age_
        # // fracpoly exact age (better fitting model, higher F statistic) - see 'Initiation1 by age'
        # worksheet, results are in 'r_init1_age' sheet
        c_multiplier = workbook['r_init1_age']

        # 'irate_1_' sheet created manually
        c_baseline = workbook['irate1_']
        # this Excel sheet is irate1_all.csv outputs from 'initiation rates_age_stcox.do'
        # Stata analysis of DHS contraception calendar data
        c_intervention = workbook['interventions']
        c_intervention = pd.DataFrame(c_intervention)
        c_intervention = c_intervention.set_index('contraception').T
        c_intervention1 = c_intervention.iloc[[0]]  # just the first row: multiplier, double brackets for df type
        c_intervention2 = c_intervention.iloc[[2]]  # just the third row: PPFP_multiplier

        c_baseline = c_baseline.drop(columns=['not_using'])
        c_baseline = c_baseline.mul(c_intervention1.iloc[0])    # intervention1 to increase each contraception uptake
        c_baseline = pd.concat([c_baseline] * len(c_multiplier), ignore_index=True)
        c_adjusted = c_baseline.mul(c_multiplier.r_init1_age, axis='index')
        c_adjusted = c_baseline + c_adjusted
        self.parameters['contraception_initiation1'] = c_adjusted.set_index(c_multiplier.age)

        # For ContraceptionSwitchingPoll.switch ----------------------------------------------------
        switching_prob = workbook['Switching'].transpose()
        # this Excel sheet is from contraception_failure_discontinuation_switching.csv outputs from
        # 'failure discontinuation switching rates.do' Stata analysis of DHS contraception calendar data
        switching_prob.columns = ['probability']
        self.parameters['contraception_switching'] = switching_prob

        switching_matrix = workbook['switching_matrix']
        # this Excel sheet is from contraception switching matrix outputs from line 144 of
        # 'failure discontinuation switching rates.do' Stata analysis of DHS contraception calendar data

        switching_matrix = switching_matrix.set_index('switchfrom')
        switching_matrix = switching_matrix.transpose()
        self.parameters['contraception_switching_matrix'] = switching_matrix

        # For ContraceptionSwitchingPoll.discontinue
        c_baseline = workbook['Discontinuation']
        # this Excel sheet is from contraception_failure_discontinuation_switching.csv outputs from
        # 'failure discontinuation switching rates.do' Stata analysis of DHS contraception calendar data
        c_multiplier = self.parameters['r_discont_age']
        c_baseline = pd.concat([c_baseline] * len(c_multiplier), ignore_index=True)
        c_adjusted = c_baseline.mul(c_multiplier.r_discont_age, axis='index')
        c_adjusted = c_baseline + c_adjusted
        self.parameters['contraception_discontinuation'] = c_adjusted.set_index(c_multiplier.age)

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.
        """
        df = population.props

        df.loc[df.is_alive, 'co_contraception'] = 'not_using'
        df.loc[df.is_alive, 'co_date_of_childbirth'] = pd.NaT
        df.loc[df.is_alive, 'is_pregnant'] = False
        df.loc[df.is_alive, 'date_of_last_pregnancy'] = pd.NaT
        df.loc[df.is_alive, 'co_unintended_preg'] = False

        # Assign contraception method
        # 1. select females aged 15-49 from population, for current year
        females1549 = df.is_alive & (df.sex == 'F') & df.age_years.between(15, 49)

        # 2. Prepare probabilities lookup table
        co_types_prob_lookup = self.parameters['fertility_schedule'].set_index('age')
        co_types_prob_lookup.drop(columns=['year', 'basefert_dhs'], inplace=True)  # drop unused columns
        co_types = list(co_types_prob_lookup.columns)

        # normalise the values so they sum to 1 and collapse into single array
        co_types_prob_lookup = co_types_prob_lookup.apply(lambda x: x.values / sum(x.values), axis=1)

        # 3. apply probabilities of each contraception type to sim population
        def pick_contraceptive(age):
            """a function that randomly chooses a contraceptive based on age"""
            return self.rng.choice(co_types, p=co_types_prob_lookup.loc[age])

        df.loc[females1549, 'co_contraception'] = df.loc[females1549, 'age_years'].apply(pick_contraceptive)

    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """
        # starting contraception, switching contraception metho, and stopping contraception:
        sim.schedule_event(ContraceptionSwitchingPoll(self), sim.date + DateOffset(months=0))

        # check all females using contraception to determine if contraception fails i.e. woman becomes
        # pregnant whilst using contraception (starts at month 0)
        sim.schedule_event(Fail(self), sim.date + DateOffset(months=0))

        # check all population to determine if pregnancy should be triggered (repeats every month)
        sim.schedule_event(PregnancyPoll(self), sim.date + DateOffset(months=1))

        # Launch the repeating event that will store statistics about the population structure
        sim.schedule_event(ContraceptionLoggingEvent(self), sim.date + DateOffset(days=0))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        df = self.sim.population.props

        df.at[child_id, 'co_contraception'] = 'not_using'
        df.at[child_id, 'co_date_of_childbirth'] = pd.NaT
        df.at[child_id, 'is_pregnant'] = False
        df.at[child_id, 'date_of_last_pregnancy'] = pd.NaT
        df.at[child_id, 'co_unintended_preg'] = False

        # Reset the mother's is_pregnant status showing that she is no longer pregnant
        df.at[mother_id, 'is_pregnant'] = False
        # TODO- commented out by joe, this way women always keep the date on which they most recently became pregnant,
        #  it is over written for new pregnancy
        # df.at[mother_id, 'date_of_last_pregnancy'] = pd.NaT

        # Initiation of mother's contraception after birth (was previously Init2 event)
        # Notes: decide what contraceptive method they have (including not_using, according to
        # initiation_rate2 (irate_2)
        # Note the irate2s are low as they are just for the month after pregnancy and
        # then for the 99.48% who 'initiate' to 'not_using' (i.e. 1 - sum(irate2s))
        # they are then subject to the usual irate1s per month
        # - see Contraception-Pregnancy.pdf schematic
        on_birth_co_probs: pd.Series = self.parameters['contraception_initiation2']

        # sample a single row of the init2 probabilities (weighted by those same probabilities)
        chosen_co = on_birth_co_probs.sample(n=1, weights=on_birth_co_probs, random_state=self.rng)

        # the index of the row is the contraception type
        df.at[mother_id, 'co_contraception'] = chosen_co.index[0]
        post_birth_contraception_summary = {
            'woman_index': mother_id,
            'co_contraception': df.at[mother_id, 'co_contraception']
        }

        logger.info(key='post_birth_contraception',
                    data=post_birth_contraception_summary,
                    description='post birth_contraception')


class ContraceptionSwitchingPoll(RegularEvent, PopulationScopeEventMixin):
    """
    Switching contraception status for population

    This event looks across:
    1) all women who are 'not_using' contraception to determine if they start using each method=
    according to initiation_rate1 (irate_1)
    2) all women who are using a contraception method to determine if they switch to another
    method according to switching_rate, and then for those that switch directs towards a new method according to
    switching_matrix
    3) all women who are using a contraception method to determine if they stop using it
    according to discontinuation_rate

    (Was previously in Init1, Switch, Discontinue events)
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        self.age_low = 15
        self.age_high = 49

    def apply(self, population):
        df = population.props

        possible_co_users = ((df.sex == 'F') &
                             df.is_alive &
                             df.age_years.between(self.age_low, self.age_high) &
                             ~df.is_pregnant)

        currently_using_co = df.index[possible_co_users &
                                      ~df.co_contraception.isin(['not_using', 'female_steralization'])]
        currently_not_using_co = df.index[possible_co_users & (df.co_contraception == 'not_using')]

        # not using -> using
        self.init1(df, currently_not_using_co)

        # using A -> using B
        self.switch(df, currently_using_co)

        # using -> not using
        self.discontinue(df, currently_using_co)

    def init1(self, df: pd.DataFrame, individuals_not_using: pd.Index):
        """check all females not using contraception to determine if contraception starts
        i.e. category should change from 'not_using'
        """
        p = self.module.parameters
        rng = self.module.rng

        c_multiplier = p['r_init_year'].at[self.sim.date.year, 'r_init_year']
        co_start_prob_by_age = p['contraception_initiation1'].mul(c_multiplier)
        co_start_prob_by_age['not_using'] = 1 - co_start_prob_by_age.sum(axis=1)

        # all the contraceptive types we can randomly choose
        co_types = list(co_start_prob_by_age.columns)

        def pick_random_contraceptive(age):
            """random selects a contraceptive using probabilities for given age"""
            return rng.choice(co_types, p=co_start_prob_by_age.loc[age])

        # select a random contraceptive for everyone not currently using
        random_co = df.loc[individuals_not_using, 'age_years'].apply(pick_random_contraceptive)

        if len(random_co):
            # get index of all those now using
            now_using_co = random_co.index[random_co != 'not_using']

            # only update entries for all those now using a contraceptive
            df.loc[now_using_co, 'co_contraception'] = random_co[now_using_co]
            # TODO: health system resource use will come in here - HSI at end of code below
            contraception_event = HSI_Contraception(self.module)
            self.sim.modules['HealthSystem'].schedule_hsi_event(contraception_event,
                                                                priority=1,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))
            logger.debug(key='debug', data='The population wide HSI event has been scheduled successfully!')

            for woman in now_using_co:
                start_contraception_summary = {
                    'woman_index': woman,
                    'age': df.at[woman, 'age_years'],
                    'co_contraception': df.at[woman, 'co_contraception']
                }

                logger.info(key='start_contraception',
                            data=start_contraception_summary,
                            description='start_contraception')

    def switch(self, df: pd.DataFrame, individuals_using: pd.Index):
        """check all females using contraception to determine if contraception Switches
        i.e. category should change from any method to a new method (not including 'not_using')
        """
        p = self.module.parameters
        rng = self.module.rng

        switching_prob = p['contraception_switching']
        switching_matrix = p['contraception_switching_matrix']

        # get the probability of switching contraceptive for all those currently using
        co_switch_prob = df.loc[individuals_using, 'co_contraception'].map(switching_prob.probability)

        # randomly select some individuals to switch contraceptive
        random_draw = rng.random_sample(size=len(individuals_using))
        switch_co = co_switch_prob.index[co_switch_prob > random_draw]

        # if no one is switching, exit
        if switch_co.empty:
            return

        # select new contraceptive using switching matrix
        new_co = transition_states(df.loc[switch_co, 'co_contraception'], switching_matrix, rng)

        # log old -> new contraception types
        for woman in switch_co:
            switch_contraception_summary = {
                'woman_index': woman,
                'co_from': df.at[woman, 'co_contraception'],
                'co_to': new_co[woman]
            }

            logger.info(key='switch_contraception',
                        data=switch_contraception_summary,
                        description='switch_contraception')

        # update contraception for all who switched
        df.loc[switch_co, 'co_contraception'] = new_co

    def discontinue(self, df: pd.DataFrame, individuals_using: pd.Index):
        """check all females using contraception to determine if contraception discontinues
        i.e. category should change to 'not_using'
        """
        p = self.module.parameters
        rng = self.module.rng

        c_multiplier = p['r_discont_year'].at[self.sim.date.year, 'r_discont_year']
        c_adjustment = p['contraception_discontinuation'].mul(c_multiplier)

        def get_prob_discontinued(row):
            """returns the probability of discontinuing contraceptive based on age and current
            contraceptive"""
            return c_adjustment.loc[row.age_years, row.co_contraception]

        # get the probability of discontinuing for all currently using
        discontinue_prob = df.loc[individuals_using].apply(get_prob_discontinued, axis=1)

        # random choose some to discontinue
        random_draw = rng.random_sample(size=len(individuals_using))
        if len(random_draw):
            co_discontinue = discontinue_prob.index[discontinue_prob > random_draw]
            df.loc[co_discontinue, 'co_contraception'] = 'not_using'

            for woman in co_discontinue:
                stop_contraception_summary = {
                    'woman_index': woman,
                }

                logger.info(key='stop_contraception',
                            data=stop_contraception_summary,
                            description='stop_contraception')


class Fail(RegularEvent, PopulationScopeEventMixin):
    """
    This event looks across all women who are using a contraception method to determine if they become pregnant
    i.e. the method fails according to failure_rate
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))  # runs every month
        self.age_low = 15
        self.age_high = 49

    def apply(self, population):
        logger.debug('Checking to see if anyone becomes pregnant whilst on contraception')

        df = population.props
        p = self.module.parameters
        rng = self.module.rng

        prob_of_failure = p['contraception_failure']

        # TODO: N.B edited by joe- women who are in labour, have been pregnant in the last month or have previously had
        #  a hysterectomy cannot get pregnant(eventually should remove women with hysterectomy from being on
        #  contraception?)

        possible_to_fail = ((df.sex == 'F') &
                            df.is_alive &
                            ~df.is_pregnant &
                            ~df.la_currently_in_labour &
                            ~df.la_has_had_hysterectomy &
                            df.age_years.between(self.age_low, self.age_high) &
                            ~df.co_contraception.isin(['not_using', 'female_steralization']) &
                            (pd.isnull(df.la_due_date_current_pregnancy) | (self.sim.date >
                                                                            df.la_due_date_current_pregnancy +
                                                                            pd.to_timedelta(42, unit='D'))))

        prob_of_failure = df.loc[possible_to_fail, 'co_contraception'].map(prob_of_failure)

        # apply increased risk of failure to under 25s
        prob_of_failure.loc[df.index[possible_to_fail & (df.age_years.between(15, 25))]] *= p['rr_fail_under25']

        # randomly select some individual's for contraceptive failure
        random_draw = rng.random_sample(size=len(prob_of_failure))
        women_co_failure = prob_of_failure.index[prob_of_failure > random_draw]

        for woman in women_co_failure:
            # this woman's contraception has failed - she is pregnant
            # Women currently in labour cannot become pregnant
            df.at[woman, 'is_pregnant'] = True
            df.at[woman, 'date_of_last_pregnancy'] = self.sim.date
            df.at[woman, 'co_unintended_preg'] = True
            self.sim.modules['Labour'].set_date_of_labour(woman)

            # outputs some logging if any pregnancy (contraception failure)
            fail_contraception_summary = {
                'woman_index': woman,
                'birth_booked': str(df.at[woman, 'co_date_of_childbirth'])
            }

            logger.info(key='fail_contraception',
                        data=fail_contraception_summary,
                        description='fail_contraception')


class PregnancyPoll(RegularEvent, PopulationScopeEventMixin):
    """
    This event looks across each woman in the population to determine who will become pregnant
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        self.age_low = 15
        self.age_high = 49

    def apply(self, population):

        logger.debug('Checking to see if anyone should become pregnant....')

        if self.sim.date > Date(2035, 1, 1):
            logger.debug('Now after 2035')

        df = population.props  # get the population dataframe

        # get the subset of women from the population dataframe and relevant characteristics
        subset = (df.sex == 'F') & df.is_alive & df.age_years.between(self.age_low, self.age_high) & ~df.is_pregnant & \
                 (df.co_contraception == 'not_using') & ~df.la_currently_in_labour & ~df.la_has_had_hysterectomy &\
                 (pd.isnull(df.la_due_date_current_pregnancy) | (self.sim.date > df.la_due_date_current_pregnancy +
                                                                 pd.to_timedelta(42, unit='D')))

        females = df.loc[subset, ['co_contraception', 'age_years']]

        # load the fertility schedule (imported datasheet from excel workbook)
        fertility_schedule = self.module.parameters['fertility_schedule']

        # --------

        # get the probability of pregnancy for each woman in the model, through merging with the fert_schedule data
        len_before_merge = len(females)
        females = females.reset_index().merge(fertility_schedule,
                                              left_on=['age_years'],
                                              # TimC: got rid of 'contraception' here as just one
                                              # basefert_dhs per age (see new 'Age_spec fertility' sheet)
                                              right_on=['age'],
                                              # TimC: got rid of 'cmeth' here as just one basefert_dhs per age
                                              # (see new 'Age_spec fertility' sheet)
                                              how='inner').set_index('person')
        assert len(females) == len_before_merge

        # flipping the coin to determine if this woman will become pregnant (basefert_dhs is in the Excel sheet)
        newly_pregnant = (self.module.rng.random_sample(size=len(females)) < females.basefert_dhs / 12)

        # the imported number is a yearly proportion. So adjust the rate accordingly
        # to the frequency with which the event is recurring
        # TODO: this should be linked to the self.frequency value

        newly_pregnant_ids = females.index[newly_pregnant]

        # updating the pregancy status for that women
        df.loc[newly_pregnant_ids, 'is_pregnant'] = True
        df.loc[newly_pregnant_ids, 'date_of_last_pregnancy'] = self.sim.date

        # loop through each newly pregnant women in order to schedule them to have labour sheduled
        for female_id in newly_pregnant_ids:
            self.sim.modules['Labour'].set_date_of_labour(female_id)

            pregnant_at_age_summary = {
                'person_id': female_id,
                'age_years': females.at[female_id, 'age_years']
            }

            logger.info(key='pregnant_at_age_',
                        data=pregnant_at_age_summary,
                        description='pregnant_at_age_')


class ContraceptionLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """Logs outputs for analysis_contraception
        """
        # run this event every 12 months (every year)
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))
        self.age_low = 15
        self.age_high = 49

    def apply(self, population):
        df = population.props

        contraception_count = df[df.is_alive & df.age_years.between(self.age_low, self.age_high)].groupby(
            'co_contraception').size()

        contraception_summary = {
            'total': sum(contraception_count),
            'not_using': contraception_count['not_using'],
            'using': sum(contraception_count) - contraception_count['not_using'],
            'pill': contraception_count['pill'],
            'IUD': contraception_count['IUD'],
            'injections': contraception_count['injections'],
            'implant': contraception_count['implant'],
            'male_condom': contraception_count['male_condom'],
            'female_sterilization': contraception_count['female_sterilization'],
            'other_modern': contraception_count['other_modern'],
            'periodic_abstinence': contraception_count['periodic_abstinence'],
            'withdrawal': contraception_count['withdrawal'],
            'other_traditional': contraception_count['other_traditional']
        }

        logger.info(key='contraception',
                    data=contraception_summary,
                    description='contraception')

        preg_counts = df[df.is_alive & df.age_years.between(self.age_low, self.age_high)].is_pregnant.value_counts()
        is_preg_count = (df.is_alive & df.age_years.between(self.age_low, self.age_high) & df.is_pregnant).sum()
        is_notpreg_count = (df.is_alive & df.age_years.between(self.age_low, self.age_high) & ~df.is_pregnant).sum()

        pregnancy_summary = {
            'total': sum(preg_counts),
            'pregnant': is_preg_count,
            'not_pregnant': is_notpreg_count
        }

        logger.info(key='pregnancy',
                    data=pregnancy_summary,
                    description='pregnancy')


class HSI_Contraception(HSI_Event, PopulationScopeEventMixin):  # whole population at once
    """"""
    def __init__(self, module):
        super().__init__(module)
        assert isinstance(module, Contraception)

        # Define the necessary information for a Population level HSI
        self.TREATMENT_ID = 'Contraception'
        self.ACCEPTED_FACILITY_LEVEL = 1    # need to comment out line 285 in healthsystem.py: assert 'ACCEPTED_FACILITY_LEVEL' not in dir(hsi_event)

    def apply(self, population, squeeze_factor):
        df = self.sim.population.props
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        # multiply each Unit_Cost by Expected_Units_Per_Case so they can be summed for all Items for each contraceptive
        # package to get cost of each contraceptive user for each contraceptive
        item_cost = self.sim.modules['HealthSystem'].parameters['Consumables']['Expected_Units_Per_Case']*\
                    self.sim.modules['HealthSystem'].parameters['Consumables']['Unit_Cost']
        consumables['item_cost'] = item_cost
        # TODO: How long are the units of each item supposed to be for though? e.g. male condoms have 120 units.

        # contraception_consumables = self.parameters['contraception_consumables']
        # contraception_users = df.co_contraception.isin(['pill', 'IUD', 'injections', 'implant', 'male_condom',  \
        #                                                 'female_sterilization', 'other_modern'])
        # consumables_pkgs = self.sim.modules['HealthSystem'].parameters['Consumables'].isin(['Pill', 'IUD',  \
        #                                                                                     'Injectable', 'Implant',  \
        #                                                                                     'Male condomm',  \
        #                                                                                     'Female sterilization',  \
        #                                                                                     'Female Condom'])
        #
        # # loop through the following for each contraceptive
        # for consumables_pkgs in contraception_users:
        #   TODO: make this loop by mapping the names of the contraception packages in Consumables to the names in contraception as per above


        # Each contraceptive done individually now pending mapping / looping as per above

        # Pill
        pill_users = df.loc[df.co_contraception == 'pill']
        pkg_code_pill = pd.unique(consumables.loc[
                                             consumables['Intervention_Pkg']
                                             == 'Pill',
                                             'Intervention_Pkg_Code'])[0]
        cost_pill = pd.Series(consumables.loc[
                                         consumables['Intervention_Pkg']
                                         == 'Pill',
                                         'item_cost']).sum()    # adds all item costs

        consumables_needed = {'Intervention_Package_Code': {pkg_code_pill: 0}, 'Item_Code': {}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self,
            cons_req_as_footprint=consumables_needed)

        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_pill]:
            df.loc[pill_users.index, 'pill_received'] = True
            df.loc[pill_users.index, 'pill_costs'] = cost_pill

        pill_counts = df.pill_received.value_counts()
        pill_costs = df.pill_costs.sum()


        # IUD
        IUD_users = df.loc[df.co_contraception == 'IUD']
        pkg_code_IUD = pd.unique(consumables.loc[
                                             consumables['Intervention_Pkg']
                                             == 'IUD',
                                             'Intervention_Pkg_Code'])[0]
        cost_IUD = pd.Series(consumables.loc[
                                         consumables['Intervention_Pkg']
                                         == 'IUD',
                                         'item_cost']).sum()    # adds all item costs

        consumables_needed = {'Intervention_Package_Code': {pkg_code_IUD: 3}, 'Item_Code': {}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self,
            cons_req_as_footprint=consumables_needed)

        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_IUD]:
            df.loc[IUD_users.index, 'IUD_received'] = True
            df.loc[IUD_users.index, 'IUD_costs'] = cost_IUD

        IUD_counts = df.IUD_received.value_counts()
        IUD_costs = df.IUD_costs.sum()


        # injections
        injections_users = df.loc[df.co_contraception == 'injections']
        pkg_code_injections = pd.unique(consumables.loc[
                                             consumables['Intervention_Pkg']
                                             == 'Injectable',
                                             'Intervention_Pkg_Code'])[0]
        cost_injections = pd.Series(consumables.loc[
                                         consumables['Intervention_Pkg']
                                         == 'Injectable',
                                         'item_cost']).sum()    # adds all item costs

        consumables_needed = {'Intervention_Package_Code': {pkg_code_injections: 2}, 'Item_Code': {}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self,
            cons_req_as_footprint=consumables_needed)

        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_injections]:
            df.loc[injections_users.index, 'injections_received'] = True
            df.loc[injections_users.index, 'injections_costs'] = cost_injections

        injections_counts = df.injections_received.value_counts()
        injections_costs = df.injections_costs.sum()


        # Implant
        implant_users = df.loc[df.co_contraception == 'implant']
        pkg_code_implant = pd.unique(consumables.loc[
                                             consumables['Intervention_Pkg']
                                             == 'Implant',
                                             'Intervention_Pkg_Code'])[0]
        cost_implant = pd.Series(consumables.loc[
                                         consumables['Intervention_Pkg']
                                         == 'Implant',
                                         'item_cost']).sum()    # adds all item costs

        consumables_needed = {'Intervention_Package_Code': {pkg_code_implant: 4}, 'Item_Code': {}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self,
            cons_req_as_footprint=consumables_needed)

        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_implant]:
            df.loc[implant_users.index, 'implant_received'] = True
            df.loc[implant_users.index, 'implant_costs'] = cost_implant

        implant_counts = df.implant_received.value_counts()
        implant_costs = df.implant_costs.sum()


        # Male condoms
        male_condom_users = df.loc[df.co_contraception == 'male_condom']
        pkg_code_male_condom = pd.unique(consumables.loc[
                                         consumables['Intervention_Pkg']
                                         == 'Male condom',
                                         'Intervention_Pkg_Code'])[0]
        cost_male_condom = pd.Series(consumables.loc[
                                         consumables['Intervention_Pkg']
                                         == 'Male condom',
                                         'item_cost']).sum()

        consumables_needed = {'Intervention_Package_Code': {pkg_code_male_condom: 1}, 'Item_Code': {}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self,
            cons_req_as_footprint=consumables_needed)

        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_male_condom]:
            df.loc[male_condom_users.index, 'male_condom_received'] = True
            df.loc[male_condom_users.index, 'male_condom_costs'] = cost_male_condom

        male_condom_counts = df.male_condom_received.value_counts()
        male_condom_costs = df.male_condom_costs.sum()


        # Female sterilization
        female_sterilization_users = df.loc[df.co_contraception == 'female_sterilization']
        pkg_code_female_sterilization = pd.unique(consumables.loc[
                                         consumables['Intervention_Pkg']
                                         == 'Female sterilization',
                                         'Intervention_Pkg_Code'])[0]
        cost_female_sterilization = pd.Series(consumables.loc[
                                         consumables['Intervention_Pkg']
                                         == 'Female sterilization',
                                         'item_cost']).sum()    # adds all item costs

        consumables_needed = {'Intervention_Package_Code': {pkg_code_female_sterilization: 5}, 'Item_Code': {}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self,
            cons_req_as_footprint=consumables_needed)

        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_female_sterilization]:
            df.loc[female_sterilization_users.index, 'female_sterilization_received'] = True
            df.loc[female_sterilization_users.index, 'female_sterilization_costs'] = cost_female_sterilization

        female_sterilization_counts = df.female_sterilization_received.value_counts()
        female_sterilization_costs = df.female_sterilization_costs.sum()


        # Other modern (Female condom)
        other_modern_users = df.loc[df.co_contraception == 'other_modern']
        pkg_code_other_modern = pd.unique(consumables.loc[
                                         consumables['Intervention_Pkg']
                                         == 'Female Condom',
                                         'Intervention_Pkg_Code'])[0]
        cost_female_condom = pd.Series(consumables.loc[
                                         consumables['Intervention_Pkg']
                                         == 'Female Condom',
                                         'item_cost']).sum()    # adds all item costs

        consumables_needed = {'Intervention_Package_Code': {pkg_code_other_modern: 7}, 'Item_Code': {}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self,
            cons_req_as_footprint=consumables_needed)

        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_other_modern]:
            df.loc[other_modern_users.index, 'female_condom_received'] = True
            df.loc[other_modern_users.index, 'female_condom_costs'] = cost_female_condom

        female_condom_counts = df.female_condom_received.value_counts()
        female_condom_costs = df.female_condom_costs.sum()


        # Summary and logging
        contraception_consumables_summary = {
            # consumables
            'pills': sum(pill_counts),
            'IUDs': sum(IUD_counts),
            'injections': sum(injections_counts),
            'implants': sum(implant_counts),
            'male_condoms': sum(male_condom_counts),
            'female_sterilizations': sum(female_sterilization_counts),
            'female_condoms': sum(female_condom_counts),
            # costs
            'pill_costs': pill_costs,
            'IUD_costs': IUD_costs,
            'injections_costs': injections_costs,
            'implant_costs': implant_costs,
            'male_condom_costs': male_condom_costs,
            'female_sterilization_costs': female_sterilization_costs,
            'female_condom_costs': female_condom_costs,
        }

        logger.info(key='contraception_consumables_summary',
                    data=contraception_consumables_summary,
                    description='contraception_consumables_summary')



    def did_not_run(self):
        logger.debug(key='debug', data='HSI__Contraception: did not run')
        # return False to prevent this event from being rescheduled if it did not run.
        return False
