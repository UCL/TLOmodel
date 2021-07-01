from pathlib import Path
import numpy as np
import pandas as pd
from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import PopulationScopeEventMixin, RegularEvent
from tlo.util import transition_states

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""
# Issues remaining:
# * unify logging events: might be helpful to unify the logs (all pregnancies, all contrapctive changes). Currently _a lot_ of separate logs, each with differennt information and different labelling
# * remove computing of costs (this is done working but will give inaccurate results as structured).
# * should hiv effect on fertility also affect the risk of failure?
# * it would also be useful to run a check that when there is no contracpetive, then the number of births matches WPP very closely (ResourceFile_ASFR_WPP.csv) (as does the simplified_birth module)
# * There is a comment "add link to unintended preg from not using" - but I think it should be that the flag for unintended pregnancy comes when there is Fail (from a method).
# * We could encapsulate each init and swtich inside an HSI pretty easily now -- shall we do this? Could make it an optional thing (argument for ```no_hsi=True``` could preserve current behaviour)
"""


class Contraception(Module):
    """
    Contraception module covering baseline contraception methods use, failure (pregnancy),
    Switching contraceptive methods, and discontinuation rates by age.
    """

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    # Declare Metadata
    METADATA = {}

    # Declare Parameters
    PARAMETERS = {
        'fertility_schedule': Parameter(Types.DATA_FRAME, 'Age specific fertility'),

        'irate1_': Parameter(Types.DATA_FRAME, "Importation of the sheet 'irate1_'"),

        'irate2_': Parameter(Types.DATA_FRAME, "Importation of the sheet 'irate2_'"),

        'Switching': Parameter(Types.DATA_FRAME, "Importation of the sheet 'Switching'"),

        'switching_matrix': Parameter(Types.DATA_FRAME, "Importation of the sheet 'switching_matrix'"),

        'Discontinuation': Parameter(Types.DATA_FRAME, "Importation of the sheet 'Discontinuation'"),

        'contraception_initiation1': Parameter(Types.DATA_FRAME,
                                               'Monthly probabilities of initiating each method of contraception from '
                                               'not using contraception'),
        # 2011-2016 rate
        'contraception_initiation2': Parameter(Types.DATA_FRAME,
                                               'Monthly probabilities of initiating each method of contraception after '
                                               'pregnancy'),
        # 2011-2016 rate
        'contraception_switching': Parameter(Types.DATA_FRAME, 'Monthly probability of switching contraceptive methpd'),
        'contraception_switching_matrix': Parameter(Types.DATA_FRAME,
                                                    'switching matrix containing probabilities of switching from each '
                                                    'method to each other method'),
        'contraception_discontinuation': Parameter(Types.DATA_FRAME,
                                                   'Monthly probabilities of discontinuation of each method of '
                                                   'contaception to not using contraception'),
        'contraception_failure': Parameter(Types.DATA_FRAME,
                                           'Monthly probabilities of failure of each contraception method to pregnancy'),
        # from Fracpoly regression:
        'r_init1_age': Parameter(Types.REAL,
                                 'Proportioniate change in probabilities of initiating each method of contraception '
                                 'from not using contraception for each age of the woman in years'),
        # from Fracpoly regression:
        'r_discont_age': Parameter(Types.REAL,
                                   'Proportioniate change in probabilities of discontinuation of each method of '
                                   'contaception to not using contraception for each age of the woman in years'),
        'rr_fail_under25': Parameter(Types.REAL, 'Increase in failure rate for under-25s'),
        'r_init_year': Parameter(Types.REAL,
                                 'proportional change in contraception initiation probabilities for each year, 2010 to'
                                 ' 2100'),
        'r_discont_year': Parameter(Types.REAL,
                                    'proportional change in contraception discontinuation probabilities for each year, '
                                    '2010 to 2100'),
        # From Marston et al 2017, Figure 1, Eastern Africa, closer to rural - effect in younger ages may be due
        # to more sex in HIV+ but that is ok as we don't model sexual activity separately)
        'r_hiv': Parameter(Types.REAL,
                           'proportional change in fertility rate for HIV+ compared to HIV- by age group'),
        'contraception_consumables': Parameter(Types.DATA_FRAME, 'contraception consumables'),
        'contraception_interventions': Parameter(Types.DATA_FRAME, 'contraception interventions'),
    }

    # Declare Properties
    PROPERTIES = {
        'co_contraception': Property(Types.CATEGORICAL, 'Current contraceptive method',
                                     categories=['not_using', 'pill', 'IUD', 'injections', 'implant', 'male_condom',
                                                 'female_sterilization', 'other_modern', 'periodic_abstinence',
                                                 'withdrawal', 'other_traditional']),
        # These are the 11 categories of contraception ('not using' + 10 methods) from the DHS analysis of initiation,
        # discontinuation, failure and switching rates.
        # 'other modern' includes Male sterilization, Female Condom, Emergency contraception
        # 'other traditional' includes lactational amenohroea (LAM),  standard days method (SDM), 'other traditional
        #  method'

        'is_pregnant': Property(Types.BOOL, 'Whether this individual is currently pregnant'),
        'date_of_last_pregnancy': Property(Types.DATE,
                                           'Date of the last pregnancy of this individual'),
        'co_unintended_preg': Property(Types.BOOL,
                                       'Most recent or current pregnancy was unintended pregnancies following contraception failure'),

        # TODO: add link to unintended preg from not using????
    }

    def read_parameters(self, data_folder):
        """
        Import the relevant sheets from the ResourceFile (an excel workbook).
        Please see documentation for description of the relationships between baseline fertility rate, intitiation
        rates, discontinuation, failure and switching rates, and being on contraception or not and being pregnant.
        """
        workbook = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_Contraception.xlsx', sheet_name=None)

        self.parameters['fertility_schedule'] = workbook['Age_spec fertility']
        # todo @TimC - is this the WPP fertility scheudle? To keep things in sync, I'd suggest we use instead: "ResourceFile_ASFR_WPP.csv"

        self.parameters['contraception_failure'] = workbook['Failure'].loc[0]
        # this Excel sheet is from contraception_failure_discontinuation_switching.csv outputs from
        # 'failure discontinuation switching rates.do' Stata analysis of DHS contraception calendar data

        # Import sheets from the workbook here (will be processed later)
        self.parameters['r_init1_age'] = workbook['r_init1_age']
        self.parameters['irate1_'] = workbook['irate1_']
        self.parameters['irate2_'] = workbook['irate2_']
        self.parameters['Switching'] = workbook['Switching']
        self.parameters['switching_matrix'] = workbook['switching_matrix']
        self.parameters['Discontinuation'] = workbook['Discontinuation']
        self.parameters['r_discont_age'] = workbook['r_discont_age']

        # from Stata analysis Step 3.5 of discontinuation & switching rates_age.do: fracpoly: regress drate_allmeth age:
        # - see 'Discontinuation by age' worksheet, results are in 'r_discont_age' sheet

        self.parameters['r_init_year'] = workbook['r_init_year'].set_index('year')

        self.parameters['r_discont_year'] = workbook['r_discont_year'].set_index('year')

        self.parameters['r_hiv'] = workbook['r_hiv']

        self.parameters['contraception_consumables'] = workbook['consumables']
        # this has data on all the contraception consumables used - currently not used as link to main consumables
        # worksheet in health system instead
        # todo - remove this?

        self.parameters['contraception_interventions'] = workbook['interventions']
        # this has multipliers of initiation rates to model effect of interventions to increase contraception uptake
        # multiplier: of r_init1 by contraception type e.g. 1.2 for pill means a 20% increase in initiation of pill
        # PPFP_multiplier: "Post Partum Family Planning" multiplier of r_init2 by contraception type e.g. 1.8 for IUD
        #   means 80% increase in IUD initiation after pregnancy/birth. These are set below within read_parameters
        #   before the simulations starts

        # additional parameters (hard-coded)
        self.parameters['rr_fail_under25'] = 2.2
        # From Guttmacher analysis. (Nb. Is not applied to those persons with female sterilization)

    def initialise_population(self, population):
        """
        * 1) Process parameters
        * 2) Set initial values for properties
        """

        # 1) Process parameters
        self.process_parameters()

        # 2) Set initial values for properties
        df = population.props

        df.loc[df.is_alive, 'co_contraception'] = 'not_using'
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
        assert set(co_types) == set(self.PROPERTIES['co_contraception'].categories)

        # normalise the values so they sum to 1 and collapse into single array
        co_types_prob_lookup = co_types_prob_lookup.apply(lambda x: x.values / sum(x.values), axis=1)

        # 3. apply probabilities of each contraception type to sim population
        def pick_contraceptive(age):
            """a function that randomly chooses a contraceptive based on age"""
            return self.rng.choice(co_types, p=co_types_prob_lookup.loc[age])

        df.loc[females1549, 'co_contraception'] = df.loc[females1549, 'age_years'].apply(pick_contraceptive)

    def initialise_simulation(self, sim):
        """
        * Schedule the recurring events: ContraceptiveSwitchingPoll, Fail, PregnancyPoll, ContraceptiveLoggingEvent
        """
        # starting contraception, switching contraception metho, and stopping contraception:
        sim.schedule_event(ContraceptionSwitchingPoll(self), sim.date)

        # check all females using contraception to determine if contraception fails i.e. woman becomes
        # pregnant whilst using contraception (starts at month 0)
        sim.schedule_event(Fail(self), sim.date)

        # check all population to determine if pregnancy should be triggered (repeats every month)
        sim.schedule_event(PregnancyPoll(self), sim.date + DateOffset(months=1))

        # Launch the repeating event that will store statistics about the population structure
        sim.schedule_event(ContraceptionLoggingEvent(self), sim.date)

    def on_birth(self, mother_id, child_id):
        """
        * 1) Initialise our properties for a newborn individual.
        * 2) Update mother's properties
        * 3) Update the mother's contraception
        """
        df = self.sim.population.props

        # 1) Initialise child's properties:
        df.loc[child_id, (
            'co_contraception',
            'is_pregnant',
            'date_of_last_pregnancy',
            'co_unintended_preg')
        ] = (
            'not_using',
            False,
            pd.NaT,
            False
        )

        # 2) Reset the mother's is_pregnant status showing that she is no longer pregnant
        df.at[mother_id, 'is_pregnant'] = False

        # 3) Initiate mother of newborn to a contracpetive
        self.select_contraceptive_following_birth(mother_id)

    def process_parameters(self):
        """Process parameters that have been read-in"""
        # todo - this could be tidied-up!
        # =================== ARRANGE INPUTS FOR USE BY REGULAR EVENTS =============================

        # For ContraceptionSwitchingPoll.init1 -----------------------------------------------------

        # from Stata analysis line 250 of initiation rates_age_stcox_2005_2016_5yrPeriods.do:
        # fracpoly: regress _d age_
        # // fracpoly exact age (better fitting model, higher F statistic) - see 'Initiation1 by age'
        # worksheet, results are in 'r_init1_age' sheet
        c_multiplier = self.parameters['r_init1_age']

        # 'irate_1_' sheet created manually
        c_baseline = self.parameters['irate1_']

        # this Excel sheet is irate1_all.csv outputs from 'initiation rates_age_stcox.do'
        # Stata analysis of DHS contraception calendar data
        c_intervention = self.parameters['contraception_interventions']
        c_intervention = pd.DataFrame(c_intervention)
        c_intervention = c_intervention.set_index('contraception').T
        c_intervention1 = c_intervention.iloc[[0]]  # just the first row: multiplier, double brackets for df type
        c_intervention2 = c_intervention.iloc[[2]]  # just the third row: PPFP_multiplier

        c_baseline = c_baseline.drop(columns=['not_using'])
        c_baseline = c_baseline.mul(c_intervention1.iloc[0])  # intervention1 to increase each contraception uptake
        c_baseline = pd.concat([c_baseline] * len(c_multiplier), ignore_index=True)
        c_adjusted = c_baseline.mul(c_multiplier.r_init1_age, axis='index')
        c_adjusted = c_baseline + c_adjusted
        self.parameters['contraception_initiation1'] = c_adjusted.set_index(c_multiplier.age)

        # For ContraceptionSwitchingPoll.switch ----------------------------------------------------
        switching_prob = self.parameters['Switching'].transpose()
        # this Excel sheet is from contraception_failure_discontinuation_switching.csv outputs from
        # 'failure discontinuation switching rates.do' Stata analysis of DHS contraception calendar data
        switching_prob.columns = ['probability']
        self.parameters['contraception_switching'] = switching_prob

        switching_matrix = self.parameters['switching_matrix']
        # this Excel sheet is from contraception switching matrix outputs from line 144 of
        # 'failure discontinuation switching rates.do' Stata analysis of DHS contraception calendar data

        switching_matrix = switching_matrix.set_index('switchfrom')
        switching_matrix = switching_matrix.transpose()
        self.parameters['contraception_switching_matrix'] = switching_matrix

        # For ContraceptionSwitchingPoll.discontinue
        c_baseline = self.parameters['Discontinuation']
        # this Excel sheet is from contraception_failure_discontinuation_switching.csv outputs from
        # 'failure discontinuation switching rates.do' Stata analysis of DHS contraception calendar data
        c_multiplier = self.parameters['r_discont_age']
        c_baseline = pd.concat([c_baseline] * len(c_multiplier), ignore_index=True)
        c_adjusted = c_baseline.mul(c_multiplier.r_discont_age, axis='index')
        c_adjusted = c_baseline + c_adjusted
        self.parameters['contraception_discontinuation'] = c_adjusted.set_index(c_multiplier.age)

        # For on_birth init2 post-partum family planning interventions
        c_baseline = self.parameters['irate2_']
        # this Excel sheet is irate2_all.csv outputs from 'initiation rates_age_stcox.do'
        # Stata analysis of DHS contraception calendar data
        c_baseline = c_baseline.drop(columns=['not_using'])
        c_adjusted = c_baseline.mul(c_intervention2.iloc[0])
        self.parameters['contraception_initiation2'] = c_adjusted.loc[0]

        # Public health costs per year of interventions - sum these annually - in a separate new event?:
        # doesn't seem possible to simply do it in the LoggingEvent as that doesn't have access to parameters
        cost_per_year1 = c_intervention.iloc[[1]]  # cost_per_year_multiplier for increasing r_init1
        cost_per_year2 = c_intervention.iloc[[3]]  # cost_per_year_multiplier for increasing r_init2 PPFP

    def select_contraceptive_following_birth(self, mother_id):
        """
        Initiation of mother's contraception after birth (was previously Init2 event)
        todo - Notes: decide what contraceptive method they have (including not_using, according to
         initiation_rate2 (irate_2)
        Note the irate2s are low as they are just for the month after pregnancy and
        then for the 99.48% who 'initiate' to 'not_using' (i.e. 1 - sum(irate2s))
        they are then subject to the usual irate1s per month
        """
        # - see Contraception-Pregnancy.pdf schematic
        on_birth_co_probs: pd.Series = self.parameters['contraception_initiation2']

        # sample a single row of the init2 probabilities (weighted by those same probabilities)
        chosen_co = on_birth_co_probs.sample(n=1, weights=on_birth_co_probs, random_state=self.rng)

        # update the contracption choice (nb. the index of the row is the contraception type)
        df = self.sim.population.props
        df.at[mother_id, 'co_contraception'] = chosen_co.index[0]

        # Log that this women had initiated this contraceptive following birth:
        logger.info(key='post_birth_contraception',
                    data={
                        'woman_index': mother_id,
                        'co_contraception': df.at[mother_id, 'co_contraception']
                    },
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

    # todo  - Should remove women with hysterectomy from being on contraception

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

        # initiating: not using -> using
        self.init1(df, currently_not_using_co)

        # switching: using A -> using B
        self.switch(df, currently_using_co)

        # discontinuing: using -> not using
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

            # Log information for each woman about the contraceptive being initiated:
            for woman in now_using_co:
                logger.info(key='start_contraception',
                            data={
                                'woman_index': woman,
                                'age': df.at[woman, 'age_years'],
                                'co_contraception': df.at[woman, 'co_contraception']
                            },
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

        # log women that are switching to a new contraceptive
        for woman in switch_co:
            logger.info(key='switch_contraception',
                        data={
                            'woman_index': woman,
                            'co_from': df.at[woman, 'co_contraception'],
                            'co_to': new_co[woman]
                        },
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

            # Log each woman that is discontinuing contraception
            for woman in co_discontinue:
                logger.info(key='stop_contraception',
                            data={
                                'woman_index': woman,
                                'co_from': df.at[woman, 'co_contraception']
                            },
                            description='stop_contraception')


class Fail(RegularEvent, PopulationScopeEventMixin):
    """
    This event looks across all women who are using a contraception method to determine if they become pregnant
    i.e. the method fails according to failure_rate.
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))  # runs every month
        self.age_low = 15
        self.age_high = 49

    def apply(self, population):
        df = population.props
        p = self.module.parameters
        rng = self.module.rng

        prob_of_failure = p['contraception_failure']

        # Get the women who are using a contracpetive that may fail and who may become pregnanct (i.e., women who are
        # not in labour, have been pregnant in the last month, have previously had a hysterectomy,
        # cannot get pregnant.)

        # todo @TimC - should the HIV risk effect manifest on the failure rate too?
        possible_to_fail = ((df.sex == 'F') &
                            df.is_alive &
                            ~df.is_pregnant &
                            ~df.la_currently_in_labour &
                            ~df.la_has_had_hysterectomy &
                            df.age_years.between(self.age_low, self.age_high) &
                            ~df.co_contraception.isin(['not_using', 'female_steralization']) &
                            ~df.la_is_postpartum &
                            (df.ps_ectopic_pregnancy == 'none')
                            )

        prob_of_failure = df.loc[possible_to_fail, 'co_contraception'].map(prob_of_failure)

        # apply increased risk of failure to under 25s
        prob_of_failure.loc[df.index[possible_to_fail & (df.age_years.between(15, 25))]] *= p['rr_fail_under25']

        # randomly select some individual's for contraceptive failure
        random_draw = rng.random_sample(size=len(prob_of_failure))
        women_co_failure = prob_of_failure.index[prob_of_failure > random_draw]

        for woman in women_co_failure:
            # this woman's contraception has failed - she is now pregnant

            # update properties:
            df.loc[woman, (
                'is_pregnant',
                'date_of_last_pregnancy',
                'co_unintended_preg'
            )] = (
                True,
                self.sim.date,
                True
            )

            # Set date of labour in the Labour module:
            self.sim.modules['Labour'].set_date_of_labour(woman)

            # Log that a pregnancy has occured following the failure of a contraceptive:
            logger.info(key='fail_contraception',
                        data={'woman_index': woman},
                        description='pregnancy following the failure of contraceptive method')


class PregnancyPoll(RegularEvent, PopulationScopeEventMixin):
    """
    This event looks across each woman who is not using a contracpetive to determine who will become pregnant.
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        self.age_low = 15
        self.age_high = 49

    def apply(self, population):
        df = population.props  # get the population dataframe

        # get subset of women who are not using a contraceptive and who may become pregnant
        subset = (
            (df.sex == 'F') &
            df.is_alive &
            df.age_years.between(self.age_low, self.age_high) &
            ~df.is_pregnant &
            (df.co_contraception == 'not_using') &
            ~df.la_currently_in_labour &
            ~df.la_has_had_hysterectomy &
            ~df.la_is_postpartum &
            (df.ps_ectopic_pregnancy == 'none')
        )
        # get properties that affect risk of pregnancy
        females = df.loc[subset, ['age_years', 'hv_inf']]

        # load the fertility schedule (imported datasheet from excel workbook)
        fertility_schedule = self.module.parameters['fertility_schedule']

        # load the age-specific effects of HIV
        frr_hiv = self.module.parameters['r_hiv']

        # get the probability of pregnancy for each woman in the model, through merging with the fert_schedule data and
        # the 'r_hiv' parameter
        len_before_merge = len(females)
        females = females.reset_index().merge(
            fertility_schedule, left_on=['age_years'], right_on=['age'], how='left'
        ).merge(
            frr_hiv, left_on=['age_years'], right_on=['age_'], how='left'
                ).set_index('person')
        assert len(females) == len_before_merge

        # probability of pregnancy
        annual_risk_of_pregnancy = females.basefert_dhs
        annual_risk_of_pregnancy.loc[females.hv_inf] *= females.frr_hiv
        monthly_risk_of_pregnancy = 1 - np.exp(-annual_risk_of_pregnancy / 12.0)

        # flipping the coin to determine which women become pregnant
        newly_pregnant_ids = females.index[(self.module.rng.rand(len(females)) < monthly_risk_of_pregnancy)]

        # updating the pregancy status for that women
        df.loc[newly_pregnant_ids, 'is_pregnant'] = True
        df.loc[newly_pregnant_ids, 'date_of_last_pregnancy'] = self.sim.date
        # todo @TimC - should we set co_unintended_preg = False in this case?

        # loop through each newly pregnant women in order to schedule them to have labour
        # todo - if we unify the logging, factorize so that all pregnancies go through same function (from Fail and from PregnancyPoll)
        for female_id in newly_pregnant_ids:
            self.sim.modules['Labour'].set_date_of_labour(female_id)

            # Log each pregnancy that occurs to a woman not using contraceptive
            logger.info(key='pregnant_at_age_',
                        data={
                            'person_id': female_id,
                            'age_years': females.at[female_id, 'age_years']
                        },
                        description='pregnany to a woman not on contraceptive')


class ContraceptionLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """Logs state of contraceptive usage in the population each year."""
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))
        self.age_low = 15
        self.age_high = 49

        self.get_costs_of_each_contraceptive()

    def get_costs_of_each_contraceptive(self):
        # cost_pill = pd.Series(consumables.loc[
        #                                  consumables['Intervention_Pkg']
        #                                  == 'Pill',
        #                                  'item_cost']).sum()    # adds all item costs

        # cost_IUD = pd.Series(consumables.loc[
        #                          consumables['Intervention_Pkg']
        #                          == 'IUD',
        #                          'item_cost']).sum()  # adds all item costs

        # cost_injections = pd.Series(consumables.loc[
        #                                  consumables['Intervention_Pkg']
        #                                  == 'Injectable',
        #                                  'item_cost']).sum()    # adds all item costs

        # cost_implant = pd.Series(consumables.loc[
        #                                  consumables['Intervention_Pkg']
        #                                  == 'Implant',
        #                                  'item_cost']).sum()    # adds all item costs

        # cost_male_condom = pd.Series(consumables.loc[
        #                                  consumables['Intervention_Pkg']
        #                                  == 'Male condom',
        #                                  'item_cost']).sum()

        # cost_female_sterilization = pd.Series(consumables.loc[
        #                                  consumables['Intervention_Pkg']
        #                                  == 'Female sterilization',
        #                                  'item_cost']).sum()    # adds all item costs

        # cost_other_modern = pd.Series(consumables.loc[
        #                                    consumables['Intervention_Pkg']
        #                                    == 'Female Condom',
        #                                    'item_cost']).sum()  # adds all item costs
        # nb- estimated to be equal to the cost of a female condom
        pass

    def apply(self, population):
        df = population.props

        # 1) Log current distributuion of persons who are pregnant / not pregnant
        # todo - @TimC - is this really useful!?!? also note that it includes men! if so could be simplified to
        #  ```logger.info(key='pregnancy', data=df.loc[df.is_alive & df.age_years.between(self.age_low, self.age_high)].is_pregnant.value_counts())```
        preg_counts = df[df.is_alive & df.age_years.between(self.age_low, self.age_high)].is_pregnant.value_counts()
        is_preg_count = (df.is_alive & df.age_years.between(self.age_low, self.age_high) & df.is_pregnant).sum()
        is_notpreg_count = (df.is_alive & df.age_years.between(self.age_low, self.age_high) & ~df.is_pregnant).sum()

        logger.info(key='pregnancy',
                    data={
                        'total': sum(preg_counts),
                        'pregnant': is_preg_count,
                        'not_pregnant': is_notpreg_count
                    },
                    description='pregnancy')

        # 2) Log usage of contracpetive (and compute costs)
        # todo @TimC - I've made this work for now, but note that this is only giving a rough estimate and its based
        #  on a cross-sectional measure taken every year and people will start and stop in the intervening time. The log
        #  already contains the full information (every person that starts and stops and the dates), so I think it'd
        #  be better to contruct estimates of cost outside of the simulation and in the analysis files. If you did want
        #  to continue using this, then to be accurate its frequency must be at lesst equal to monthly.

        c_intervention = self.module.parameters['contraception_interventions']

        # Public health costs per year of interventions - sum these annually below:
        c_intervention = pd.DataFrame(c_intervention)
        c_intervention = c_intervention.set_index('contraception').T
        cost_per_year1 = c_intervention.iloc[1]  # cost_per_year_multiplier for increasing r_init1
        cost_per_year2 = c_intervention.iloc[3]  # cost_per_year_multiplier for increasing r_init2 PPFP

        # Costs for each contraceptive
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        # multiply each Unit_Cost by Expected_Units_Per_Case so they can be summed for all Items for each contraceptive
        # package to get cost of each contraceptive user for each contraceptive
        item_cost = self.sim.modules['HealthSystem'].parameters['Consumables']['Expected_Units_Per_Case'] * \
            self.sim.modules['HealthSystem'].parameters['Consumables']['Unit_Cost']
        consumables['item_cost'] = item_cost

        # # Pill

        # pill_users = df.loc[df.co_contraception == 'pill']
        # # df.loc[pill_users.index, 'pill_costs'] = cost_pill
        # # pill_costs = df.pill_costs.sum()

        # # IUD

        # IUD_users = df.loc[df.co_contraception == 'IUD']
        # # df.loc[IUD_users.index, 'IUD_costs'] = cost_IUD
        # # IUD_costs = df.IUD_costs.sum()

        # # Injections

        # injections_users = df.loc[df.co_contraception == 'injections']
        # # df.loc[injections_users.index, 'injections_costs'] = cost_injections
        # # injections_costs = df.injections_costs.sum()

        # # Implants

        # implant_users = df.loc[df.co_contraception == 'implant']
        # # df.loc[implant_users.index, 'implant_costs'] = cost_implant
        # # implant_costs = df.implant_costs.sum()

        # # Male condoms

        # male_condom_users = df.loc[df.co_contraception == 'male_condom']
        # # df.loc[male_condom_users.index, 'male_condom_costs'] = cost_male_condom
        # # male_condom_costs = df.male_condom_costs.sum()

        # # Female Sterilization

        # female_sterilization_users = df.loc[df.co_contraception == 'female_sterilization']
        # # df.loc[female_sterilization_users.index, 'female_sterilization_costs'] = cost_female_sterilization
        # female_sterilization_costs = df.female_sterilization_costs.sum()

        # Other Modern

        # other_modern_users = df.loc[df.co_contraception == 'other_modern']
        # # df.loc[other_modern_users.index, 'female_condom_costs'] = cost_female_condom
        # female_condom_costs = df.female_condom_costs.sum()
        #

        # contraception_count = df[df.is_alive & df.age_years.between(self.age_low, self.age_high)].groupby(
        #     'co_contraception').size()
        #
        # contraception_summary = {
        #     'total': sum(contraception_count),
        #     'not_using': contraception_count['not_using'],
        #     'using': sum(contraception_count) - contraception_count['not_using'],
        #     'pill': contraception_count['pill'],
        #     'IUD': contraception_count['IUD'],
        #     'injections': contraception_count['injections'],
        #     'implant': contraception_count['implant'],
        #     'male_condom': contraception_count['male_condom'],
        #     'female_sterilization': contraception_count['female_sterilization'],
        #     'female_condom': contraception_count['other_modern'],
        #     'periodic_abstinence': contraception_count['periodic_abstinence'],
        #     'withdrawal': contraception_count['withdrawal'],
        #     'other_traditional': contraception_count['other_traditional'],
        #
        #     # costs
        #     'public_health_costs1': sum(cost_per_year1),
        #     'public_health_costs2': sum(cost_per_year2),
        #     'pill_costs': pill_costs,
        #     'IUD_costs': IUD_costs,
        #     'injections_costs': injections_costs,
        #     'implant_costs': implant_costs,
        #     'male_condom_costs': male_condom_costs,
        #     'female_sterilization_costs': female_sterilization_costs,
        #     'female_condom_costs': female_condom_costs,
        # }

        # logger.info(key='contraception_summary',
        #             data=contraception_summary,
        #             description='contraception_summary')
