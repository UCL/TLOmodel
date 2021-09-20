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
# * We could encapsulate each init and swtich inside an HSI pretty easily now -- shall we do this? Could make it an
optional thing (argument for ```no_hsi=True``` could preserve current behaviour)
# todo  - Should remove women with hysterectomy from being on contraception??
"""


class Contraception(Module):
    """
    Contraception module covering baseline contraception methods use, failure (pregnancy),
    Switching contraceptive methods, and discontinuation rates by age.
    """

    INIT_DEPENDENCIES = {'Demography', 'HealthSystem'}
    ADDITIONAL_DEPENDENCIES = {'Labour', 'PregnancySupervisor', 'Hiv'}

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
                                           'Monthly probabilities of failure of each contraception method to '
                                           'pregnancy'),
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
        'contraception_interventions': Parameter(Types.DATA_FRAME, 'contraception interventions'),
    }

    # Declare Properties
    PROPERTIES = {
        'co_contraception': Property(Types.CATEGORICAL, 'Current contraceptive method',
                                     categories=[
                                         'not_using',
                                         'pill',
                                         'IUD',
                                         'injections',
                                         'implant',
                                         'male_condom',
                                         'female_sterilization',
                                         'other_modern',
                                         'periodic_abstinence',
                                         'withdrawal',
                                         'other_traditional'
                                     ]),
        # These are the 11 categories of contraception ('not using' + 10 methods) from the DHS analysis of initiation,
        # discontinuation, failure and switching rates.
        # 'other modern' includes Male sterilization, Female Condom, Emergency contraception
        # 'other traditional' includes lactational amenohroea (LAM),  standard days method (SDM), 'other traditional
        #  method'),

        'is_pregnant': Property(Types.BOOL, 'Whether this individual is currently pregnant'),
        'date_of_last_pregnancy': Property(Types.DATE, 'Date of the last pregnancy of this individual'),
        'co_unintended_preg': Property(Types.BOOL, 'Most recent or current pregnancy was unintended (following '
                                                   'contraception failure)'),
    }

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.all_contraception_states = set(self.PROPERTIES['co_contraception'].categories)

    def read_parameters(self, data_folder):
        """
        Import the relevant sheets from the ResourceFile (an excel workbook).
        Please see documentation for description of the relationships between baseline fertility rate, intitiation
        rates, discontinuation, failure and switching rates, and being on contraception or not and being pregnant.
        """
        # todo - something in here is generrating a warning? Yes, not sure what, the message is:
        #   /Users/timothycolbourn/opt/anaconda3/envs/tlo38/lib/python3.8/site-packages/openpyxl/worksheet/
        #   _reader.py:308: UserWarning: Unknown extension is not supported and will be removed warn(msg)

        workbook = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_Contraception.xlsx', sheet_name=None)

        self.parameters['fertility_schedule'] = workbook['Age_spec fertility']
        # todo @TimC - is this the WPP fertility scheudle? To keep things in sync, I'd suggest we use instead:
        #  "ResourceFile_ASFR_WPP.csv"  Response: this needs to be baseline fertility without contraceptives so I don't
        #  think we can use WPP here. It matches fairly well though (see my workbook calibration.xlsx)

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
        df.loc[df.is_alive, 'co_unintended_preg'] = False  # todo should this be np.nan? Response: this is BOOL like
        #   'is_pregnant' two lines above, so should be False?

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
        sim.schedule_event(ContraceptionPoll(self), sim.date)

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
            False  # todo should this be np.nan? Response: this is BOOL like
            #   'is_pregnant' two lines above, so should be False?
        )

        # 2) Reset the mother's is_pregnant status showing that she is no longer pregnant
        df.at[mother_id, 'is_pregnant'] = False

        # 3) Initiate mother of newborn to a contracpetive
        self.select_contraceptive_following_birth(mother_id)

    def process_parameters(self):
        """Process parameters that have been read-in"""
        # todo - this could be tidied-up! Response:  ok, though no time right now sorry and don't want to break it now
        #  it's doing what I want it too :)
        # =================== ARRANGE INPUTS FOR USE BY REGULAR EVENTS =============================

        # For ContraceptionPoll.init1 -----------------------------------------------------

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

        # For ContraceptionPoll.switch ----------------------------------------------------
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

        # For ContraceptionPoll.discontinue
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

    def select_contraceptive_following_birth(self, mother_id):
        """
        Initiation of mother's contraception after birth (was previously Init2 event)
        Decide what contraceptive method they have (including not_using, according to
         initiation_rate2 (irate_2)
        Note the irate2s are low as they are just for the month after pregnancy and
        then for the 99.48% who 'initiate' to 'not_using' (i.e. 1 - sum(irate2s))
        they are then subject to the usual irate1s per month
        """
        # - see Contraception-Pregnancy.pdf schematic
        on_birth_co_probs: pd.Series = self.parameters['contraception_initiation2']

        # sample a single row of the init2 probabilities (weighted by those same probabilities)
        chosen_co = on_birth_co_probs.sample(n=1, weights=on_birth_co_probs, random_state=self.rng)

        # update the contraception choice (nb. the index of the row is the contraception type)
        df = self.sim.population.props
        df.at[mother_id, 'co_contraception'] = chosen_co.index[0]

        # though don't allow female sterilization to any woman below 30
        is_younger_woman = df.at[mother_id, 'age_years'] < 30
        female_sterilization = chosen_co.index == 'female_sterilization'
        if is_younger_woman & female_sterilization == [True]:
            df.at[mother_id, 'co_contraception'] = 'not_using'

        # Log that this women had initiated this contraceptive following birth:
        self.log_contraception_change(mother_id,
                                      old='not_using',
                                      new=df.at[mother_id, 'co_contraception'],
                                      init_after_pregnancy=True)

    def log_contraception_change(self, woman_id: int, old, new, init_after_pregnancy=False):
        """Log a start / stop / switch of contraception. """
        assert old in self.all_contraception_states
        assert new in self.all_contraception_states

        df = self.sim.population.props
        woman = df.loc[woman_id]
        logger.info(key='contraception',
                    data={
                        'woman': woman_id,
                        'age': woman['age_years'],
                        'switch_from': old,
                        'switch_to': new,
                        'init_after_pregnancy': init_after_pregnancy
                    },
                    description='All changes in contraception use'
                    )


class ContraceptionPoll(RegularEvent, PopulationScopeEventMixin):
    """
    The regular poll (monthly) for the Contraceptive Module.
    * Determines contraceptive use
    * Pregnancy
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        self.age_low = 15
        self.age_high = 49

    def apply(self, population):
        """Update contraceptive method and determine who will become pregnant."""

        # Determine who will become pregnant:
        self.update_pregnancy()

        # Update contraception:
        self.update_contraceptive()

    def update_contraceptive(self):
        """ Determine women that will start, stop or switch contraceptive method.
        1) For all women who are 'not_using' contraception to determine if they start using each method according to
        initiation_rate1 (irate_1).
        2) For all women who are using a contraception method to determine if they switch to another method according to
        switching_rate, and then for those that switch directs towards a new method according to switching_matrix.
        3) For all women who are using a contraception method to determine if they stop using it according to
        discontinuation_rate
        """

        df = self.sim.population.props

        possible_co_users = ((df.sex == 'F') &
                             df.is_alive &
                             df.age_years.between(self.age_low, self.age_high) &
                             ~df.is_pregnant)

        currently_using_co = df.index[possible_co_users &
                                      ~df.co_contraception.isin(['not_using', 'female_steralization'])]
        currently_not_using_co = df.index[possible_co_users & (df.co_contraception == 'not_using')]

        # initiating: not using -> using
        self.initiate(df, currently_not_using_co)

        # switching: using A -> using B
        self.switch(df, currently_using_co)

        # discontinuing: using -> not using
        self.discontinue(df, currently_using_co)

    def initiate(self, df: pd.DataFrame, individuals_not_using: pd.Index):
        """check all females not using contraception to determine if contraception starts
        i.e. category should change from 'not_using'
        """
        # exit if there are no individuals currenlty not using a contraceptive:
        if not len(individuals_not_using):
            return

        p = self.module.parameters
        rng = self.module.rng

        c_multiplier = p['r_init_year'].at[self.sim.date.year, 'r_init_year1']
        co_start_prob_by_age = p['contraception_initiation1'].mul(c_multiplier)
        co_start_prob_by_age['not_using'] = 1 - co_start_prob_by_age.sum(axis=1)

        # all the contraceptive types we can randomly choose
        co_types = list(co_start_prob_by_age.columns)

        def pick_random_contraceptive(age):
            """random selects a contraceptive using probabilities for given age"""
            return rng.choice(co_types, p=co_start_prob_by_age.loc[age])

        # select a random contraceptive for everyone not currently using
        random_co = df.loc[individuals_not_using, 'age_years'].apply(pick_random_contraceptive)

        # though don't allow female sterilization to any woman below 30
        younger_women = df.loc[random_co.index, 'age_years'] < 30
        female_sterilization = random_co.loc[younger_women.loc[younger_women].index] == 'female_sterilization'
        random_co.loc[female_sterilization.loc[female_sterilization].index] = 'not_using'

        # get index of all those now using
        now_using_co = random_co.index[random_co != 'not_using']

        # only update entries for all those now using a contraceptive
        df.loc[now_using_co, 'co_contraception'] = random_co[now_using_co]

        # log women that are starting a new contraceptive
        for woman in now_using_co:
            self.module.log_contraception_change(woman, old='not_using', new=df.at[woman, 'co_contraception'])

    def switch(self, df: pd.DataFrame, individuals_using: pd.Index):
        """check all females using contraception to determine if contraception Switches
        i.e. category should change from any method to a new method (not including 'not_using')
        """
        # exit if there are no individuals currenlty using a contraceptive:
        if not len(individuals_using):
            return

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

        # though don't allow female sterilization to any woman below 30
        younger_women = df.loc[new_co.index, 'age_years'] < 30
        female_sterilization = new_co.loc[younger_women.loc[younger_women].index] == 'female_sterilization'
        # make them switch to injections instead
        # Todo: ideally this should revert to no switch (i.e. stay on the method they are using, though couldn't see
        #  how to do this; note that (reverting to 'not_using' throws an error in discontinuation as we are in
        #  individuals_using here)
        new_co.loc[female_sterilization.loc[female_sterilization].index] = 'injections'

        # log women that are switching to a new contraceptive
        for woman in switch_co:
            self.module.log_contraception_change(woman, old=df.at[woman, 'co_contraception'], new=new_co[woman])

        # update contraception for all who switched
        df.loc[switch_co, 'co_contraception'] = new_co

    def discontinue(self, df: pd.DataFrame, individuals_using: pd.Index):
        """check all females using contraception to determine if contraception discontinues
        i.e. category should change to 'not_using'
        """
        # exit if there are no individuals currenlty using a contraceptive:
        if not len(individuals_using):
            return

        p = self.module.parameters
        rng = self.module.rng

        c_multiplier = p['r_discont_year'].at[self.sim.date.year, 'r_discont_year1']
        c_adjustment = p['contraception_discontinuation'].mul(c_multiplier)

        def get_prob_discontinued(row):
            """returns the probability of discontinuing contraceptive based on age and current
            contraceptive"""
            return c_adjustment.loc[row.age_years, row.co_contraception]

        # get the probability of discontinuing for all currently using
        discontinue_prob = df.loc[individuals_using].apply(get_prob_discontinued, axis=1)

        # random choose some to discontinue
        co_discontinue = discontinue_prob.index[discontinue_prob > rng.rand(len(individuals_using))]

        # Log information for each woman about the contraceptive being initiated:
        for woman in co_discontinue:
            self.module.log_contraception_change(woman, old=df.at[woman, 'co_contraception'], new='not_using')

        # Update contraception property:
        df.loc[co_discontinue, 'co_contraception'] = 'not_using'

    def update_pregnancy(self):
        """Determine who will become pregnant"""

        # Determine pregnancy for those on a contraceptive ("unintentional pregnancy")
        self.pregnancy_for_those_on_contraceptive()

        # Determine pregnancy for those not on contraceptive ("intentional pregnancy")
        self.pregnancy_for_those_not_on_contraceptive()

    def pregnancy_for_those_on_contraceptive(self):
        """Look across all women who are using a contraception method to determine if they become pregnant i.e. the
         method fails according to failure_rate."""

        def apply(self, population):  # Todo: why is apply greyed out here in Python? is it somehow not being used?!
            df = population.props
            p = self.module.parameters
            rng = self.module.rng

            prob_of_failure = p['contraception_failure']

            # Get the women who are using a contracpetive that may fail and who may become pregnanct (i.e., women who
            # are not in labour, have been pregnant in the last month, have previously had a hysterectomy, cannot get
            # pregnant.)

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
            # todo convert 12mo to 1mo probability of pregnancy - just checked and failure rates are already monthly so
            #  not sure why this to do comment was added, I think it can be deleted
            random_draw = rng.random_sample(size=len(prob_of_failure))
            women_co_failure = prob_of_failure.index[prob_of_failure > random_draw]

            # Effect these women to be pregnant now:
            self.set_new_pregnancy(women_id=women_co_failure)

    def pregnancy_for_those_not_on_contraceptive(self):
        """
        This event looks across each woman who is not using a contracpetive to determine who will become pregnant.
        """
        df = self.sim.population.props  # get the population dataframe

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

        # Effect these women to be pregnant now:
        self.set_new_pregnancy(women_id=newly_pregnant_ids)

    def set_new_pregnancy(self, women_id: list):
        """Effect that these women are now pregnancy and enter to the log"""
        df = self.sim.population.props

        for w in women_id:
            woman = df.loc[w]

            # Determine if this is unintended or not.  For now let this be the simple rule of if it 'unintended' if
            # the women is using a contraceptive.
            # todo - @TimC - update this logic as you see fit! response: this looks good, thanks Tim!
            unintended = (woman['co_contraception'] != 'not_using')

            # Update properties:
            df.loc[w, (
                'is_pregnant',
                'date_of_last_pregnancy',
                'co_unintended_preg'
            )] = (
                True,
                self.sim.date,
                unintended
            )

            # Set date of labour in the Labour module:
            self.sim.modules['Labour'].set_date_of_labour(w)

            # Log that a pregnancy has occured following the failure of a contraceptive:
            logger.info(key='pregnancy',
                        data={
                            'woman_index': w,
                            'age_years': woman['age_years'],
                            'contraception': woman['co_contraception'],
                            'unintended_preg': unintended
                        },
                        description='pregnancy following the failure of contraceptive method')

        # todo: @TimC/Joe - should a woman who is now pregnant stop any contraceptive method she may be on (apart from
        #  'female_sterilization')? Response, yes, and even female_sterilization should be set back to not_using I think
        #  given it failed. Though currently there is a zero probability of failre of female_sterlization


class ContraceptionLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """Logs state of contraceptive usage in the population each year."""
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))
        self.age_low = 15
        self.age_high = 49

        self.get_costs_of_each_contraceptive()

    def get_costs_of_each_contraceptive(self):
        """Get the cost for a year's useage of the consumable"""

        # Costs for each contraceptive
        # We multiply each Unit_Cost by Expected_Units_Per_Case so they can be summed for all Items for each
        # contraceptive package to get cost of each contraceptive user for each contraceptive.
        # todo @TimC - do you want each to represent a year's use? Response: Yes please
        # NB. Cost of "other modern method" is estimated to be equal to the cost of a female condom
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        item_cost = consumables['Expected_Units_Per_Case'] * consumables['Unit_Cost']

        self.costs = dict()
        self.costs['pill'] = item_cost.loc[consumables['Intervention_Pkg'] == 'Pill'].sum()
        self.costs['IUD'] = item_cost.loc[consumables['Intervention_Pkg'] == 'IUD'].sum()
        self.costs['injections'] = item_cost.loc[consumables['Intervention_Pkg'] == 'Injectable'].sum()
        self.costs['implant'] = item_cost.loc[consumables['Intervention_Pkg'] == 'Implant'].sum()
        self.costs['male_condom'] = item_cost.loc[consumables['Intervention_Pkg'] == 'Male condom'].sum()
        self.costs['female_sterilization'] = item_cost.loc[
            consumables['Intervention_Pkg'] == 'Female sterilization'].sum()
        self.costs['other_modern'] = item_cost.loc[consumables['Intervention_Pkg'] == 'Female Condom'].sum()

        assert set(self.costs.keys()).issubset(self.module.all_contraception_states)
        assert set(self.costs.keys()) == set(['male_condom', 'injections', 'other_modern', 'IUD', 'pill',
                                              'female_sterilization', 'implant'])

    def apply(self, population):
        df = population.props

        # Log usage of contracpetive
        num_using = df.loc[df.is_alive & (df.sex == 'F'), 'co_contraception'].value_counts().to_dict()
        logger.info(key='contraception_use_yearly_summary',
                    data=num_using,
                    description='Counts of women on each type of contraceptive at a point each time.')

        # Log costs associated with the consumables used for this pattern of usage (if annualised)
        # todo @TimC - I've made this work for now, but note that this is only giving a rough estimate and its based
        #  on a cross-sectional measure taken every year and people will start and stop in the intervening time. The log
        #  already contains the full information (every person that starts and stops and the dates), so I think it'd
        #  be better to contruct estimates of cost outside of the simulation and in the analysis files. If you did want
        #  to continue using this, then to be accurate its frequency must be at lesst equal to monthly. I haven't made
        #  that change to allow your script files to still work, but if you do, you would need to adjust the cost calcs.
        #  Also - note that the HSI system automatically keeps track of all consumables etc, so either way I think it's
        #  not neccessary, but this working works as intended, I think.
        #   Response: Thanks Tim, maybe as per your comment on Git it's worth adding in the HSIs instead - very happy
        #   for you to do this if it's relatively easy for you (your comment: "We could encapsulate each init and
        #   swtich inside an HSI pretty easily now -- shall we do this? Could make it an optional thing
        #   (argument for no_hsi=True could preserve current behaviour)"

        # Public health costs per year of interventions - sum these annually below:
        c_intervention = self.module.parameters['contraception_interventions']
        c_intervention = c_intervention.set_index('contraception').T
        cost_per_year1 = c_intervention.iloc[1]  # cost_per_year_multiplier for increasing r_init1
        cost_per_year2 = c_intervention.iloc[3]  # cost_per_year_multiplier for increasing r_init2 PPFP

        costs = {
            'public_health_costs1': sum(cost_per_year1),
            'public_health_costs2': sum(cost_per_year2)
        }

        # Get cost for provisioning of each type of contraceptive
        for method in self.costs.keys():
            costs.update({
                f"{method}_annual_cost": num_using[method] * self.costs[method]
            })

        logger.info(key='contraception_costs_yearly_summary',
                    data=costs,
                    description='Annual cost (if current pattern of usaage is annualised) for the consumables required'
                                'for each contraceptive method')
