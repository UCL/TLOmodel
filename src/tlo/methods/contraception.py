from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods.healthsystem import HSI_Event
from tlo.util import random_date, transition_states

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Contraception(Module):
    """Contraception module covering baseline contraception methods use, failure (pregnancy),
    Switching contraceptive methods, and discontinuation rates by age."""

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
                           'Proportional change in fertility rate for HIV+ compared to HIV- by age group'),
        'contraception_interventions': Parameter(Types.DATA_FRAME, 'contraception interventions'),
        'days_between_appts_for_maintenance': Parameter(Types.INT, 'The number of days between Family Planning '
                                                                   'appointments when the woman is maintianng the use'
                                                                   'of her current contraceptive')
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
        # These are the 11 categories of contraception ('not_using' + 10 methods) from the DHS analysis of initiation,
        # discontinuation, failure and switching rates.
        # 'other modern' includes Male sterilization, Female Condom, Emergency contraception
        # 'other traditional' includes lactational amenohroea (LAM),  standard days method (SDM), 'other traditional
        #  method'),

        'is_pregnant': Property(Types.BOOL, 'Whether this individual is currently pregnant'),
        'date_of_last_pregnancy': Property(Types.DATE, 'Date of the last pregnancy of this individual'),
        'co_unintended_preg': Property(Types.BOOL, 'Most recent or current pregnancy was unintended (following '
                                                   'contraception failure)'),
        'date_of_last_appt_for_maintenance': Property(Types.DATE,
                                                      'The date of the most recent Family Planning appointment at which'
                                                      'the current method was provided (pd.NaT if the person is not'
                                                      'using any contraceptive method). This is used to determine if '
                                                      'a Family Planning appointment is needed to maintain the person'
                                                      'on the contraceptive. If the person is to maintain use of the '
                                                      'current contraceptive, they will have an HSI if the days elapsed'
                                                      'since this value exceeds the parameter '
                                                      '`days_between_appts_for_maintenance`.'
                                                      )
    }

    def __init__(self, name=None, resourcefilepath=None, use_healthsystem=True):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.all_contraception_states = set(self.PROPERTIES['co_contraception'].categories)
        self.states_that_may_require_HSI_to_switch_to = {'male_condom', 'injections', 'other_modern', 'IUD', 'pill',
                                                         'female_sterilization', 'implant'}
        assert self.states_that_may_require_HSI_to_switch_to.issubset(self.all_contraception_states)

        self.use_healthsystem = use_healthsystem  # True: initiation and switches to contracption require an HSI;
        # False: initiation and switching do not occur through an HSI
        self.cons_codes = dict()  # (Will store the consumables codes for use in the HSI)

        self.rng2 = None  # This will be a second random number generator, used for things to do with scheduling HSI

    def read_parameters(self, data_folder):
        """
        Import the relevant sheets from the ResourceFile (an excel workbook).
        Please see documentation for description of the relationships between baseline fertility rate, intitiation
        rates, discontinuation, failure and switching rates, and being on contraception or not and being pregnant.
        """
        workbook = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_Contraception.xlsx', sheet_name=None)

        self.parameters['fertility_schedule'] = workbook['Age_spec fertility']

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

        self.parameters['days_between_appts_for_maintenance'] = 180

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
        df.loc[df.is_alive, 'date_of_last_appt_for_maintenance'] = pd.NaT

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

        # 4. Give a notional date on which the last appointment occured for those that need them
        needs_appts = females1549 & df['co_contraception'].isin(self.states_that_may_require_HSI_to_switch_to)
        df.loc[needs_appts, 'date_of_last_appt_for_maintenance'] = pd.Series([
            random_date(
                self.sim.date - pd.DateOffset(days=self.parameters['days_between_appts_for_maintenance']),
                self.sim.date - pd.DateOffset(days=1),
                self.rng) for _ in range(len(needs_appts))
        ])

    def initialise_simulation(self, sim):
        """
        * Schedule the recurring events: ContraceptiveSwitchingPoll, Fail, PregnancyPoll, ContraceptiveLoggingEvent
        * Retreive the consumables codes for the consumables used
        """
        # Launch the repeating event that will store statistics about the population structure
        sim.schedule_event(ContraceptionLoggingEvent(self), sim.date)

        # starting contraception, switching contraception metho, and stopping contraception:
        sim.schedule_event(ContraceptionPoll(self), sim.date)

        # Retrieve the consumables codes for the consumables used
        if self.use_healthsystem:
            self.cons_codes = self.get_item_code_for_each_contraceptive()

        # Create second random number generator
        self.rng2 = np.random.RandomState(self.rng.randint(2 ** 31 - 1))

    def on_birth(self, mother_id, child_id):
        """
        * 1) Initialise our properties for a newborn individual.
        * 2) Update mother's properties
        * 3) Update the mother's contraception following the birth
        """
        df = self.sim.population.props

        # 1) Initialise child's properties:
        df.loc[child_id, (
            'co_contraception',
            'is_pregnant',
            'date_of_last_pregnancy',
            'co_unintended_preg',
            'date_of_last_appt_for_maintenance'
        )
        ] = (
            'not_using',
            False,
            pd.NaT,
            False,
            pd.NaT
        )

        # 2) Reset the mother's is_pregnant status showing that she is no longer pregnant
        df.at[mother_id, 'is_pregnant'] = False

        # 3) Initiate mother of newborn to a contracpetive
        self.select_contraceptive_following_birth(mother_id)

    def process_parameters(self):
        """Process parameters that have been read-in"""
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
        # The switching matrix is conditonnal on there being a switch (so there is no "not_using" and no switch_from
        # "female_sterilization"
        switching_matrix = switching_matrix.set_index('switchfrom')
        switching_matrix = switching_matrix.transpose()

        assert set(switching_matrix.columns) == (self.all_contraception_states - {"not_using", "female_sterilization"})
        assert set(switching_matrix.index) == (self.all_contraception_states - {"not_using"})
        assert np.isclose(1.0, switching_matrix.sum()).all()
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
        self.parameters['contraception_initiation2'] = \
            self.parameters['irate2_'].iloc[0] / self.parameters['irate2_'].iloc[0].sum()
        assert set(self.parameters['contraception_initiation2'].index) == self.all_contraception_states
        assert np.isclose(1.0, self.parameters['contraception_initiation2'].sum())

    def select_contraceptive_following_birth(self, mother_id):
        """
        Initiation of mother's contraception after birth.
        Decide what contraceptive method they have (including not_using, according to
         initiation_rate2 (irate_2)
        Note the irate2s are low as they are just for the month after pregnancy and
        then for the 99.48% who 'initiate' to 'not_using' (i.e. 1 - sum(irate2s))
        they are then subject to the usual irate1s per month
        """

        # sample a single row of the `init2` probabilities
        new_contraceptive = self.rng.choice(
            self.parameters['contraception_initiation2'].index,
            p=self.parameters['contraception_initiation2'].values
        )

        # ... but don't allow female sterilization to any woman below 30: reset to 'not_using'
        if (self.sim.population.props.at[mother_id, 'age_years'] < 30) and (
            new_contraceptive == 'female_sterilization'
        ):
            new_contraceptive = 'not_using'

        # Do the change in contraceptive
        self.schedule_batch_of_contraceptive_changes(ids=[mother_id], old=['not_using'], new=[new_contraceptive])

    def get_item_code_for_each_contraceptive(self):
        """Get the item_code for each contraceptive"""
        # NB. The consumable female condom is used for the contraceptive state of "other modern method"

        lookup = self.sim.modules['HealthSystem'].parameters['Consumables'][
            ['Intervention_Pkg', 'Intervention_Pkg_Code']
        ].drop_duplicates().set_index('Intervention_Pkg')['Intervention_Pkg_Code'].to_dict()

        _cons_codes = dict()
        _cons_codes['pill'] = lookup['Pill']
        _cons_codes['IUD'] = lookup['IUD']
        _cons_codes['injections'] = lookup['Injectable']
        _cons_codes['implant'] = lookup['Implant']
        _cons_codes['male_condom'] = lookup['Male condom']
        _cons_codes['female_sterilization'] = lookup['Female sterilization']
        _cons_codes['other_modern'] = lookup['Female Condom']

        assert set(_cons_codes.keys()) == set(self.states_that_may_require_HSI_to_switch_to)
        return _cons_codes

    def schedule_batch_of_contraceptive_changes(self, ids, old, new):
        """Enact the change in contraception, either through scheduling HSI or instantaneously.
        ids: pd.Index of the woman for whom the contraceptive state is changing
        old: itterable giving the corresponding contraceptive state being switched from
        new: itterable giving the corresponding contraceptive state being switched to

        It is assumed that even with the option `self.use_healthsystem=True` that switches to certain methods do not
        require the use of HSI (these are in `states_that_may_require_HSI_to_switch_to`)."""

        df = self.sim.population.props
        date_today = self.sim.date
        days_between_appts = self.parameters['days_between_appts_for_maintenance']

        date_of_last_appt = df.loc[ids, "date_of_last_appt_for_maintenance"].to_dict()

        for _woman_id, _old, _new in zip(ids, old, new):

            # If the new method requires an HSI to be implemented, schedule the HSI:
            if (
                self.use_healthsystem and
                (_new in self.states_that_may_require_HSI_to_switch_to) and
                (
                    (_old != _new) or
                    pd.isnull(date_of_last_appt[_woman_id]) or
                    ((date_today - date_of_last_appt[_woman_id]).days >= days_between_appts)
                )
            ):
                # If this is a change, or its maintenance and time for an appointment, schedule an appointment
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    hsi_event=HSI_Contraception_FamilyPlanningAppt(
                        person_id=_woman_id,
                        module=self,
                        old_contraceptive=_old,
                        new_contraceptive=_new
                    ),
                    topen=random_date(
                        # scatter dates over the month
                        self.sim.date, self.sim.date + pd.DateOffset(days=28), self.rng2),
                    tclose=None,
                    priority=1
                )
            else:
                # Otherwise, implement the change immediately (ignore things which aren't changes)
                if _old != _new:
                    self.do_and_log_individual_contraception_change(woman_id=_woman_id, old=_old, new=_new)
                else:
                    pass  # no need to do anything if the old is the same as the new and no HSI needed.

    def do_and_log_individual_contraception_change(self, woman_id: int, old, new):
        """Implement and then log a start / stop / switch of contraception. """
        assert old in self.all_contraception_states
        assert new in self.all_contraception_states

        df = self.sim.population.props

        # Do the change
        df.at[woman_id, "co_contraception"] = new

        # Log the change
        woman = df.loc[woman_id]
        logger.info(key='contraception_change',
                    data={
                        'woman_id': woman_id,
                        'age_years': woman['age_years'],
                        'switch_from': old,
                        'switch_to': new,
                    },
                    description='All changes in contraception use'
                    )


class ContraceptionPoll(RegularEvent, PopulationScopeEventMixin):
    """The regular poll (monthly) for the Contraceptive Module.
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
        self.initiate(currently_not_using_co)

        # continue/discontinue/switch: using --> using/not using
        self.discontinue_switch_or_continue(currently_using_co)

    def initiate(self, individuals_not_using: pd.Index):
        """Check all females not using contraception to determine if contraception starts
        i.e. category should change from 'not_using'"""
        # Exit if there are no individuals currently not using a contraceptive:
        if not len(individuals_not_using):
            return

        df = self.sim.population.props
        p = self.module.parameters
        rng = self.module.rng

        c_multiplier = p['r_init_year'].at[self.sim.date.year, 'r_init_year1']
        co_start_prob_by_age = p['contraception_initiation1'].mul(c_multiplier)
        co_start_prob_by_age['not_using'] = 1 - co_start_prob_by_age.sum(axis=1)

        # all the contraceptive types we can randomly choose
        co_types = list(co_start_prob_by_age.columns)

        # select a random contraceptive for everyone not currently using
        random_co = df.loc[individuals_not_using, 'age_years'].apply(
            lambda age: rng.choice(co_types, p=co_start_prob_by_age.loc[age])
        )

        # ... but don't allow female sterilization to any woman below 30: default to "not_using"
        random_co.loc[(random_co == 'female_sterilization') &
                      (df.loc[random_co.index, 'age_years'] < 30)] = "not_using"

        # Make pd.Series only containing those who are starting
        now_using_co = random_co.loc[random_co != 'not_using']

        # Do the contraceptive change
        if len(now_using_co) > 0:
            self.module.schedule_batch_of_contraceptive_changes(
                ids=now_using_co.index,
                old=['not_using'] * len(now_using_co),
                new=now_using_co.values
            )

    def discontinue_switch_or_continue(self, individuals_using: pd.Index):
        """Check all females currently using contraception to determine if they discontinue it, switch to a different
        one, or keep using the same one."""
        # exit if there are no individuals currently using a contraceptive:
        if not len(individuals_using):
            return

        p = self.module.parameters
        rng = self.module.rng
        df = self.sim.population.props

        # -- Discontinuation
        # Probability of discontinuing:
        prob_discont_by_age_and_method = p['contraception_discontinuation'].mul(
            p['r_discont_year'].at[self.sim.date.year, 'r_discont_year1']
        )

        # get the probability of discontinuing for all currently using a contraceptive
        discontinue_prob = df.loc[individuals_using].apply(
            lambda row: prob_discont_by_age_and_method.loc[row.age_years, row.co_contraception], axis=1
        )

        # randomly choose who will discontinue
        co_discontinue_idx = discontinue_prob.index[discontinue_prob > rng.rand(len(individuals_using))]

        # Do the contraceptive change
        if len(co_discontinue_idx) > 0:
            self.module.schedule_batch_of_contraceptive_changes(
                ids=co_discontinue_idx,
                old=df.loc[co_discontinue_idx, 'co_contraception'].values,
                new=['not_using'] * len(co_discontinue_idx)
            )

        # -- Switches and Continuations for that that do not Discontinue:
        individuals_eligible_for_continue_or_switch = individuals_using.drop(co_discontinue_idx)

        # Get the probability of switching contraceptive for all those currently using
        co_switch_prob = df.loc[individuals_eligible_for_continue_or_switch, 'co_contraception'].map(
            p['contraception_switching']['probability']
        )

        # randomly select who will switch contraceptive and who will remain on their current contraceptive
        will_switch = co_switch_prob > rng.random_sample(size=len(individuals_eligible_for_continue_or_switch))
        switch_idx = individuals_eligible_for_continue_or_switch[will_switch]
        continue_idx = individuals_eligible_for_continue_or_switch[~will_switch]

        # For that do switch, select the new contraceptive using switching matrix
        new_co = transition_states(df.loc[switch_idx, 'co_contraception'], p['contraception_switching_matrix'], rng)

        # ... but don't allow female sterilization to any woman below 30 (instead, they will continue on current method)
        to_not_switch_to_sterilization = \
            new_co.index[(new_co == 'female_sterilization') & (df.loc[new_co.index, 'age_years'] < 30)]
        new_co = new_co.drop(to_not_switch_to_sterilization)
        continue_idx = continue_idx.append([to_not_switch_to_sterilization])

        # Do the contraceptive change for those switching
        if len(new_co) > 0:
            self.module.schedule_batch_of_contraceptive_changes(
                ids=new_co.index,
                old=df.loc[new_co.index, 'co_contraception'].values,
                new=new_co.values
            )

        # Do the contraceptive "change" for those not switching (this is so that an HSI may be logged and if the HSI
        #  cannot occur the person discontinues use of the method).
        if len(continue_idx) > 0:
            current_contraception = df.loc[continue_idx, 'co_contraception'].values
            self.module.schedule_batch_of_contraceptive_changes(
                ids=continue_idx,
                old=current_contraception,
                new=current_contraception
            )

    def update_pregnancy(self):
        """Determine who will become pregnant"""

        # Determine pregnancy for those on a contraceptive ("unintentional pregnancy")
        self.pregnancy_for_those_on_contraceptive()

        # Determine pregnancy for those not on contraceptive ("intentional pregnancy")
        self.pregnancy_for_those_not_on_contraceptive()

    def pregnancy_for_those_on_contraceptive(self):
        """Look across all women who are using a contraception method to determine if they become pregnant i.e. the
         method fails according to failure_rate."""

        df = self.module.sim.population.props
        p = self.module.parameters
        rng = self.module.rng

        prob_of_failure = p['contraception_failure']

        # Get the women who are using a contraceptive that may fail and who may become pregnant (i.e., women who
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

            # Determine if this is unintended or not: it is 'unintended' if the women is using a contraceptive at the
            # when she becomes pregnant.
            unintended = (woman['co_contraception'] != 'not_using')

            # Update properties (including that she is no longer on any contraception)
            df.loc[w, (
                'co_contraception',
                'is_pregnant',
                'date_of_last_pregnancy',
                'co_unintended_preg'
            )] = (
                'not_using',
                True,
                self.sim.date,
                unintended
            )

            # Set date of labour in the Labour module:
            self.sim.modules['Labour'].set_date_of_labour(w)

            # Log that a pregnancy has occurred following the failure of a contraceptive:
            logger.info(key='pregnancy',
                        data={
                            'woman_id': w,
                            'age_years': woman['age_years'],
                            'contraception': woman['co_contraception'],
                            'unintended': unintended
                        },
                        description='pregnancy following the failure of contraceptive method')


class ContraceptionLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """Logs state of contraceptive usage in the population at a point in time."""
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        df = population.props

        # Log summary of usage of contraceptives
        logger.info(key='contraception_use_yearly_summary',
                    data=df.loc[df.is_alive & (df.sex == 'F'), 'co_contraception'].value_counts().sort_index().to_dict(),
                    description='Counts of women on each type of contraceptive at a point each time (yearly).')


class HSI_Contraception_FamilyPlanningAppt(HSI_Event, IndividualScopeEventMixin):
    """HSI event for the starting a contraceptive method, maintaining use of a method of a contraceptive, or switching
     between contraceptives."""

    def __init__(self, module, person_id, old_contraceptive, new_contraceptive):
        super().__init__(module, person_id=person_id)
        self.old_contraceptive = old_contraceptive
        self.new_contraceptive = new_contraceptive

        self.TREATMENT_ID = "Contracpetion_StartOrSwitch"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'FamPlan': 1})
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        """If the consumable is available, enact the change in contraception and log it"""

        if not self.sim.population.props.at[person_id, 'is_alive']:
            return

        # Record the date that Family Planning Appointment happened for this person
        self.sim.population.props.at[person_id, "date_of_last_appt_for_maintenance"] = self.sim.date

        # Record use of consumables and default the person to "not_using" if the consumable is not available:
        cons_available = self.get_all_consumables(
            pkg_codes=self.module.cons_codes[self.new_contraceptive]
        )
        _new_contraceptive = self.new_contraceptive if cons_available else "not_using"

        # If the old method is the same as the new method, do nothing else (not even logging): this is an HSI that is
        # maintaining the woman on the same method she has been using.
        if self.old_contraceptive == _new_contraceptive:
            return

        # Do the change:
        self.module.do_and_log_individual_contraception_change(
            woman_id=self.target,
            old=self.old_contraceptive,
            new=_new_contraceptive
        )

    def never_ran(self):
        """If this HSI never ran, the person defaults to "not_using" a contraceptive."""
        if not self.sim.population.props.at[self.target, 'is_alive']:
            return

        self.module.do_and_log_individual_contraception_change(
            woman_id=self.target,
            old=self.old_contraceptive,
            new="not_using"
        )
