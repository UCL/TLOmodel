import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import Date, DateOffset, Module, Parameter, Property, Types, logging
from tlo.analysis.utils import flatten_multi_index_series_into_dict_for_logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods.healthsystem import HSI_Event
from tlo.util import random_date, sample_outcome, transition_states

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Contraception(Module):
    """Contraception module covering baseline contraception methods use, failure (i.e., pregnancy),
    Switching contraceptive methods, and discontinuation rates by age.
    Calibration is done in two stages:
    (i) A scaling factor on the risk of pregnancy (`scaling_factor_on_monthly_risk_of_pregnancy`) is used to induce the
     correct number of age-specific births initially, given the initial pattern of contraceptive use;
    (ii) Trends over time in the risk of starting (`time_age_trend_in_initiation`) and stopping
    (`time_age_trend_in_stopping`) of contraceptives are used to induce the correct trend in the number of births."""

    INIT_DEPENDENCIES = {'Demography'}

    OPTIONAL_INIT_DEPENDENCIES = {'HealthSystem'}

    ADDITIONAL_DEPENDENCIES = {'Labour', 'Hiv'}

    METADATA = {}

    PARAMETERS = {
        'Method_Use_In_2010': Parameter(Types.DATA_FRAME,
                                        'Proportion of women using each method in 2010, by age.'),
        'Pregnancy_NotUsing_In_2010': Parameter(Types.DATA_FRAME,
                                                'Probability per year of a women not on contraceptive becoming '
                                                'pregnant, by age.'),
        'Pregnancy_NotUsing_HIVeffect': Parameter(Types.DATA_FRAME,
                                                  'Relative probability of becoming pregnant whilst not using a '
                                                  'a contraceptive for HIV-positive women compared to HIV-negative '
                                                  'women.'),
        'Failure_ByMethod': Parameter(Types.DATA_FRAME,
                                      'Probability per month of a women on a contraceptive becoming pregnant, by '
                                      'method.'),
        'rr_fail_under25': Parameter(Types.REAL,
                                     'The relative risk of becoming pregnant whilst using a contraceptive for woman '
                                     'younger than 25 years compared to older women.'),
        'Initiation_ByMethod': Parameter(Types.DATA_FRAME,
                                         'Probability per month of a women who is not using any contraceptive method of'
                                         ' starting use of a method, by method.'),
        'Interventions_Pop': Parameter(Types.DATA_FRAME,
                                       'Pop (population scale contraception intervention) intervention multiplier.'),
        'Interventions_PPFP': Parameter(Types.DATA_FRAME,
                                        'PPFP (post-partum family planning) intervention multiplier.'),
        'Initiation_ByAge': Parameter(Types.DATA_FRAME,
                                      'The effect of age on the probability of starting use of contraceptive (add one '
                                      'for multiplicative effect).'),
        'Initiation_AfterBirth': Parameter(Types.DATA_FRAME,
                                           'The probability of a woman starting a contraceptive immidiately after birth'
                                           ', by method.'),
        'Discontinuation_ByMethod': Parameter(Types.DATA_FRAME,
                                              'The probability per month of discontinuing use of a method, by method.'),
        'Discontinuation_ByAge': Parameter(Types.DATA_FRAME,
                                           'The effect of age on the probability of discontinuing use of contraceptive '
                                           '(add one for multiplicative effect).'),

        'Prob_Switch_From': Parameter(Types.DATA_FRAME,
                                      'The probability per month that a women switches from one form of contraceptive '
                                      'to another, conditional that she will not discontinue use of the method.'),
        'Prob_Switch_From_And_To': Parameter(Types.DATA_FRAME,
                                             'The probability of switching to a new method, by method, conditional that'
                                             ' the woman will switch to a new method.'),
        'days_between_appts_for_maintenance': Parameter(Types.LIST,
                                                        'The number of days between successive family planning '
                                                        'appointments for women that are maintaining the use of a '
                                                        'method (for each method in'
                                                        'sorted(self.states_that_may_require_HSI_to_maintain_on), i.e.'
                                                        '[IUD, implant, injections, other_modern, pill]).'),
        'age_specific_fertility_rates': Parameter(
            Types.DATA_FRAME, 'Data table from official source (WPP) for age-specific fertility rates and calendar '
                              'period.'),

        'scaling_factor_on_monthly_risk_of_pregnancy': Parameter(
            Types.LIST, "Scaling factor (by age-group: 15-19, 20-24, ..., 45-49) on the monthly risk of pregnancy and "
                        "contraceptive failure rate. This value is found through calibration so that, at the beginning "
                        "of the simulation, the age-specific monthly probability of a woman having a live birth matches"
                        " the WPP age-specific fertility rate value for the same year."),

        'max_number_of_runs_of_hsi_if_consumable_not_available': Parameter(
            Types.INT, "The maximum number of time an HSI can run (repeats occur if the consumables are not "
                       "available)."),

        'max_days_delay_between_decision_to_change_method_and_hsi_scheduled': Parameter(
            Types.INT, "The maximum delay (in days) between the decision for a contraceptive to change and the `topen` "
                       "date of the HSI that is scheduled to effect the change (when using the healthsystem)."),

        # TODO: needs to be fixed to be BOOL parameter (together with resolving the temporary fix below)
        'use_interventions': Parameter(
            Types.STRING, "if 'On': FP interventions (pop & ppfp) are simulated since 'interventions_start_date',"
                          " if 'Off': FP interventions (pop & ppfp) are not simulated."),

        'interventions_start_date': Parameter(
            Types.DATE, "The date since which the FP interventions (pop & ppfp) are implemented, if at all (ie, if "
                        "use_interventions==True")
    }

    all_contraception_states = {
        'not_using', 'pill', 'IUD', 'injections', 'implant', 'male_condom', 'female_sterilization', 'other_modern',
        'periodic_abstinence', 'withdrawal', 'other_traditional'
    }
    # These are the 11 categories of contraception ('not_using' + 10 methods) from the DHS analysis of initiation,
    # discontinuation, failure and switching rates.
    # 'other modern' includes Male sterilization, Female Condom, Emergency contraception;
    # 'other traditional' includes lactational amenohroea (LAM),  standard days method (SDM), 'other traditional
    #  method').
    contraceptives_initiated_with_additional_items = {
        'pill', 'IUD', 'injections', 'implant', 'female_sterilization'
    }
    # These are methods for which additional items ('co_initiation') are used to initiate the method after not using any
    # contraceptive or after using a method which is not in this category.

    PROPERTIES = {
        'co_contraception': Property(Types.CATEGORICAL, 'Current contraceptive method',
                                     categories=sorted(all_contraception_states)),
        'is_pregnant': Property(Types.BOOL, 'Whether this individual is currently pregnant'),
        'date_of_last_pregnancy': Property(Types.DATE, 'Date that the most recent or current pregnancy began.'),
        'co_unintended_preg': Property(Types.BOOL, 'Whether the most recent or current pregnancy was unintended.'),
        'co_date_of_last_fp_appt': Property(Types.DATE,
                                            'The date of the most recent Family Planning appointment. This is used to '
                                            'determine if a Family Planning appointment is needed to maintain the '
                                            'person on their current contraceptive. If the person is to maintain use of'
                                            ' the current contraceptive, they will have an HSI only if the days elapsed'
                                            ' since this value exceeds the method-specific parameter '
                                            '`days_between_appts_for_maintenance`.'
                                            )
    }

    def __init__(self, name=None, resourcefilepath=None, use_healthsystem=True, run_update_contraceptive=True):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        self.use_healthsystem = use_healthsystem  # if True: initiation and switches to contraception require an HSI;
        # if False: initiation and switching do not occur through an HSI

        self.run_update_contraceptive = run_update_contraceptive  # If 'False' prevents any logic occurring for
        #                                                           updating/changing/maintaining contraceptive methods.

        self.states_that_may_require_HSI_to_switch_to = {'male_condom', 'injections', 'other_modern', 'IUD', 'pill',
                                                         'female_sterilization', 'implant'}
        self.states_that_may_require_HSI_to_maintain_on = {'male_condom', 'injections', 'other_modern', 'IUD', 'pill',
                                                           'implant'}

        assert self.states_that_may_require_HSI_to_switch_to.issubset(self.all_contraception_states)
        assert self.states_that_may_require_HSI_to_maintain_on.issubset(self.states_that_may_require_HSI_to_switch_to)

        self.processed_params = dict()  # (Will store the processed data for rates/probabilities of outcomes).
        self.cons_codes = dict()  # (Will store the consumables codes for use in the HSI)
        self.rng2 = None  # (Will be a second random number generator, used for things to do with scheduling HSI)

        self._women_ids_sterilized_below30 = set()  # The ids of women who had female sterilization initiated when they
        #                                             were less than 30 years old.

    def read_parameters(self, data_folder):
        """Import the relevant sheets from the ResourceFile (excel workbook) and declare values for other parameters
        (CSV ResourceFile).
        """
        workbook = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_Contraception.xlsx', sheet_name=None)

        # Import selected sheets from the workbook as the parameters
        sheet_names = [
            'Method_Use_In_2010',
            'Pregnancy_NotUsing_In_2010',
            'Pregnancy_NotUsing_HIVeffect',
            'Failure_ByMethod',
            'Initiation_ByAge',
            'Initiation_ByMethod',
            'Interventions_Pop',
            'Interventions_PPFP',
            'Initiation_AfterBirth',
            'Discontinuation_ByMethod',
            'Discontinuation_ByAge',
            'Prob_Switch_From',
            'Prob_Switch_From_And_To',
        ]

        for sheet in sheet_names:
            self.parameters[sheet] = workbook[sheet]

        # Declare values for other parameters
        self.load_parameters_from_dataframe(pd.read_csv(
            Path(self.resourcefilepath) / 'ResourceFile_ContraceptionParams.csv'
        ))

        # Temporary fix of loading the boolean  parameter 'use_interventions':
        co_params = pd.read_csv(
            Path(self.resourcefilepath) / 'ResourceFile_ContraceptionParams.csv'
        )
        self.parameters['use_interventions_loaded'] =\
            co_params['value'].loc[co_params['parameter_name'] == 'use_interventions'].values[0]
        self.parameters['use_interventions'] = self.parameters['use_interventions_loaded'] == 'On'
        if self.parameters['use_interventions_loaded'] not in ('On', 'Off'):
            warnings.warn(UserWarning(f"'use_intervention' parameter set to 'Off' (=False) as the given value "
                                      f"{self.parameters['use_interventions_loaded']} was not recognised."))
            logger.info(
                key="use_intervention",
                data={"input_given": self.parameters['use_interventions_loaded'],
                      "value_used": self.parameters['use_interventions']
                      },
                description='Unrecognised value given for use_intervention parameter replaced.'
            )

        # Import the Age-specific fertility rate data from WPP
        self.parameters['age_specific_fertility_rates'] = \
            pd.read_csv(Path(self.resourcefilepath) / 'demography' / 'ResourceFile_ASFR_WPP.csv')

        # Import 2010 pop and count numbs of women 15-49 & 30-49
        pop_2010 = self.sim.modules["Demography"].parameters["pop_2010"].copy()

        female1549_in_2010 = \
            pop_2010.loc[(pop_2010.Sex == 'F') & pop_2010.Age.between(15, 49), ['Age', 'Count']].groupby('Age').sum()
        self.n_female1549_in_2010 = female1549_in_2010.sum()
        self.n_female3049_in_2010 = female1549_in_2010.loc[female1549_in_2010.index >= 30].sum()

    def pre_initialise_population(self):
        """Process parameters before initialising population and simulation"""
        self.processed_params = self.process_params()

    def initialise_population(self, population):
        """Set initial values for properties"""

        # 1) Set default values for properties
        df = population.props
        df.loc[df.is_alive, 'co_contraception'] = 'not_using'
        df.loc[df.is_alive, 'is_pregnant'] = False
        df.loc[df.is_alive, 'date_of_last_pregnancy'] = pd.NaT
        df.loc[df.is_alive, 'co_unintended_preg'] = False
        df.loc[df.is_alive, 'co_date_of_last_fp_appt'] = pd.NaT

        # 2) Assign contraception method
        # Select females aged 15-49 from population, for current year
        females1549 = df.is_alive & (df.sex == 'F') & df.age_years.between(15, 49)
        p_method = self.processed_params['initial_method_use']
        df.loc[females1549, 'co_contraception'] = df.loc[females1549, 'age_years'].apply(
            lambda _age_years: self.rng.choice(p_method.columns, p=p_method.loc[_age_years])
        )

        # 3) Give a notional date on which the last appointment occurred for those that need them
        needs_appts = females1549 & df['co_contraception'].isin(self.states_that_may_require_HSI_to_maintain_on)
        states_to_maintain_on = sorted(self.states_that_may_require_HSI_to_maintain_on)
        df.loc[needs_appts, 'co_date_of_last_fp_appt'] = df.loc[needs_appts, 'co_contraception'].astype('string').apply(
            lambda _co_contraception: random_date(
                self.sim.date - pd.DateOffset(
                    days=self.parameters['days_between_appts_for_maintenance']
                    [states_to_maintain_on.index(_co_contraception)]),
                self.sim.date - pd.DateOffset(days=1),
                self.rng
            )
        )

    def initialise_simulation(self, sim):
        """
        * Schedule the ContraceptionPoll and ContraceptionLoggingEvent
        * Retrieve the consumables codes for the consumables used
        * Create second random number generator
        * Schedule births to occur during the first 9 months of the simulation
        """

        # Schedule the first occurrence of the Logging event to occur at the beginning of the simulation
        sim.schedule_event(ContraceptionLoggingEvent(self), sim.date)

        # Schedule first occurrences of Contraception Poll to occur at the beginning of the simulation
        sim.schedule_event(ContraceptionPoll(self, run_update_contraceptive=self.run_update_contraceptive), sim.date)

        # Retrieve the consumables codes for the consumables used
        if self.use_healthsystem:
            self.cons_codes = self.get_item_code_for_each_contraceptive()

        # Create second random number generator
        self.rng2 = np.random.RandomState(self.rng.randint(2 ** 31 - 1))

        # Schedule births to occur during the first 9 months of the simulation
        self.schedule_births_for_first_9_months()

        if self.parameters['use_interventions']:
            # Schedule StartInterventions event to update parameters when FP interventions are introduced
            sim.schedule_event(StartInterventions(self), Date(self.parameters['interventions_start_date']))

            # Log initiation date of interventions and implementation costs
            logger.info(key='interventions_start_date',
                        data={
                            'date_co_interv_implemented': self.parameters['interventions_start_date']
                        },
                        description='The date when parameters are updated to enable the FP interventions.'
                        )

    def on_birth(self, mother_id, child_id):
        """
        * 1) Formally end the pregnancy
        * 2) Initialise properties for the newborn
        """
        df = self.sim.population.props

        if mother_id >= 0:  # check if direct birth look for positive mother ids
            self.end_pregnancy(person_id=mother_id)

        # Initialise child's properties:
        new_properties = {
            'co_contraception': 'not_using',
            'is_pregnant': False,
            'date_of_last_pregnancy': pd.NaT,
            'co_unintended_preg': False,
            'co_date_of_last_fp_appt': pd.NaT,
        }
        df.loc[child_id, new_properties.keys()] = new_properties.values()

    def end_pregnancy(self, person_id):
        """End the pregnancy. Reset pregnancy status and may initiate a contraceptive method.
        This is called by `on_birth` in this module and by Labour/Pregnancy modules for births that do result in live
        birth."""

        assert self.sim.population.props.at[person_id, 'co_contraception'] \
               not in self.contraceptives_initiated_with_additional_items
        # TODO: Shouldn't it be even == "not_using"?
        #  It is not always the case when used along with Joe's rmnch modules, why?
        self.sim.population.props.at[person_id, 'is_pregnant'] = False
        person_age = self.sim.population.props.at[person_id, 'age_years']
        self.select_contraceptive_following_birth(person_id, person_age)

    def process_params(self):
        """Process parameters that have been read-in."""

        processed_params = dict()

        def expand_to_age_years(values_by_age_groups, ages_by_year):
            _d = dict(zip(['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49'], values_by_age_groups))
            return np.array(
                [_d[self.sim.modules['Demography'].AGE_RANGE_LOOKUP[_age_year]] for _age_year in ages_by_year]
            )

        def initial_method_use():
            """Generate the distribution of method use by age for the start of the simulation."""
            p_method = self.parameters['Method_Use_In_2010'].set_index('age').rename_axis('age_years')

            # Normalise so that the sum within each age is 1.0
            p_method = p_method.div(p_method.sum(axis=1), axis=0)
            assert np.isclose(1.0, p_method.sum(axis=1)).all()

            # Check correct format
            assert set(p_method.columns) == set(self.all_contraception_states)
            assert (p_method.index == range(15, 50)).all()

            return p_method

        def avoid_sterilization_below30(probs):
            """Prevent women below 30 years having female sterilization and adjust the probability for women 30 and over
            to preserve the overall probability of initiating sterilization."""
            # Input 'probs' must include probs for all methods including 'not_using'
            assert set(probs.index) == set(self.all_contraception_states)

            # Prevent women below 30 years having 'female_sterilization'
            probs_below30 = probs.copy()
            probs_below30['female_sterilization'] = 0.0
            # Scale so that the probability of all outcomes sum to 1.0
            probs_below30 = probs_below30 / probs_below30.sum()
            assert np.isclose(1.0, probs_below30.sum())

            # Increase prob of 'female_sterilization' in older women accordingly
            probs_30plus = probs.copy()
            probs_30plus['female_sterilization'] = \
                probs.loc['female_sterilization'] / \
                (self.n_female3049_in_2010 / self.n_female1549_in_2010)
            # Scale so that the probability of all outcomes sum to 1.0
            probs_30plus = probs_30plus / probs_30plus.sum()
            assert np.isclose(1.0, probs_30plus.sum())

            return probs_below30, probs_30plus

        def contraception_initiation():
            """Generate the probability per month of a woman initiating onto each contraceptive, by the age (in whole
             years)."""

            # Probability of initiation by method per month (average over all ages)
            p_init_by_method = self.parameters['Initiation_ByMethod'].loc[0]

            # Prevent women below 30 years having 'female_sterilization' while preserving the overall probability of
            # 'female_sterilization' initiation
            p_init_by_method_below30, p_init_by_method_30plus = avoid_sterilization_below30(p_init_by_method)

            # Effect of age
            age_effect = 1.0 + self.parameters['Initiation_ByAge'].set_index('age')['r_init1_age'].rename_axis(
                "age_years")

            # Year effect
            year_effect = time_age_trend_in_initiation()

            def apply_age_year_effects(probs_below30, probs_30plus):
                # Assemble into age-specific data-frame:
                probs_by_method_below30 = probs_below30.copy().drop('not_using')
                probs_by_method_30plus = probs_30plus.copy().drop('not_using')
                p_init = dict()
                for year in year_effect.index:

                    p_init_this_year = dict()
                    for a in age_effect.index:
                        if a < 30:
                            p_init_this_year[a] = probs_by_method_below30 * age_effect.at[a] * year_effect.at[year, a]
                        else:
                            p_init_this_year[a] = probs_by_method_30plus * age_effect.at[a] * year_effect.at[year, a]
                    p_init_this_year_df = pd.DataFrame.from_dict(p_init_this_year, orient='index')

                    # Check correct format of age/method data-frame
                    assert set(p_init_this_year_df.columns) == set(self.all_contraception_states - {'not_using'})
                    assert (p_init_this_year_df.index == range(15, 50)).all()
                    assert (p_init_this_year_df >= 0.0).all().all()

                    p_init[year] = p_init_this_year_df

                return p_init

            return apply_age_year_effects(p_init_by_method_below30, p_init_by_method_30plus)

        def contraception_switch():
            """Get the probability per month of a woman switching to contraceptive method, given that she is currently
            using a different one."""

            # Get the probability per month of the woman making a switch (to anything)
            p_switch_from = self.parameters['Prob_Switch_From'].loc[0]

            # Get the probability that the woman switches to a new contraceptive (given that she will switch to
            # something different).
            # Columns = "current method"; Row = "new method"
            switching_matrix = self.parameters['Prob_Switch_From_And_To'].set_index('switchfrom').transpose()

            # Prevent women below 30 years having 'female_sterilization'
            switching_matrix_below30 = switching_matrix.copy()
            switching_matrix_below30.loc['female_sterilization', :] = 0.0
            switching_matrix_below30 = switching_matrix_below30.apply(lambda col: col / col.sum())

            assert set(switching_matrix_below30.columns) == (
                self.all_contraception_states - {"not_using", "female_sterilization"})
            assert set(switching_matrix_below30.index) == (self.all_contraception_states - {"not_using"})
            assert np.isclose(1.0, switching_matrix_below30.sum(axis=0)).all()

            # Increase prob of 'female_sterilization' in older women accordingly
            new_fs_probs_30plus = switching_matrix.loc['female_sterilization', :] /\
                float(
                    self.n_female3049_in_2010 / self.n_female1549_in_2010
                )
            switching_matrix_except_fs = switching_matrix.loc[switching_matrix.index != 'female_sterilization']
            switching_matrix_30plus = switching_matrix_except_fs.apply(lambda col: col / col.sum())
            switching_matrix_30plus = switching_matrix_30plus * (1 - new_fs_probs_30plus)
            switching_matrix_30plus.loc[new_fs_probs_30plus.name] = new_fs_probs_30plus

            assert set(switching_matrix_30plus.columns) == (
                self.all_contraception_states - {"not_using", "female_sterilization"})
            assert set(switching_matrix_30plus.index) == (self.all_contraception_states - {"not_using"})
            assert np.isclose(1.0, switching_matrix_30plus.sum(axis=0)).all()

            return p_switch_from, switching_matrix_below30, switching_matrix_30plus

        def contraception_stop():
            """Get the probability per month of a woman stopping use of contraceptive method."""

            # Get data from read-in Excel sheets
            p_stop_by_method = self.parameters['Discontinuation_ByMethod'].loc[0]
            age_effect = 1.0 + self.parameters['Discontinuation_ByAge'].set_index('age')['r_discont_age'].rename_axis(
                "age_years")
            year_effect = time_age_trend_in_stopping()

            # Probability of initiation by age for each method
            p_stop = dict()
            for year in year_effect.index:
                p_stop_this_year = dict()
                for a in age_effect.index:
                    p_stop_this_year[a] = p_stop_by_method * age_effect.at[a] * year_effect.at[year, a]
                p_stop_this_year_df = pd.DataFrame.from_dict(p_stop_this_year, orient='index')

                # Check correct format of age/method data-frame
                assert set(p_stop_this_year_df.columns) == set(self.all_contraception_states - {'not_using'})
                assert (p_stop_this_year_df.index == range(15, 50)).all()
                assert (p_stop_this_year_df >= 0.0).all().all()

                p_stop[year] = p_stop_this_year_df

            return p_stop

        def time_age_trend_in_initiation():
            """The age-specific effect of calendar year on the probability of starting use of contraceptive
            (multiplicative effect). Values are chosen to induce a trend in age-specific fertility consistent with
             the WPP estimates."""

            _years = np.arange(2010, 2101)
            _ages = np.arange(15, 50)

            _init_over_time = np.exp(+0.05 * np.minimum(2020 - 2010, (_years - 2010))) * np.maximum(1.0, np.exp(
                +0.01 * (_years - 2020)))
            _init_over_time_modification_by_age = 1.0 / expand_to_age_years([1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], _ages)
            _init = np.outer(_init_over_time, _init_over_time_modification_by_age)

            return pd.DataFrame(index=_years, columns=_ages, data=_init)

        def time_age_trend_in_stopping():
            """The age-specific effect of calendar year on the probability of discontinuing use of contraceptive
            (multiplicative effect). Values are chosen to induce a trend in age-specific fertility consistent with
            the WPP estimates."""

            _years = np.arange(2010, 2101)
            _ages = np.arange(15, 50)

            _discont_over_time = np.exp(-0.05 * np.minimum(2020 - 2010, (_years - 2010))) * np.minimum(1.0, np.exp(
                -0.01 * (_years - 2020)))
            _discont_over_time_modification_by_age = expand_to_age_years([1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], _ages)
            _discont = np.outer(_discont_over_time, _discont_over_time_modification_by_age)

            return pd.DataFrame(index=_years, columns=_ages, data=_discont)

        def contraception_initiation_after_birth():
            """Get the probability of a woman starting a contraceptive following giving birth. Avoid sterilization in
            women below 30 years old."""

            # Get data from read-in Excel sheets
            p_start_after_birth = self.parameters['Initiation_AfterBirth'].loc[0]

            return avoid_sterilization_below30(p_start_after_birth)

        def scaling_factor_on_monthly_risk_of_pregnancy():
            """A scaling factor on the monthly risk of pregnancy, chosen to give the correct number of live-births
            initially, given the initial pattern of contraceptive use."""

            # first scaling factor is that worked out from the calibration script
            scaling_factor_as_dict = dict(zip(
                ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49'],
                self.parameters['scaling_factor_on_monthly_risk_of_pregnancy']
            ))

            AGE_RANGE_LOOKUP = self.sim.modules['Demography'].AGE_RANGE_LOOKUP
            _ages = range(15, 50)
            return pd.Series(
                index=_ages,
                data=[scaling_factor_as_dict[AGE_RANGE_LOOKUP[_age_year]] for _age_year in _ages]
            )

        def pregnancy_no_contraception():
            """Get the probability per month of a woman becoming pregnant if she is not using any contraceptive method.
            """

            # Get the probability of being pregnant if not HIV-positive
            p_pregnancy_no_contraception_per_month_nohiv = self.parameters['Pregnancy_NotUsing_In_2010'] \
                .set_index('age')['AnnualProb'].rename_axis('age_years').apply(convert_annual_prob_to_monthly_prob)

            # Compute the probability of being pregnant if HIV-positive
            p_pregnancy_no_contraception_per_month_hiv = (
                p_pregnancy_no_contraception_per_month_nohiv *
                self.parameters['Pregnancy_NotUsing_HIVeffect'].set_index('age_')['RR_pregnancy']
            )

            # Create combined dataframe
            p_pregnancy_no_contraception_per_month = pd.concat({
                'hv_inf_False': p_pregnancy_no_contraception_per_month_nohiv,
                'hv_inf_True': p_pregnancy_no_contraception_per_month_hiv}, axis=1)

            assert (p_pregnancy_no_contraception_per_month.index == range(15, 50)).all()
            assert set(p_pregnancy_no_contraception_per_month.columns) == {'hv_inf_True', 'hv_inf_False'}
            assert np.isclose(
                self.parameters['Pregnancy_NotUsing_In_2010']['AnnualProb'].values,
                1.0 - np.power(1.0 - p_pregnancy_no_contraception_per_month['hv_inf_False'], 12)
            ).all()

            return p_pregnancy_no_contraception_per_month.mul(scaling_factor_on_monthly_risk_of_pregnancy(), axis=0)

        def pregnancy_with_contraception():
            """Get the probability per month of a woman becoming pregnant if she is using a contraceptive method."""
            p_pregnancy_by_method_per_month = self.parameters['Failure_ByMethod'].loc[0]

            # Create rates that are age-specific (using self.parameters['rr_fail_under25'])
            p_pregnancy_with_contraception_per_month = pd.DataFrame(
                index=range(15, 50),
                columns=sorted(self.all_contraception_states - {"not_using"})
            )
            p_pregnancy_with_contraception_per_month.loc[15, :] = p_pregnancy_by_method_per_month
            p_pregnancy_with_contraception_per_month = p_pregnancy_with_contraception_per_month.ffill()
            p_pregnancy_with_contraception_per_month.loc[
                p_pregnancy_with_contraception_per_month.index < 25
                ] *= self.parameters['rr_fail_under25']

            assert (p_pregnancy_with_contraception_per_month.index == range(15, 50)).all()
            assert set(p_pregnancy_with_contraception_per_month.columns) == set(
                self.all_contraception_states - {"not_using"})
            assert (0.0 == p_pregnancy_with_contraception_per_month['female_sterilization']).all()

            return p_pregnancy_with_contraception_per_month.mul(scaling_factor_on_monthly_risk_of_pregnancy(), axis=0)

        processed_params['initial_method_use'] = initial_method_use()
        processed_params['p_start_per_month'] = contraception_initiation()
        processed_params['p_switch_from_per_month'], \
            processed_params['p_switching_to_below30'], processed_params['p_switching_to_30plus'] =\
            contraception_switch()
        processed_params['p_stop_per_month'] = contraception_stop()
        processed_params['p_start_after_birth_below30'], processed_params['p_start_after_birth_30plus'] =\
            contraception_initiation_after_birth()

        processed_params['p_pregnancy_no_contraception_per_month'] = pregnancy_no_contraception()
        processed_params['p_pregnancy_with_contraception_per_month'] = pregnancy_with_contraception()

        return processed_params

    def update_params_for_interventions(self):
        """Updates process parameters to enable FP interventions."""

        processed_params = self.processed_params

        def expand_to_age_years(values_by_age_groups, ages_by_year):
            _d = dict(zip(['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49'], values_by_age_groups))
            return np.array(
                [_d[self.sim.modules['Demography'].AGE_RANGE_LOOKUP[_age_year]] for _age_year in ages_by_year]
            )

        def time_age_trend_in_initiation():
            """The age-specific effect of calendar year on the probability of starting use of contraceptive
            (multiplicative effect). Values are chosen to induce a trend in age-specific fertility consistent with
             the WPP estimates."""

            _years = np.arange(2010, 2101)
            _ages = np.arange(15, 50)

            _init_over_time = np.exp(+0.05 * np.minimum(2020 - 2010, (_years - 2010))) * np.maximum(1.0, np.exp(
                +0.01 * (_years - 2020)))
            _init_over_time_modification_by_age = 1.0 / expand_to_age_years([1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], _ages)
            _init = np.outer(_init_over_time, _init_over_time_modification_by_age)

            return pd.DataFrame(index=_years, columns=_ages, data=_init)

        def avoid_sterilization_below30(probs):
            """Prevent women below 30 years having female sterilization and adjust the probability for women 30 and over
            to preserve the overall probability of initiating sterilization."""
            # Input 'probs' must include probs for all methods including 'not_using'
            assert set(probs.index) == set(self.all_contraception_states)

            # Prevent women below 30 years having 'female_sterilization'
            probs_below30 = probs.copy()
            probs_below30['female_sterilization'] = 0.0
            # Scale so that the probability of all outcomes sum to 1.0
            probs_below30 = probs_below30 / probs_below30.sum()
            assert np.isclose(1.0, probs_below30.sum())

            # Increase prob of 'female_sterilization' in older women accordingly
            probs_30plus = probs.copy()
            probs_30plus['female_sterilization'] = \
                probs.loc['female_sterilization'] / \
                (self.n_female3049_in_2010 / self.n_female1549_in_2010)
            # Scale so that the probability of all outcomes sum to 1.0
            probs_30plus = probs_30plus / probs_30plus.sum()
            assert np.isclose(1.0, probs_30plus.sum())

            return probs_below30, probs_30plus

        def contraception_initiation_with_interv():
            """Generate the probability per month of a woman initiating onto each contraceptive, by the age (in whole
             years) if FP interventions are applied."""

            # Probability of initiation by method per month (average over all ages)
            p_init_by_method = self.parameters['Initiation_ByMethod'].loc[0]

            # Prevent women below 30 years having 'female_sterilization' while preserving the overall probability of
            # 'female_sterilization' initiation
            p_init_by_method_below30, p_init_by_method_30plus = avoid_sterilization_below30(p_init_by_method)

            # Effect of age
            age_effect = 1.0 + self.parameters['Initiation_ByAge'].set_index('age')['r_init1_age'].rename_axis(
                "age_years")

            # Year effect
            year_effect = time_age_trend_in_initiation()

            def apply_intervention_age_year_effects(probs_below30, probs_30plus):
                # Apply Pop intervention
                probs_by_method_below30 = \
                    probs_below30.copy().drop('not_using').mul(self.parameters['Interventions_Pop'].loc[0])
                probs_by_method_30plus = \
                    probs_30plus.copy().drop('not_using').mul(self.parameters['Interventions_Pop'].loc[0])
                # Assemble into age-specific data-frame:
                p_init = dict()
                for year in year_effect.index:

                    p_init_this_year = dict()
                    for a in age_effect.index:
                        if a < 30:
                            p_init_this_year[a] = probs_by_method_below30 * age_effect.at[a] * year_effect.at[year, a]
                        else:
                            p_init_this_year[a] = probs_by_method_30plus * age_effect.at[a] * year_effect.at[year, a]
                    p_init_this_year_df = pd.DataFrame.from_dict(p_init_this_year, orient='index')

                    # Check correct format of age/method data-frame
                    assert set(p_init_this_year_df.columns) == set(self.all_contraception_states - {'not_using'})
                    assert (p_init_this_year_df.index == range(15, 50)).all()
                    assert (p_init_this_year_df >= 0.0).all().all()

                    p_init[year] = p_init_this_year_df

                return p_init

            return apply_intervention_age_year_effects(p_init_by_method_below30, p_init_by_method_30plus)

        def contraception_initiation_after_birth_with_interv():
            """Get the probability of a woman starting a contraceptive following giving birth if FP interventions are
            applied. Avoid sterilization in women below 30 years old."""

            # Get initiation probabilities of contraception methods after birth from read-in Excel sheet
            p_start_after_birth = self.parameters['Initiation_AfterBirth'].loc[0].drop('not_using')

            # Apply PPFP intervention multipliers
            p_start_after_birth = p_start_after_birth.mul(self.parameters['Interventions_PPFP'].loc[0])

            # Add 'not_using' to initiation probabilities of contraception methods after birth
            p_start_after_birth = pd.concat(
                (
                    pd.Series((1.0 - p_start_after_birth.sum()), index=['not_using']),
                    p_start_after_birth
                )
            )

            return avoid_sterilization_below30(p_start_after_birth)

        processed_params['p_start_per_month'] = contraception_initiation_with_interv()
        processed_params['p_start_after_birth_below30'], processed_params['p_start_after_birth_30plus'] =\
            contraception_initiation_after_birth_with_interv()

        return processed_params

    def select_contraceptive_following_birth(self, mother_id, mother_age):
        """Initiation of mother's contraception after birth."""

        # Allocate the woman to a contraceptive status
        if mother_age < 30:
            probs_below30 = self.processed_params['p_start_after_birth_below30']
            new_contraceptive = self.rng.choice(probs_below30.index, p=probs_below30.values)
        else:
            probs_30plus = self.processed_params['p_start_after_birth_30plus']
            new_contraceptive = self.rng.choice(probs_30plus.index, p=probs_30plus.values)

        # Do the change in contraceptive
        self.schedule_batch_of_contraceptive_changes(ids=[mother_id], old=['not_using'], new=[new_contraceptive])

    def get_item_code_for_each_contraceptive(self):
        """Get the item_code for each contraceptive and for contraceptive initiation."""
        # TODO: update with optional items (currently all considered essential)

        get_items_from_pkg = self.sim.modules['HealthSystem'].get_item_codes_from_package_name

        _cons_codes = dict()
        # items for each method that requires an HSI to switch to
        _cons_codes['pill'] = get_items_from_pkg('Pill')
        _cons_codes['male_condom'] = get_items_from_pkg('Male condom')
        _cons_codes['other_modern'] = get_items_from_pkg('Female Condom')
        # NB. The consumable female condom is used for the contraceptive state of "other_modern method"
        _cons_codes['IUD'] = get_items_from_pkg('IUD')
        _cons_codes['injections'] = get_items_from_pkg('Injectable')
        _cons_codes['implant'] = get_items_from_pkg('Implant')
        _cons_codes['female_sterilization'] = get_items_from_pkg('Female sterilization')
        assert set(_cons_codes.keys()) == set(self.states_that_may_require_HSI_to_switch_to)
        # items used when initiating a modern reliable method after not using or switching from non-reliable method
        _cons_codes['co_initiation'] = get_items_from_pkg('Contraception initiation')

        return _cons_codes

    def schedule_batch_of_contraceptive_changes(self, ids, old, new):
        """Enact the change in contraception, either through editing properties instantaneously or by scheduling HSI.
        ids: pd.Index of the woman for whom the contraceptive state is changing
        old: iterable giving the corresponding contraceptive state being switched from
        new: iterable giving the corresponding contraceptive state being switched to

        It is assumed that even with the option `self.use_healthsystem=True` that switches to certain methods do not
        require the use of HSI (these are not in `states_that_may_require_HSI_to_switch_to`)."""

        df = self.sim.population.props
        date_today = self.sim.date
        days_between_appts = self.parameters['days_between_appts_for_maintenance']

        date_of_last_appt = df.loc[ids, "co_date_of_last_fp_appt"].to_dict()
        states_to_maintain_on = sorted(self.states_that_may_require_HSI_to_maintain_on)

        for _woman_id, _old, _new in zip(ids, old, new):
            if (_new == 'female_sterilization') and (df.loc[_woman_id, 'age_years'] < 30):
                self._women_ids_sterilized_below30.add(_woman_id)

            # Does this change require an HSI?
            is_a_switch = _old != _new
            reqs_appt = _new in self.states_that_may_require_HSI_to_switch_to if is_a_switch \
                else _new in self.states_that_may_require_HSI_to_maintain_on
            if (not is_a_switch) & reqs_appt:
                due_appt = (pd.isnull(date_of_last_appt[_woman_id]) or
                            (date_today - date_of_last_appt[_woman_id]).days >=
                            days_between_appts[states_to_maintain_on.index(_old)]
                            )
            do_appt = self.use_healthsystem and reqs_appt and (is_a_switch or due_appt)

            # If the new method requires an HSI to be implemented, schedule the HSI:
            if do_appt:
                # If this is a change, or its maintenance and time for an appointment, schedule an appointment
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    hsi_event=HSI_Contraception_FamilyPlanningAppt(
                        person_id=_woman_id,
                        module=self,
                        new_contraceptive=_new
                    ),
                    # select start_date for 0 max day delay; start_date or later for >=1 max day delay:
                    topen=random_date(
                        self.sim.date,
                        self.sim.date + pd.DateOffset(
                            days=self.parameters[
                                     'max_days_delay_between_decision_to_change_method_and_hsi_scheduled'] + 1),
                        self.rng2),
                    tclose=None,
                    priority=1
                )
            else:
                # Otherwise, implement the change immediately:
                if _old != _new:
                    self.do_and_log_individual_contraception_change(woman_id=_woman_id, old=_old, new=_new)
                else:
                    pass  # No need to do anything if the old is the same as the new and no HSI needed.

    def do_and_log_individual_contraception_change(self, woman_id: int, old, new):
        """Implement and then log a start / stop / switch of contraception. """
        assert old in self.all_contraception_states
        assert new in self.all_contraception_states

        df = self.sim.population.props

        # Do the change
        df.at[woman_id, "co_contraception"] = new

        # Log the change
        logger.info(key='contraception_change',
                    data={
                        'woman_id': woman_id,
                        'age_years': df.at[woman_id, 'age_years'],
                        'switch_from': old,
                        'switch_to': new,
                    },
                    description='All changes in contraception use'
                    )

    def schedule_births_for_first_9_months(self):
        """Schedule births to occur during the first 9 months of the simulation. This is necessary because at initiation
        no women are pregnant, so the first births generated endogenously (through pregnancy -> gestation -> labour)
        occur after 9 months of simulation time. This method examines age-specific fertility rate data and causes there
        to be the appropriate number of births, scattered uniformly over the first 9 months of the simulation. These are
        "direct live births" that are not subjected to any of the processes (e.g. risk of loss of pregnancy, or risk of
        death to mother) represented in the `PregnancySupervisor`, `CareOfWomenDuringPregnancy` or `Labour`.
        When initialising population ensured person_id=0 is a man, so can safely exclude person_id=0 from choice of
        direct birth mothers without loss of generality."""

        risk_of_birth = get_medium_variant_asfr_from_wpp_resourcefile(
            dat=self.parameters['age_specific_fertility_rates'], months_exposure=9)

        df = self.sim.population.props
        # don't use person_id=0 for direct birth mother
        prob_birth = df.loc[(df.index != 0) & (df.sex == 'F') & df.is_alive & ~df.is_pregnant]['age_range'].map(
            risk_of_birth[self.sim.date.year]).fillna(0)

        # determine which women will get pregnant
        give_birth_women_ids = prob_birth.index[
            (self.rng.random_sample(size=len(prob_birth)) < prob_birth)
        ]

        # schedule births, passing negative of mother's id:
        for _id in give_birth_women_ids:
            self.sim.schedule_event(DirectBirth(person_id=_id * (-1), module=self),
                                    random_date(self.sim.date, self.sim.date + pd.DateOffset(months=9), self.rng)
                                    )

    def on_simulation_end(self):
        """Do tasks at the end of the simulation: Raise warning and enter to log about women_ids who are sterilized
        when under 30 years old."""
        if self._women_ids_sterilized_below30:
            warnings.warn(UserWarning(f"IDs of women for whom sterilization was initiated when they were under 30:/n"
                                      f"{self._women_ids_sterilized_below30}"))
            logger.info(
                key="women_ids_sterilized_below30",
                data={"ids": self._women_ids_sterilized_below30}
            )


class DirectBirth(Event, IndividualScopeEventMixin):
    """Do birth, with the mother_id set to person_id*(-1) to reflect that this was a `DirectBirth`."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        assert person_id < 0  # check that mother is correctly logged as direct birth mother
        self.sim.do_birth(person_id)  # use actual id for mother


class ContraceptionPoll(RegularEvent, PopulationScopeEventMixin):
    """The regular poll (monthly) for the Contraceptive Module:
    * Determines contraceptive start / stops / switches
    * Determines the onset of pregnancy
    """

    def __init__(self, module, run_do_pregnancy=True, run_update_contraceptive=True):
        super().__init__(module, frequency=DateOffset(months=1))
        self.age_low = 15
        self.age_high = 49

        self.run_do_pregnancy = run_do_pregnancy  # (Provided for testing only)
        self.run_update_contraceptive = run_update_contraceptive  # (Provided for testing only)

    def apply(self, population):
        """Determine who will become pregnant and update contraceptive method."""

        # Determine who will become pregnant, given current contraceptive method
        if self.run_do_pregnancy:
            self.update_pregnancy()

        # Update contraception method
        if self.run_update_contraceptive:
            self.update_contraceptive()

    def update_contraceptive(self):
        """ Determine women that will start, stop or switch contraceptive method."""

        df = self.sim.population.props

        possible_co_users = ((df.sex == 'F') &
                             df.is_alive &
                             df.age_years.between(self.age_low, self.age_high) &
                             ~df.is_pregnant)

        currently_using_co = df.index[possible_co_users &
                                      ~df.co_contraception.isin(['not_using', 'female_sterilization'])]
        currently_not_using_co = df.index[possible_co_users & (df.co_contraception == 'not_using')]

        # initiating: not using -> using
        self.initiate(currently_not_using_co)

        # continue/discontinue/switch: using --> using/not using
        self.discontinue_switch_or_continue(currently_using_co)

        # put everyone older than `age_high` onto not_using:
        df.loc[
            (df.sex == 'F') &
            df.is_alive &
            (df.age_years > self.age_high) &
            (df.co_contraception != 'not_using'),
            'co_contraception'] = 'not_using'

    def initiate(self, individuals_not_using: pd.Index):
        """Check all females not using contraception to determine if contraception starts
        (i.e. category change from 'not_using' to something else).
        """

        # Exit if there are no individuals currently not using a contraceptive:
        if not len(individuals_not_using):
            return

        df = self.sim.population.props
        pp = self.module.processed_params
        rng = self.module.rng

        # Get probability of each individual starting on each contraceptive
        probs = df.loc[individuals_not_using, ['age_years']].merge(
            pp['p_start_per_month'][self.sim.date.year],
            how='left',
            left_on='age_years',
            right_index=True
        ).drop(columns=['age_years'])

        # Determine if individual will start contraceptive
        will_initiate = sample_outcome(probs=probs, rng=rng)

        # Do the contraceptive change
        if len(will_initiate) > 0:
            self.module.schedule_batch_of_contraceptive_changes(
                ids=list(will_initiate),
                old=['not_using'] * len(will_initiate),
                new=list(will_initiate.values())
            )

    def discontinue_switch_or_continue(self, individuals_using: pd.Index):
        """Check all females currently using contraception to determine if they discontinue it, switch to a different
        one, or keep using the same one."""

        # Exit if there are no individuals currently using a contraceptive:
        if not len(individuals_using):
            return

        df = self.sim.population.props
        pp = self.module.processed_params
        rng = self.module.rng

        # Get the probability of discontinuation for each individual (depends on age and current method)
        prob = df.loc[individuals_using, ['age_years', 'co_contraception']].apply(
            lambda row: pp['p_stop_per_month'][self.sim.date.year].at[row.age_years, row.co_contraception],
            axis=1
        )

        # Determine if each individual will discontinue
        will_stop_idx = prob.index[prob > rng.rand(len(prob))]

        # Do the contraceptive change
        if len(will_stop_idx) > 0:
            self.module.schedule_batch_of_contraceptive_changes(
                ids=will_stop_idx,
                old=df.loc[will_stop_idx, 'co_contraception'].values,
                new=['not_using'] * len(will_stop_idx)
            )

        # 2) -- Switches and Continuations for those who do not Discontinue:
        individuals_eligible_for_continue_or_switch = individuals_using.drop(will_stop_idx)

        # Get the probability of switching contraceptive for all those currently using
        switch_prob = df.loc[individuals_eligible_for_continue_or_switch, 'co_contraception'].map(
            pp['p_switch_from_per_month']
        )

        # Randomly select who will switch contraceptive and who will remain on their current contraceptive
        will_switch = switch_prob > rng.random_sample(size=len(individuals_eligible_for_continue_or_switch))
        switch_idx = individuals_eligible_for_continue_or_switch[will_switch]
        switch_idx_below30 = switch_idx[df.loc[switch_idx, 'age_years'] < 30]
        switch_idx_30plus = switch_idx.drop(switch_idx_below30)
        continue_idx = individuals_eligible_for_continue_or_switch[~will_switch]

        # For that do switch, select the new contraceptive using switching matrix
        new_co_below30 = transition_states(
            df.loc[switch_idx_below30, 'co_contraception'], pp['p_switching_to_below30'], rng
        )
        new_co_30plus = transition_states(
            df.loc[switch_idx_30plus, 'co_contraception'], pp['p_switching_to_30plus'], rng
        )
        new_co = pd.concat([new_co_below30, new_co_30plus])

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
        """Look across all women who are using a contraception method to determine if they become pregnant (i.e., the
         method fails)."""

        df = self.module.sim.population.props
        pp = self.module.processed_params
        rng = self.module.rng

        prob_failure_per_month = pp['p_pregnancy_with_contraception_per_month']

        # Get the women who are using a contraceptive that may fail and who may become pregnant (i.e., women who
        # are not in labour, have been pregnant in the last month, have previously had a hysterectomy, can get
        # pregnant.)
        possible_to_fail = (
            df.is_alive
            & (df.sex == 'F')
            & ~df.is_pregnant
            & df.age_years.between(self.age_low, self.age_high)
            & ~df.co_contraception.isin(['not_using', 'female_sterilization'])
            & ~df.la_currently_in_labour
            & ~df.la_has_had_hysterectomy
            & ~df.la_is_postpartum
            & ~df.ps_ectopic_pregnancy.isin(['not_ruptured', 'ruptured'])
        )

        if possible_to_fail.sum():
            # Get probability of method failure for each individual
            prob_of_failure = df.loc[possible_to_fail, ['age_years', 'co_contraception']].apply(
                lambda row: prob_failure_per_month.at[row.age_years, row.co_contraception],
                axis=1
            )

            # Determine if there will be a contraceptive failure for each individual
            idx_failure = prob_of_failure.index[prob_of_failure > rng.random_sample(size=len(prob_of_failure))]

            # Effect these women to be pregnant
            self.set_new_pregnancy(women_id=idx_failure)

    def pregnancy_for_those_not_on_contraceptive(self):
        """Look across all woman who are not using a contraceptive to determine who will become pregnant."""

        df = self.module.sim.population.props
        pp = self.module.processed_params
        rng = self.module.rng

        # Get the subset of women who are not using a contraceptive and who may become pregnant
        subset = (
            df.is_alive
            & (df.sex == 'F')
            & ~df.is_pregnant
            & df.age_years.between(self.age_low, self.age_high)
            & (df.co_contraception == 'not_using')
            & ~df.la_currently_in_labour
            & ~df.la_has_had_hysterectomy
            & ~df.la_is_postpartum
            & ~df.ps_ectopic_pregnancy.isin(['not_ruptured', 'ruptured'])
        )

        if subset.sum():
            # Get the probability of pregnancy for each individual
            prob_pregnancy = df.loc[subset, ['age_years', 'hv_inf']].apply(
                lambda row: pp['p_pregnancy_no_contraception_per_month'].at[
                    row.age_years, 'hv_inf_True' if row.hv_inf else 'hv_inf_False'
                ],
                axis=1)

            # Determine if there will be a pregnancy for each individual
            idx_pregnant = prob_pregnancy.index[prob_pregnancy > rng.rand(len(prob_pregnancy))]

            # Effect these women to be pregnant
            self.set_new_pregnancy(women_id=idx_pregnant)

    def set_new_pregnancy(self, women_id: list):
        """Effect that these women are now pregnancy and enter to the log"""
        df = self.sim.population.props

        for w in women_id:
            woman = df.loc[w, ['co_contraception', 'age_years']]
            method_before_pregnancy = woman['co_contraception']

            # Determine if this is unintended or not. (We say that it is 'unintended' if the women is using a
            # contraceptive when she becomes pregnant.)
            unintended = (method_before_pregnancy != 'not_using')

            # Update properties (including that she is no longer on any contraception; and store the method used prior
            # pregnancy).
            new_properties = {
                'co_contraception': 'not_using',
                'is_pregnant': True,
                'date_of_last_pregnancy': self.sim.date,
                'co_unintended_preg': unintended,
            }
            df.loc[w, new_properties.keys()] = new_properties.values()

            # Set date of labour in the Labour module
            self.sim.modules['Labour'].set_date_of_labour(w)

            # Log that a pregnancy has occurred
            logger.info(key='pregnancy',
                        data={
                            'woman_id': w,
                            'age_years': woman['age_years'],
                            'contraception': method_before_pregnancy,
                            'unintended': unintended
                        },
                        description='pregnancy following the failure of contraceptive method')


class ContraceptionLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """Logs state of contraceptive usage in the population at a point in time."""
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        df = population.props

        # Log summary of usage of contraceptives (without age-breakdown)
        # (NB. sort_index ensures the resulting dict has keys in the same order, which is requirement of the logging.)
        logger.info(key='contraception_use_summary',
                    data=df.loc[
                        df.is_alive & (df.sex == 'F') & df.age_years.between(15, 49), 'co_contraception'
                    ].value_counts().sort_index().to_dict(),
                    description='Counts of women on each type of contraceptive at a point in time.')

        # Log summary of usage of contraceptives (with age-breakdown)
        logger.info(key='contraception_use_summary_by_age',
                    data=flatten_multi_index_series_into_dict_for_logging(
                        df.loc[
                            df.is_alive & (df.sex == 'F') & df.age_years.between(15, 49)
                            ].groupby(by=['co_contraception', 'age_range']).size().sort_index()
                    ),
                    description='Counts of women, by age-range, on each type of contraceptive at a point in time.')


class HSI_Contraception_FamilyPlanningAppt(HSI_Event, IndividualScopeEventMixin):
    """HSI event for the starting a contraceptive method, maintaining use of a method of a contraceptive, or switching
     between contraceptives."""

    def __init__(self, module, person_id, new_contraceptive):
        super().__init__(module, person_id=person_id)

        _facility_level = '2' if new_contraceptive in ('implant', 'female_sterilization') else '1a'

        self.new_contraceptive = new_contraceptive
        self._number_of_times_run = 0

        self.TREATMENT_ID = "Contraception_Routine"
        self.ACCEPTED_FACILITY_LEVEL = _facility_level

    @property
    def EXPECTED_APPT_FOOTPRINT(self):
        """Return the expected appt footprint based on contraception method and whether the HSI has been rescheduled."""
        person_id = self.target
        current_method = self.sim.population.props.loc[person_id].co_contraception
        if self._number_of_times_run > 0:  # if it is to re-schedule due to unavailable consumables
            return self.make_appt_footprint({})
        # if to switch to a method
        elif self.new_contraceptive in ['female_sterilization']:
            return self.make_appt_footprint({'MinorSurg': 1})
        elif self.new_contraceptive != current_method:
            return self.make_appt_footprint({'FamPlan': 1})
        # if to maintain on a method
        elif self.new_contraceptive in ['injections', 'IUD', 'implant']:
            return self.make_appt_footprint({'FamPlan': 1})
        elif self.new_contraceptive in ['male_condom', 'other_modern', 'pill']:
            return self.make_appt_footprint({'PharmDispensing': 1})
        else:
            return self.make_appt_footprint({})
            warnings.warn(UserWarning("Assumed empty footprint for Contraception Routine appt because couldn't find"
                                      "actual case."))

    def apply(self, person_id, squeeze_factor):
        """If the relevant consumable is available, do change in contraception and log it"""

        person = self.sim.population.props.loc[person_id]
        current_method = person.co_contraception

        if not (person.is_alive and not person.is_pregnant):
            return

        # Record the date that Family Planning Appointment happened for this person
        self.sim.population.props.at[person_id, "co_date_of_last_fp_appt"] = self.sim.date

        # Determine essential and optional items
        # TODO: we don't distinguish essential X optional for contraception methods yet, will need to update once we do
        items_essential = self.module.cons_codes[self.new_contraceptive]
        items_optional = {}
        # Record use of consumables and default the person to "not_using" if the consumable is not available.
        # If initiating use of a modern contraceptive method except condoms (after not using any or using non-modern
        # contraceptive or using condoms), "co_initiation" items are used along with the method consumables.
        if (
            (current_method not in self.module.contraceptives_initiated_with_additional_items)
            and (self.new_contraceptive in self.module.contraceptives_initiated_with_additional_items)
        ):
            items_optional = self.module.cons_codes["co_initiation"]
            cons_available = self.get_consumables(
                item_codes=items_essential,
                optional_item_codes=items_optional,
                return_individual_results=True
            )
        else:
            cons_available = self.get_consumables(
                item_codes=items_essential,
                return_individual_results=True
            )

        items_all = {**items_essential, **items_optional}

        # Determine whether the contraception is administrated (ie all essential items are available),
        # if so do log the availability of all items, if not set the contraception to "not_using":
        co_administrated = all(v for k, v in cons_available.items() if k in items_essential)

        if co_administrated:
            # if not running contraception module at debug level, it's pointless to save the (not) available items
            if logger.isEnabledFor(logging.DEBUG):
                # Do logging of consumables for the `Consumables` module. Although this is duplicative of the detailed
                # logging of consumables in the `HealthSystem` module, it is done here too as the file generated by the
                # `HealthSystem` module is extremely large and not usable for the very long runs that are needed for the
                # analyses using the `Contraception` module.
                items_available = {k: v for k, v in items_all.items() if cons_available[k]}
                items_not_available = {k: v for k, v in items_all.items() if not cons_available[k]}
                logger.debug(key='Contraception_consumables',
                             data={
                                 'TREATMENT_ID': (self.TREATMENT_ID if self.TREATMENT_ID is not None else ""),
                                 'Item_Available': str(items_available),
                                 'Item_NotAvailable': str(items_not_available),
                             },
                             # NB. Casting the data to strings because logger complains with dict of varying sizes/keys
                             description="Record of each contraception consumable item if the contraceptive is "
                                         "administrated."
                             )

            _new_contraceptive = self.new_contraceptive
        else:
            _new_contraceptive = "not_using"

        if current_method != _new_contraceptive:
            # Do the change:
            self.module.do_and_log_individual_contraception_change(
                woman_id=self.target,
                old=current_method,
                new=_new_contraceptive
            )
            # (N.B. If the current method is the same as the new method, there is no logging.)

        # If the intended change was not possible due to non-available consumable, reschedule the appointment
        if (not co_administrated) and (
            self._number_of_times_run < self.module.parameters['max_number_of_runs_of_hsi_if_consumable_not_available']
        ):
            self.reschedule()

    def post_apply_hook(self):
        self._number_of_times_run += 1

    def reschedule(self):
        """Schedule for this same HSI_Event to occur tomorrow."""
        self.module.sim.modules['HealthSystem'].schedule_hsi_event(hsi_event=self,
                                                                   topen=self.sim.date + pd.DateOffset(days=1),
                                                                   tclose=None,
                                                                   priority=1)

    def never_ran(self):
        """If this HSI never ran, the person defaults to "not_using" a contraceptive."""
        person = self.sim.population.props.loc[self.target]

        if not person.is_alive:
            return

        self.module.do_and_log_individual_contraception_change(
            woman_id=self.target,
            old=person.co_contraception,  # Current Method
            new="not_using"
        )


class StartInterventions(Event, PopulationScopeEventMixin):
    """
    This event is run by Contraception module exactly once when interventions are introduced, or never if interventions
    are not used. If run, it updates parameters changed to reflect the FP interventions to be used from now on.
    """

    def __init__(self, module):
        super().__init__(module)
        assert isinstance(module, Contraception)

    def apply(self, population):

        # Update module parameters to enable interventions
        self.module.processed_params = self.module.update_params_for_interventions()


# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
#
# Accessory modules for testing / debugging
#
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

class SimplifiedPregnancyAndLabour(Module):
    """Simplified module to replace `Labour`, 'PregnancySupervisor` and other associated module, for use in
    testing/calibrating the Contraception Module. The module calls itself Labour and provides the method
    `set_date_of_labour`, which is called by the Contraception Module at the onset of pregnancy. It schedules an event
    for the end of the pregnancy (approximately 9 months later) which may or may not result in a live birth."""

    INIT_DEPENDENCIES = {'Contraception'}

    ALTERNATIVE_TO = {'Labour'}

    METADATA = {}

    PARAMETERS = {
        'prob_live_birth': Parameter(Types.REAL, 'Probability that a pregnancy results in a live birth.')
    }

    PROPERTIES = {
        'la_currently_in_labour': Property(Types.BOOL, 'whether this woman is currently in labour'),
        'la_has_had_hysterectomy': Property(Types.BOOL, 'whether this woman has had a hysterectomy as treatment for a '
                                                        'complication of labour, and therefore is unable to conceive'),
        'la_is_postpartum': Property(Types.BOOL, 'Whether a woman is in the postpartum period, from delivery until '
                                                 'day +42 (6 weeks)'),
        'ps_ectopic_pregnancy': Property(Types.CATEGORICAL, 'Whether a woman is experiencing ectopic pregnancy and'
                                                            ' its current state',
                                         categories=['none', 'not_ruptured', 'ruptured']
                                         )
    }

    def __init__(self, *args):
        super().__init__(name='Labour')

    def read_parameters(self, *args):
        parameter_dataframe = pd.read_excel(self.sim.modules['Contraception'].resourcefilepath /
                                            'ResourceFile_Contraception.xlsx',
                                            sheet_name='simplified_labour_parameters')
        self.load_parameters_from_dataframe(parameter_dataframe)

    def initialise_population(self, population):
        df = population.props
        df.loc[df.is_alive, 'la_currently_in_labour'] = False
        df.loc[df.is_alive, 'la_has_had_hysterectomy'] = False
        df.loc[df.is_alive, 'la_is_postpartum'] = False
        df.loc[df.is_alive, 'ps_ectopic_pregnancy'] = np.NAN

    def initialise_simulation(self, *args):
        pass

    def on_birth(self, mother_id, child_id):
        df = self.sim.population.props
        df.at[child_id, 'la_currently_in_labour'] = False
        df.at[child_id, 'la_has_had_hysterectomy'] = False
        df.at[child_id, 'la_is_postpartum'] = False
        df.at[child_id, 'ps_ectopic_pregnancy'] = np.NAN

    def set_date_of_labour(self, person_id):
        """This is a drop-in replacement for the method in Labour that triggers the processes that determine the outcome
        of a pregnancy."""

        self.sim.schedule_event(EndOfPregnancyEvent(module=self,
                                                    person_id=person_id,
                                                    live_birth=(self.rng.rand() < self.parameters['prob_live_birth'])
                                                    ),
                                random_date(
                                    self.sim.date + pd.DateOffset(months=9) - pd.DateOffset(days=14),
                                    self.sim.date + pd.DateOffset(months=9) + pd.DateOffset(days=14),
                                    self.rng)
                                )


class EndOfPregnancyEvent(Event, IndividualScopeEventMixin):
    """This event signals the end of the pregnancy, which may or may not result in a live-birth"""

    def __init__(self, module, person_id, live_birth):
        super().__init__(module, person_id=person_id)
        self.live_birth = live_birth

    def apply(self, person_id):
        """End pregnancy and do live birth if needed"""

        if self.live_birth:
            self.sim.do_birth(person_id)
        else:
            self.sim.modules['Contraception'].end_pregnancy(person_id)


# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
#
# Helper functions
#
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

def get_medium_variant_asfr_from_wpp_resourcefile(dat: pd.DataFrame, months_exposure: int) -> dict:
    """Process the data on age-specific fertility rates into a form that can be used to quickly map
    age-ranges to an age-specific fertility rate (for the "Medium Variant" in the WPP data source).
    :param dat: Raw form of the data in `ResourceFile_ASFR_WPP.csv`
    :param months_exposure: The time (in integer number of months) over which the risk of pregnancy should be
    computed.
    :returns: a dict, keyed by year, giving a dataframe of risk of pregnancy over a period, by age """

    dat = dat.drop(dat[~dat.Variant.isin(['WPP_Estimates', 'WPP_Medium variant'])].index)
    dat['Period-Start'] = dat['Period'].str.split('-').str[0].astype(int)
    dat['Period-End'] = dat['Period'].str.split('-').str[1].astype(int)
    years = range(min(dat['Period-Start'].values), 1 + max(dat['Period-End'].values))

    # Convert the rates for asfr (rate of live-birth per year) to a rate per the frequency of this event repeating
    dat['asfr_per_period'] = convert_annual_prob_to_monthly_prob(dat['asfr'], num_months=months_exposure)

    asfr = dict()  # format is {year: {age-range: asfr}}
    for year in years:
        asfr[year] = dat.loc[
            (year >= dat['Period-Start']) & (year <= dat['Period-End'])
            ].set_index('Age_Grp')['asfr_per_period'].to_dict()

    return asfr


def convert_annual_prob_to_monthly_prob(p_annual, num_months=1):
    return 1.0 - ((1.0 - p_annual) ** (num_months / 12.0))


def convert_monthly_prob_to_annual_prob(p_monthly):
    return 1.0 - ((1.0 - p_monthly) ** 12.0)
