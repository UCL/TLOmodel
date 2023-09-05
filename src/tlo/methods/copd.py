from pathlib import Path
from typing import Dict, List

import pandas as pd

from tlo import Module, Parameter, Property, Types, logging
from tlo.analysis.utils import flatten_multi_index_series_into_dict_for_logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata
from tlo.methods.causes import Cause
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.symptommanager import Symptom
from tlo.util import random_date

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch_lungfunction_cats = list(range(7))

gbd_causes_of_copd_represented_in_this_module = {
    'Chronic obstructive pulmonary disease',
}


class Copd(Module):
    """The module responsible for determining Chronic Obstructive Pulmonary Diseases (COPD) status and outcomes.
     and initialises parameters and properties associated with COPD plus functions and events related to COPD."""

    INIT_DEPENDENCIES = {'SymptomManager', 'Lifestyle'}
    ADDITIONAL_DEPENDENCIES = set()

    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN,
    }

    CAUSES_OF_DEATH = {
        # Chronic Obstructive Pulmonary Diseases
        f'COPD_cat{lung_f_category}': Cause(gbd_causes=sorted(gbd_causes_of_copd_represented_in_this_module),
                                            label='COPD') for lung_f_category in ch_lungfunction_cats
    }

    CAUSES_OF_DISABILITY = {
        # Chronic Obstructive Pulmonary Diseases
        'COPD': Cause(gbd_causes=sorted(gbd_causes_of_copd_represented_in_this_module),
                      label='COPD')
    }

    PARAMETERS = {
        'prob_mod_exacerb': Parameter(
            Types.LIST, 'probability of moderate exacerbation given lung function, for each lung function category. '
        ),
        'prob_sev_exacerb': Parameter(
            Types.LIST, 'probability of severe exacerbation given lung function, for each lung function category.'
        ),
        'prob_progress_to_next_cat': Parameter(
            Types.REAL, 'probability of changing from a lower lung function category to a higher lung function category'
        ),

        'prob_will_die_sev_exacerbation_ge80': Parameter(
            Types.REAL, 'probability that a person equal to or greater than 80 years will die of severe exacerbation '
        ),
        'prob_will_die_sev_exacerbation_7079': Parameter(
            Types.REAL, 'probability that a person between 70 to 79 years will die of severe exacerbation '
        ),
        'prob_will_die_sev_exacerbation_6069': Parameter(
            Types.REAL, 'probability that a person between 60 to 99 years will die of severe exacerbation '
        ),
        'prob_will_die_sev_exacerbation_5059': Parameter(
            Types.REAL, 'probability that a person between 50 to 59 years will die of severe exacerbation '
        ),
        'prob_will_die_sev_exacerbation_4049': Parameter(
            Types.REAL, 'probability that a person between 40 to 49 years will die of severe exacerbation '
        ),
        'prob_will_die_sev_exacerbation_3039': Parameter(
            Types.REAL, 'probability that a person between 30 to 39 years will die of severe exacerbation '
        ),
        'prob_will_die_sev_exacerbation_lt30': Parameter(
            Types.REAL, 'probability that a person less than 30 years will die of severe exacerbation '
        ),

        'eff_oxygen': Parameter(
            Types.REAL, 'the effect of oxygen on individuals with severe exacerbation '
        ),
        'rel_risk_progress_per_higher_cat': Parameter(
            Types.REAL, 'relative risk of progression to the next higher level of lung function for each higher level '
        ),
        'rel_risk_tob': Parameter(
            Types.REAL, 'relative effect of tobacco on the rate of lung function progression'
        ),
        'rel_risk_wood_burn_stove': Parameter(
            Types.REAL, 'relative risk of wood burn stove on the rate of lung function progression'
        ),
        'prob_not_tob_lung_func_15_39': Parameter(
            Types.LIST, 'probability of lung function categories in individuals aged 15-39 and do not smoke tobacco'
        ),
        'prob_not_tob_lung_func_40_59': Parameter(
            Types.LIST, 'probability of lung function categories in individuals aged 40-59 and do not smoke tobacco'
        ),
        'prob_not_tob_lung_func_gr59': Parameter(
            Types.LIST, 'probability of lung function categories in individuals aged 60+ and do not smoke tobacco'
        ),

        'prob_tob_lung_func_15_39': Parameter(
            Types.LIST, 'probability of lung function categories in individuals aged 15-39 who smoke tobacco'
        ),
        'prob_tob_lung_func_40_59': Parameter(
            Types.LIST, 'probability of lung function categories in individuals aged 40-59 who smoke tobacco'
        ),
        'prob_tob_lung_func_gr59': Parameter(
            Types.LIST, 'probability of lung function categories in individuals aged 60+ who smoke tobacco'
        )
    }

    PROPERTIES = {
        'ch_lungfunction': Property(
            Types.CATEGORICAL, 'Lung function of the person.'
                               '0 for those under 15; on a 7-point scale for others, from 0 (Perfect) to 6 (End-Stage'
                               ' COPD).', categories=ch_lungfunction_cats, ordered=True,
        ),
        'ch_will_die_this_episode': Property(
            Types.BOOL, 'Whether this person will die during a current severe exacerbation'
        ),
        'ch_has_inhaler': Property(
            Types.BOOL, 'Whether the person is currently using an inhaler'
        ),
    }

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = Path(resourcefilepath)
        self.models = None  # Will hold pointer to helper class containing models
        self.item_codes = None  # Will hold dict containing the item_codes for consumables needed in the HSI Events.

    def read_parameters(self, data_folder):
        """ Read all parameters and define symptoms (if any)"""
        self.load_parameters_from_dataframe(pd.read_csv(self.resourcefilepath / 'ResourceFile_Copd.csv'))
        self.define_symptoms()

    def pre_initialise_population(self):
        """Things to do before processing the population:
         * Generate the models
         """
        self.models = CopdModels(self.parameters, self.rng)

    def initialise_population(self, population):
        """ Set initial values of properties values for all individuals"""
        df = population.props
        df.loc[df.is_alive, 'ch_lungfunction'] = self.models.init_lung_function(df.loc[df.is_alive])
        df.loc[df.is_alive, 'ch_will_die_this_episode'] = False
        df.loc[df.is_alive, 'ch_has_inhaler'] = False

    def initialise_simulation(self, sim):
        """ Get ready for simulation start:
         * Schedule the main polling event
         * Look-up item codes for consumables
        """
        sim.schedule_event(CopdPollEvent(self), sim.date)
        self.lookup_item_codes()

    def on_birth(self, mother_id, child_id):
        """ Initialise COPD properties for a newborn individual."""
        props = {
            'ch_lungfunction': self.models.at_birth_lungfunction(child_id),
            'ch_will_die_this_episode': False,
            'ch_has_inhaler': False
        }
        self.sim.population.props.loc[child_id, props.keys()] = props.values()

    def report_daly_values(self):
        """Return disability weight for alive persons, based on the current status of ch_lungfunction."""
        df = self.sim.population.props
        return df.loc[df.is_alive, 'ch_lungfunction'].map(self.models.disability_weight_given_lungfunction)

    def define_symptoms(self):
        """Define and register Symptoms"""
        self.sim.modules['SymptomManager'].register_symptom(
            Symptom('breathless_moderate'),
            Symptom.emergency('breathless_severe'),
        )

    def lookup_item_codes(self):
        """Look-up the item-codes for the consumables needed in the HSI Events for this module."""
        # todo: Need to look-up these item-codes.
        self.item_codes = {
            'bronchodilater_inhaler': 293,
            'steroid_inhaler': 294,
            'oxygen': 127,
            'aminophylline': 292,
            'amoxycillin': 125,
            'prednisolone': 291
        }

    def do_logging(self):
        """Log current states."""
        df = self.sim.population.props
        counts = df.loc[df.is_alive].groupby(by=['sex', 'age_range', 'li_tob', 'ch_lungfunction']).size()
        logger.info(
            key='copd_prevalence',
            description='Proportion of alive persons in each COPD category currently (by age and sex)',
            data=flatten_multi_index_series_into_dict_for_logging(counts)
        )

    def give_inhaler(self, person_id: int, hsi_event: HSI_Event):
        """Give inhaler if person does not already have one"""
        df = self.sim.population.props
        has_inhaler = df.at[person_id, 'ch_has_inhaler']
        if not has_inhaler:
            if hsi_event.get_consumables(self.item_codes['bronchodilater_inhaler']):
                df.at[person_id, 'ch_has_inhaler'] = True

    def do_when_present_with_breathless(self, person_id: int, hsi_event: HSI_Event):
        """What to do when a person presents at the generic first appt HSI with a symptom of `breathless_severe` or
        `breathless_moderate`.
        * If severe --> give the inhaler and schedule the HSI for Treatment
        * Otherwise --> just give inhaler.
        """
        self.give_inhaler(hsi_event=hsi_event, person_id=person_id)

        if 'breathless_severe' in self.sim.modules['SymptomManager'].has_what(person_id):
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_Copd_TreatmentOnSevereExacerbation(module=self, person_id=person_id),
                priority=0,
                topen=self.sim.date,
                tclose=None,
            )


class CopdModels:
    """Helper class containing the models needed for the Copd module."""

    def __init__(self, params, rng):
        self.params = params
        self.rng = rng

        # The chance (in a 3-month period) of progressing to the next (greater) category of ch_lungfunction
        self.__Prob_Progress_LungFunction__ = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            self.params['prob_progress_to_next_cat'],
            Predictor(
                'ch_lungfunction'
            ).when(0, (self.params['rel_risk_progress_per_higher_cat'] ** 0))
            .when(1, (self.params['rel_risk_progress_per_higher_cat'] ** 1))
            .when(2, (self.params['rel_risk_progress_per_higher_cat'] ** 2))
            .when(3, (self.params['rel_risk_progress_per_higher_cat'] ** 3))
            .when(4, (self.params['rel_risk_progress_per_higher_cat'] ** 4))
            .when(5, (self.params['rel_risk_progress_per_higher_cat'] ** 5)),
            Predictor(
                'li_tob'
            ).when(True, self.params['rel_risk_tob']),
            Predictor('li_wood_burn_stove').when(True, self.params['rel_risk_wood_burn_stove'])
        )
        # The probability (in a 3-month period) of having a Moderate Exacerbation
        self.__Prob_ModerateExacerbation__ = LinearModel.multiplicative(
            Predictor(
                'ch_lungfunction'
            ).when(0, self.params['prob_mod_exacerb'][0])
            .when(1, self.params['prob_mod_exacerb'][1])
            .when(2, self.params['prob_mod_exacerb'][2])
            .when(3, self.params['prob_mod_exacerb'][3])
            .when(4, self.params['prob_mod_exacerb'][4])
            .when(5, self.params['prob_mod_exacerb'][5])
            .when(6, self.params['prob_mod_exacerb'][6])
        )

        # The probability (in a 3-month period) of having a Severe Exacerbation
        self.__Prob_SevereExacerbation__ = LinearModel.multiplicative(
            Predictor(
                'ch_lungfunction'
            ).when(0, self.params['prob_sev_exacerb'][0])
            .when(1, self.params['prob_sev_exacerb'][1])
            .when(2, self.params['prob_sev_exacerb'][2])
            .when(3, self.params['prob_sev_exacerb'][3])
            .when(4, self.params['prob_sev_exacerb'][4])
            .when(5, self.params['prob_sev_exacerb'][5])
            .when(6, self.params['prob_sev_exacerb'][6])
        )

        self.__Prob_Will_Die_SevereExacerbation__ = LinearModel.multiplicative(
            Predictor(
                'age_years'
            ).when('>=80', self.params['prob_will_die_sev_exacerbation_ge80'])
            .when('.between(70, 79)', self.params['prob_will_die_sev_exacerbation_7079'])
            .when('.between(60, 69)', self.params['prob_will_die_sev_exacerbation_6069'])
            .when('.between(50, 59)', self.params['prob_will_die_sev_exacerbation_5059'])
            .when('.between(40, 49)', self.params['prob_will_die_sev_exacerbation_4049'])
            .when('.between(30, 39)', self.params['prob_will_die_sev_exacerbation_3039'])
            .when('<30', self.params['prob_will_die_sev_exacerbation_lt30'])
        )
        self.__Prob_OxygenEffect_On_Mortality__ = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            self.params['eff_oxygen'],
            Predictor(
                'age_years'
            ).when('>=80', self.params['prob_will_die_sev_exacerbation_ge80'])
            .when('.between(70, 79)', self.params['prob_will_die_sev_exacerbation_7079'])
            .when('.between(60, 69)', self.params['prob_will_die_sev_exacerbation_6069'])
            .when('.between(50, 59)', self.params['prob_will_die_sev_exacerbation_5059'])
            .when('.between(40, 49)', self.params['prob_will_die_sev_exacerbation_4049'])
            .when('.between(30, 39)', self.params['prob_will_die_sev_exacerbation_3039'])
            .when('<30', self.params['prob_will_die_sev_exacerbation_lt30'])
        )

    def init_lung_function(self, df: pd.DataFrame) -> pd.Series:
        """Returns the values for ch_lungfunction for an initial population described in `df`
        """
        # store lung function categories
        cats = ch_lungfunction_cats

        # lung function categories for non-smokers
        idx_15_39_not_tob = df.index[(df.age_years.between(15, 39)) & ~df.li_tob]
        cats_15_39_not_tob = dict(
            zip(idx_15_39_not_tob, self.rng.choice(cats, p=self.params['prob_not_tob_lung_func_15_39'],
                                                   size=len(idx_15_39_not_tob))))

        idx_40_59_not_tob = df.index[(df.age_years.between(40, 59)) & ~df.li_tob]
        cats_40_59_not_tob = dict(
            zip(idx_40_59_not_tob, self.rng.choice(cats, p=self.params['prob_not_tob_lung_func_40_59'],
                                                   size=len(idx_40_59_not_tob))))

        idx_gr59_not_tob = df.index[(df.age_years >= 60) & ~df.li_tob]
        cats_gr59_not_tob = dict(
            zip(idx_gr59_not_tob, self.rng.choice(cats, p=self.params['prob_not_tob_lung_func_gr59'],
                                                  size=len(idx_gr59_not_tob))))

        # lung function categories for smokers
        idx_15_39_tob = df.index[(df.age_years.between(15, 39)) & df.li_tob]
        cats_15_39_tob = dict(zip(idx_15_39_tob, self.rng.choice(cats, p=self.params['prob_tob_lung_func_15_39'],
                                                                 size=len(idx_15_39_tob))))

        idx_40_59_tob = df.index[(df.age_years.between(40, 59)) & df.li_tob]
        cats_40_59_tob = dict(zip(idx_40_59_tob, self.rng.choice(cats, p=self.params['prob_tob_lung_func_40_59'],
                                                                 size=len(idx_40_59_tob))))

        idx_gr59_tob = df.index[(df.age_years >= 60) & df.li_tob]
        cats_gr59_tob = dict(zip(idx_gr59_tob, self.rng.choice(cats, p=self.params['prob_tob_lung_func_gr59'],
                                                               size=len(idx_gr59_tob))))

        # For under-15s, assign the category that would be given at birth
        idx_notover15 = df.index[df.age_years < 15]
        cats_for_under15s = {idx: self.at_birth_lungfunction(idx) for idx in idx_notover15}

        return pd.Series(index=df.index, data={**cats_for_under15s, **cats_15_39_not_tob, **cats_40_59_not_tob,
                                               **cats_gr59_not_tob, **cats_15_39_tob, **cats_40_59_tob, **cats_gr59_tob
                                               })

    def at_birth_lungfunction(self, person_id: int) -> int:
        """ Returns value for ch_lungfunction for the person at birth."""
        return 0

    def prob_livesaved_given_treatment(self, df: pd.DataFrame, oxygen: bool, aminophylline: bool):
        """ Returns the probability that a treatment prevents death during an exacerbation, according to the treatment
        provided (we're considering oxygen only as there is no evidence of a mortality benefits on aminophylline) """
        if oxygen:
            return self.__Prob_OxygenEffect_On_Mortality__.predict(df, self.rng)
        else:
            return False

    @property
    def disability_weight_given_lungfunction(self) -> Dict:
        """ Returns `dict` with the mapping between a lung_function and a disability weight """
        return {
            0: 0.0,
            1: 0.0,
            2: 0.0,
            3: 0.0,
            4: 0.02,
            5: 0.2,
            6: 0.4,
        }

    def will_progres_to_next_cat_of_lungfunction(self, df: pd.DataFrame) -> List:
        """Returns the indices (corresponding to the person_id) that will progress to the next category up of
         `ch_lungfunction`, based on the probability of progression in a 3-month period."""
        will_progress_to_next_category = self.__Prob_Progress_LungFunction__.predict(df, self.rng,
                                                                                     squeeze_single_row_output=False)
        return will_progress_to_next_category[will_progress_to_next_category].index.to_list()

    def will_get_moderate_exacerbation(self, df: pd.DataFrame) -> List:
        """Returns the indices (corresponding to the person_id) that will have a Moderate Exacerbation, based on the
        probability of having a Severe Exacerbation in a 3-month period"""
        will_get_ex = self.__Prob_ModerateExacerbation__.predict(df, self.rng, squeeze_single_row_output=False)
        return will_get_ex[will_get_ex].index.to_list()

    def will_get_severe_exacerbation(self, df: pd.DataFrame) -> List:
        """Returns the indices (corresponding to the person_id) that will have a Severe Exacerbation, based on the
        probability of having a Severe Exacerbation in a 3-month period"""
        will_get_ex = self.__Prob_SevereExacerbation__.predict(df, self.rng, squeeze_single_row_output=False)
        return will_get_ex[will_get_ex].index.to_list()

    def will_die_given_severe_exacerbation(self, df: pd.DataFrame) -> bool:
        """Return bool indicating if a person will die due to a severe exacerbation.
        :param df: pandas dataframe """
        return self.__Prob_Will_Die_SevereExacerbation__.predict(df, self.rng)


def eligible_to_progress_to_next_lung_function(df: pd.DataFrame) -> pd.Series:
    """ Returns a pd.Series with the same index as `df` and with value `True` where individuals are eligible to progress
     to the next level of ch_lungfunction (i.e., alive, aged 15+, and not in the highest category already).
    :param df: an individual population dataframe """
    return (
        df.is_alive & (df.age_years >= 15) & (df['ch_lungfunction'] != ch_lungfunction_cats[-1])
    )


class CopdPollEvent(RegularEvent, PopulationScopeEventMixin):
    """An event that controls the COPD infection process and logs current states. It repeats every 3 months."""

    def __init__(self, module):
        super().__init__(module, frequency=pd.DateOffset(months=3))

    def apply(self, population):
        """
         * Progress persons to higher categories of ch_lungfunction
         * Schedules Exacerbation (Moderate / Severe) events.
         * Log current states.
        """
        df = population.props

        # call a function to make individuals progress to a next higher lung function
        self.progress_to_next_lung_function(df)

        # Schedule Moderate Exacerbation
        for idx in self.module.models.will_get_moderate_exacerbation(df.loc[df.is_alive]):
            self.sim.schedule_event(CopdExacerbationEvent(self.module, idx, severe=False),
                                    self.gen_random_date_in_next_three_months())

        # Schedule Severe Exacerbation (persons can have a moderate and severe exacerbation in the same 3-month period)
        for idx in self.module.models.will_get_severe_exacerbation(df.loc[df.is_alive]):
            self.sim.schedule_event(CopdExacerbationEvent(self.module, idx, severe=True),
                                    self.gen_random_date_in_next_three_months())

        # Logging
        self.module.do_logging()

    def gen_random_date_in_next_three_months(self):
        """Returns a datetime for a day that is chosen randomly to be within the next 3 months."""
        return random_date(self.sim.date, self.sim.date + pd.DateOffset(months=3), self.module.rng)

    @staticmethod
    def increment_category(ser: pd.Series) -> pd.Series:
        """Returns a pd.Series with same index as `ser` but with the categories shifted to next higher one."""
        new_codes = ser.cat.codes + 1
        ser.values[:] = new_codes
        return ser

    def progress_to_next_lung_function(self, df):
        """ make individuals progress to a next higher lung function """
        idx_will_progress_to_next_category = self.module.models.will_progres_to_next_cat_of_lungfunction(
            df.loc[eligible_to_progress_to_next_lung_function(df)])
        df.loc[idx_will_progress_to_next_category, 'ch_lungfunction'] = self.increment_category(
            df.loc[idx_will_progress_to_next_category, 'ch_lungfunction'])


class CopdExacerbationEvent(Event, IndividualScopeEventMixin):
    """An Exacerbation, which may be 'Severe' (severe=True) or 'Moderate' (severe=False).
     * A Moderate Exacerbation will not cause death but may cause non-emergency healthcare seeking;
     * A Severe Exacerbation may cause death and may cause emergency healthcare seeking.
    """

    def __init__(self, module, person_id, severe: bool = False):
        super().__init__(module, person_id=person_id)
        self.severe = severe

    def apply(self, person_id):
        df = self.sim.population.props
        if not df.at[person_id, 'is_alive']:
            return

        # Onset symptom (that will auto-resolve in two days)
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=person_id,
            symptom_string='breathless_severe' if self.severe else 'breathless_moderate',
            add_or_remove='+',
            disease_module=self.module,
            duration_in_days=2,
        )

        if self.severe:
            # Work out if the person will die of this exacerbation (if not treated). If they die, they die the next day.
            if self.module.models.will_die_given_severe_exacerbation(df.iloc[[person_id]]):
                df.at[person_id, "ch_will_die_this_episode"] = True
                self.sim.schedule_event(CopdDeath(self.module, person_id), self.sim.date + pd.DateOffset(days=1))


class CopdDeath(Event, IndividualScopeEventMixin):
    """This event will cause the person to die to 'COPD' if the person's property `ch_will_die_this_episode` is True.
    It is scheduled when a Severe Exacerbation is lethal, but can be cancelled if a subsequent treatment is successful.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props
        person = df.loc[person_id, ['is_alive', 'age_years', 'ch_will_die_this_episode', 'ch_lungfunction']]
        # Check if an individual should still die and, if so, cause the death
        if person.is_alive and person.ch_will_die_this_episode:
            self.sim.modules['Demography'].do_death(
                individual_id=person_id,
                cause=f'COPD_cat{person.ch_lungfunction}',
                originating_module=self.module,
            )


class HSI_Copd_TreatmentOnSevereExacerbation(HSI_Event, IndividualScopeEventMixin):
    """ HSI event for issuing treatment to all individuals with severe exacerbation. We first check the availability of
    oxygen at the default facility(1a). If no oxygen is found we refer individuals to the next higher level facility
    until either we find oxygen or we are at the highest facility level. Here facility levels are ["1a", "1b", "2"]"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.facility_levels_index = 0  # an index to facility levels
        self.all_facility_levels = ["1a", "1b", "2"]  # available facility levels

        self.TREATMENT_ID = "Copd_Treatment"
        self.ACCEPTED_FACILITY_LEVEL = self.all_facility_levels[self.facility_levels_index]
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 2})

    def apply(self, person_id, squeeze_factor):
        """What to do when someone presents for care with an exacerbation.
         * Provide treatment: whatever is available at this facility at this time (no referral).
        """
        df = self.sim.population.props
        if not self.get_consumables(self.module.item_codes['oxygen']):
            # refer to the next higher facility if the current facility has no oxygen
            self.facility_levels_index += 1
            if self.facility_levels_index >= len(self.all_facility_levels):
                return
            self.ACCEPTED_FACILITY_LEVEL = self.all_facility_levels[self.facility_levels_index]
            self.sim.modules['HealthSystem'].schedule_hsi_event(self, topen=self.sim.date, priority=0)

        else:
            # Give oxygen and AminoPhylline, if possible, ... and cancel death if the treatment is successful.
            prob_treatment_success = self.module.models.prob_livesaved_given_treatment(
                df=df.iloc[[person_id]],
                oxygen=self.get_consumables(self.module.item_codes['oxygen']),
                aminophylline=self.get_consumables(self.module.item_codes['aminophylline'])
            )

            if prob_treatment_success:
                df.at[person_id, 'ch_will_die_this_episode'] = False
