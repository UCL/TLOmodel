from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.analysis.utils import flatten_nested_dict
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import Metadata
from tlo.methods.causes import Cause
from tlo.methods.hsi_event import HSI_Event
from tlo.methods.hsi_generic_first_appts import GenericFirstAppointmentsMixin
from tlo.methods.symptommanager import Symptom
from tlo.util import random_date, read_csv_files

if TYPE_CHECKING:
    from tlo.methods.hsi_generic_first_appts import HSIEventScheduler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Measles(Module, GenericFirstAppointmentsMixin):
    """This module represents measles infections and disease."""

    INIT_DEPENDENCIES = {'Demography', 'HealthSystem', 'SymptomManager'}

    OPTIONAL_INIT_DEPENDENCIES = {'HealthBurden'}

    # declare metadata
    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_HEALTHBURDEN,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_SYMPTOMMANAGER
    }

    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        "Measles":
            Cause(gbd_causes={'Measles'},
                  label='Measles')
    }

    # Declare Causes of Disability
    CAUSES_OF_DISABILITY = {
        "Measles":
            Cause(gbd_causes={'Measles'},
                  label='Measles')
    }

    PARAMETERS = {
        "beta_baseline": Parameter(
            Types.REAL, "Baseline measles transmission probability"),
        "beta_scale": Parameter(
            Types.REAL, "Scale value for measles transmission probability sinusoidal function"),
        "phase_shift": Parameter(
            Types.REAL, "Phase shift for measles transmission probability sinusoidal function"),
        "period": Parameter(
            Types.REAL, "Period for measles transmission probability sinusoidal function"),
        "vaccine_efficacy_1": Parameter(
            Types.REAL, "Efficacy of first measles vaccine dose against measles infection"),
        "vaccine_efficacy_2": Parameter(
            Types.REAL, "Efficacy of second measles vaccine dose against measles infection"),
        "symptom_prob": Parameter(
            Types.DATA_FRAME, "Probability of each symptom with measles infection"),
        "odds_ratio_health_seeking_in_children_rash": Parameter(
            Types.REAL, "Odds ratio seeking care in children rash"),
        "odds_ratio_health_seeking_in_adults_rash": Parameter(
            Types.REAL, "Odds ratio seeking care in adults rash"),
        "odds_ratio_health_seeking_in_children_otitis_media": Parameter(
            Types.REAL, "Odds ratio seeking care in children otitis media"),
        "odds_ratio_health_seeking_in_adults_otitis_media": Parameter(
            Types.REAL, "Odds ratio seeking care in adults otitis media"),
        "reduction_in_death_risk_measle_treated": Parameter(
            Types.REAL, "Reduction in death risk when measles are treated"
        ),
        "maternal_immunity_age_threshold": Parameter(
            Types.REAL, "Age threshold below which children are protected by maternal immunity"
        ),
        "age_min_symptoms_constant_rate": Parameter(
            Types.INT, "Maximum age for measles symptom probability lookup"
        ),
        "death_timing_min_days": Parameter(
            Types.INT, "Minimum days from symptom onset to death"
        ),
        "death_timing_max_days": Parameter(
            Types.INT, "Maximum days from symptom onset to death"
        ),
        "natural_resolution_min_days": Parameter(
            Types.INT, "Minimum days from symptom onset to natural resolution"
        ),
        "natural_resolution_max_days": Parameter(
            Types.INT, "Maximum days from symptom onset to natural resolution"
        ),
        "symptom_onset_min_days": Parameter(
            Types.INT, "Minimum days from infection to symptom onset (incubation period)"
        ),
        "symptom_onset_max_days": Parameter(
            Types.INT, "Maximum days from infection to symptom onset (incubation period)"
        ),
        "sympt_resolution_after_treatment": Parameter(
            Types.INT, "Days until symptom resolution after treatment"
        ),
        "main_polling_event_frequency_months": Parameter(
            Types.INT, "Measles main polling event frequency months"
        ),
        "case_fatality_rate": Parameter(
            Types.DICT, "Case fatality rate for measles infection"),
    }

    PROPERTIES = {
        "me_has_measles": Property(Types.BOOL, "Measles infection status"),
        "me_date_measles": Property(Types.DATE, "Date of onset of measles"),
        "me_on_treatment": Property(Types.BOOL, "Currently on treatment for measles complications"),
    }

    def __init__(self, name=None):

        super().__init__(name)

        # Declare the symptoms that this module will use:
        self.symptoms = {
            'rash',
            'fever',
            'diarrhoea',
            'encephalitis',
            'otitis_media',
            'respiratory_symptoms',  # pneumonia
            'eye_complaint'
        }
        self.symptom_probs = None  # (will store the probabilities of symptom onset by age)

        self.consumables = None  # (will store item_codes for consumables used in HSI)

    def read_parameters(self, resourcefilepath: Optional[Path] = None):
        """Read parameter values from file
        """
        workbook = read_csv_files(resourcefilepath/'ResourceFile_Measles', files=None)
        self.load_parameters_from_dataframe(workbook["parameters"])

        self.parameters["symptom_prob"] = workbook["symptoms"]
        self.declare_parameter_metadata(parameter_name="symptom_prob", param_label="local")

        self.parameters["case_fatality_rate"] = workbook["cfr"].set_index('age')["probability"].to_dict()
        self.declare_parameter_metadata(parameter_name="case_fatality_rate", param_label="universal")

        # moderate symptoms all mapped to moderate_measles, pneumonia/encephalitis mapped to severe_measles

        p = self.parameters
        if "HealthBurden" in self.sim.modules.keys():
            self.parameters["daly_wts"] = {
                "rash": self.sim.modules["HealthBurden"].get_daly_weight(sequlae_code=205),
                "fever": self.sim.modules["HealthBurden"].get_daly_weight(sequlae_code=205),
                "diarrhoea": self.sim.modules["HealthBurden"].get_daly_weight(sequlae_code=205),
                "encephalitis": self.sim.modules["HealthBurden"].get_daly_weight(sequlae_code=206),
                "otitis_media": self.sim.modules["HealthBurden"].get_daly_weight(sequlae_code=205),
                "respiratory_symptoms": self.sim.modules["HealthBurden"].get_daly_weight(sequlae_code=206),
                "eye_complaint": self.sim.modules["HealthBurden"].get_daly_weight(sequlae_code=205),
            }

        # Declare symptoms that this module will cause and which are not included in the generic symptoms:
        self.sim.modules['SymptomManager'].register_symptom(
            Symptom(name='rash',
                    odds_ratio_health_seeking_in_children=p['odds_ratio_health_seeking_in_children_rash'],
                    odds_ratio_health_seeking_in_adults=p['odds_ratio_health_seeking_in_adults_rash'])
        )

        self.sim.modules['SymptomManager'].register_symptom(
            Symptom(name='otitis_media',
                    odds_ratio_health_seeking_in_children=p['odds_ratio_health_seeking_in_children_otitis_media'],
                    odds_ratio_health_seeking_in_adults=p['odds_ratio_health_seeking_in_adults_otitis_media'])
        )

        self.sim.modules['SymptomManager'].register_symptom(Symptom.emergency('encephalitis'))

    def pre_initialise_population(self):
        self.process_parameters()

    def initialise_population(self, population):
        """Set our property values for the initial population.
        set whole population to measles-free for 1st jan
        """
        df = population.props

        df.loc[df.is_alive, "me_has_measles"] = False  # default: no individuals infected
        df.loc[df.is_alive, "me_date_measles"] = pd.NaT
        df.loc[df.is_alive, "me_on_treatment"] = False

    def initialise_simulation(self, sim):
        """Schedule measles event to start straight away. Each month it will assign new infections"""

        sim.schedule_event(MeaslesEvent(self), sim.date)
        sim.schedule_event(MeaslesLoggingEvent(self), sim.date)
        sim.schedule_event(MeaslesLoggingFortnightEvent(self), sim.date)
        sim.schedule_event(MeaslesLoggingAnnualEvent(self), sim.date)

        # Look-up item_codes for the consumables used in the associated HSI
        self.consumables = {
            'vit_A':
                self.sim.modules['HealthSystem'].get_item_code_from_item_name("Vitamin A, caplet, 100,000 IU"),
            'severe_diarrhoea':
                self.sim.modules['HealthSystem'].get_item_code_from_item_name("ORS, sachet"),
            'severe_pneumonia':
                self.sim.modules['HealthSystem'].get_item_code_from_item_name("Oxygen, 1000 liters, primarily with "
                                                                              "oxygen cylinders")
        }


    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual
        assume all newborns are uninfected

        :param mother_id: the ID for the mother for this child
        :param child_id: the ID for the new child
        """
        df = self.sim.population.props  # shortcut to the population props dataframe
        df.at[child_id, "me_has_measles"] = False
        df.at[child_id, "me_date_measles"] = pd.NaT
        df.at[child_id, "me_on_treatment"] = False

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        df = self.sim.population.props
        health_values = pd.Series(index=df.index[df.is_alive], data=0.0)

        for symptom, daly_wt in self.parameters["daly_wts"].items():
            health_values.loc[
                self.sim.modules["SymptomManager"].who_has(symptom)] += daly_wt

        return health_values

    def report_prevalence(self):
        # This reports age- and sex-specific prevalence of measles for all individuals
        df = self.sim.population.props
        alive_df = df[df['is_alive']]

        measles_df = alive_df[alive_df['me_has_measles']]
        prevalence_counts = (
            measles_df.groupby(['age_range', 'sex']).size().unstack(fill_value=0)
        )
        prevalence_by_age_group_sex = (prevalence_counts / len(alive_df)).to_dict(orient='index')

        return flatten_nested_dict(prevalence_by_age_group_sex)

    def process_parameters(self):
        """Process the parameters (following being read-in) prior to the simulation starting.
        Make `self.symptom_probs` to be a dictionary keyed by age, with values of dictionaries keyed by symptoms and
        the probability of symptom onset."""
        p = self.parameters
        probs = self.parameters["symptom_prob"].set_index(["age", "symptom"])["probability"]
        self.symptom_probs = {level: probs.loc[(level, slice(None))].to_dict() for level in probs.index.levels[0]}

        # Check that a sensible value for a probability of symptom onset is declared for each symptom and for each age
        # up to and including age  for which symptom probabilities stabilizes (age >= 'age_min_symptoms_constant_rate')
        for _age in range(p["age_min_symptoms_constant_rate"] + 1):
            assert set(self.symptoms) == set(self.symptom_probs.get(_age).keys())
            assert all([0.0 <= x <= 1.0 for x in self.symptom_probs.get(_age).values()])

    def do_at_generic_first_appt(
        self,
        person_id: int,
        symptoms: List[str],
        schedule_hsi_event: HSIEventScheduler,
        **kwargs,
    ) -> None:
        if "rash" in symptoms:
            event = HSI_Measles_Treatment(person_id=person_id, module=self)
            schedule_hsi_event(event, priority=0, topen=self.sim.date)


class MeaslesEvent(RegularEvent, PopulationScopeEventMixin):
    """ MeaslesEvent runs every month and creates a number of new infections which are scattered across the month.
    * Seasonality is captured by the risk of infection changing according to the month.
    * Vaccination lowers an individual's likelihood of getting the infection (one dose will be 85% protective and two
      doses will be 99% protective).
    """

    def __init__(self, module):
        p = module.parameters
        super().__init__(module, frequency=DateOffset(months=p['main_polling_event_frequency_months']))
        assert isinstance(module, Measles)

    def apply(self, population):
        df = population.props
        p = self.module.parameters
        rng = self.module.rng

        month = self.sim.date.month  # integer month

        # transmission probability follows a sinusoidal function with peak in May
        # value is per person per month
        trans_prob = p["beta_baseline"] * (1 + p["beta_scale"] *
                                           math.cos((2 * math.pi * (month - p["phase_shift"])) / p["period"]))

        # get individual levels of protection due to vaccine
        protected_by_vaccine = pd.Series(1, index=df.index)  # all fully susceptible

        if "Epi" in self.sim.modules:
            protected_by_vaccine.loc[~df.is_alive] = 0  # not susceptible
            protected_by_vaccine.loc[(df.va_measles == 1)] *= (1 - p["vaccine_efficacy_1"])  # partially susceptible
            protected_by_vaccine.loc[(df.va_measles > 1)] *= (1 - p["vaccine_efficacy_2"])  # partially susceptible

        # Find persons to be newly infected (no risk to children under maternal immunity threshold )
        new_inf = df.index[~df.me_has_measles & (df.age_exact_years >= p["maternal_immunity_age_threshold"]) &
                           (rng.random_sample(size=len(df)) < (trans_prob * protected_by_vaccine))]

        logger.debug(key="MeaslesEvent",
                     data=f"Measles Event: new infections scheduled for {new_inf}")

        # if any are new cases
        if new_inf.any():
            # schedule the infections
            for person_index in new_inf:
                self.sim.schedule_event(
                    MeaslesOnsetEvent(self.module, person_index),
                    random_date(start=self.sim.date,
                      end=self.sim.date + pd.DateOffset(months=p['main_polling_event_frequency_months']), rng=rng)
                )


class MeaslesOnsetEvent(Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        """Onset a case of Measles"""

        df = self.sim.population.props  # shortcut to the dataframe
        rng = self.module.rng
        p = self.module.parameters

        if not df.at[person_id, "is_alive"]:
            return

        ref_age = df.at[person_id, "age_years"].clip(max=p["age_min_symptoms_constant_rate"])

        # Determine if the person has "untreated HIV", which is defined as a person in any stage of HIV but not on
        # successful treatment currently.

        logger.debug(key="MeaslesOnsetEvent",
                     data=f"MeaslesOnsetEvent: new infections scheduled for {person_id}")

        df.at[person_id, "me_has_measles"] = True
        df.at[person_id, "me_date_measles"] = self.sim.date
        df.at[person_id, "me_on_treatment"] = False

        symp_onset = self.assign_symptoms(ref_age)  # Assign symptoms for this person
        prob_death = self.get_prob_death(ref_age)  # Look-up the probability of death for this person

        # Schedule either the DeathEvent of the SymptomResolution event, depending on the expected outcome of this case
        if rng.random_sample() < prob_death:
            logger.debug(key="MeaslesOnsetEvent",
                         data=f"This is MeaslesOnsetEvent, scheduling measles death for {person_id}")

            # make that death event
            death_event = MeaslesDeathEvent(self.module, person_id=person_id)

            # schedule the death
            self.sim.schedule_event(
                death_event, symp_onset +
                             DateOffset(days=rng.randint(p["death_timing_min_days"], p["death_timing_max_days"])))

        else:
            # schedule symptom resolution without treatment - this only occurs if death doesn't happen first
            symp_resolve = (symp_onset +
                            DateOffset(days=rng.randint(p["natural_resolution_min_days"],
                                                        p["natural_resolution_max_days"])))
            self.sim.schedule_event(MeaslesSymptomResolveEvent(self.module, person_id), symp_resolve)

    def assign_symptoms(self, _age):
        """Assign symptoms for this case and returns the date on which symptom onset.
        (Parameter values specify that everybody gets rash, fever and eye complain.)"""

        rng = self.module.rng
        p = self.module.parameters
        person_id = self.target
        symptom_probs_for_this_person = self.module.symptom_probs.get(_age)
        date_of_symp_onset = self.sim.date + DateOffset(days=rng.randint(p["symptom_onset_min_days"],
                                                                         p["symptom_onset_max_days"]))

        symptoms_to_onset = [
            _symp for (_symp, _prob), _rand in zip(
                symptom_probs_for_this_person.items(), rng.random_sample(len(symptom_probs_for_this_person))
            ) if _rand < _prob
        ]

        # schedule symptoms onset
        self.sim.modules["SymptomManager"].change_symptom(
            person_id=person_id,
            symptom_string=symptoms_to_onset,
            add_or_remove="+",
            disease_module=self.module,
            date_of_onset=date_of_symp_onset,
        )

        return date_of_symp_onset

    def get_prob_death(self, _age):
        """Returns the probability of death for this person based on their age."""
        p = self.module.parameters
        return p["case_fatality_rate"].get(_age)


class MeaslesSymptomResolveEvent(Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        """ this event is called by MeaslesOnsetEvent and HSI_Measles_Treatment
        """

        df = self.sim.population.props  # shortcut to the dataframe

        logger.debug(key="MeaslesSymptomResolve Event",
                     data=f"MeaslesSymptomResolveEvent: symptoms resolved for {person_id}")

        # check if person still alive, has measles (therefore has symptoms)
        if df.at[person_id, "is_alive"] and df.at[person_id, "me_has_measles"]:
            # clear symptoms
            self.sim.modules["SymptomManager"].clear_symptoms(
                person_id=person_id, disease_module=self.module)

            # change measles status
            df.at[person_id, "me_has_measles"] = False

            # change treatment status if needed
            if df.at[person_id, "me_on_treatment"]:
                df.at[person_id, "me_on_treatment"] = False


class MeaslesDeathEvent(Event, IndividualScopeEventMixin):
    """
    Performs the Death operation on an individual and logs it.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props
        p = self.module.parameters

        if not df.at[person_id, "is_alive"]:
            return

        # reduction in risk of death if being treated for measles complications
        # check still infected (symptoms not resolved)
        if df.at[person_id, "me_has_measles"]:

            if df.at[person_id, "me_on_treatment"]:

                # Certain death (100%) is reduced by specified amount
                p_death_with_treatment = 1. - p['reduction_in_death_risk_measle_treated']

                # If below that probability, death goes ahead
                if self.module.rng.random_sample() < p_death_with_treatment:
                    logger.debug(key="MeaslesDeathEvent",
                                 data=f"MeaslesDeathEvent: scheduling death for treated {person_id} on {self.sim.date}")

                    self.sim.modules['Demography'].do_death(individual_id=person_id,
                                                            cause="Measles",
                                                            originating_module=self.module)

            else:
                logger.debug(key="MeaslesDeathEvent",
                             data=f"MeaslesDeathEvent: scheduling death for untreated {person_id} on {self.sim.date}")

                self.sim.modules['Demography'].do_death(individual_id=person_id,
                                                        cause="Measles",
                                                        originating_module=self.module)


# ---------------------------------------------------------------------------------
# Health System Interaction Events
# ---------------------------------------------------------------------------------


class HSI_Measles_Treatment(HSI_Event, IndividualScopeEventMixin):
    """
    Health System Interaction Event
    It is the event when a person with diagnosed measles receives treatment
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Measles)

        self.TREATMENT_ID = "Measles_Treatment"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({
            ('Under5OPD' if self.sim.population.props.at[person_id, "age_years"] < 5 else 'Over5OPD'): 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'

    def apply(self, person_id, squeeze_factor):
        logger.debug(key="HSI_Measles_Treatment",
                     data=f"HSI_Measles_Treatment: treat person {person_id} for measles")

        df = self.sim.population.props
        symptoms = self.sim.modules["SymptomManager"].has_what(person_id=person_id)
        p = self.module.parameters

        # for non-complicated measles
        item_codes = [self.module.consumables['vit_A']]

        # for measles with severe diarrhoea
        if "diarrhoea" in symptoms:
            item_codes.append(self.module.consumables['severe_diarrhoea'])

        # for measles with pneumonia
        if "respiratory_symptoms" in symptoms:
            item_codes.append(self.module.consumables['severe_pneumonia'])

        # request the treatment
        if self.get_consumables(item_codes):
            logger.debug(key="HSI_Measles_Treatment",
                         data=f"HSI_Measles_Treatment: giving required measles treatment to person {person_id}")

            if "respiratory_symptoms" in symptoms:
                self.add_equipment({'Oxygen concentrator', 'Oxygen cylinder, with regulator'})

            # modify person property which is checked when scheduled death occurs (or shouldn't occur)
            df.at[person_id, "me_on_treatment"] = True

            # schedule symptom resolution following treatment
            self.sim.schedule_event(MeaslesSymptomResolveEvent(self.module, person_id),
                                    self.sim.date + DateOffset(days=p["sympt_resolution_after_treatment"]))

    def did_not_run(self):
        logger.debug(key="HSI_Measles_Treatment",
                     data="HSI_Measles_Treatment: did not run"
                     )
        pass


# ---------------------------------------------------------------------------------
# Logging Events
# ---------------------------------------------------------------------------------

class MeaslesLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        self.repeat = 1
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        """Log Summary Statistics Monthly"""

        df = population.props
        now = self.sim.date

        # ------------------------------------ INCIDENCE ------------------------------------

        # infected in the last time-step
        # incidence rate per 1000 people per month
        # include those cases that have died in the case load
        tmp = len(
            df.loc[(df.me_date_measles > (now - DateOffset(months=self.repeat)))]
        )
        pop = len(df[df.is_alive])

        inc_1000people = (tmp / pop) * 1000

        incidence_summary = {
            "number_new_cases": tmp,
            "population": pop,
            "inc_1000people": inc_1000people,
        }

        logger.info(key="incidence",
                    data=incidence_summary,
                    description="summary of measles incidence per 1000 people per month")


class MeaslesLoggingFortnightEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        self.repeat = 2
        super().__init__(module, frequency=DateOffset(weeks=self.repeat))

    def apply(self, population):
        """Log Summary Statistics Every Two Weeks"""

        df = population.props

        # ------------------------------------ SYMPTOMS ------------------------------------
        # this will check for all measles cases in the past two weeks (average symptom duration)
        # and look at current symptoms
        # so if symptoms have resolved they won't be included

        symptom_list = self.module.symptoms
        symptom_output = dict()
        symptom_output['Key'] = symptom_list

        # currently infected and has rash (every case)
        tmp = len(
            df.index[df.is_alive & df.me_has_measles & (df.sy_rash > 0)]
        )

        # get distribution of all symptoms
        # only measles running currently, no other causes of symptoms
        # if they have died, who_has does not count them
        for symptom in symptom_list:
            # sum who has each symptom
            number_with_symptom = len(self.sim.modules['SymptomManager'].who_has(symptom))
            if tmp:
                proportion_with_symptom = number_with_symptom / tmp
            else:
                proportion_with_symptom = 0.0
            symptom_output[symptom] = proportion_with_symptom

        logger.info(key="measles_symptoms",
                    data=symptom_output,
                    description="summary of measles symptoms each month")


class MeaslesLoggingAnnualEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        self.repeat = 1
        super().__init__(module, frequency=DateOffset(years=self.repeat))

    def apply(self, population):
        """Log Summary Statistics Annually"""

        df = population.props
        now = self.sim.date

        # get annual distribution of cases by age-range

        # ------------------------------------ ANNUAL INCIDENCE ------------------------------------

        # infected in the last time-step

        age_count = df[df.is_alive].groupby('age_range').size()

        logger.info(key='pop_age_range', data=age_count.to_dict(), description="population sizes by age range")

        # get the numbers infected by age group
        infected_age_counts = df[(df.me_date_measles > (now - DateOffset(months=self.repeat)))].groupby(
            'age_range').size()
        total_infected = len(
            df.loc[(df.me_date_measles > (now - DateOffset(months=self.repeat)))]
        )
        if total_infected:
            prop_infected_by_age = infected_age_counts / total_infected
        else:
            prop_infected_by_age = infected_age_counts.astype("float")  # just output the series of zeros by age group

        logger.info(key='measles_incidence_age_range', data=prop_infected_by_age.to_dict(),
                    description="measles incidence by age group")
