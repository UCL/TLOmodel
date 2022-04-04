import math
import os

import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import Metadata
from tlo.methods.causes import Cause
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.symptommanager import Symptom

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Measles(Module):
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
        "prob_severe": Parameter(
            Types.REAL, "Probability of severe measles infection, requiring hospitalisation"),
        "risk_death_on_treatment": Parameter(
            Types.REAL, "Risk of scheduled death occurring if on treatment for measles complications"),
        "symptom_prob": Parameter(
            Types.DATA_FRAME, "Probability of each symptom with measles infection"),
    }

    PROPERTIES = {
        "me_has_measles": Property(Types.BOOL, "Measles infection status"),
        "me_date_measles": Property(Types.DATE, "Date of onset of measles"),
        "me_on_treatment": Property(Types.BOOL, "Currently on treatment for measles complications"),
    }

    # Declaration of the specific symptoms that this module will use
    SYMPTOMS = {
        'rash',  # moderate symptoms, will trigger healthcare seeking in community/district facility
        'encephalitis',
        'otitis_media'
    }

    def __init__(self, name=None, resourcefilepath=None):

        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # Store the symptoms that this module will use:
        self.symptoms = {
            'rash',
            'fever',
            'diarrhoea',
            'encephalitis',
            'otitis_media',
            'respiratory_symptoms',  # pneumonia
            'eye_complaint'
        }

        self.consumables = None  # (will store item_codes for consumables used in HSI)

    def read_parameters(self, data_folder):
        """Read parameter values from file
        """

        workbook = pd.read_excel(
            os.path.join(self.resourcefilepath, "ResourceFile_Measles.xlsx"),
            sheet_name=None,
        )
        self.load_parameters_from_dataframe(workbook["parameters"])

        self.parameters["symptom_prob"] = workbook["symptoms"]

        # moderate symptoms all mapped to moderate_measles, pneumonia/encephalitis mapped to severe_measles
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
                    odds_ratio_health_seeking_in_children=2.5,
                    odds_ratio_health_seeking_in_adults=2.5)  # non-emergencies
        )

        self.sim.modules['SymptomManager'].register_symptom(
            Symptom(name='otitis_media',
                    odds_ratio_health_seeking_in_children=2.5,
                    odds_ratio_health_seeking_in_adults=2.5)  # non-emergencies
        )

        self.sim.modules['SymptomManager'].register_symptom(
            Symptom(
                name='encephalitis',
                emergency_in_adults=True,
                emergency_in_children=True
            )
        )

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
                self.sim.modules['HealthSystem'].get_item_code_from_item_name("Oxygen, 1000 liters, primarily with oxygen cylinders")
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

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """
        pass

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


class MeaslesEvent(RegularEvent, PopulationScopeEventMixin):
    """ MeaslesEvent runs every year and creates a number of new infections which are scattered across the year
    seasonality is captured using a cosine function
    vaccination lowers an individual's likelihood of getting the disease
    assume one dose will be 85% protective and 2 doses will be 99% protective
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        assert isinstance(module, Measles)

    def apply(self, population):
        df = population.props
        p = self.module.parameters
        rng = self.module.rng

        month = self.sim.date.month  # integer month

        # transmission probability follows a sinusoidal function with peak in May
        # value is per person per month
        trans_prob = p["beta_baseline"] * (1 + p["beta_scale"] *
                                           math.cos((2 + math.pi * (month - p["phase_shift"])) / p["period"]))

        # children under 6 months protected by maternal immunity
        # get individual levels of protection due to vaccine
        protected_by_vaccine = pd.Series(1, index=df.index)  # all fully susceptible

        if "Epi" in self.sim.modules:
            protected_by_vaccine.loc[~df.is_alive] = 0  # not susceptible
            protected_by_vaccine.loc[(df.va_measles == 1)] *= (1 - p["vaccine_efficacy_1"])  # partially susceptible
            protected_by_vaccine.loc[(df.va_measles > 1)] *= (1 - p["vaccine_efficacy_2"])  # partially susceptible

        new_inf = df.index[~df.me_has_measles & (df.age_exact_years >= 0.5) &
                           (rng.random_sample(size=len(df)) < (trans_prob * protected_by_vaccine))]

        logger.debug(key="MeaslesEvent",
                     data=f"Measles Event: new infections scheduled for {new_inf}")

        # if any are new cases
        if new_inf.any():
            # schedule the infections
            for person_index in new_inf:
                self.sim.schedule_event(
                    MeaslesOnsetEvent(self.module, person_index),
                    self.sim.date + DateOffset(rng.random_integers(low=0, high=28, size=1))
                )


class MeaslesOnsetEvent(Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):

        df = self.sim.population.props  # shortcut to the dataframe
        p = self.module.parameters
        rng = self.module.rng

        if not df.at[person_id, "is_alive"]:
            return

        symptom_prob = p["symptom_prob"]

        logger.debug(key="MeaslesOnsetEvent",
                     data=f"MeaslesOnsetEvent: new infections scheduled for {person_id}")

        df.at[person_id, "me_has_measles"] = True
        df.at[person_id, "me_date_measles"] = self.sim.date

        # assign symptoms
        symptom_list = sorted(self.module.symptoms)

        ref_age = df.at[person_id, "age_years"]
        # age limit for symptom data is 30 years
        if ref_age > 30:
            ref_age = 30

        # read probabilities of symptoms by age
        symptom_prob = symptom_prob.loc[symptom_prob.age == ref_age, ["symptom", "probability"]]
        # map to a series of probabilities indexed by symptom key
        symptom_prob = symptom_prob.set_index("symptom").probability

        # symptom onset
        symp_onset = self.sim.date + DateOffset(days=rng.randint(7, 21))

        # everybody gets rash, fever and eye complaint, other symptoms assigned with age-specific probability

        # random sample which symptoms a person will have
        active_symptoms = [sym for sym in symptom_list if rng.uniform() < symptom_prob[sym]]
        # schedule symptoms onset
        self.sim.modules["SymptomManager"].change_symptom(
            person_id=person_id,
            symptom_string=active_symptoms,
            add_or_remove="+",
            disease_module=self.module,
            date_of_onset=symp_onset,
            duration_in_days=14,  # same duration for all symptoms
        )

        # schedule symptom resolution without treatment - this only occurs if death doesn't happen first
        symp_resolve = symp_onset + DateOffset(days=rng.randint(7, 14))
        self.sim.schedule_event(MeaslesSymptomResolveEvent(self.module, person_id), symp_resolve)

        # todo separate probability for people with HIV

        # probability of death
        if rng.uniform() < symptom_prob["death"]:
            logger.debug(key="MeaslesOnsetEvent",
                         data=f"This is MeaslesOnsetEvent, scheduling measles death for {person_id}")

            # make that death event
            death_event = MeaslesDeathEvent(self.module, person_id=person_id)

            # schedule the death
            self.sim.schedule_event(
                death_event, symp_onset + DateOffset(days=rng.randint(3, 7)))


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
        if df.at[person_id, "is_alive"] & df.at[person_id, "me_has_measles"]:
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

        if not df.at[person_id, "is_alive"]:
            return

        # reduction in risk of death if being treated for measles complications
        # check still infected (symptoms not resolved)
        if df.at[person_id, "me_has_measles"]:

            if df.at[person_id, "me_on_treatment"]:
                reduction_in_death_risk = self.module.rng.uniform(low=0.2, high=0.6, size=1)

                if self.module.rng.rand() < reduction_in_death_risk:
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

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["Over5OPD"] = 1  # This requires one out patient appt

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Measles_Treatment"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(key="HSI_Measles_Treatment",
                     data=f"HSI_Measles_Treatment: treat person {person_id} for measles")

        df = self.sim.population.props
        symptoms = self.sim.modules["SymptomManager"].has_what(person_id)

        # for non-complicated measles
        item_codes = [self.module.consumables['vit_A']]

        # for measles with severe diarrhoea
        if "diarrhoea" in symptoms:
            item_codes += self.module.consumables['severe_diarrhoea']

        # for measles with pneumonia
        if "respiratory_symptoms" in symptoms:
            item_codes += self.module.consumables['severe_pneumonia']

        # request the treatment
        if self.get_consumables(item_codes):
            logger.debug(key="HSI_Measles_Treatment",
                         data=f"HSI_Measles_Treatment: giving required measles treatment to person {person_id}")

            # modify person property which is checked when scheduled death occurs (or shouldn't occur)
            df.at[person_id, "me_on_treatment"] = True

            # schedule symptom resolution following treatment: assume perfect treatment
            # also changes treatment status back to False
            self.sim.schedule_event(MeaslesSymptomResolveEvent(self.module, person_id),
                                    self.sim.date + DateOffset(days=7))

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
        # get some summary statistics
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
                proportion_with_symptom = 0
            symptom_output[symptom] = proportion_with_symptom

        logger.info(key="measles_symptoms",
                    data=symptom_output,
                    description="summary of measles symptoms each month")


class MeaslesLoggingAnnualEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        self.repeat = 1
        super().__init__(module, frequency=DateOffset(years=self.repeat))

    def apply(self, population):
        # get some summary statistics
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
            prop_infected_by_age = infected_age_counts  # just output the series of zeros by age group

        logger.info(key='measles_incidence_age_range', data=prop_infected_by_age.to_dict(),
                    description="measles incidence by age group")
