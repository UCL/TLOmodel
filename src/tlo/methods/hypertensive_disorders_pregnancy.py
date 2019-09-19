import logging

import numpy as np
import pandas as pd
from pathlib import Path

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods.demography import InstantaneousDeath

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HypertensiveDisordersOfPregnancy(Module):
    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
    """
   This module manages the hypertensive disorders of pregnancy including pre-eclampsia/eclampsia, gestational 
   hypertension and superimposed pre-eclampsia/eclampsia. Chronic hypertension in pregnant women is managed by the 
   hypertension module
    """

    PARAMETERS = {

        'base_prev_pe': Parameter(
            Types.REAL, 'prevalence of pre-eclampsia in women who are already pregnant at baseline'),
        'prob_pre_eclamp': Parameter(
            Types.REAL, 'baseline probability of pre-eclampsia during pregnancy'),
        'rr_pre_eclamp_nulip': Parameter(
            Types.REAL, 'relative risk of pre-eclampsia in nuliparous women'),
        'rr_pre_eclamp_chron_htn': Parameter(
            Types.REAL, 'relative risk of pre-eclampsia in women with chronic hypertension'),
        'rr_pre_eclamp_prev_pe': Parameter(
            Types.REAL, 'relative risk of pre- eclampsia in women who have previous suffered from pre-eclampsia'),
        'rr_pre_eclamp_multip_preg': Parameter(
            Types.REAL, 'relative risk of pre-eclampsia in women who are pregnant with twins'),
        'rr_pre_eclamp_diabetes': Parameter(
            Types.REAL, 'relative risk of pre-eclampsia in women with diabetes'),
        'rr_pre_eclamp_high_bmi': Parameter(
            Types.REAL, 'relative risk of pre-eclampsia in women with a high BMI prior to pregnancy'),
        'base_prev_gest_htn': Parameter(
            Types.REAL, 'prevalence of gestational hypertension in women who are already pregnant at baseline'),
        'prob_gest_htn':Parameter(
            Types.REAL, 'probability of a pregnant woman developing gestational hypertension during pregnancy')

    }

    PROPERTIES = {
        'hp_htn_preg': Property(Types.CATEGORICAL, 'None, Chronic Hypertension, Gestational Hypertension, Pre-eclampsia,'
                                                   'Superimposed Pre-eclampsia',
                                                   categories=['none', 'chronic_htn', 'gest_htn', 'pre_eclampsia',
                                                               'sup_pre_eclamp']),
        'hp_pe_specific_symptoms': Property(
            Types.CATEGORICAL, 'Level of symptoms for pre-eclampsia',
            categories=['none', 'headache normal vision', 'headache visual disturbance', 'neurological disturbance']),
        # TODO: review in light of new symptom tracker
        'hp_pe_unified_symptom_code': Property(
            Types.CATEGORICAL,
            'Level of symptoms on the standardised scale (governing health-care seeking): '
            '0=None; 1=Mild; 2=Moderate; 3=Severe; 4=Extreme_Emergency',
            categories=[0, 1, 2, 3, 4]),
        'hp_prev_pre_eclamp': Property(Types.BOOL, 'Whether this woman has suffered from pre-eclampsia in a previous'
                                                   ' pregnancy'),
        'hp_anti_htns': Property(Types.BOOL, 'Whether this woman is currently on anti-hypertensive medication for high '
                                             'blood pressure')

    }

    def read_parameters(self, data_folder):
        params = self.parameters

        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_HypertensiveDisordersOfPregnancy.xlsx',
                            sheet_name='parameter_values')

        dfd.set_index('parameter_name', inplace=True)

        params['base_prev_pe'] = dfd.loc['base_prev_pe', 'value']
        params['levels_pe_symptoms'] = pd.DataFrame(
            data={
                'level_of_symptoms': ['none',
                                      'headache normal vision',
                                      'headache visual disturbance',
                                      'neurological disturbance'],
                'probability': [0.25, 0.25, 0.25, 0.25]})
        params['prob_pre_eclamp'] = dfd.loc['prob_pre_eclamp', 'value']
        params['rr_pre_eclamp_nulip'] = dfd.loc['rr_pre_eclamp_nulip', 'value']
        params['rr_pre_eclamp_chron_htn'] = dfd.loc['rr_pre_eclamp_chron_htn', 'value']
        params['rr_pre_eclamp_prev_pe'] = dfd.loc['rr_pre_eclamp_prev_pe', 'value']
        params['rr_pre_eclamp_multip_preg'] = dfd.loc['rr_pre_eclamp_multip_preg', 'value']
        params['rr_pre_eclamp_diabetes'] = dfd.loc['rr_pre_eclamp_diabetes', 'value']
        params['rr_pre_eclamp_high_bmi'] = dfd.loc['rr_pre_eclamp_high_bmi', 'value']
        params['base_prev_gest_htn'] = dfd.loc['base_prev_gest_htn', 'value']
        params['prob_gest_htn'] = dfd.loc['prob_gest_htn', 'value']

        # get the DALY weight that this module will use from the weight database (these codes are just random!)
#        if 'HealthBurden' in self.sim.modules.keys():
#            p['daly_wt_mild_sneezing'] = self.sim.modules['HealthBurden'].get_daly_weight(50)
#            p['daly_wt_coughing'] = self.sim.modules['HealthBurden'].get_daly_weight(50)
#            p['daly_wt_advanced'] = self.sim.modules['HealthBurden'].get_daly_weight(589)

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        df = population.props  # a shortcut to the dataframe storing data for individiuals
        params = self.parameters

        df.loc[df.sex == 'F', 'hp_htn_preg'] = 'none'
        df.loc[df.sex == 'F', 'hp_prev_pre_eclamp'] = False
        df.loc[df.sex == 'F', 'hp_anti_htns'] = False

        # CHRONIC HTN at baseline is covered by mikaela's module

        # ============================= PRE-ECLAMPSIA/SUPER IMPOSED PE (at baseline) ===================================

        # First we apply the baseline prevalence of pre-eclampsia to women who are pregnant at baseline
        preg_women = df.index[df.is_alive & df.is_pregnant & (df.sex == 'F')]  # TODO: limit to GA >20 weeks
        random_draw = pd.Series(self.sim.rng.random_sample(size=len(preg_women)), index=preg_women)

        eff_prob_pe = pd.Series(params['base_prev_pe'], index=preg_women)
        dfx = pd.concat((random_draw, eff_prob_pe), axis=1)
        dfx.columns = ['random_draw', 'eff_prob_pe']
        idx_pe = dfx.index[dfx.eff_prob_pe > dfx.random_draw]  # TODO: create 2 indexes (with and without hypertension)

        df.loc[idx_pe, 'hp_htn_preg'] = 'pre_eclampsia'
        df.loc[idx_pe, 'hp_prev_pre_eclamp'] = True

        # next we assign level of symptoms to women who have pre-eclampsia
        level_of_symptoms = params['levels_pe_symptoms']
        symptoms = self.rng.choice(level_of_symptoms.level_of_symptoms,
                                   size=len(idx_pe),
                                   p=level_of_symptoms.probability)
        df.loc[idx_pe, 'hp_pe_specific_symptoms'] = symptoms

        # ============================= GESTATIONAL HYPERTENSION (at baseline) ========================================

        preg_women_no_pe = df.index[df.is_alive & df.is_pregnant & (df.sex == 'F') & (df.hp_htn_preg =='none')]  # TODO: limit to GA >20 weeks
        random_draw = pd.Series(self.sim.rng.random_sample(size=len(preg_women_no_pe)), index=preg_women_no_pe)

        eff_prob_gh = pd.Series(params['base_prev_gest_htn'], index=preg_women_no_pe)
        dfx = pd.concat((random_draw, eff_prob_gh), axis=1)
        dfx.columns = ['random_draw', 'eff_prob_gh']
        idx_pe = dfx.index[dfx.eff_prob_gh > dfx.random_draw]
        df.loc[idx_pe, 'hp_htn_preg'] = 'gest_htn'

        # TODO: symptom free?

    def initialise_simulation(self, sim):

        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        # add the basic event
        event = CongenitalAnomalyEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=1))

        # add an event to log to screen
        sim.schedule_event(MockitisLoggingEvent(self), sim.date + DateOffset(months=6))



    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the ID for the mother for this child
        :param child_id: the ID for the new child
        """

        df = self.sim.population.props  # shortcut to the population props dataframe

        # Initialise all the properties that this module looks after:


    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug('This is Mockitis, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        logger.debug('This is mockitis reporting my health values')

        df = self.sim.population.props  # shortcut to population properties dataframe

        p = self.parameters


class CongenitalAnomalyEvent(RegularEvent, PopulationScopeEventMixin):
    """
    This event is occurrs once durin
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):

        logger.debug('This is MockitisEvent, tracking the disease progression of the population.')

        df = population.props



class MockitisDeathEvent(Event, IndividualScopeEventMixin):
    """
    This is the death event for mockitis
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe

        # Apply checks to ensure that this death should occur
        if df.at[person_id, 'mi_status'] == 'C':

            # Fire the centralised death event:
            death = InstantaneousDeath(self.module, person_id, cause='Mockitis')
            self.sim.schedule_event(death, self.sim.date)


# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# Health System Interaction Events

class HSI_Mockitis_PresentsForCareWithSevereSymptoms(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    It is first appointment that someone has when they present to the healthcare system with the severe
    symptoms of Mockitis.
    If they are aged over 15, then a decision is taken to start treatment at the next appointment.
    If they are younger than 15, then another initial appointment is scheduled for then are 15 years old.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Mockitis_PresentsForCareWithSevereSymptoms'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = self.sim.modules['HealthSystem'].get_blank_cons_footprint()
        self.ACCEPTED_FACILITY_LEVELS = [0]     # This enforces that the apppointment must be run at that facility-level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):

        logger.debug('This is HSI_Mockitis_PresentsForCareWithSevereSymptoms, a first appointment for person %d',
                     person_id)

        df = self.sim.population.props  # shortcut to the dataframe

        if df.at[person_id, 'age_years'] >= 15:
            logger.debug(
                '...This is HSI_Mockitis_PresentsForCareWithSevereSymptoms: \
                there should now be treatment for person %d',
                person_id)
            event = HSI_Mockitis_StartTreatment(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=2,
                                                                topen=self.sim.date,
                                                                tclose=None)

        else:
            logger.debug(
                '...This is HSI_Mockitis_PresentsForCareWithSevereSymptoms: there will not be treatment for person %d',
                person_id)

            date_turns_15 = self.sim.date + DateOffset(years=np.ceil(15 - df.at[person_id, 'age_exact_years']))
            event = HSI_Mockitis_PresentsForCareWithSevereSymptoms(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=2,
                                                                topen=date_turns_15,
                                                                tclose=date_turns_15 + DateOffset(months=12))


class HSI_Mockitis_StartTreatment(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.

    It is appointment at which treatment for mockitiis is inititaed.

    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient appt
        the_appt_footprint['NewAdult'] = 1  # Plus, an amount of resources similar to an HIV initiation

        # Get the consumables required
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(consumables.loc[consumables[
                                                  'Intervention_Pkg'] ==
                                              'First line treatment for new TB cases for adults',
                                              'Intervention_Pkg_Code'])[0]
        pkg_code2 = pd.unique(consumables.loc[consumables[
                                                  'Intervention_Pkg'] ==
                                              'MDR notification among previously treated patients',
                                              'Intervention_Pkg_Code'])[0]

        item_code1 = \
            pd.unique(consumables.loc[consumables['Items'] == 'Ketamine hydrochloride 50mg/ml, 10ml', 'Item_Code'])[0]
        item_code2 = pd.unique(consumables.loc[consumables['Items'] == 'Underpants', 'Item_Code'])[0]

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1, pkg_code2],
            'Item_Code': [item_code1, item_code2]
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Mockitis_Treatment_Initiation'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = [1, 2]  # Enforces that this apppointment must happen at those facility-levels
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        logger.debug('This is HSI_Mockitis_StartTreatment: initiating treatent for person %d', person_id)
        df = self.sim.population.props
        treatmentworks = self.module.rng.rand() < self.module.parameters['p_cure']

        if treatmentworks:
            df.at[person_id, 'mi_is_infected'] = False
            df.at[person_id, 'mi_status'] = 'P'

            # (in this we nullify the death event that has been scheduled.)
            df.at[person_id, 'mi_scheduled_date_death'] = pd.NaT

            df.at[person_id, 'mi_date_cure'] = self.sim.date
            df.at[person_id, 'mi_specific_symptoms'] = 'none'
            df.at[person_id, 'mi_unified_symptom_code'] = 0
            pass

        # Create a follow-up appointment
        target_date_for_followup_appt = self.sim.date + DateOffset(months=6)

        logger.debug(
            '....This is HSI_Mockitis_StartTreatment: scheduling a follow-up appointment for person %d on date %s',
            person_id, target_date_for_followup_appt)

        followup_appt = HSI_Mockitis_TreatmentMonitoring(self.module, person_id=person_id)

        # Request the heathsystem to have this follow-up appointment
        self.sim.modules['HealthSystem'].schedule_hsi_event(followup_appt,
                                                            priority=2,
                                                            topen=target_date_for_followup_appt,
                                                            tclose=target_date_for_followup_appt + DateOffset(weeks=2)
                                                            )


class HSI_Mockitis_TreatmentMonitoring(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.

    It is appointment at which treatment for mockitis is monitored.
    (In practise, nothing happens!)

    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient appt
        the_appt_footprint['NewAdult'] = 1  # Plus, an amount of resources similar to an HIV initiation

        # Get the consumables required
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(consumables.loc[consumables[
                                                  'Intervention_Pkg'] ==
                                              'First line treatment for new TB cases for adults',
                                              'Intervention_Pkg_Code'])[0]
        pkg_code2 = pd.unique(consumables.loc[consumables[
                                                  'Intervention_Pkg'] ==
                                              'MDR notification among previously treated patients',
                                              'Intervention_Pkg_Code'])[0]

        item_code1 = \
            pd.unique(consumables.loc[consumables['Items'] == 'Ketamine hydrochloride 50mg/ml, 10ml', 'Item_Code'])[0]
        item_code2 = pd.unique(consumables.loc[consumables['Items'] == 'Underpants', 'Item_Code'])[0]

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1, pkg_code2],
            'Item_Code': [item_code1, item_code2]
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Mockitis_TreatmentMonitoring'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = ['*']   # Allows this HSI to occur at any facility-level
        self.ALERT_OTHER_DISEASES = ['*']

    def apply(self, person_id):
        # There is a follow-up appoint happening now but it has no real effect!

        # Create the next follow-up appointment....
        target_date_for_followup_appt = self.sim.date + DateOffset(months=6)

        logger.debug(
            '....This is HSI_Mockitis_StartTreatment: scheduling a follow-up appointment for person %d on date %s',
            person_id, target_date_for_followup_appt)

        followup_appt = HSI_Mockitis_TreatmentMonitoring(self.module, person_id=person_id)

        # Request the heathsystem to have this follow-up appointment
        self.sim.modules['HealthSystem'].schedule_hsi_event(followup_appt,
                                                            priority=2,
                                                            topen=target_date_for_followup_appt,
                                                            tclose=target_date_for_followup_appt + DateOffset(weeks=2))


# ---------------------------------------------------------------------------------


class MockitisLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """Produce a summmary of the numbers of people with respect to their 'mockitis status'
        """
        # run this event every month
        self.repeat = 6
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props

        infected_total = df.loc[df.is_alive, 'mi_is_infected'].sum()
        proportion_infected = infected_total / len(df)

        mask: pd.Series = (df.loc[df.is_alive, 'mi_date_infected'] >
                           self.sim.date - DateOffset(months=self.repeat))
        infected_in_last_month = mask.sum()
        mask = (df.loc[df.is_alive, 'mi_date_cure'] > self.sim.date - DateOffset(months=self.repeat))
        cured_in_last_month = mask.sum()

        counts = {'N': 0, 'T1': 0, 'T2': 0, 'P': 0}
        counts.update(df.loc[df.is_alive, 'mi_status'].value_counts().to_dict())

        logger.info('%s|summary|%s', self.sim.date,
                    {
                        'TotalInf': infected_total,
                        'PropInf': proportion_infected,
                        'PrevMonth': infected_in_last_month,
                        'Cured': cured_in_last_month,
                    })

        logger.info('%s|status_counts|%s', self.sim.date, counts)
