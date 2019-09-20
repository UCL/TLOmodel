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
        'hp_pre_eclampsia': Property(Types.CATEGORICAL, 'None, Early Pre-eclampsia (<33 wks), '
                                                        'Late Pre-eclampsia (>33wks),',
                                                   categories=['none', 'early_pre_eclamp', 'late_pre_eclamp']),
        'hp_pre_eclamp_sev': Property(Types.CATEGORICAL, 'Severity of disease:None = N, Mild = M, Severe = S, '
                                                         'Eclampsia = E',
                                                   categories=['N', 'M', 'S', 'E']),
        'hp_gest_htn': Property(Types.BOOL, 'whether this woman has developed gestational hypertension'),
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
                'probability': [0.30, 0.40, 0.20, 0.10]})  # DUMMY
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

        df.loc[df.sex == 'F', 'hp_pre_eclampsia'] = 'none'
        df.loc[df.sex == 'F', 'hp_pre_eclamp_sev'] = 'N'
        df.loc[df.sex == 'F', 'hp_gest_htn'] = False
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
        # TODO: death schedule? or complication event...?

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

    def initialise_simulation(self, sim):

        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        # add the basic event
        event = HypertensiveDisordersOfPregnancyEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=1)) # TODO: will this capture the right people?

        # add an event to log to screen
        sim.schedule_event(HypertensiveDisordersOfPregnancyLoggingEvent(self), sim.date + DateOffset(months=6))


    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the ID for the mother for this child
        :param child_id: the ID for the new child
        """

        df = self.sim.population.props  # shortcut to the population props dataframe

        # Initialise all the properties that this module looks after:

        if df.at[child_id, 'sex'] == 'F':
            df.at[child_id, 'hp_pre_eclampsia'] = 'none'
            df.at[child_id, 'hp_pre_eclamp_sev'] = 'N'
            df.at[child_id, 'hp_gest_htn'] = False
            df.at[child_id, 'hp_prev_pre_eclamp'] = False
            df.at[child_id, 'hp_anti_htns'] = False

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug('This is HypertensiveDisordersOfPregnancy, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)

    def report_daly_values(self):

        logger.debug('This is HypertensiveDisordersOfPregnancy reporting my health values')

        df = self.sim.population.props  # shortcut to population properties dataframe

        p = self.parameters


class HypertensiveDisordersOfPregnancyEvent(RegularEvent, PopulationScopeEventMixin):
    """
    This event is occurrs once durin
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        df = population.props
        params = self.module.parameters

        logger.debug('This is HypertensiveDisodersOfPregnancy, managing the incidence of new cases in the pregnant'
                     ' population.')

        # 1.) NEW CASES PE/SPE

        susceptible_women = df.index[df.is_alive & df.is_pregnant & (df.hp_pre_eclampsia == 'none') &
                                     (df.ac_gestational_age < 4)]  # do we allow gestational hypertension to become PE

        eff_prob_pe = pd.Series(params['prob_pre_eclamp'], index=susceptible_women)

        # RISK FACTORS:
        eff_prob_pe.loc[df.is_alive & df.is_pregnant & (df.hp_pre_eclampsia =='none') & (df.la_parity == 0)] \
            *= params['rr_pre_eclamp_nulip']
        eff_prob_pe.loc[df.is_alive & df.is_pregnant & (df.hp_pre_eclampsia == 'none') & df.hp_prev_pre_eclamp] \
            *= params['rr_pre_eclamp_prev_pe']

        # TODO: chronic HTN (creating superimposed), BMI, DIABETES, TWINS

        selected = susceptible_women[eff_prob_pe > self.module.rng.random_sample(size=len(eff_prob_pe))]
        for person in selected:
            diff = 20 - df.at[person, 'ac_gestational_age']  # Ensure onset is after 20 weeks gestation
            time_until_onset = diff + np.random.exponential(scale=4, size=1)
            days_till_pe = pd.to_timedelta(time_until_onset, unit='w')  # this is days for some reason
            onset_date= self.sim.date + days_till_pe
            self.sim.schedule_event(PreeclampsiaEvent(self.module, person, cause='pre_eclampsia'), onset_date)

        # DISTRIBUTION FOR DATE OF ONSET? - if so this would be an individual event? (LISONKOVA ET AL FIGURE A?)
        # TODO: should include late onset and early onset as outcomes are worse for early onset...(need evidence)

# 2.) NEW CASES GH

# 3.) PROGRESSION OF PE?--> ECLAMPSIA

# 4.) CARE SEEKING (ADDITIONAL TO ANC?- undiagnosed but symptomatic?)


class PreeclampsiaEvent (Event, IndividualScopeEventMixin):
    """This event manages the onsent of pre-eclampsia in pregnant women previously determined as suceptible """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters

        logger.info('This is PreeclampsiaEvent, person %d has developed pre-eclampsia during her pregnancy',
                    individual_id)

        if df.at[individual_id, 'ac_gestational_age'] <= 33:
            df.at[individual_id, 'hp_pre_eclampsia'] = 'early_pre_eclamp'
            df.at[individual_id,'hp_prev_pre_eclamp'] = True

        if df.at[individual_id, 'ac_gestational_age'] > 33:
            df.at[individual_id, 'hp_pre_eclampsia'] = 'late_pre_eclamp'
            df.at[individual_id,'hp_prev_pre_eclamp'] = True

        # care seeking here?

class HypertensiveDisordersOfPregnancyDeathEvent(Event, IndividualScopeEventMixin):
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



class HypertensiveDisordersOfPregnancyLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """Produce a summmary of the numbers of people with respect to their 'mockitis status'
        """
        # run this event every month
        self.repeat = 6
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props


