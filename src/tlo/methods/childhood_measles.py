import logging

import pandas as pd
from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods.iCCM import HSI_Sick_Child_Seeks_Care_From_HSA

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ChildhoodMeasles(Module):
    PARAMETERS = {
        'base_prev_measles': Parameter
        (Types.REAL,
         'initial prevalence of non-complicated measles, among children aged 2-11 months,'
         'HIV negative, no SAM'
         ),
        'rp_measles_agelt2mo': Parameter
        (Types.REAL,
         'relative prevalence of non-complicated measles for age < 2 months'
         ),
        'rp_measles_age12to23mo': Parameter
        (Types.REAL,
         'relative prevalence of non-complicated measles for age 12 to 23 months'
         ),
        'rp_measles_age24to59mo': Parameter
        (Types.REAL,
         'relative prevalence of non-complicated measles for age 24 to 59 months'
         ),
        'rp_measles_HIV': Parameter
        (Types.REAL,
         'relative prevalence of non-complicated measles for HIV positive'
         ),
        'rp_measles_SAM': Parameter
        (Types.REAL,
         'relative prevalence of non-complicated measles for severe acute malnutrition'
         ),
        'rp_measles_excl_breast': Parameter
        (Types.REAL,
         'relative prevalence of non-complicated measles for exclusive breastfeeding upto 6 months'
         ),
        'rp_measles_cont_breast': Parameter
        (Types.REAL,
         'relative prevalence of non-complicated measles for continued breastfeeding upto 23 months'
         ),
        'base_prev_complicated_measles': Parameter
        (Types.REAL,
         'initial prevalence of measles with eye or mouth complications, among children aged 2-11 months,'
         'HIV negative, no SAM'
         ),
        'rp_complicated_measles_agelt2mo': Parameter
        (Types.REAL,
         'relative prevalence of measles with eye or mouth complications for age < 2 months'
         ),
        'rp_complicated_measles_age12to23mo': Parameter
        (Types.REAL,
         'relative prevalence of measles with eye or mouth complications for age 12 to 23 months'
         ),
        'rp_complicated_measles_age24to59mo': Parameter
        (Types.REAL,
         'relative prevalence of measles with eye or mouth complications for age 24 to 59 months'
         ),
        'rp_complicated_measles_HIV': Parameter
        (Types.REAL,
         'relative prevalence of measles with eye or mouth complications for HIV positive'
         ),
        'rp_complicated_measles_SAM': Parameter
        (Types.REAL,
         'relative prevalence of measles with eye or mouth complications for severe acute malnutrition'
         ),
        'rp_complicated_measles_excl_breast': Parameter
        (Types.REAL,
         'relative prevalence of measles with eye or mouth complications for exclusive breastfeeding upto 6 months'
         ),
        'rp_complicated_measles_cont_breast': Parameter
        (Types.REAL,
         'relative prevalence of measles with eye or mouth complications for continued breastfeeding upto 23 months'
         ),
        'base_prev_severe_measles': Parameter
        (Types.REAL,
         'initial prevalence of measles with eye or mouth complications, among children aged 2-11 months,'
         'HIV negative, no SAM'
         ),
        'rp_severe_measles_agelt2mo': Parameter
        (Types.REAL,
         'relative prevalence of severe measles for age < 2 months'
         ),
        'rp_severe_measles_age12to23mo': Parameter
        (Types.REAL,
         'relative prevalence of severe measles for age 12 to 23 months'
         ),
        'rp_severe_measles_age24to59mo': Parameter
        (Types.REAL,
         'relative prevalence of severe measles for age 24 to 59 months'
         ),
        'rp_severe_measles_HIV': Parameter
        (Types.REAL,
         'relative prevalence of severe measles for HIV positive'
         ),
        'rp_severe_measles_SAM': Parameter
        (Types.REAL,
         'relative prevalence of severe measles for severe acute malnutrition'
         ),
        'rp_severe_measles_excl_breast': Parameter
        (Types.REAL,
         'relative prevalence of severe measles for exclusive breastfeeding upto 6 months'
         ),
        'rp_severe_measles_cont_breast': Parameter
        (Types.REAL,
         'relative prevalence of severe measles for continued breastfeeding upto 23 months'
         ),
        'base_incidence_measles': Parameter
        (Types.REAL,
         'baseline incidence of non-complicated measles, among children aged 2-11 months, '
         'HIV negative, no SAM'
         ),
        'rr_measles_agelt2mo': Parameter
        (Types.REAL,
         'relative rate of non-complicated measles for age <2 months'
         ),
        'rr_measles_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of non-complicated measles for age 12 to 23 months'
         ),
        'rr_measles_age24to59mo': Parameter
        (Types.REAL,
         'relative rate of non-complicated measles for age 24 to 59 months'
         ),
        'rr_measles_HIV': Parameter
        (Types.REAL,
         'relative rate of non-complicated measles for HIV positive'
         ),
        'rr_measles_SAM': Parameter
        (Types.REAL,
         'relative rate of non-complicated measles for severe acute malnutrition'
         ),
        'rr_measles_excl_breast': Parameter
        (Types.REAL,
         'relative rate of non-complicated measles for exclusive breastfeeding upto 6 months'
         ),
        'rr_measles_cont_breast': Parameter
        (Types.REAL,
         'relative rate of non-complicated measles for continued breastfeeding upto 23 months'
         ),
        'r_progress_to_complicated_measles': Parameter
        (Types.REAL,
         'rate of progression from non-complicated measles to measles with eye or mouth complications '
         'among children aged 2-11 months, HIV negative, no SAM'
         ),
        'rr_progress_complicated_measles_agelt2mo': Parameter
        (Types.REAL,
         'rate of progression from non-complicated measles to measles with eye or mouth complications '
         'for children aged <2 months'
         ),
        'rr_progress_complicated_measles_age12to23mo': Parameter
        (Types.REAL,
         'rate of progression from non-complicated measles to measles with eye or mouth complications '
         'for children aged 12 to 23 months'
         ),
        'rr_progress_complicated_measles_age24to59mo': Parameter
        (Types.REAL, 'relative rate of progression to measles with eye or mouth complications for age 24 to 59 months'
         ),
        'rr_progress_complicated_measles_HIV': Parameter
        (Types.REAL,
         'relative risk of progression to measles with eye or mouth complications for HIV positive status'
         ),
        'rr_progress_complicated_measles_SAM': Parameter
        (Types.REAL,
         'relative rate of progression to measles with eye or mouth complications for severe acute malnutrition'
         ),
        'r_progress_to_severe_measles': Parameter
        (Types.REAL,
         'rate of progression from measles with eye or mouth complications to severe measles '
         'among children aged 2-11 months, HIV negative, no SAM'
         ),
        'rr_progress_severe_measles_agelt2mo': Parameter
        (Types.REAL,
         'rate of progression from non-complicated measles to measles with eye or mouth complications '
         'for children aged <2 months'
         ),
        'rr_progress_severe_measles_age12to23mo': Parameter
        (Types.REAL,
         'rate of progression from non-complicated measles to measles with eye or mouth complications '
         'for children aged 12 to 23 months'
         ),
        'rr_progress_severe_measles_age24to59mo': Parameter
        (Types.REAL, 'relative rate of progression to measles with eye or mouth complications for age 24 to 59 months'
         ),
        'rr_progress_severe_measles_HIV': Parameter
        (Types.REAL,
         'relative risk of progression to measles with eye or mouth complications for HIV positive status'
         ),
        'rr_progress_severe_measles_SAM': Parameter
        (Types.REAL,
         'relative rate of progression to measles with eye or mouth complications for severe acute malnutrition'
         ),
        'r_death_measles': Parameter
        (Types.REAL,
         'death rate from measles disease among children aged 2-11 months, HIV negative, no SAM '
         ),
        'rr_death_measles_agelt2mo': Parameter
        (Types.REAL,
         'relative rate of death from measles disease for age < 2 months'
         ),
        'rr_death_measles_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of death from measles disease for age 12 to 23 months'
         ),
        'rr_death_measles_age24to59mo': Parameter
        (Types.REAL,
         'relative rate of death from measles disease for age 24 to 59 months'
         ),
        'rr_death_measles_HIV': Parameter
        (Types.REAL,
         'relative rate of death from measles disease for HIV positive'
         ),
        'rr_death_measles_SAM': Parameter
        (Types.REAL,
         'relative rate of death from measles disease for severe acute malnutrition'
         ),
    }

    PROPERTIES = {
        'mi_measles_status': Property(Types.CATEGORICAL, 'measles infection - status',
                                      categories=['none', 'measles', 'complicated measles', 'severe pneumonia']),
        'has_hiv': Property(Types.BOOL, 'temporary property - has hiv'),
        'malnutrition': Property(Types.BOOL, 'temporary property - malnutrition status'),
        'exclusive_breastfeeding': Property(Types.BOOL, 'temporary property - exclusive breastfeeding upto 6 mo'),
        'continued_breastfeeding': Property(Types.BOOL, 'temporary property - continued breastfeeding 6mo-2years'),
        'mi_measles_death': Property(Types.BOOL, 'death from measles disease'),
        'mi_generalised_rash': Property(Types.BOOL, 'measles symptoms - generalised rash'),
        'mi_fever': Property(Types.BOOL, 'measles symptoms - fever'),
        'mi_cough_running_nose_redeyes': Property(Types.BOOL, 'measles symptoms - cough, running nose, or red eyes'),
        'mi_mouth_ulcers': Property(Types.BOOL, 'complicated measles symptoms - mouth ulcers'),
        'mi_pus_draining_from_eye': Property(Types.BOOL, 'complicated measles symptoms - pus draining from the eye'),
        'mi_deep_extensive_mouth_ulcers': Property(Types.BOOL, 'severe measles symptoms - deep or extensive mouth ulcers'),
        'mi_clouding_cornea': Property(Types.BOOL, 'severe measles symptoms - clouding of the cornea')
    }

    TREATMENT_ID = 'Vitamin A'

    def read_parameters(self, data_folder):
        p = self.parameters

        p['base_prev_measles'] = 0.3
        p['rp_measles_agelt2mo'] = 0.7
        p['rp_measles_age12to23mo'] = 1.2
        p['rp_measles_age24to59mo'] = 1.3
        p['rp_measles_HIV'] = 1.4
        p['rp_measles_SAM'] = 1.3
        p['rp_measles_excl_breast'] = 0.3
        p['rp_measles_cont_breast'] = 0.7
        p['base_prev_complicated_measles'] = 0.2
        p['rp_complicated_measles_agelt2mo'] = 0.8
        p['rp_complicated_measles_age12to23mo'] = 1.2
        p['rp_complicated_measles_age24to59mo'] = 1.3
        p['rp_complicated_measles_HIV'] = 1.4
        p['rp_complicated_measles_SAM'] = 1.3
        p['rp_complicated_measles_excl_breast'] = 0.3
        p['rp_complicated_measles_cont_breast'] = 0.7
        p['base_prev_severe_measles'] = 0.2
        p['rp_severe_measles_agelt2mo'] = 1.2
        p['rp_severe_measles_age12to23mo'] = 1.2
        p['rp_severe_measles_age24to59mo'] = 1.3
        p['rp_severe_measles_HIV'] = 1.4
        p['rp_severe_measles_SAM'] = 1.3
        p['rp_severe_measles_excl_breast'] = 0.3
        p['rp_severe_measles_cont_breast'] = 0.7
        p['base_incidence_measles'] = 0.3
        p['rr_measles_agelt2mo'] = 0.7
        p['rr_measles_age12to23mo'] = 1.2
        p['rr_measles_age24to59mo'] = 1.3
        p['rr_measles_HIV'] = 1.4
        p['rr_measles_SAM'] = 1.3
        p['rr_measles_excl_breast'] = 0.3
        p['rr_measles_cont_breast'] = 0.7
        p['r_progress_to_complicated_measles'] = 0.5
        p['rr_progress_complicated_measles_agelt2mo'] = 0.9
        p['rr_progress_complicated_measles_age12to23mo'] = 1.2
        p['rr_progress_complicated_measles_age24to59mo'] = 1.2
        p['rr_progress_complicated_measles_HIV'] = 1.5
        p['rr_progress_complicated_measles_SAM'] = 1.4
        p['r_progress_to_severe_measles'] = 0.5
        p['rr_progress_severe_measles_agelt2mo'] = 1.2
        p['rr_progress_severe_measles_age12to23mo'] = 1.2
        p['rr_progress_severe_measles_age24to59mo'] = 1.2
        p['rr_progress_severe_measles_HIV'] = 1.5
        p['rr_progress_severe_measles_SAM'] = 1.4
        p['r_death_measles'] = 0.7
        p['rr_death_measles_agelt2mo'] = 1.2
        p['rr_death_measles_age12to23mo'] = 1.1
        p['rr_death_measles_age24to59mo'] = 1.4
        p['rr_death_measles_HIV'] = 1.4
        p['rr_death_measles_SAM'] = 1.4
        p['init_prop_measles_status'] = [0.4, 0.3, 0.2]

    def initialise_population(self, population):
        df = population.props
        m = self
        rng = m.rng

        # defaults
        df['mi_measles_status'] = 'none'
        df['malnutrition'] = False
        df['has_HIV'] = False
        df['exclusive_breastfeeding'] = False
        df['continued_breastfeeding'] = False
        df['mi_measles_death'] = False

        # --------------------------------------------------------------------------------------------------------
        # ----------------------------- ASSIGN VALUES OF MEASLES STATUS AT BASELINE ------------------------------
        # --------------------------------------------------------------------------------------------------------

        under5_idx = df.index[(df.age_years < 5) & df.is_alive]

        # create data-frame of the probabilities of mi_measles_status for children
        # aged 2-11 months, HIV negative, no SAM
        p_measles_status = pd.Series(self.init_prop_measles_status[0], index=under5_idx)
        p_complic_measles_status = pd.Series(self.init_prop_measles_status[1], index=under5_idx)
        p_severe_measles_status = pd.Series(self.init_prop_measles_status[2], index=under5_idx)

        # create probabilities of pneumonia for all age under
        p_measles_status.loc[
            (df.age_exact_years >= 1) & (df.age_exact_years < 2) & df.is_alive] *= self.rp_measles_age12to23mo
        p_measles_status.loc[
            (df.age_exact_years >= 2) & (df.age_exact_years < 5) & df.is_alive] *= self.rp_measles_age24to59mo
        p_measles_status.loc[
            (df.has_hiv == True) & (df.age_years < 5) & df.is_alive] *= self.rp_measles_HIV
        p_measles_status.loc[
            (df.malnutrition == True) & (df.age_years < 5) & df.is_alive] *= self.rp_measles_SAM
        p_measles_status.loc[
            (df.exclusive_breastfeeding == True) & (df.age_exact_years <= 0.5) & df.is_alive] \
            *= self.rp_measles_excl_breast
        p_measles_status.loc[
            (df.continued_breastfeeding == True) & (df.age_exact_years > 0.5) & (df.age_exact_years < 2) &
            df.is_alive] *= self.rp_measles_cont_breast

        # create probabilities of severe pneumonia for all age under 5
        p_complic_measles_status.loc[
            (df.age_exact_years < 0.1667) & df.is_alive] *= self.rp_complicated_measles_agelt2mo
        p_complic_measles_status.loc[
            (df.age_exact_years >= 1) & (df.age_exact_years < 2) & df.is_alive] *= self.rp_complicated_measles_age12to23mo
        p_complic_measles_status.loc[
            (df.age_exact_years >= 2) & (df.age_exact_years < 5) & df.is_alive] *= self.rp_complicated_measles_age24to59mo
        p_complic_measles_status.loc[
            (df.has_hiv == True) & (df.age_years < 5) & df.is_alive] *= self.rp_complicated_measles_HIV
        p_complic_measles_status.loc[
            (df.malnutrition == True) & (df.age_years < 5) & df.is_alive] *= self.rp_complicated_measles_SAM
        p_complic_measles_status.loc[
            (df.exclusive_breastfeeding == True) & (df.age_exact_years <= 0.5) & df.is_alive] \
            *= self.rp_complicated_measles_excl_breast
        p_complic_measles_status.loc[
            (df.continued_breastfeeding == True) & (df.age_exact_years > 0.5) & (df.age_exact_years < 2) &
            df.is_alive] *= self.rp_complicated_measles_cont_breast

        # create probabilities of very severe pneumonia for all age under 5
        p_severe_measles_status.loc[
            (df.age_exact_years < 0.1667) & df.is_alive] *= self.rp_severe_measles_agelt2mo
        p_severe_measles_status.loc[
            (df.age_exact_years >= 1) & (df.age_exact_years < 2) & df.is_alive] *= self.rp_severe_measles_age12to23mo
        p_severe_measles_status.loc[
            (df.age_exact_years >= 2) & (df.age_exact_years < 5) & df.is_alive] *= self.rp_severe_measles_age24to59mo
        p_severe_measles_status.loc[
            (df.has_hiv == True) & (df.age_years < 5) & df.is_alive] *= self.rp_severe_measles_HIV
        p_severe_measles_status.loc[
            (df.malnutrition == True) & (df.age_years < 5) & df.is_alive] *= self.rp_severe_measles_SAM
        p_severe_measles_status.loc[
            (df.exclusive_breastfeeding == True) & (df.age_exact_years <= 0.5) & df.is_alive] \
            *= self.rp_severe_measles_excl_breast
        p_severe_measles_status.loc[
            (df.continued_breastfeeding == True) & (df.age_exact_years > 0.5) & (df.age_exact_years < 2) &
            df.is_alive] *= self.rp_severe_measles_cont_breast

        random_draw = pd.Series(rng.random_sample(size=len(under5_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive])

        # create a temporary dataframe called dfx to hold values of probabilities and random draw
        dfx = pd.concat([p_measles_status, p_complic_measles_status, p_severe_measles_status, random_draw], axis=1)
        dfx.columns = ['p_measles', 'p_complicated_measles', 'p_severe_measles', 'random_draw']

        dfx['p_none'] = 1 - (dfx.p_measles + dfx.p_complicated_measles + dfx.p_severe_measles)

        # based on probabilities of being in each category, define cut-offs to determine status from
        # random draw uniform(0,1)

        # assign baseline values of ri_resp_infection_stat based on probabilities and value of random draw

        idx_none = dfx.index[dfx.p_none > dfx.random_draw]
        idx_measles = dfx.index[(dfx.p_none < dfx.random_draw) & ((dfx.p_none + dfx.p_measles) > dfx.random_draw)]
        idx_complic_measles = dfx.index[((dfx.p_none + dfx.p_measles) < dfx.random_draw) &
                                         (dfx.p_none + dfx.p_measles + dfx.p_complicated_measles) > dfx.random_draw]
        idx_severe_measles = dfx.index[
            ((dfx.p_none + dfx.p_measles + dfx.p_complicated_measles) < dfx.random_draw) &
            (dfx.p_none + dfx.p_measles + dfx.p_complicated_measles + dfx.p_severe_measles) > dfx.random_draw]

        df.loc[idx_none, 'mi_measles_status'] = 'none'
        df.loc[idx_measles, 'mi_measles_status'] = 'measles'
        df.loc[idx_complic_measles, 'mi_measles_status'] = 'complicated pneumonia'
        df.loc[idx_severe_measles, 'mi_measles_status'] = 'severe measles'

    def initialise_simulation(self, sim):

        # add the basic event
        sim.schedule_event(MeaslesEvent(self), sim.date + DateOffset(months=3))

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

    def on_birth(self, mother_id, child_id):
        pass

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug('This is Pneumonia, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)

    def report_daly_values(self):
        pass


class MeaslesEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=3))

    def apply(self, population):

        # logger.debug('This is MeaslesEvent, tracking the disease progression of the population.')

        df = population.props
        m = self.module
        rng = m.rng

        # --------------------------------------------------------------------------------------------------------
        # UPDATING FOR CHILDREN UNDER 5 WITH CURRENT STATUS 'NONE' TO NON-COMPLICATED MEASLES
        # --------------------------------------------------------------------------------------------------------

        eff_prob_mi_measles = pd.Series(m.base_incidence_measles,
                                        index=df.index[
                                              df.is_alive & (df.mi_measles_status == 'none') &
                                              (df.age_exact_years > 0.1667) & (df.age_years < 5)])

        eff_prob_mi_measles.loc[df.is_alive & (df.mi_measles_status == 'none') &
                                (df.age_exact_years < 0.1667)] *= m.rr_measles_agelt2mo
        eff_prob_mi_measles.loc[df.is_alive & (df.mi_measles_status == 'none') &
                                (df.age_exact_years >= 1) & (df.age_exact_years < 2)] *= m.rr_measles_age12to23mo
        eff_prob_mi_measles.loc[df.is_alive & (df.mi_measles_status == 'none') &
                                (df.age_exact_years >= 2) & (df.age_exact_years < 5)] *= m.rr_measles_age24to59mo
        eff_prob_mi_measles.loc[df.is_alive & (df.mi_measles_status == 'none') &
                                (df.has_hiv == True) & (df.age_years < 5)] *= m.rr_measles_HIV
        eff_prob_mi_measles.loc[df.is_alive & (df.mi_measles_status == 'none') &
                                df.malnutrition == True & (df.age_years < 5)] *= m.rr_measles_SAM
        eff_prob_mi_measles.loc[df.is_alive & (df.mi_measles_status == 'none') &
                                df.exclusive_breastfeeding == True &
                                (df.age_exact_years <= 0.5)] *= m.rr_measles_excl_breast
        eff_prob_mi_measles.loc[df.is_alive & (df.continued_breastfeeding == True) &
                                (df.age_exact_years > 0.5) & (df.age_exact_years < 2)] *= m.rr_measles_cont_breast

        mi_current_none_idx = \
            df.index[
                df.is_alive & (df.mi_measles_status == 'none') & (df.age_years < 5)]

        random_draw_01 = pd.Series(rng.random_sample(size=len(mi_current_none_idx)),
                                   index=df.index[(df.age_years < 5) & df.is_alive & (df.mi_measles_status == 'none')])

        dfx = pd.concat([eff_prob_mi_measles, random_draw_01], axis=1)
        dfx.columns = ['eff_prob_mi_measles', 'random_draw_01']
        idx_incident_measles = dfx.index[dfx.eff_prob_mi_measles > dfx.random_draw_01]
        df.loc[idx_incident_measles, 'mi_measles_status'] = 'measles'

        # # # # # # # # # SYMPTOMS FROM NON-COMPLICATED MEASLES # # # # # # # # # # # # # # # # # #

        mi_current_measles_idx = df.index[df.is_alive & (df.age_years < 5) & (df.mi_measles_status == 'measles')]
        # generalised rash
        for individual in mi_current_measles_idx:
            df.at[individual, 'mi_generalised_rash'] = True

        # fever
        eff_prob_fever = pd.Series(0.89, index=mi_current_measles_idx)
        random_draw = pd.Series(rng.random_sample(size=len(mi_current_measles_idx)),
                                index=df.index[(df.age_exact_years > 0.1667) & (df.age_years < 5) & df.is_alive &
                                               (df.mi_measles_status == 'measles')])
        dfx = pd.concat([eff_prob_fever, random_draw], axis=1)
        dfx.columns = ['eff_prob_fever', 'random number']
        idx_fever = dfx.index[dfx.eff_prob_fever > random_draw]
        df.loc[idx_fever, 'mi_fever'] = True

        # cough, running nose or red eyes
        eff_prob_cough_running_nose_redeyes = pd.Series(0.89, index=mi_current_measles_idx)
        random_draw = pd.Series(rng.random_sample(size=len(mi_current_measles_idx)),
                                index=df.index[(df.age_exact_years > 0.1667) &
                                               (df.age_years < 5) & df.is_alive & (df.mi_measles_status == 'measles')])
        dfx = pd.concat([eff_prob_cough_running_nose_redeyes, random_draw], axis=1)
        dfx.columns = ['eff_prob_cough_running_nose_redeyes', 'random number']
        idx_cough_running_nose_redeyes = dfx.index[dfx.eff_prob_cough_running_nose_redeyes > random_draw]
        df.loc[idx_cough_running_nose_redeyes, 'mi_cough_running_nose_redeyes'] = True

        # --------------------------------------------------------------------------------------------------------
        # SEEKING CARE FOR NON-COMPLICATED MEASLES
        # --------------------------------------------------------------------------------------------------------

        measles_symptoms = df.index[df.is_alive & (df.mi_generalised_rash == True) and
                                    (df.mi_cough_running_nose_redeyes == True) | (df.mi_fever == True)]

        seeks_care = pd.Series(data=False, index=measles_symptoms)
        for individual in measles_symptoms:
            prob = self.sim.modules['HealthSystem'].get_prob_seek_care(individual, symptom_code=1)
            seeks_care[individual] = self.module.rng.rand() < prob
            event = HSI_Sick_Child_Seeks_Care_From_HSA(self.module, person_id=individual)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=2,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(weeks=2)
                                                                )

        # --------------------------------------------------------------------------------------------------------
        # UPDATING FOR CHILDREN UNDER 5 WITH CURRENT STATUS 'MEASLES' TO 'MEASLES WITH EYE OR MOUTH COMPLICATIONS'
        # --------------------------------------------------------------------------------------------------------

        eff_prob_prog_complic_measles = pd.Series(m.r_progress_to_complicated_measles,
                                                   index=df.index[df.is_alive & (df.mi_measles_status == 'measles')
                                                                  & (df.age_years < 5)])

        eff_prob_prog_complic_measles.loc[df.is_alive & (df.mi_measles_status == 'measles') &
                                          (df.age_exact_years < 0.1667)] *= m.rr_progress_complicated_measles_agelt2mo
        eff_prob_prog_complic_measles.loc[df.is_alive & (df.mi_measles_status == 'measles') &
                                          (df.age_exact_years >= 1) &
                                          (df.age_exact_years < 2)] *= m.rr_progress_complicated_measles_age12to23mo
        eff_prob_prog_complic_measles.loc[df.is_alive & (df.mi_measles_status == 'measles') &
                                          (df.age_exact_years >= 2) & (df.age_exact_years < 5)] *= \
            m.rr_progress_complicated_measles_age24to59mo
        eff_prob_prog_complic_measles.loc[df.is_alive & (df.mi_measles_status == 'measles') &
                                          df.has_hiv == True & (df.age_years < 5)] *= \
            m.rr_progress_complicated_measles_HIV
        eff_prob_prog_complic_measles.loc[df.is_alive & (df.mi_measles_status == 'measles') &
                                          df.malnutrition == True & (df.age_years < 5)] *= \
            m.rr_progress_complicated_measles_SAM

        mi_current_measles_idx = df.index[df.is_alive & (df.age_years < 5) & (df.mi_measles_status == 'measles')]

        random_draw_03 = pd.Series(rng.random_sample(size=len(mi_current_measles_idx)),
                                   index=df.index[(df.age_years < 5) & df.is_alive &
                                                  (df.mi_measles_status == 'measles')])

        dfx = pd.concat([eff_prob_prog_complic_measles, random_draw_03], axis=1)
        dfx.columns = ['eff_prob_prog_complicated_measles', 'random_draw_03']
        idx_mi_progress_complic_measles = dfx.index[dfx.eff_prob_prog_complicated_measles > dfx.random_draw_03]
        df.loc[idx_mi_progress_complic_measles, 'mi_measles_status'] = ' complicated measles'

        # # # # # # # # # SYMPTOMS FROM MEASLES WITH EYE OR MOUTH COMPLICATIONS # # # # # # # # #

        mi_current_complic_measles_idx = df.index[df.is_alive & (df.age_years < 5) &
                                          (df.mi_measles_status == 'complicated measles')]
        # generalised rash
        for individual in mi_current_complic_measles_idx:
            df.at[individual, 'mi_generalised_rash'] = True

        # fever
        eff_prob_fever = pd.Series(0.89, index=mi_current_complic_measles_idx)
        random_draw = pd.Series(rng.random_sample(size=len(mi_current_complic_measles_idx)),
                                index=df.index[(df.age_exact_years > 0.1667) & (df.age_years < 5) & df.is_alive &
                                               (df.mi_measles_status == 'complicated measles')])
        dfx = pd.concat([eff_prob_fever, random_draw], axis=1)
        dfx.columns = ['eff_prob_fever', 'random number']
        idx_fever = dfx.index[dfx.eff_prob_fever > random_draw]
        df.loc[idx_fever, 'mi_fever'] = True

        # cough, running nose or red eyes
        eff_prob_cough_running_nose_redeyes = pd.Series(0.89, index=mi_current_complic_measles_idx)
        random_draw = pd.Series(rng.random_sample(size=len(mi_current_complic_measles_idx)),
                                index=df.index[(df.age_exact_years > 0.1667) &
                                               (df.age_years < 5) & df.is_alive & (df.mi_measles_status == 'complicated measles')])
        dfx = pd.concat([eff_prob_cough_running_nose_redeyes, random_draw], axis=1)
        dfx.columns = ['eff_prob_cough_running_nose_redeyes', 'random number']
        idx_cough_running_nose_redeyes = dfx.index[dfx.eff_prob_cough_running_nose_redeyes > random_draw]
        df.loc[idx_cough_running_nose_redeyes, 'mi_cough_running_nose_redeyes'] = True

        # mouth ulcers
        eff_prob_mouth_ulcers = pd.Series(0.89, index=mi_current_complic_measles_idx)
        random_draw = pd.Series(rng.random_sample(size=len(mi_current_complic_measles_idx)),
                                index=df.index[(df.age_exact_years > 0.1667) & (df.age_years < 5) & df.is_alive &
                                               (df.mi_measles_status == 'complicated measles')])
        dfx = pd.concat([eff_prob_mouth_ulcers, random_draw], axis=1)
        dfx.columns = ['eff_prob_mouth_ulcers', 'random number']
        idx_mi_mouth_ulcers = dfx.index[dfx.eff_prob_mouth_ulcers > random_draw]
        df.loc[idx_mi_mouth_ulcers, 'mi_mouth_ulcers'] = True

        # pus draining from the eye
        eff_prob_pus_draining_from_eye = pd.Series(0.89, index=mi_current_complic_measles_idx)
        random_draw = pd.Series(rng.random_sample(size=len(mi_current_complic_measles_idx)),
                                index=df.index[(df.age_exact_years > 0.1667) & (df.age_years < 5) & df.is_alive &
                                               (df.mi_measles_status == 'complicated measles')])
        dfx = pd.concat([eff_prob_pus_draining_from_eye, random_draw], axis=1)
        dfx.columns = ['eff_prob_pus_draining_from_eye', 'random number']
        idx_pus_draining_from_eye = dfx.index[dfx.eff_prob_pus_draining_from_eye > random_draw]
        df.loc[idx_pus_draining_from_eye, 'mi_pus_draining_from_eye'] = True

        # --------------------------------------------------------------------------------------------------------
        # SEEKING CARE FOR MEASLES WITH EYE OR MOUTH COMPLICATIONS
        # --------------------------------------------------------------------------------------------------------

        measles_with_complications_symptoms = \
            df.index[df.is_alive & (df.mi_generalised_rash == True) and (df.mi_cough_running_nose_redeyes == True) |
                     (df.mi_fever == True) & (df.mi_mouth_ulcers == True) | (df.mi_pus_draining_from_eye == True)]

        seeks_care = pd.Series(data=False, index=measles_symptoms)
        for individual in measles_with_complications_symptoms:
            prob = self.sim.modules['HealthSystem'].get_prob_seek_care(individual, symptom_code=1)
            seeks_care[individual] = self.module.rng.rand() < prob
            event = HSI_Sick_Child_Seeks_Care_From_HSA(self.module, person_id=individual)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=2,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(weeks=2)
                                                                )

        # --------------------------------------------------------------------------------------------------------
        # UPDATING CURRENT STATUS 'MEASLES WITH EYE OR MOUTH COMPLICATIONS' TO 'SEVERE COMPLICATED MEASLES'
        # --------------------------------------------------------------------------------------------------------

        eff_prob_prog_severe_measles = pd.Series(m.r_progress_to_complicated_measles,
                                                   index=df.index[df.is_alive & (df.mi_measles_status == 'complicated measles')
                                                                  & (df.age_years < 5)])

        eff_prob_prog_severe_measles.loc[df.is_alive & (df.mi_measles_status == 'complicated measles') &
                                          (df.age_exact_years < 0.1667)] *= m.rr_progress_complicated_measles_agelt2mo
        eff_prob_prog_severe_measles.loc[df.is_alive & (df.mi_measles_status == 'complicated measles') &
                                          (df.age_exact_years >= 1) &
                                          (df.age_exact_years < 2)] *= m.rr_progress_complicated_measles_age12to23mo
        eff_prob_prog_severe_measles.loc[df.is_alive & (df.mi_measles_status == 'complicated measles') &
                                          (df.age_exact_years >= 2) & (df.age_exact_years < 5)] *= \
            m.rr_progress_severe_measles_age24to59mo
        eff_prob_prog_severe_measles.loc[df.is_alive & (df.mi_measles_status == 'complicated measles') &
                                          df.has_hiv == True & (df.age_years < 5)] *= \
            m.rr_progress_severe_measles_HIV
        eff_prob_prog_severe_measles.loc[df.is_alive & (df.mi_measles_status == 'complicated measles') &
                                          df.malnutrition == True & (df.age_years < 5)] *= \
            m.rr_progress_severe_measles_SAM

        mi_current_complic_measles_idx = df.index[df.is_alive & (df.age_years < 5) & (df.mi_measles_status == 'complicated measles')]

        random_draw_03 = pd.Series(rng.random_sample(size=len(mi_current_complic_measles_idx)),
                                   index=df.index[(df.age_years < 5) & df.is_alive &
                                                  (df.mi_measles_status == 'complicated measles')])

        dfx = pd.concat([eff_prob_prog_severe_measles, random_draw_03], axis=1)
        dfx.columns = ['eff_prob_prog_severe_measles', 'random_draw_03']
        idx_mi_progress_severe_measles = dfx.index[dfx.eff_prob_prog_severe_measles > dfx.random_draw_03]
        df.loc[idx_mi_progress_severe_measles, 'mi_measles_status'] = 'severe measles'

        # # # # # # # # # SYMPTOMS FROM SEVERE COMPLICATED MEASLES # # # # # # # # # # # # # # # # # #

        mi_current_severe_measles_idx = df.index[df.is_alive & (df.age_years < 5) &
                                                  (df.mi_measles_status == 'severe measles')]
        # generalised rash
        for individual in mi_current_severe_measles_idx:
            df.at[individual, 'mi_generalised_rash'] = True

        # fever
        eff_prob_fever = pd.Series(0.89, index=mi_current_severe_measles_idx)
        random_draw = pd.Series(rng.random_sample(size=len(mi_current_severe_measles_idx)),
                                index=df.index[(df.age_exact_years > 0.1667) & (df.age_years < 5) & df.is_alive &
                                               (df.mi_measles_status == 'severe measles')])
        dfx = pd.concat([eff_prob_fever, random_draw], axis=1)
        dfx.columns = ['eff_prob_fever', 'random number']
        idx_fever = dfx.index[dfx.eff_prob_fever > random_draw]
        df.loc[idx_fever, 'mi_fever'] = True

        # cough, running nose or red eyes
        eff_prob_cough_running_nose_redeyes = pd.Series(0.89, index=mi_current_severe_measles_idx)
        random_draw = pd.Series(rng.random_sample(size=len(mi_current_severe_measles_idx)),
                                index=df.index[(df.age_exact_years > 0.1667) &
                                               (df.age_years < 5) & df.is_alive & (
                                                       df.mi_measles_status == 'severe measles')])
        dfx = pd.concat([eff_prob_cough_running_nose_redeyes, random_draw], axis=1)
        dfx.columns = ['eff_prob_cough_running_nose_redeyes', 'random number']
        idx_cough_running_nose_redeyes = dfx.index[dfx.eff_prob_cough_running_nose_redeyes > random_draw]
        df.loc[idx_cough_running_nose_redeyes, 'mi_cough_running_nose_redeyes'] = True

        # pus draining from the eye
        eff_prob_pus_draining_from_eye = pd.Series(0.89, index=mi_current_severe_measles_idx)
        random_draw = pd.Series(rng.random_sample(size=len(mi_current_severe_measles_idx)),
                                index=df.index[(df.age_exact_years > 0.1667) & (df.age_years < 5) & df.is_alive &
                                               (df.mi_measles_status == 'severe measles')])
        dfx = pd.concat([eff_prob_pus_draining_from_eye, random_draw], axis=1)
        dfx.columns = ['eff_prob_pus_draining_from_eye', 'random number']
        idx_pus_draining_from_eye = dfx.index[dfx.eff_prob_pus_draining_from_eye > random_draw]
        df.loc[idx_pus_draining_from_eye, 'mi_pus_draining_from_eye'] = True

        # deep or extensive mouth ulcers
        eff_prob_deep_extensive_mouth_ulcers = pd.Series(0.89, index=mi_current_severe_measles_idx)
        random_draw = pd.Series(rng.random_sample(size=len(mi_current_severe_measles_idx)),
                                index=df.index[(df.age_exact_years > 0.1667) & (df.age_years < 5) & df.is_alive &
                                               (df.mi_measles_status == 'severe measles')])
        dfx = pd.concat([eff_prob_deep_extensive_mouth_ulcers, random_draw], axis=1)
        dfx.columns = ['eff_prob_deep_extensive_mouth_ulcers', 'random number']
        idx_mi_deep_extensive_mouth_ulcers = dfx.index[dfx.eff_prob_deep_extensive_mouth_ulcers > random_draw]
        df.loc[idx_mi_deep_extensive_mouth_ulcers, 'mi_deep_extensive_mouth_ulcers'] = True

        # clouding of the cornea
        eff_prob_clouding_cornea = pd.Series(0.89, index=mi_current_severe_measles_idx)
        random_draw = pd.Series(rng.random_sample(size=len(mi_current_severe_measles_idx)),
                                index=df.index[(df.age_exact_years > 0.1667) & (df.age_years < 5) & df.is_alive &
                                               (df.mi_measles_status == 'severe measles')])
        dfx = pd.concat([eff_prob_clouding_cornea, random_draw], axis=1)
        dfx.columns = ['eff_prob_clouding_cornea', 'random number']
        idx_mi_clouding_cornea = dfx.index[dfx.eff_prob_clouding_cornea > random_draw]
        df.loc[idx_mi_clouding_cornea, 'mi_clouding_cornea'] = True

        # --------------------------------------------------------------------------------------------------------
        # SEEKING CARE FOR SEVERE COMPLICATED MEASLES
        # --------------------------------------------------------------------------------------------------------

        severe_measles_symptoms = \
            df.index[df.is_alive & (df.mi_generalised_rash == True) and (df.mi_cough_running_nose_redeyes == True) |
                     (df.mi_fever == True) & (df.mi_deep_extensive_mouth_ulcers == True) | (df.mi_clouding_cornea == True)]

        seeks_care = pd.Series(data=False, index=measles_symptoms)
        for individual in severe_measles_symptoms:
            prob = self.sim.modules['HealthSystem'].get_prob_seek_care(individual, symptom_code=1)
            seeks_care[individual] = self.module.rng.rand() < prob
            event = HSI_Sick_Child_Seeks_Care_From_HSA(self.module, person_id=individual)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=2,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(weeks=2)
                                                                )


class MeaslesDeath_Event(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        m = self.module
        rng = m.rng

        eff_prob_death_measles = \
            pd.Series(m.r_death_measles,
                      index=df.index[
                          df.is_alive & (df.mi_measles_status == 'severe measles') & (df.age_years < 5)])
        eff_prob_death_measles.loc[df.is_alive & (df.mi_measles_status == 'severe measles') &
                                   (df.age_years < 5)] *= m.rr_death_measles_agelt2mo
        eff_prob_death_measles.loc[df.is_alive & (df.mi_measles_status == 'severe measles') &
                                   (df.age_exact_years >= 1) & (df.age_exact_years < 2)] *= \
            m.rr_death_measles_age12to23mo
        eff_prob_death_measles.loc[df.is_alive & (df.mi_measles_status == 'severe measles') &
                                   (df.age_exact_years >= 2) & (df.age_exact_years < 5)] *= \
            m.rr_death_measles_age24to59mo
        eff_prob_death_measles.loc[df.is_alive & (df.mi_measles_status == 'severe measles') &
                                   df.has_hiv == True & (df.age_years < 5)] *= m.rr_death_measles_HIV
        eff_prob_death_measles.loc[df.is_alive & (df.mi_measles_status == 'severe measles') &
                                   df.malnutrition == True & (df.age_years < 5)] *= m.rr_death_measles_SAM

        mi1_current_severe_measles_idx = \
            df.index[df.is_alive & (df.mi_measles_status == 'severe measles') & (df.age_years < 5)]

        random_draw_04 = pd.Series(rng.random_sample(size=len(mi1_current_severe_measles_idx)),
                                   index=df.index[(df.age_years < 5) & df.is_alive &
                                                  (df.mi_measles_status == 'severe measles')])

        dfx = pd.concat([eff_prob_death_measles, random_draw_04], axis=1)
        dfx.columns = ['eff_prob_death_measles', 'random_draw_04']
        dfx['measles_death'] = False

        if dfx.loc[dfx.eff_prob_death_measles > dfx.random_draw_04]:
            dfx['measles_death'] = True
            df.loc[mi1_current_severe_measles_idx, 'mi_measles_death'] = dfx['measles_death']
            for individual_id in df.index[df.ri_pneumonia_death]:
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id, 'ChildhoodMeasles'),
                                        self.sim.date)
        else:
            df.loc[mi1_current_severe_measles_idx, 'mi_measles_status'] = 'none'
