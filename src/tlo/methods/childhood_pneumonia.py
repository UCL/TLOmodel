"""
Childhood Pneumonia module
Documentation: 04 - Methods Repository/Method_Child_RespiratoryInfection.xlsx
"""
import logging

import pandas as pd
from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ChildhoodPneumonia(Module):
    PARAMETERS = {
        'base_prev_pneumonia': Parameter
        (Types.REAL,
         'initial prevalence of pneumonia, among children aged 2-11 months,'
         ' HIV negative, no SAM, no indoor air pollution '
         ),
        'rp_pneumonia_agelt2mo': Parameter
        (Types.REAL,
         'relative prevalence of pneumonia for age < 2 months'
         ),
        'rp_pneumonia_age12to23mo': Parameter
        (Types.REAL,
         'relative prevalence of pneumonia for age 12 to 23 months'
         ),
        'rp_pneumonia_age24to59mo': Parameter
        (Types.REAL,
         'relative prevalence of pneumonia for age 24 to 59 months'
         ),
        'rp_pneumonia_HIV': Parameter
        (Types.REAL,
         'relative prevalence of pneumonia for HIV positive'
         ),
        'rp_pneumonia_malnutrition': Parameter
        (Types.REAL,
         'relative prevalence of pneumonia for severe acute malnutrition'
         ),
        'rp_pneumonia_IAP': Parameter
        (Types.REAL, 'relative prevalence of pneumonia for indoor air pollution'
         ),
        'base_incidence_pneumonia': Parameter
        (Types.REAL,
         'baseline incidence of pneumonia, among children aged 2-11 months, '
         'HIV negative, no SAM, no indoor air pollution '
         ),
        'rr_pneumonia_agelt2mo': Parameter
        (Types.REAL,
         'relative rate of pneumonia for age < 2 months'
         ),
        'rr_pneumonia_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of pneumonia for age 12 to 23 months'
         ),
        'rr_pneumonia_age24to59mo': Parameter
        (Types.REAL,
         'relative rate of pneumonia for age 24 to 59 months'
         ),
        'rr_pneumonia_HIV': Parameter
        (Types.REAL,
         'relative rate of pneumonia for HIV positive'
         ),
        'rr_pneumonia_malnutrition': Parameter
        (Types.REAL,
         'relative rate of pneumonia for severe acute malnutrition'
         ),
        'rr_pneumonia_IAP': Parameter
        (Types.REAL,
         'relative rate of pneumonia for indoor air pollution'
         ),
        'base_prev_severe_pneumonia': Parameter
        (Types.REAL,
         'initial prevalence of severe pneumonia, among children aged 3-11 months, '
         'HIV negative, normal weight, no SAM, no indoor air pollution '
         ),
        'rp_severe_pneum_agelt2mo': Parameter
        (Types.REAL, 'relative prevalence of severe pneumonia for age <2 months'
         ),
        'rp_severe_pneum_age12to23mo': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for age 12 to 23 months'
         ),
        'rp_severe_pneum_age24to59mo': Parameter
        (Types.REAL, 'relative prevalence of severe pneumonia for age 24 to 59 months'
         ),
        'rp_severe_pneum_HIV': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for HIV positive status'
         ),
        'rp_severe_pneum_malnutrition': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for severe acute malnutrition'
         ),
        'rp_severe_pneum_IAP': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for indoor air pollution'
         ),
        'base_incidence_severe_pneum': Parameter
        (Types.REAL,
         'baseline incidence of severe pneumonia, among children aged 2-11 months, '
         'HIV negative, no SAM, no indoor air pollution '
         ),
        'rr_severe_pneum_agelt2mo': Parameter
        (Types.REAL,
         'relative rate of severe pneumonia for age <2 months'
         ),
        'rr_severe_pneum_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of severe pneumonia for age 12 to 23 months'
         ),
        'rr_severe_pneum_age24to59mo': Parameter
        (Types.REAL,
         'relative rate of severe pneumonia for age 24 to 59 months'
         ),
        'rr_severe_pneum_HIV': Parameter
        (Types.REAL,
         'relative rate of severe pneumonia for HIV positive status'
         ),
        'rr_severe_pneum_malnutrition': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for severe acute malnutrition'
         ),
        'rr_severe_pneum_IAP': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for indoor air pollution'
         ),
        'r_progress_to_severe_pneum': Parameter
        (Types.REAL,
         'probability of progressing from pneumonia to severe pneumonia among children aged 2-11 months,'
         ' HIV negative, normal weight, no SAM, no indoor air pollution'
         ),
        'rr_progress_severe_pneum_agelt2mo': Parameter
        (Types.REAL,
         'relative rate of progression to severe pneumonia for age <2 months'
         ),
        'rr_progress_severe_pneum_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of progression to severe pneumonia for age 12 to 23 months'
         ),
        'rr_progress_severe_pneum_age24to59mo': Parameter
        (Types.REAL, 'relative rate of progression to severe pneumonia for age 24 to 59 months'
         ),
        'rr_progress_severe_pneum_HIV': Parameter
        (Types.REAL,
         'relative risk of progression to severe pneumonia for HIV positive status'
         ),
        'rr_progress_severe_pneum_malnutrition': Parameter
        (Types.REAL,
         'relative rate of progression to severe pneumonia for severe acute malnutrition'
         ),
        'rr_progress_severe_pneum_IAP': Parameter
        (Types.REAL,
         'relative risk of progression to severe pneumonia for indoor air pollution'
         ),
        'r_death_pneumonia': Parameter
        (Types.REAL,
         'death rate from pneumonia among children aged 2-11 months, '
         'HIV negative, no SAM, no indoor air pollution'
         ),
        'rr_death_pneumonia_agelt2months': Parameter
        (Types.REAL,
         'relative risk of common cold for age < 2 months'
         ),
        'rr_death_pneumonia_age12to23mo': Parameter
        (Types.REAL,
         'relative risk of death from pneumonia for age 12 to 23 months'
         ),
        'rr_death_pneumonia_age24to59mo': Parameter
        (Types.REAL,
         'relative risk of death from pneumonia for age 24 to 59 months'
         ),
        'rr_death_pneumonia_HIV': Parameter
        (Types.REAL,
         'relative risk of death from pneumonia for HIV positive'
         ),
        'rr_death_pneumonia_malnutrition': Parameter
        (Types.REAL,
         'relative risk of death from pneumonia for severe acute malnutrition'
         ),
        'rr_death_pneumonia_IAP': Parameter
        (Types.REAL,
         'relative risk of death from pneumonia for indoor air pollution'
         ),
        'rr_death_pneumonia_treatment_adherence': Parameter
        (Types.REAL,
         'relative risk of death from pneumonia for completed treatment'
         ),
        'r_recovery_pneumonia': Parameter
        (Types.REAL,
         'recovery rate from pneumonia among children aged 2-11 months, '
         'HIV negative, no SAM, no indoor air pollution '
         ),
        'rr_recovery_pneumonia_agelt2mo': Parameter
        (Types.REAL,
         'relative rate of recovery from pneumonia for age < 2 months'
         ),
        'rr_recovery_pneumonia_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of recovery from pneumonia for age between 12 to 23 months'
         ),
        'rr_recovery_pneumonia_age24to59mo': Parameter
        (Types.REAL, 'relative rate of recovery from pneumonia for age between 24 to 59 months'
         ),
        'rr_recovery_pneumonia_HIV': Parameter
        (Types.REAL,
         'relative rate of recovery from pneumonia for HIV positive status'
         ),
        'rr_recovery_pneumonia_malnutrition': Parameter
        (Types.REAL,
         'relative rate of recovery from pneumonia for acute malnutrition'
         ),
        'r_recovery_severe_pneumonia': Parameter
        (Types.REAL,
         'baseline recovery rate from severe pneumonia among children ages 2 to 11 months, '
         'HIV negative, no SAM, no indoor air pollution'
         ),
        'rr_recovery_severe_pneum_agelt2mo': Parameter
        (Types.REAL,
         'relative rate of recovery from severe pneumonia for age <2 months'
         ),
        'rr_recovery_severe_pneum_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of recovery from severe pneumonia for age between 12 to 23 months'
         ),
        'rr_recovery_severe_pneum_age24to59mo': Parameter
        (Types.REAL,
         'relative rate of recovery from severe pneumonia for age between 24 to 59 months'
         ),
        'rr_recovery_severe_pneum_HIV': Parameter
        (Types.REAL,
         'relative rate of recovery from severe pneumonia for HIV positive status'
         ),
        'rr_recovery_severe_pneum_malnutrition': Parameter
        (Types.REAL,
         'relative rate of recovery from severe pneumonia for acute malnutrition'
         ),
        'init_prop_pneumonia_status': Parameter
        (Types.LIST,
         'initial proportions in ri_pneumonia_status categories '
         'for children aged 2-11 months, HIV negative, no SAM, no indoor air pollution'
         )
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'ri_pneumonia_status': Property(Types.CATEGORICAL, 'lower respiratory infection - pneumonia status',
                                        categories=['none', 'pneumonia', 'severe pneumonia']),
        'has_hiv': Property(Types.BOOL, 'temporary property - has hiv'),
        'malnutrition': Property(Types.BOOL, 'temporary property - malnutrition status'),
        'indoor_air_pollution': Property(Types.BOOL, 'temporary property - indoor air pollution'),
        'siblings': Property(Types.BOOL, 'temporary property - number of siblings'),
        'HHhandwashing': Property(Types.BOOL, 'temporary property - household handwashing'),
    }

    def read_parameters(self, data_folder):
        """ Setup parameters values used by the module
        """
        p = self.parameters

        p['base_prev_pneumonia'] = 0.2
        p['rp_pneumonia_agelt2mo'] = 1.2
        p['rp_pneumonia_age12to23mo'] = 0.8
        p['rp_pneumonia_age24to59mo'] = 0.5
        p['rp_pneumonia_HIV'] = 1.4
        p['rp_pneumonia_malnutrition'] = 1.25
        p['rp_pneumonia_IAP'] = 1.1
        p['base_incidence_pneumonia'] = 0.0015
        p['rr_pneumonia_agelt2mo'] = 1.2
        p['rr_pneumonia_age12to23mo'] = 0.8
        p['rr_pneumonia_age24to59mo'] = 0.5
        p['rr_pneumonia_HIV'] = 1.4
        p['rr_pneumonia_malnutrition'] = 1.25
        p['rr_pneumonia_IAP'] = 1.1
        p['base_prev_severe_pneumonia'] = 0.1
        p['rp_severe_pneum_agelt2mo'] = 1.3
        p['rp_severe_pneum_age12to23mo'] = 0.8
        p['rp_severe_pneum_age24to59mo'] = 0.5
        p['rp_severe_pneum_HIV'] = 1.3
        p['rp_severe_pneum_malnutrition'] = 1.3
        p['rp_severe_pneum_IAP'] = 1.1
        p['rr_severe_pneum_agelt2mo'] = 1.3
        p['rr_severe_pneum_age12to23mo'] = 0.8
        p['rr_severe_pneum_age24to59mo'] = 0.5
        p['rr_severe_pneum_HIV'] = 1.3
        p['rr_severe_pneum_malnutrition'] = 1.3
        p['rr_severe_pneum_IAP'] = 1.1
        p['r_progress_to_severe_pneumonia'] = 0.05
        p['rr_progress_severe_pneum_agelt2mo'] = 1.3
        p['rr_progress_severe_pneum_age12to23mo'] = 0.9
        p['rr_progress_severe_pneum_age24to59mo'] = 0.6
        p['rr_progress_severe_pneum_HIV'] = 1.2
        p['rr_progress_severe_pneum_malnutrition'] = 1.1
        p['rr_progress_severe_pneum_IAP'] = 1.08
        p['r_death_pneumonia'] = 0.2
        p['rr_death_pneumonia_agelt2mo'] = 1.2
        p['rr_death_pneumonia_age12to23mo'] = 0.8
        p['rr_death_pneumonia_age24to59mo'] = 0.04
        p['rr_death_pneumonia_HIV'] = 1.4
        p['rr_death_pneumonia_malnutrition'] = 1.3
        p['rr_death_pneumonia_IAP'] = 1.1
        p['rr_death_pneumonia_treatment_adherence'] = 1.4
        p['r_recovery_pneumonia'] = 0.5
        p['rr_recovery_pneumonia_agelt2mo'] = 0.3
        p['rr_recovery_pneumonia_age12to23mo'] = 0.7
        p['rr_recovery_pneumonia_age24to59mo'] = 0.8
        p['rr_recovery_pneumonia_HIV'] = 0.3
        p['rr_recovery_pneumonia_malnutrition'] = 0.4
        p['rr_recovery_pneumonia_treatment_adherence'] = 0.6
        p['r_recovery_severe_pneumonia'] = 0.2
        p['rr_recovery_severe_pneum_agelt2mo'] = 0.6
        p['rr_recovery_severe_pneum_age12to23mo'] = 1.2
        p['rr_recovery_severe_pneum_age24to59mo'] = 1.5
        p['rr_recovery_severe_pneum_HIV'] = 0.5
        p['rr_recovery_severe_pneum_malnutrition'] = 0.6
        p['init_prop_pneumonia_status'] = [0.2, 0.1]

    def initialise_population(self, population):
        """Set our property values for the initial population.
        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.
        :param population: the population of individuals
        """
        df = population.props  # a shortcut to the data-frame storing data for individuals
        m = self
        rng = m.rng

        # -------------------- DEFAULTS ------------------------------------------------------------

        df['ri_pneumonia_status'] = 'none'
        df['malnutrition'] = False
        df['has_HIV'] = False
        df['indoor_air_pollution'] = False
        df['HHhandwashing'] = False
        df['siblings'] = False

        # -------------------- ASSIGN VALUES OF RESPIRATORY INFECTION STATUS AT BASELINE -----------

        under5_idx = df.index[(df.age_years < 5) & df.is_alive]

        # create data-frame of the probabilities of ri_pneumonia_status for children
        # aged 2-11 months, HIV negative, no SAM, no indoor air pollution
        p_pneumonia_status = pd.Series(0.2, index=under5_idx)
        p_sev_pneum_status = pd.Series(0.1, index=under5_idx)

        # create probabilities of pneumonia for all age under 5
        p_pneumonia_status.loc[(df.age_exact_years < 0.1667) & df.is_alive] *= self.rp_pneumonia_agelt2mo
        p_pneumonia_status.loc[
            (df.age_exact_years >= 1) & (df.age_exact_years < 2) & df.is_alive] *= self.rp_pneumonia_age12to23mo
        p_pneumonia_status.loc[
            (df.age_exact_years >= 2) & (df.age_exact_years < 5) & df.is_alive] *= self.rp_pneumonia_age24to59mo
        p_pneumonia_status.loc[(df.has_hiv == True) & (df.age_years < 5) & df.is_alive] *= self.rp_pneumonia_HIV
        p_pneumonia_status.loc[
            (df.malnutrition == True) & (df.age_years < 5) & df.is_alive] *= self.rp_pneumonia_malnutrition
        p_pneumonia_status.loc[
            (df.indoor_air_pollution == True) & (df.age_years < 5) & df.is_alive] *= self.rp_pneumonia_IAP

        # create probabilities of severe pneumonia for all age under 5
        p_sev_pneum_status.loc[(df.age_exact_years < 0.1667) & df.is_alive] *= self.rp_severe_pneum_agelt2mo
        p_sev_pneum_status.loc[
            (df.age_exact_years >= 1) & (df.age_exact_years < 2) & df.is_alive] *= self.rp_severe_pneum_age12to23mo
        p_sev_pneum_status.loc[
            (df.age_exact_years >= 2) & (df.age_exact_years < 5) & df.is_alive] *= self.rp_severe_pneum_age24to59mo
        p_sev_pneum_status.loc[(df.has_hiv == True) & (df.age_years < 5) & df.is_alive] *= self.rp_severe_pneum_HIV
        p_sev_pneum_status.loc[
            (df.malnutrition == True) & (df.age_years < 5) & df.is_alive] *= self.rp_severe_pneum_malnutrition
        p_sev_pneum_status.loc[(df.indoor_air_pollution == True) & (
            df.age_years < 5) & df.is_alive] *= self.rp_severe_pneum_IAP

        random_draw = pd.Series(rng.random_sample(size=len(under5_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive])

        # create a temporary dataframe called dfx to hold values of probabilities and random draw
        dfx = pd.concat([p_pneumonia_status, p_sev_pneum_status, random_draw], axis=1)
        dfx.columns = ['p_pneumonia', 'p_severe_pneumonia', 'random_draw']

        dfx['p_none'] = 1 - (dfx.p_pneumonia + dfx.p_severe_pneumonia)

        # based on probabilities of being in each category, define cut-offs to determine status from
        # random draw uniform(0,1)

        # assign baseline values of ri_resp_infection_stat based on probabilities and value of random draw

        idx_none = dfx.index[dfx.p_none > dfx.random_draw]
        idx_pneumonia = dfx.index[(dfx.p_none < dfx.random_draw) &
                                  ((dfx.p_none + dfx.p_pneumonia) > dfx.random_draw)]
        idx_severe_pneumonia = dfx.index[((dfx.p_none + dfx.p_pneumonia) < dfx.random_draw) &
                                         (dfx.p_none + dfx.p_pneumonia + dfx.p_severe_pneumonia)
                                         > dfx.random_draw]

        df.loc[idx_none, 'ri_pneumonia_status'] = 'none'
        df.loc[idx_pneumonia, 'ri_pneumonia_status'] = 'pneumonia'
        df.loc[idx_severe_pneumonia, 'ri_pneumonia_status'] = 'severe pneumonia'

    def initialise_simulation(self, sim):
        """
        Get ready for simulation start.
        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        # add the basic event
        event = RespInfectionEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(weeks=1))

        # add an event to log to screen
        sim.schedule_event(RespInfectionLoggingEvent(self), sim.date + DateOffset(weeks=1))

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        This is called by the simulation whenever a new person is born.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """

        df = self.sim.population.props

        df.at[child_id, 'ri_pneumonia_status'] = 'none'

class RespInfectionEvent(RegularEvent, PopulationScopeEventMixin):
    """
    Regular event that updates all Respiratory Infection properties for population
    Regular events automatically reschedule themselves at a fixed frequency,
    and thus implement discrete timestep type behaviour. The frequency is
    specified when calling the base class constructor in our __init__ method.
    """

    def __init__(self, module):
        """schedule to run every 7 days
        note: if change this offset from 1 week need to consider code conditioning on age.years_exact
        We need to pass the frequency at which we want to occur to the base class
        constructor using super(). We also pass the module that created this event,
        so that random number generators can be scoped per-module.
        :param module: the module that created this event
        """
        super().__init__(module, frequency=DateOffset(days=7))

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props
        m = self.module
        rng = m.rng

        # ------------------- UPDATING OF LOWER RESPIRATORY INFECTION - PNEUMONIA STATUS OVER TIME -------------------

        # updating for children under 5 with current status 'none'

        pn_current_none_idx = df.index[df.is_alive & (df.ri_pneumonia_status == 'none')
                                       & (df.age_years < 5)]
        pn_current_none_agelt2mo_idx = df.index[df.is_alive & (df.ri_pneumonia_status == 'none')
                                                & (df.age_exact_years < 0.1667)]
        pn_current_none_age12to23mo_idx = df.index[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                                   (df.age_exact_years >= 1) & (df.age_exact_years < 2)]
        pn_current_none_age24to59mo_idx = df.index[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                                   (df.age_exact_years >= 2) & (df.age_exact_years < 5)]
        pn_current_none_HHhandwashing_idx = df.index[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                                     df.HHhandwashing & (df.age_years < 5)]
        pn_current_none_HIV_idx = df.index[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                           (df.has_hiv) & (df.age_years < 5)]
        pn_current_none_malnutrition_idx = df.index[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                                    df.malnutrition & (df.age_years < 5)]
        pn_current_none_siblings_idx = df.index[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                                (df.siblings) & (df.age_years < 5)]
        pn_current_none_IAP_idx = df.index[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                                            df.indoor_air_pollution & (df.age_years < 5)]
        pn_current_none_wealth_idx = df.index[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                              (df.li_wealth == '5') & (df.age_years < 5)]

        eff_prob_ri_pneumonia = pd.Series(m.base_incidence_pneumonia,
                                          index=df.index[
                                              df.is_alive & (df.ri_pneumonia_status == 'pneumonia') & (
                                                  df.age_years < 5)])

        eff_prob_ri_pneumonia.loc[pn_current_none_agelt2mo_idx] *= m.rr_pneumonia_agelt2mo
        eff_prob_ri_pneumonia.loc[pn_current_none_age12to23mo_idx] *= m.rr_pneumonia_age12to23mo
        eff_prob_ri_pneumonia.loc[pn_current_none_age24to59mo_idx] *= m.rr_pneumonia_age24to59mo
        eff_prob_ri_pneumonia.loc[pn_current_none_HHhandwashing_idx] *= m.rr_pneumonia_HHhandwashing
        eff_prob_ri_pneumonia.loc[pn_current_none_HIV_idx] *= m.rr_pneumonia_HIV
        eff_prob_ri_pneumonia.loc[pn_current_none_malnutrition_idx] *= m.rr_pneumonia_malnutrition
        eff_prob_ri_pneumonia.loc[pn_current_none_siblings_idx] *= m.rr_pneumonia_siblings
        eff_prob_ri_pneumonia.loc[pn_current_none_wealth_idx] *= m.rr_pneumonia_wealth

        random_draw = pd.Series(rng.random_sample(size=len(pn_current_none_idx)),
                                index=df.index[
                                    (df.age_years < 5) & df.is_alive & (df.ri_pneumonia_status == 'none')])

        dfx = pd.concat([eff_prob_ri_pneumonia, random_draw], axis=1)
        dfx.columns = ['eff_prob_ri_pneumonia', 'random_draw']
        idx_incident_pneumonia = dfx.index[dfx.eff_prob_ri_pneumonia > dfx.random_draw]
        df.loc[idx_incident_pneumonia, 'ri_pneumonia_status'] = 'pneumonia'

        eff_prob_ri_severe_pneumonia = pd.Series(m.base_incidence_severe_pneum,
                                                 index=df.index[
                                                     df.is_alive & (df.ri_pneumonia_status == 'severe pneumonia') & (
                                                         df.age_years < 5)])

        eff_prob_ri_severe_pneumonia.loc[pn_current_none_agelt2mo_idx] *= m.rr_severe_pneum_agelt2mo
        eff_prob_ri_severe_pneumonia.loc[pn_current_none_age12to23mo_idx] *= m.rr_severe_pneum_age12to23mo
        eff_prob_ri_severe_pneumonia.loc[pn_current_none_age24to59mo_idx] *= m.rr_severe_pneum_age24to59mo
        eff_prob_ri_severe_pneumonia.loc[pn_current_none_HHhandwashing_idx] *= m.rr_severe_pneum_HHhandwashing
        eff_prob_ri_severe_pneumonia.loc[pn_current_none_HIV_idx] *= m.rr_severe_pneum_HIV
        eff_prob_ri_severe_pneumonia.loc[pn_current_none_malnutrition_idx] *= m.rr_severe_pneum_malnutrition
        eff_prob_ri_severe_pneumonia.loc[pn_current_none_siblings_idx] *= m.rr_severe_pneum_siblings
        eff_prob_ri_severe_pneumonia.loc[pn_current_none_wealth_idx] *= m.rr_severe_pneum_wealth

        random_draw = pd.Series(rng.random_sample(size=len(pn_current_none_idx)),
                                index=df.index[
                                    (df.age_years < 5) & df.is_alive & (df.ri_pneumonia_status == 'none')])

        dfx = pd.concat([eff_prob_ri_pneumonia, random_draw], axis=1)
        dfx.columns = ['eff_prob_ri_severe_pneumonia', 'random_draw']
        idx_incident_severe_pneumonia = dfx.index[dfx.eff_prob_ri_severe_pneumonia > dfx.random_draw]
        df.loc[idx_incident_severe_pneumonia, 'ri_pneumonia_status'] = 'severe pneumonia'

        # ---------- updating for children under 5 with current status 'pneumonia' to 'severe pneumonia'----------

        pn_current_pneumonia_idx = df.index[df.is_alive & (df.ri_pneumonia_status == 'pneumonia')
                                            & (df.age_years < 5)]
        pn_current_pneumonia_agelt2mo_idx = df.index[df.is_alive & (df.ri_pneumonia_status == 'pneumonia')
                                                     & (df.age_exact_years < 0.1667)]
        pn_current_pneumonia_age12to23mo_idx = df.index[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                                                        (df.age_exact_years >= 1) & (df.age_exact_years < 2)]
        pn_current_pneumonia_age24to59mo_idx = df.index[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                                                        (df.age_exact_years >= 2) & (df.age_exact_years < 5)]
        pn_current_pneumonia_HHhandwashing_idx = df.index[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                                                        (df.HHhandwashing) & (df.age_years < 5)]
        pn_current_pneumonia_HIV_idx = df.index[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                                                (df.has_hiv) & (df.age_years < 5)]
        pn_current_pneumonia_malnutrition_idx = df.index[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                                                         (df.malnutrition) & (df.age_years < 5)]
        pn_current_pneumonia_IAP_idx = df.index[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                                                                 (df.indoor_air_pollution) & (df.age_years < 5)]
        pn_current_pneumonia_wealth_idx = df.index[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                                                   (df.li_wealth) & (df.age_years < 5)]

        eff_prob_prog_severe_pneumonia = pd.Series(m.r_progress_to_severe_penumonia,
                                                   index=df.index[df.is_alive & (df.ri_pneumonia_status == 'pneumonia')
                                                                  & (df.age_years < 5)])
        eff_prob_prog_severe_pneumonia.loc[pn_current_pneumonia_agelt2mo_idx] *=\
            m.rr_progress_severe_pneumonia_agelt2mo
        eff_prob_prog_severe_pneumonia.loc[pn_current_pneumonia_age12to23mo_idx] *=\
            m.rr_progress_severe_pneumonia_age_12to23mo
        eff_prob_prog_severe_pneumonia.loc[pn_current_pneumonia_age24to59mo_idx] *=\
            m.rr_progress_severe_pneumonia_age_24to59mo
        eff_prob_prog_severe_pneumonia.loc[pn_current_pneumonia_HHhandwashing_idx] *= \
            m.rr_progress_severe_pneumonia_HHhandwashing
        eff_prob_prog_severe_pneumonia.loc[pn_current_pneumonia_HIV_idx] *=\
            m.rr_progress_severe_pneumonia_HIV
        eff_prob_prog_severe_pneumonia.loc[pn_current_pneumonia_malnutrition_idx] *=\
            m.rr_progress_severe_pneumonia_malnutrition
        eff_prob_prog_severe_pneumonia.loc[pn_current_pneumonia_IAP_idx] *= \
            m.rr_progress_severe_pneumonia_indoor_air_pollution
        eff_prob_prog_severe_pneumonia.loc[pn_current_pneumonia_wealth_idx] *= \
            m.rr_progress_severe_pneumonia_wealth

        random_draw = pd.Series(rng.random_sample(size=len(pn_current_pneumonia_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive &
                                               (df.resp_infection_stat == 'pneumonia')])
        dfx = pd.concat([eff_prob_ri_severe_pneumonia, random_draw], axis=1)
        dfx.columns = ['eff_prob_prog_severe_pneumonia', 'random_draw']
        idx_ri_progress_severe_pneumonia = dfx.index[dfx.eff_prob_prog_severe_pneumonia > dfx.random_draw]
        df.loc[idx_ri_progress_severe_pneumonia, 'ri_pneumonia_status'] = 'severe pneumonia'

        # -------------------- UPDATING OF RI_PNEUMONIA_STATUS RECOVERY OVER TIME --------------------------------



        # -------------------- DEATH FROM PNEUMONIA DISEASE ---------------------------------------

        stage4_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage4')]
        random_draw = m.rng.random_sample(size=len(stage4_idx))
        df.loc[stage4_idx, 'ca_oesophageal_cancer_death'] = (random_draw < m.r_death_oesoph_cancer)

        # todo - this code dealth with centrally
        dead_oes_can_idx = df.index[df.ca_oesophageal_cancer_death]
        df.loc[dead_oes_can_idx, 'is_alive'] = False


class RespInfectionLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """Handles lifestyle logging"""

    def __init__(self, module):
        """schedule logging to repeat every 3 months
        """
        self.repeat = 3
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        # get some summary statistics
        df = population.props

        # calculate incidence of oesophageal cancer diagnosis in people aged > 60+
        # (this includes people diagnosed with dysplasia, but diagnosis rate at this stage is very low)

        incident_oes_cancer_diagnosis_agege60_idx = df.index[df.ca_incident_oes_cancer_diagnosis_this_3_month_period
                                                             & (df.age_years >= 60)]
        agege60_without_diagnosed_oes_cancer_idx = df.index[(df.age_years >= 60) & ~df.ca_oesophagus_diagnosed]

        incidence_per_year_oes_cancer_diagnosis = (4 * 100000 * len(incident_oes_cancer_diagnosis_agege60_idx)) / \
                                                  len(agege60_without_diagnosed_oes_cancer_idx)

        incidence_per_year_oes_cancer_diagnosis = round(incidence_per_year_oes_cancer_diagnosis, 3)

        #      logger.debug('%s|person_one|%s',
        #                     self.sim.date,
        #                     df.loc[0].to_dict())

        #       logger.info('%s|ca_oesophagus|%s',
        #                   self.sim.date,
        #                   df[df.is_alive].groupby(['ca_oesophagus']).size().to_dict())

        # note below remove is_alive
        #       logger.info('%s|ca_oesophagus_death|%s',
        #                   self.sim.date,
        #                   df[df.age_years >= 20].groupby(['ca_oesophageal_cancer_death']).size().to_dict())

        logger.info('%s|ca_incident_oes_cancer_diagnosis_this_3_month_period|%s',
                    self.sim.date,
                    incidence_per_year_oes_cancer_diagnosis)

#       logger.info('%s|ca_oesophagus_diagnosed|%s',
#                   self.sim.date,
#                   df[df.age_years >= 20].groupby(['ca_oesophagus', 'ca_oesophagus_diagnosed']).size().to_dict())

#       logger.info('%s|ca_oesophagus|%s',
#                   self.sim.date,
#                   df[df.is_alive].groupby(['age_range', 'ca_oesophagus']).size().to_dict())
