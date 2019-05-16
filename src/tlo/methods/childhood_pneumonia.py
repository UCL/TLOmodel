"""
Childhood Pneumonia module
Documentation: 04 - Methods Repository/Method_Child_RespiratoryInfection.xlsx
"""
import logging

import pandas as pd
from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent
from tlo.methods import demography

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ChildhoodPneumonia(Module):
    PARAMETERS = {
        'base_prev_pneumonia': Parameter
        (Types.REAL,
         'initial prevalence of non-severe pneumonia, among children aged 2-11 months,'
         'HIV negative, no SAM, not exclusively breastfeeding or continued breatfeeding, '
         'no household handwashing, no indoor air pollution, wealth level 3'
         ),
        'rp_pneumonia_agelt2mo': Parameter
        (Types.REAL,
         'relative prevalence of non-severe pneumonia for age < 2 months'
         ),
        'rp_pneumonia_age12to23mo': Parameter
        (Types.REAL,
         'relative prevalence of non-severe pneumonia for age 12 to 23 months'
         ),
        'rp_pneumonia_age24to59mo': Parameter
        (Types.REAL,
         'relative prevalence of non-severe pneumonia for age 24 to 59 months'
         ),
        'rp_pneumonia_HIV': Parameter
        (Types.REAL,
         'relative prevalence of pneumonia for HIV positive'
         ),
        'rp_pneumonia_SAM': Parameter
        (Types.REAL,
         'relative prevalence of non-severe pneumonia for severe acute malnutrition'
         ),
        'rp_pneumonia_excl_breast': Parameter
        (Types.REAL,
         'relative prevalence of non-severe pneumonia for exclusive breastfeeding upto 6 months'
         ),
        'rp_pneumonia_cont_breast': Parameter
        (Types.REAL,
         'relative prevalence of non-severe pneumonia for continued breastfeeding upto 23 months'
         ),
        'rp_pneumonia_HHhandwashing': Parameter
        (Types.REAL,
         'relative prevalence of non-severe pneumonia for household handwashing'
         ),
        'rp_pneumonia_IAP': Parameter
        (Types.REAL,
         'relative prevalence of non-severe pneumonia for indoor air pollution'
         ),
        'rp_pneumonia_wealth1': Parameter
        (Types.REAL,
         'relative prevalence of non-severe pneumonia for wealth level 1'
         ),
        'rp_pneumonia_wealth2': Parameter
        (Types.REAL,
         'relative prevalence of non-severe pneumonia for wealth level 2'
         ),
        'rp_pneumonia_wealth4': Parameter
        (Types.REAL,
         'relative prevalence of non-severe pneumonia for wealth level 4'
         ),
        'rp_pneumonia_wealth5': Parameter
        (Types.REAL,
         'relative prevalence of non-severe pneumonia for wealth level 5'
         ),
        'base_incidence_pneumonia': Parameter
        (Types.REAL,
         'baseline incidence of non-severe pneumonia, among children aged 2-11 months, '
         'HIV negative, no SAM, not exclusively breastfeeding or continued breatfeeding, '
         'no household handwashing, no indoor air pollution, wealth level 3'
         ),
        'rr_pneumonia_agelt2mo': Parameter
        (Types.REAL,
         'relative rate of non-severe pneumonia for age < 2 months'
         ),
        'rr_pneumonia_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of non-severe pneumonia for age 12 to 23 months'
         ),
        'rr_pneumonia_age24to59mo': Parameter
        (Types.REAL,
         'relative rate of non-severe pneumonia for age 24 to 59 months'
         ),
        'rr_pneumonia_HIV': Parameter
        (Types.REAL,
         'relative rate of non-severe pneumonia for HIV positive'
         ),
        'rr_pneumonia_SAM': Parameter
        (Types.REAL,
         'relative rate of non-severe pneumonia for severe acute malnutrition'
         ),
        'rr_pneumonia_excl_breast': Parameter
        (Types.REAL,
         'relative rate of non-severe pneumonia for exclusive breastfeeding upto 6 months'
         ),
        'rr_pneumonia_cont_breast': Parameter
        (Types.REAL,
         'relative rate of non-severe pneumonia for continued breastfeeding upto 23 months'
         ),
        'rr_pneumonia_HHhandwashing': Parameter
        (Types.REAL,
         'relative rate of non-severe pneumonia for household handwashing'
         ),
        'rr_pneumonia_IAP': Parameter
        (Types.REAL,
         'relative rate of non-severe pneumonia for indoor air pollution'
         ),
        'rr_pneumonia_wealth1': Parameter
        (Types.REAL,
         'relative rate of non-severe pneumonia for wealth level 1'
         ),
        'rr_pneumonia_wealth2': Parameter
        (Types.REAL,
         'relative rate of non-severe pneumonia for wealth level 2'
         ),
        'rr_pneumonia_wealth4': Parameter
        (Types.REAL,
         'relative rate of non-severe pneumonia for wealth level 4'
         ),
        'rr_pneumonia_wealth5': Parameter
        (Types.REAL,
         'relative rate of non-severe pneumonia for wealth level 5'
         ),
        'base_prev_severe_pneumonia': Parameter
        (Types.REAL,
         'initial prevalence of severe pneumonia, among children aged 2-11 months,'
         'HIV negative, no SAM, not exclusively breastfeeding or continued breastfeeding, '
         'no household handwashing, no indoor air pollution, wealth level 3'
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
        'rp_severe_pneum_SAM': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for severe acute malnutrition'
         ),
        'rp_severe_pneum_excl_breast': Parameter
        (Types.REAL,
         'relative prevalence of non-severe pneumonia for exclusive breastfeeding upto 6 months'
         ),
        'rp_severe_pneum_cont_breast': Parameter
        (Types.REAL,
         'relative prevalence of non-severe pneumonia for continued breastfeeding upto 23 months'
         ),
        'rp_severe_pneum_HHhandwashing': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for household handwashing'
         ),
        'rp_severe_pneum_IAP': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for indoor air pollution'
         ),
        'rp_severe_pneum_wealth1': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for wealth level 1'
         ),
        'rp_severe_pneum_wealth2': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for wealth level 2'
         ),
        'rp_severe_pneum_wealth4': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for wealth level 4'
         ),
        'rp_severe_pneum_wealth5': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for wealth level 5'
         ),
        'base_incidence_severe_pneum': Parameter
        (Types.REAL,
         'baseline incidence of severe pneumonia, among children aged 2-11 months, '
         'HIV negative, no SAM, not exclusively breastfeeding or continued breastfeeding, '
         'no household handwashing, no indoor air pollution, wealth level 3'
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
        'rr_severe_pneum_SAM': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for severe acute malnutrition'
         ),
        'rr_severe_pneum_excl_breast': Parameter
        (Types.REAL,
         'relative rate of non-severe pneumonia for exclusive breastfeeding upto 6 months'
         ),
        'rr_severe_pneum_cont_breast': Parameter
        (Types.REAL,
         'relative rate of non-severe pneumonia for continued breastfeeding upto 23 months'
         ),
        'rr_severe_pneum_HHhandwashing': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for household handwashing'
         ),
        'rr_severe_pneum_IAP': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for indoor air pollution'
         ),
        'rr_severe_pneum_wealth1': Parameter
        (Types.REAL,
         'relative rate of severe pneumonia for wealth level 1'
         ),
        'rr_severe_pneum_wealth2': Parameter
        (Types.REAL,
         'relative rate of severe pneumonia for wealth level 2'
         ),
        'rr_severe_pneum_wealth4': Parameter
        (Types.REAL,
         'relative rate of severe pneumonia for wealth level 4'
         ),
        'rr_severe_pneum_wealth5': Parameter
        (Types.REAL,
         'relative rate of severe pneumonia for wealth level 5'
         ),
        'r_progress_to_severe_pneum': Parameter
        (Types.REAL,
         'probability of progressing from non-severe to severe pneumonia among children aged 2-11 months, '
         'HIV negative, no SAM, wealth level 3'
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
        'rr_progress_severe_pneum_SAM': Parameter
        (Types.REAL,
         'relative rate of progression to severe pneumonia for severe acute malnutrition'
         ),
        'rr_progress_severe_pneum_wealth1': Parameter
        (Types.REAL,
         'relative rate of progression to severe pneumonia for wealth level 1'
         ),
        'rr_progress_severe_pneum_wealth2': Parameter
        (Types.REAL,
         'relative rate of progression to severe pneumonia for wealth level 2'
         ),
        'rr_progress_severe_pneum_wealth4': Parameter
        (Types.REAL,
         'relative rate of progression to severe pneumonia for wealth level 4'
         ),
        'rr_progress_severe_pneum_wealth5': Parameter
        (Types.REAL,
         'relative rate of progression to severe pneumonia for wealth level 5'
         ),
        'r_death_pneumonia': Parameter
        (Types.REAL,
         'death rate from pneumonia disease among children aged 2-11 months, '
         'HIV negative, no SAM, wealth level 3 '
         ),
        'rr_death_pneumonia_agelt2months': Parameter
        (Types.REAL,
         'relative rate of death from pneumonia disease for age < 2 months'
         ),
        'rr_death_pneumonia_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of death from pneumonia disease for age 12 to 23 months'
         ),
        'rr_death_pneumonia_age24to59mo': Parameter
        (Types.REAL,
         'relative rate of death from pneumonia disease for age 24 to 59 months'
         ),
        'rr_death_pneumonia_HIV': Parameter
        (Types.REAL,
         'relative rate of death from pneumonia disease for HIV positive'
         ),
        'rr_death_pneumonia_SAM': Parameter
        (Types.REAL,
         'relative rate of death from pneumonia disease for severe acute malnutrition'
         ),
        'rr_death_pneumonia_wealth1': Parameter
        (Types.REAL,
         'relative rate of death from pneumonia disease for wealth level 1'
         ),
        'rr_death_pneumonia_wealth2': Parameter
        (Types.REAL,
         'relative rate of death from pneumonia disease for wealth level 2'
         ),
        'rr_death_pneumonia_wealth4': Parameter
        (Types.REAL,
         'relative rate of death from pneumonia disease for wealth level 4'
         ),
        'rr_death_pneumonia_wealth5': Parameter
        (Types.REAL,
         'relative rate of death from pneumonia disease for wealth level 5'
         ),
        'init_prop_pneumonia_status': Parameter
        (Types.LIST,
         'initial proportions in ri_pneumonia_status categories '
         'for children aged 2-11 months, HIV negative, no SAM, '
         'not exclusively breastfeeding or continued breastfeeding, '
         'no household handwashing, no indoor air pollution, wealth level 3'
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
        'exclusive_breastfeeding': Property(Types.BOOL, 'temporary property - exclusive breastfeeding upto 6 mo'),
        'continued_breastfeeding': Property(Types.BOOL, 'temporary property - continued breastfeeding 6mo-2years'),
        'ri_pneumonia_death': Property(Types.BOOL, 'death from pneumonia disease')
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
        p['rp_pneumonia_SAM'] = 1.25
        p['rp_pneumonia_excl_breast'] = 0.5
        p['rp_pneumonia_cont_breast'] = 0.7
        p['rp_pneumonia_HHhandwashing'] = 0.5
        p['rp_pneumonia_IAP'] = 1.1
        p['rp_pneumonia_wealth1'] = 0.8
        p['rp_pneumonia_wealth2'] = 0.9
        p['rp_pneumonia_wealth4'] = 1.2
        p['rp_pneumonia_wealth5'] = 1.3
        p['base_incidence_pneumonia'] = 0.5
        p['rr_pneumonia_agelt2mo'] = 1.2
        p['rr_pneumonia_age12to23mo'] = 0.8
        p['rr_pneumonia_age24to59mo'] = 0.5
        p['rr_pneumonia_HIV'] = 1.4
        p['rr_pneumonia_SAM'] = 1.25
        p['rr_pneumonia_excl_breast'] = 0.6
        p['rr_pneumonia_cont_breast'] = 0.8
        p['rr_pneumonia_HHhandwashing'] = 0.5
        p['rr_pneumonia_IAP'] = 1.1
        p['rr_pneumonia_wealth1'] = 0.8
        p['rr_pneumonia_wealth2'] = 0.9
        p['rr_pneumonia_wealth4'] = 1.2
        p['rr_pneumonia_wealth5'] = 1.3
        p['base_prev_severe_pneumonia'] = 0.5
        p['rp_severe_pneum_agelt2mo'] = 1.3
        p['rp_severe_pneum_age12to23mo'] = 0.8
        p['rp_severe_pneum_age24to59mo'] = 0.5
        p['rp_severe_pneum_HIV'] = 1.3
        p['rp_severe_pneum_SAM'] = 1.3
        p['rp_severe_pneum_excl_breast'] = 0.5
        p['rp_severe_pneum_cont_breast'] = 0.7
        p['rp_severe_pneum_HHhandwashing'] = 0.8
        p['rp_severe_pneum_IAP'] = 1.1
        p['rp_severe_pneum_wealth1'] = 0.8
        p['rp_severe_pneum_wealth2'] = 0.9
        p['rp_severe_pneum_wealth4'] = 1.1
        p['rp_severe_pneum_wealth5'] = 1.2
        p['base_incidence_severe_pneum'] = 0.5
        p['rr_severe_pneum_agelt2mo'] = 1.3
        p['rr_severe_pneum_age12to23mo'] = 0.8
        p['rr_severe_pneum_age24to59mo'] = 0.5
        p['rr_severe_pneum_HIV'] = 1.3
        p['rr_severe_pneum_SAM'] = 1.3
        p['rr_severe_pneum_excl_breast'] = 0.6
        p['rr_severe_pneum_cont_breast'] = 0.8
        p['rr_severe_pneum_HHhandwashing'] = 0.3
        p['rr_severe_pneum_IAP'] = 1.1
        p['rr_severe_pneum_wealth1'] = 0.8
        p['rr_severe_pneum_wealth2'] = 0.9
        p['rr_severe_pneum_wealth4'] = 1.1
        p['rr_severe_pneum_wealth5'] = 1.2
        p['r_progress_to_severe_pneum'] = 0.05
        p['rr_progress_severe_pneum_agelt2mo'] = 1.3
        p['rr_progress_severe_pneum_age12to23mo'] = 0.9
        p['rr_progress_severe_pneum_age24to59mo'] = 0.6
        p['rr_progress_severe_pneum_HIV'] = 1.2
        p['rr_progress_severe_pneum_SAM'] = 1.1
        p['rr_progress_severe_pneum_wealth1'] = 0.8
        p['rr_progress_severe_pneum_wealth2'] = 0.9
        p['rr_progress_severe_pneum_wealth4'] = 1.1
        p['rr_progress_severe_pneum_wealth5'] = 1.3
        p['r_death_pneumonia'] = 0.5
        p['rr_death_pneumonia_agelt2mo'] = 1.2
        p['rr_death_pneumonia_age12to23mo'] = 0.8
        p['rr_death_pneumonia_age24to59mo'] = 0.04
        p['rr_death_pneumonia_HIV'] = 1.4
        p['rr_death_pneumonia_SAM'] = 1.3
        p['rr_death_pneumonia_wealth1'] = 0.7
        p['rr_death_pneumonia_wealth2'] = 0.8
        p['rr_death_pneumonia_wealth4'] = 1.2
        p['rr_death_pneumonia_wealth5'] = 1.3
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
        df['exclusive_breastfeeding'] = False
        df['continued_breastfeeding'] = False
        df['ri_pneumonia_death'] = False

        # -------------------- ASSIGN VALUES OF RESPIRATORY INFECTION STATUS AT BASELINE -------------

        under5_idx = df.index[(df.age_years < 5) & df.is_alive]

        # create data-frame of the probabilities of ri_pneumonia_status for children
        # aged 2-11 months, HIV negative, no SAM, no indoor air pollution
        p_pneumonia_status = pd.Series(self.init_prop_pneumonia_status[0], index=under5_idx)
        p_sev_pneum_status = pd.Series(self.init_prop_pneumonia_status[1], index=under5_idx)

        # create probabilities of pneumonia for all age under 5
        p_pneumonia_status.loc[
            (df.age_exact_years < 0.1667) & df.is_alive] *= self.rp_pneumonia_agelt2mo
        p_pneumonia_status.loc[
            (df.age_exact_years >= 1) & (df.age_exact_years < 2) & df.is_alive] *= self.rp_pneumonia_age12to23mo
        p_pneumonia_status.loc[
            (df.age_exact_years >= 2) & (df.age_exact_years < 5) & df.is_alive] *= self.rp_pneumonia_age24to59mo
        p_pneumonia_status.loc[
            (df.has_hiv == True) & (df.age_years < 5) & df.is_alive] *= self.rp_pneumonia_HIV
        p_pneumonia_status.loc[
            (df.malnutrition == True) & (df.age_years < 5) & df.is_alive] *= self.rp_pneumonia_SAM
        p_pneumonia_status.loc[
            (df.exclusive_breastfeeding == True) & (df.age_exact_years <= 0.5) & df.is_alive] \
            *= self.rp_pneumonia_excl_breast
        p_pneumonia_status.loc[
            (df.continued_breastfeeding == True) & (df.age_exact_years > 0.5) & (df.age_exact_years < 2) &
            df.is_alive] *= self.rp_pneumonia_cont_breast
        p_pneumonia_status.loc[
            (df.li_wood_burn_stove == False) & (df.age_years < 5) & df.is_alive] *= self.rp_pneumonia_IAP
        p_pneumonia_status.loc[
            (df.li_wealth == 1) & (df.age_years < 5) & df.is_alive] *= self.rp_pneumonia_wealth1
        p_pneumonia_status.loc[
            (df.li_wealth == 2) & (df.age_years < 5) & df.is_alive] *= self.rp_pneumonia_wealth2
        p_pneumonia_status.loc[
            (df.li_wealth == 4) & (df.age_years < 5) & df.is_alive] *= self.rp_pneumonia_wealth4
        p_pneumonia_status.loc[
            (df.li_wealth == 5) & (df.age_years < 5) & df.is_alive] *= self.rp_pneumonia_wealth5

        # create probabilities of severe pneumonia for all age under 5
        p_sev_pneum_status.loc[
            (df.age_exact_years < 0.1667) & df.is_alive] *= self.rp_severe_pneum_agelt2mo
        p_sev_pneum_status.loc[
            (df.age_exact_years >= 1) & (df.age_exact_years < 2) & df.is_alive] *= self.rp_severe_pneum_age12to23mo
        p_sev_pneum_status.loc[
            (df.age_exact_years >= 2) & (df.age_exact_years < 5) & df.is_alive] *= self.rp_severe_pneum_age24to59mo
        p_sev_pneum_status.loc[
            (df.has_hiv == True) & (df.age_years < 5) & df.is_alive] *= self.rp_severe_pneum_HIV
        p_sev_pneum_status.loc[
            (df.malnutrition == True) & (df.age_years < 5) & df.is_alive] *= self.rp_severe_pneum_SAM
        p_sev_pneum_status.loc[
            (df.exclusive_breastfeeding == True) & (df.age_exact_years <= 0.5) & df.is_alive] \
            *= self.rp_severe_pneum_excl_breast
        p_sev_pneum_status.loc[
            (df.continued_breastfeeding == True) & (df.age_exact_years > 0.5) & (df.age_exact_years < 2) &
            df.is_alive] *= self.rp_severe_pneum_cont_breast
        p_sev_pneum_status.loc[
            (df.li_wood_burn_stove == False) & (df.age_years < 5) & df.is_alive] *= self.rp_severe_pneum_IAP
        p_sev_pneum_status.loc[
            (df.li_wealth == 1) & (df.age_years < 5) & df.is_alive] *= self.rp_severe_pneum_wealth1
        p_sev_pneum_status.loc[
            (df.li_wealth == 2) & (df.age_years < 5) & df.is_alive] *= self.rp_severe_pneum_wealth2
        p_sev_pneum_status.loc[
            (df.li_wealth == 4) & (df.age_years < 5) & df.is_alive] *= self.rp_severe_pneum_wealth4
        p_sev_pneum_status.loc[
            (df.li_wealth == 5) & (df.age_years < 5) & df.is_alive] *= self.rp_severe_pneum_wealth5

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
        idx_pneumonia = dfx.index[(dfx.p_none < dfx.random_draw) & ((dfx.p_none + dfx.p_pneumonia) > dfx.random_draw)]
        idx_severe_pneumonia = dfx.index[((dfx.p_none + dfx.p_pneumonia) < dfx.random_draw) &
                                         (dfx.p_none + dfx.p_pneumonia + dfx.p_severe_pneumonia) > dfx.random_draw]

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
        event_pneumonia = PneumoniaEvent(self)
        sim.schedule_event(event_pneumonia, sim.date + DateOffset(weeks=2))

        # add an event to log to screen
        sim.schedule_event(PneumoniaLoggingEvent(self), sim.date + DateOffset(weeks=2))

        # add the basic event
        event_severe_pneumonia = SeverePneumoniaEvent(self)
        sim.schedule_event(event_severe_pneumonia, sim.date + DateOffset(weeks=2))

        # add an event to log to screen
        sim.schedule_event(SeverePneumoniaLoggingEvent(self), sim.date + DateOffset(weeks=2))

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        This is called by the simulation whenever a new person is born.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        pass


class PneumoniaEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(weeks=2))

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props
        m = self.module
        rng = m.rng

        # ------------------- UPDATING OF LOWER RESPIRATORY INFECTION - PNEUMONIA STATUS OVER TIME -------------------

        # updating for children under 5 with current status 'none' to non-severe pneumonia

        eff_prob_ri_pneumonia = pd.Series(m.base_incidence_pneumonia,
                                          index=df.index[
                                              df.is_alive & (df.ri_pneumonia_status == 'none') & (
                                                  df.age_years < 5)])

        eff_prob_ri_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                  (df.age_exact_years < 0.1667)] *= m.rr_pneumonia_agelt2mo
        eff_prob_ri_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                     (df.age_exact_years >= 1) & (df.age_exact_years < 2)] *= m.rr_pneumonia_age12to23mo
        eff_prob_ri_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                     (df.age_exact_years >= 2) & (df.age_exact_years < 5)] *= m.rr_pneumonia_age24to59mo
        eff_prob_ri_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                     df.li_no_access_handwashing == False & (df.age_years < 5)] *= m.rr_pneumonia_HHhandwashing
        eff_prob_ri_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                     (df.has_hiv == True) & (df.age_years < 5)] *= m.rr_pneumonia_HIV
        eff_prob_ri_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                     df.malnutrition == True & (df.age_years < 5)] *= m.rr_pneumonia_SAM
        eff_prob_ri_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                     df.exclusive_breastfeeding == True & (df.age_exact_years <= 0.5)] *= m.rr_pneumonia_excl_breast
        eff_prob_ri_pneumonia.loc[df.is_alive & (df.continued_breastfeeding == True) &
                                  (df.age_exact_years > 0.5) & (df.age_exact_years < 2)] *= m.rr_pneumonia_cont_breast
        eff_prob_ri_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                     df.li_wood_burn_stove == False & (df.age_years < 5)] *= m.rr_pneumonia_IAP
        eff_prob_ri_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                     (df.li_wealth == 1) & (df.age_years < 5)] *= m.rr_pneumonia_wealth1
        eff_prob_ri_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                     (df.li_wealth == 2) & (df.age_years < 5)] *= m.rr_pneumonia_wealth2
        eff_prob_ri_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                     (df.li_wealth == 4) & (df.age_years < 5)] *= m.rr_pneumonia_wealth4
        eff_prob_ri_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                     (df.li_wealth == 5) & (df.age_years < 5)] *= m.rr_pneumonia_wealth5

        pn_current_none_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'none') & (df.age_years < 5)]

        random_draw_01 = pd.Series(rng.random_sample(size=len(pn_current_none_idx)),
                                   index=df.index[
                                       (df.age_years < 5) & df.is_alive & (df.ri_pneumonia_status == 'none')])

        dfx = pd.concat([eff_prob_ri_pneumonia, random_draw_01], axis=1)
        dfx.columns = ['eff_prob_ri_pneumonia', 'random_draw_01']

        idx_incident_pneumonia = dfx.index[dfx.eff_prob_ri_pneumonia > dfx.random_draw_01]

        df.loc[idx_incident_pneumonia, 'ri_pneumonia_status'] = 'pneumonia'

        # ---------- updating for children under 5 with current status 'pneumonia' to 'severe pneumonia'----------

        eff_prob_prog_severe_pneumonia = pd.Series(m.r_progress_to_severe_pneum,
                                                   index=df.index[df.is_alive & (df.ri_pneumonia_status == 'pneumonia')
                                                                  & (df.age_years < 5)])

        eff_prob_prog_severe_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                                           (df.age_years < 5)] *= m.rr_progress_severe_pneum_agelt2mo
        eff_prob_prog_severe_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                                           (df.age_exact_years >= 1) & (
                                                   df.age_exact_years < 2)] *= m.rr_progress_severe_pneum_age12to23mo
        eff_prob_prog_severe_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                                           (df.age_exact_years >= 2) & (df.age_exact_years < 5)] *= \
            m.rr_progress_severe_pneum_age24to59mo
        eff_prob_prog_severe_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                                           df.has_hiv == True & (df.age_years < 5)] *= \
            m.rr_progress_severe_pneum_HIV
        eff_prob_prog_severe_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                                           df.malnutrition == True & (df.age_years < 5)] *= \
            m.rr_progress_severe_pneum_SAM
        eff_prob_prog_severe_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                                           (df.li_wealth == 1) & (df.age_years < 5)] *= \
            m.rr_progress_severe_pneum_wealth1
        eff_prob_prog_severe_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                                           (df.li_wealth == 2) & (df.age_years < 5)] *= \
            m.rr_progress_severe_pneum_wealth2
        eff_prob_prog_severe_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                                           (df.li_wealth == 4) & (df.age_years < 5)] *= \
            m.rr_progress_severe_pneum_wealth4
        eff_prob_prog_severe_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                                           (df.li_wealth == 5) & (df.age_years < 5)] *= \
            m.rr_progress_severe_pneum_wealth5

        pn_current_pneumonia_idx = df.index[df.is_alive & (df.age_years < 5) & (df.ri_pneumonia_status == 'pneumonia')]

        random_draw_03 = pd.Series(rng.random_sample(size=len(pn_current_pneumonia_idx)),
                                   index=df.index[(df.age_years < 5) & df.is_alive &
                                                  (df.ri_pneumonia_status == 'pneumonia')])
        dfx = pd.concat([eff_prob_prog_severe_pneumonia, random_draw_03], axis=1)
        dfx.columns = ['eff_prob_prog_severe_pneumonia', 'random_draw_03']
        idx_ri_progress_severe_pneumonia = dfx.index[dfx.eff_prob_prog_severe_pneumonia > dfx.random_draw_03]
        df.loc[idx_ri_progress_severe_pneumonia, 'ri_pneumonia_status'] = 'severe pneumonia'

        # -------------------- UPDATING OF NON-SEVERE PNEUMONIA RECOVERY OVER TIME --------------------------------
        # recovery from non-severe pneumonia

        after_progression_pneumonia_idx = df.index[df.is_alive &
                                                   (df.ri_pneumonia_status == 'pneumonia') & (df.age_years < 5)]

        if self.sim.date + DateOffset(weeks=2):
            df.loc[after_progression_pneumonia_idx, 'ri_pneumonia_status'] == 'none'


class SeverePneumoniaEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(weeks=2))

    def apply(self, population):
        df = population.props
        m = self.module
        rng = m.rng

        # updating for children under 5 with current status 'none' to severe pneumonia

        eff_prob_ri_severe_pneumonia = pd.Series(m.base_incidence_severe_pneum,
                                                  index=df.index[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                                                 (df.age_years < 5)])
        eff_prob_ri_severe_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                         (df.age_exact_years < 0.1667)] *= m.rr_severe_pneum_agelt2mo
        eff_prob_ri_severe_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                         (df.age_exact_years >= 1) & (
                                                  df.age_exact_years < 2)] *= m.rr_severe_pneum_age12to23mo
        eff_prob_ri_severe_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                         (df.age_exact_years >= 2) & (
                                                  df.age_exact_years < 5)] *= m.rr_severe_pneum_age24to59mo
        eff_prob_ri_severe_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                         df.li_no_access_handwashing == False & (
                                                  df.age_years < 5)] *= m.rr_severe_pneum_HHhandwashing
        eff_prob_ri_severe_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                         (df.has_hiv == True) & (df.age_years < 5)] *= m.rr_severe_pneum_HIV
        eff_prob_ri_severe_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                         df.malnutrition == True & (df.age_years < 5)] *= m.rr_severe_pneum_SAM
        eff_prob_ri_severe_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                         df.exclusive_breastfeeding == True & (
                                                 df.age_exact_years <= 0.5)] *= m.rr_severe_pneum_excl_breast
        eff_prob_ri_severe_pneumonia.loc[df.is_alive & (df.continued_breastfeeding == True) &
                                         (df.age_exact_years > 0.5) & (
                                                  df.age_exact_years < 2)] *= m.rr_severe_pneum_cont_breast
        eff_prob_ri_severe_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                         df.li_wood_burn_stove == False & (df.age_years < 5)] *= m.rr_severe_pneum_IAP
        eff_prob_ri_severe_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                         (df.li_wealth == 1) & (df.age_years < 5)] *= m.rr_pneumonia_wealth1
        eff_prob_ri_severe_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                         (df.li_wealth == 2) & (df.age_years < 5)] *= m.rr_pneumonia_wealth2
        eff_prob_ri_severe_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                         (df.li_wealth == 4) & (df.age_years < 5)] *= m.rr_pneumonia_wealth4
        eff_prob_ri_severe_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                         (df.li_wealth == 5) & (df.age_years < 5)] *= m.rr_pneumonia_wealth5

        pn_current_none_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'none') & (df.age_years < 5)]

        random_draw_02 = pd.Series(rng.random_sample(size=len(pn_current_none_idx)),
                                    index=df.index[
                                        (df.age_years < 5) & df.is_alive & (df.ri_pneumonia_status == 'none')])

        dfx = pd.concat([eff_prob_ri_severe_pneumonia, random_draw_02], axis=1)
        dfx.columns = ['eff_prob_ri_severe_pneumonia', 'random_draw_02']

        idx_incident_severe_pneumonia = dfx.index[dfx.eff_prob_ri_severe_pneumonia > dfx.random_draw_02]

        df.loc[idx_incident_severe_pneumonia, 'ri_pneumonia_status'] = 'severe pneumonia'

        # ---------------------------- DEATH FROM SEVERE PNEUMONIA DISEASE ---------------------------------------

        eff_prob_death_pneumonia = \
            pd.Series(m.r_death_pneumonia,
                      index=df.index[df.is_alive & (df.ri_pneumonia_status == 'severe pneumonia') & (df.age_years < 5)])
        eff_prob_death_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'severe pneumonia') &
                     (df.age_years < 5)] *= m.rr_death_pneumonia_agelt2mo
        eff_prob_death_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'severe pneumonia') &
                     (df.age_exact_years >= 1) & (df.age_exact_years < 2)] *= \
            m.rr_death_pneumonia_age12to23mo
        eff_prob_death_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'severe pneumonia') &
                     (df.age_exact_years >= 2) & (df.age_exact_years < 5)] *= \
            m.rr_death_pneumonia_age24to59mo
        eff_prob_death_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'severe pneumonia') &
                     df.has_hiv == True & (df.age_years < 5)] *= \
            m.rr_death_pneumonia_HIV
        eff_prob_death_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'severe pneumonia') &
                     df.malnutrition == True & (df.age_years < 5)] *= \
            m.rr_death_pneumonia_SAM

        pn1_current_severe_pneumonia_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'severe pneumonia') & (df.age_years < 5)]

        random_draw_05 = pd.Series(rng.random_sample(size=len(pn1_current_severe_pneumonia_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive &
                                               (df.ri_pneumonia_status == 'severe pneumonia')])

        dfx = pd.concat([eff_prob_death_pneumonia, random_draw_05], axis=1)
        dfx.columns = ['eff_prob_death_pneumonia', 'random_draw_05']

        dfx['pneumonia_death'] = False
        dfx.loc[dfx.eff_prob_death_pneumonia > dfx.random_draw_05, 'pneumonia_death'] = True
        df.loc[pn1_current_severe_pneumonia_idx, 'ri_pneumonia_death'] = dfx['pneumonia_death']

        death_this_period = df.index[df.ri_pneumonia_death]
        for individual_id in death_this_period:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id, 'ChildhoodPneumonia'),
                                    self.sim.date)

        # -------------------- UPDATING OF SEVERE PNEUMONIA RECOVERY OVER TIME -------------------------------
        # recovery from severe pneumonia

        after_death_sev_pneumonia_idx = df.index[df.is_alive &
                                                 (df.ri_pneumonia_status == 'severe pneumonia') & (df.age_years < 5)]

        if self.sim.date + DateOffset(weeks=2):
            df.loc[after_death_sev_pneumonia_idx, 'ri_pneumonia_status'] == 'none'


class PneumoniaLoggingEvent(RegularEvent, PopulationScopeEventMixin):
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

        df = population.props
        m = self.module
        rng = m.rng

        all_children_pneumonia_idx = df.index[df.age_exact_years < 5 & df.ri_pneumonia_status == 'pneumonia']
        for child_p in all_children_pneumonia_idx:
            logger.info('%s|acquired_pneumonia|%s',
                        self.sim.date,
                        {
                            'child_index': child_p,
                            'progression': df.at[child_p, 'severe pneumonia'],
                        })


class SeverePneumoniaLoggingEvent(RegularEvent, PopulationScopeEventMixin):
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
