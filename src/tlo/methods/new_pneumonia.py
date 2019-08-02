"""
Childhood pneumonia module
Documentation: 04 - Methods Repository/Method_Child_RespiratoryInfection.xlsx
"""
import logging

import pandas as pd
from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class NewPneumonia(Module):
    PARAMETERS = {
        'base_incidence_pneumonia_by_RSV': Parameter
        (Types.LIST, 'incidence of pneumonia caused by Respiratory Syncytial Virus in age groups 0-11, 12-59 months'
         ),
        'base_incidence_pneumonia_by_rhinovirus': Parameter
        (Types.LIST, 'incidence of pneumonia caused by ST-ETEC in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_incidence_pneumonia_by_hMPV': Parameter
        (Types.LIST, 'incidence of pneumonia caused by campylobacter spp in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_incidence_pneumonia_by_parainfluenza': Parameter
        (Types.LIST, 'incidence of pneumonia caused by cryptosporidium in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_incidence_pneumonia_by_streptococcus': Parameter
        (Types.LIST, 'incidence of pneumonia caused by adenovirus 40/41 in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_incidence_pneumonia_by_hib': Parameter
        (Types.LIST, 'incidence of pneumonia caused by ST-ETEC in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_incidence_pneumonia_by_TB': Parameter
        (Types.LIST, 'incidence of pneumonia caused by TB in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_incidence_pneumonia_by_staph': Parameter
        (Types.LIST, 'incidence of pneumonia caused by Staphylococcus aureus in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_incidence_pneumonia_by_influenza': Parameter
        (Types.LIST, 'incidence of pneumonia caused by shigella spp in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_incidence_pneumonia_by_jirovecii': Parameter
        (Types.LIST, 'incidence of pneumonia caused by Respiratory Syncytial Virus in age groups 0-11, 12-59 months'
         ),
        'rr_ri_pneumonia_HHhandwashing': Parameter
        (Types.REAL, 'relative rate of pneumonia with household handwashing with soap'
         ),
        'rr_ri_pneumonia_HIV': Parameter
        (Types.REAL, 'relative rate of pneumonia for HIV positive status'
         ),
        'rr_ri_pneumonia_SAM': Parameter
        (Types.REAL, 'relative rate of pneumonia for severe malnutrition'
         ),
        'rr_ri_pneumonia_excl_breast': Parameter
        (Types.REAL, 'relative rate of pneumonia for exclusive breastfeeding upto 6 months'
         ),
        'rr_ri_pneumonia_cont_breast': Parameter
        (Types.REAL, 'relative rate of pneumonia for continued breastfeeding 6 months to 2 years'
         ),
        'rr_ri_pneumonia_indoor_air_pollution': Parameter
        (Types.REAL, 'relative rate of pneumonia for indoor air pollution'
         ),
        'rr_ri_pneumonia_pneumococcal_vaccine': Parameter
        (Types.REAL, 'relative rate of pneumonia for pneumonococcal vaccine'
         ),
        'rr_ri_pneumonia_hib#_vaccine': Parameter
        (Types.REAL, 'relative rate of pneumonia for hib vaccine'
         ),
    }

    PROPERTIES = {
        'ri_pneumonia_status': Property(Types.BOOL, 'symptomatic ALRI - pneumonia disease'
                                        ),
        'ri_pneumonia_severity': Property
        (Types.CATEGORICAL, 'severity of pneumonia disease',
         categories=['none', 'non-severe pneumonia', 'severe pneumonia', 'very severe pneumonia']
         ),
        'ri_pneumonia_pathogen': Property
        (Types.CATEGORICAL, 'attributable pathogen for pneumonia',
         categories=['RSV', 'rhinovirus', 'hMPV', 'parainfluenza', 'streptococcus',
                     'hib', 'TB', 'staph', 'P. jirovecii']
         ),
        'has_hiv': Property(Types.BOOL, 'temporary property - has hiv'),
        'malnutrition': Property(Types.BOOL, 'temporary property - malnutrition status'),
        'exclusive_breastfeeding': Property(Types.BOOL, 'temporary property - exclusive breastfeeding upto 6 mo'),
        'continued_breastfeeding': Property(Types.BOOL, 'temporary property - continued breastfeeding 6mo-2years'),
        'pneumococcal_vaccination': Property(Types.BOOL, 'temporary property - streptococcus pneumoniae vaccine'),
        'hib_vaccination': Property(Types.BOOL, 'temporary property - H. influenzae type b vaccine'),
        'influenza_vaccination': Property(Types.BOOL, 'temporary property - flu vaccine'),
        # symptoms of diarrhoea for care seeking
        'ri_pneumonia_death': Property(Types.BOOL, 'death from pneumonia disease'),
        'date_of_acquiring_pneumonia': Property(Types.DATE, 'date of acquiring pneumonia infection'),
        'pn_fever': Property(Types.BOOL, 'fever from non-severe pneumonia, severe pneumonia or very severe pneumonia'),
        'pn_cough': Property(Types.BOOL, 'cough from non-severe pneumonia, severe pneumonia or very severe pneumonia'),
        'pn_difficult_breathing': Property
        (Types.BOOL, 'difficult breathing from non-severe pneumonia, severe pneumonia or very severe pneumonia'),
        'pn_fast_breathing': Property(Types.BOOL, 'fast breathing from non-severe pneumonia'),
        'pn_chest_indrawing': Property(Types.BOOL, 'chest indrawing from severe pneumonia or very severe pneumonia'),
        'pn_any_general_danger_sign': Property
        (Types.BOOL, 'any danger sign - lethargic/uncounscious, not able to drink/breastfeed, convulsions and vomiting everything'),
        'pn_stridor_in_calm_child': Property(Types.BOOL, 'stridor in calm child from very severe pneumonia')
    }

    def read_parameters(self, data_folder):
        """ Setup parameters values used by the module
        """
        p = self.parameters

        p['base_incidence_pneumonia_by_RSV'] = [0.061, 0.02225]
        p['base_incidence_pneumonia_by_rhinovirus'] = [0.061, 0.02225]
        p['base_incidence_pneumonia_by_hMPV'] = [0.061, 0.02225]
        p['base_incidence_pneumonia_by_parainfluenza'] = [0.061, 0.02225]
        p['base_incidence_pneumonia_by_streptococcus'] = [0.061, 0.02225]
        p['base_incidence_pneumonia_by_hib'] = [0.061, 0.02225]
        p['base_incidence_pneumonia_by_TB'] = [0.061, 0.02225]
        p['base_incidence_pneumonia_by_staph'] = [0.061, 0.02225]
        p['base_incidence_pneumonia_by_influenza'] = [0.061, 0.02225]
        p['base_incidence_pneumonia_by_jirovecii'] = [0.061, 0.02225]
        p['rr_ri_pneumonia_HHhandwashing'] = 0.5
        p['rr_ri_pneumonia_HIV'] = 1.4
        p['rr_ri_pneumonia_SAM'] = 1.5
        p['rr_ri_pneumonia_excl_breast'] = 0.5
        p['rr_ri_pneumonia_cont_breast'] = 0.9
        p['rr_ri_pneumonia_indoor_air_pollution'] = 0.8
        p['rr_ri_pneumonia_pneumococcal_vaccine'] = 0.5
        p['rr_ri_pneumonia_hib_vaccine'] = 0.
        p['rr_ri_pneumonia_influenza_vaccine'] = 0.5
        p['rotavirus_AWD'] = 0.95
        p['shigella_AWD'] = 0.403
        p['adenovirus_AWD'] = 0.69
        p['crypto_AWD'] = 0.95
        p['campylo_AWD'] = 0.58
        p['ETEC_AWD'] = 0.97
        p['prob_dysentery_become_persistent'] = 0.4
        p['prob_watery_diarr_become_persistent'] = 0.2
        p['rr_bec_persistent_age12to23'] = 1.4
        p['rr_bec_persistent_age24to59'] = 1.4
        p['rr_bec_persistent_HIV'] = 1.7
        p['rr_bec_persistent_SAM'] = 1.8
        p['rr_bec_persistent_excl_breast'] = 0.4
        p['rr_bec_persistent_cont_breast'] = 0.4

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
        now = self.sim.date

        # DEFAULTS
        df['ri_pneumonia_status'] = False
        df['ri_pneumonia_severity'] = 'none'
        df['malnutrition'] = False
        df['has_hiv'] = False
        df['exclusive_breastfeeding'] = False
        df['continued_breastfeeding'] = False

    def initialise_simulation(self, sim):
        """
        Get ready for simulation start.
        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        # add the basic event for dysentery ---------------------------------------------------
        event_pneumonia = PneumoniaEvent(self)
        sim.schedule_event(event_pneumonia, sim.date + DateOffset(months=3))

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        This is called by the simulation whenever a new person is born.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """

        pass

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug('This is Pneumonia, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        logger.debug('This is diarrhoea reporting my health values')

        df = self.sim.population.props
        p = self.parameters


class PneumoniaEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=3))

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props
        m = self.module
        rng = m.rng

        no_pneumonia0 = df.is_alive & (df.ri_pneumonia_status == False) & (df.age_exact_years < 1)
        no_pneumonia1 = df.is_alive & (df.ri_pneumonia_status == False) &\
                        (df.age_exact_years >= 1) & (df.age_exact_years < 5)
        no_pneumonia_under5 = df.is_alive & (df.ri_pneumonia_status == False) & (df.age_years < 5)
        current_no_pneumonia = df.index[no_pneumonia_under5]

        pneumonia_RSV0 = pd.Series(m.base_incidence_pneumonia_by_RSV[0], index=df.index[no_pneumonia0])
        pneumonia_RSV1 = pd.Series(m.base_incidence_pneumonia_by_RSV[1], index=df.index[no_pneumonia1])

        pneumonia_rhinovirus0 = pd.Series(m.base_incidence_pneumonia_by_rhinovirus[0], index=df.index[no_pneumonia0])
        pneumonia_rhinovirus1 = pd.Series(m.base_incidence_pneumonia_by_rhinovirus[1], index=df.index[no_pneumonia1])

        pneumonia_hMPV0 = pd.Series(m.base_incidence_pneumonia_by_hMPV[0], index=df.index[no_pneumonia0])
        pneumonia_hMPV1 = pd.Series(m.base_incidence_pneumonia_by_hMPV[1], index=df.index[no_pneumonia1])

        pneumonia_parainfluenza0 = pd.Series(m.base_incidence_pneumonia_by_parainfluenza[0], index=df.index[no_pneumonia0])
        pneumonia_parainfluenza1 = pd.Series(m.base_incidence_pneumonia_by_parainfluenza[1], index=df.index[no_pneumonia1])

        pneumonia_streptococcus0 = pd.Series(m.base_incidence_pneumonia_by_streptococcus[0],
                                             index=df.index[no_pneumonia0])
        pneumonia_streptococcus1 = pd.Series(m.base_incidence_pneumonia_by_streptococcus[1],
                                             index=df.index[no_pneumonia1])

        pneumonia_hib0 = pd.Series(m.base_incidence_pneumonia_by_hib[0], index=df.index[no_pneumonia0])
        pneumonia_hib1 = pd.Series(m.base_incidence_pneumonia_by_hib[1], index=df.index[no_pneumonia1])

        pneumonia_TB0 = pd.Series(m.base_incidence_pneumonia_by_TB[0], index=df.index[no_pneumonia0])
        pneumonia_TB1 = pd.Series(m.base_incidence_pneumonia_by_TB[1], index=df.index[no_pneumonia1])

        pneumonia_staph0 = pd.Series(m.base_incidence_pneumonia_by_staph[0], index=df.index[no_pneumonia0])
        pneumonia_staph1 = pd.Series(m.base_incidence_pneumonia_by_staph[1], index=df.index[no_pneumonia1])

        pneumonia_influenza0 = pd.Series(m.base_incidence_pneumonia_by_influenza[0], index=df.index[no_pneumonia0])
        pneumonia_influenza1 = pd.Series(m.base_incidence_pneumonia_by_influenza[1], index=df.index[no_pneumonia1])

        pneumonia_jirovecii0 = pd.Series(m.base_incidence_pneumonia_by_jirovecii[0], index=df.index[no_pneumonia0])
        pneumonia_jirovecii1 = pd.Series(m.base_incidence_pneumonia_by_jirovecii[1], index=df.index[no_pneumonia1])

        # concatenating plus sorting
        eff_prob_RSV = pd.concat([pneumonia_RSV0, pneumonia_RSV1], axis=0).sort_index()
        eff_prob_rhinovirus = pd.concat([pneumonia_rhinovirus0, pneumonia_rhinovirus1], axis=0).sort_index()
        eff_prob_hMPV = pd.concat([pneumonia_hMPV0, pneumonia_hMPV1], axis=0).sort_index()
        eff_prob_parainfluenza = pd.concat([pneumonia_parainfluenza0, pneumonia_parainfluenza1], axis=0).sort_index()
        eff_prob_streptococcus = pd.concat([pneumonia_streptococcus0, pneumonia_streptococcus1], axis=0).sort_index()
        eff_prob_hib = pd.concat([pneumonia_hib0, pneumonia_hib1], axis=0).sort_index()
        eff_prob_TB = pd.concat([pneumonia_TB0, pneumonia_TB1], axis=0).sort_index()
        eff_prob_staph = pd.concat([pneumonia_staph0, pneumonia_staph1], axis=0).sort_index()
        eff_prob_influenza = pd.concat([pneumonia_influenza0, pneumonia_influenza1], axis=0).sort_index()
        eff_prob_jirovecii = pd.concat([pneumonia_jirovecii0, pneumonia_jirovecii1], axis=0).sort_index()

        eff_prob_all_pathogens = \
            pd.concat([eff_prob_RSV, eff_prob_rhinovirus, eff_prob_hMPV, eff_prob_parainfluenza, eff_prob_streptococcus,
                       eff_prob_hib, eff_prob_TB, eff_prob_staph, eff_prob_influenza, eff_prob_jirovecii], axis=1)

        eff_prob_all_pathogens.loc[no_pneumonia_under5 & df.li_no_access_handwashing == False] \
            *= m.rr_gi_diarrhoea_HHhandwashing
        eff_prob_all_pathogens.loc[no_pneumonia_under5 & (df.has_hiv == True)] \
            *= m.rr_gi_diarrhoea_HIV
        eff_prob_all_pathogens.loc[no_pneumonia_under5 & df.malnutrition == True] \
            *= m.rr_gi_diarrhoea_SAM
        eff_prob_all_pathogens.loc[
            no_pneumonia_under5 & df.exclusive_breastfeeding == True & (df.age_exact_years <= 0.5)] \
            *= m.rr_ri_pneumonia_excl_breast
        eff_prob_all_pathogens.loc[
            no_pneumonia_under5 & df.continued_breastfeeding == True & (df.age_exact_years > 0.5) &
            (df.age_exact_years < 2)] *= m.rr_ri_pneumonia_cont_breast
        eff_prob_all_pathogens.loc[
            no_pneumonia_under5 & (df.li_wood_burn_stove == False)] *= m.rr_ri_pneumonia_indoor_air_pollution
        eff_prob_all_pathogens.loc[
            no_pneumonia_under5 & (df.pneumococcal_vaccination == True)] *= m.rr_ri_pneumonia_pneumococcal_vaccine
        eff_prob_all_pathogens.loc[
            no_pneumonia_under5 & (df.hib_vaccination == True)] *= m.rr_ri_pneumonia_hib_vaccine
        eff_prob_all_pathogens.loc[
            no_pneumonia_under5 & (df.influenza_vaccination == True)] *= m.rr_ri_pneumonia_influenza_vaccine

        random_draw1 = pd.Series(rng.random_sample(size=len(current_no_pneumonia)), index=current_no_pneumonia)
        pneumonia_by_RSV = eff_prob_RSV > random_draw1
        pneum_RSV_idx = eff_prob_RSV.index[pneumonia_by_RSV]
        df.loc[pneum_RSV_idx, 'ri_pneumonia_pathogen'] = 'RSV'
        df.loc[pneum_RSV_idx, 'ri_pneumonia_status'] = True

        random_draw2 = pd.Series(rng.random_sample(size=len(current_no_pneumonia)), index=current_no_pneumonia)
        pneumonia_by_rhinovirus = eff_prob_rhinovirus > random_draw2
        pneum_rhinovirus_idx = eff_prob_rhinovirus.index[pneumonia_by_rhinovirus]
        df.loc[pneum_rhinovirus_idx, 'ri_pneumonia_pathogen'] = 'rhinovirus'
        df.loc[pneum_rhinovirus_idx, 'ri_pneumonia_status'] = True

        random_draw3 = pd.Series(rng.random_sample(size=len(current_no_pneumonia)), index=current_no_pneumonia)
        pneumonia_by_hMPV = eff_prob_hMPV > random_draw3
        pneum_hMPV_idx = eff_prob_hMPV.index[pneumonia_by_hMPV]
        df.loc[pneum_hMPV_idx, 'ri_pneumonia_pathogen'] = 'hMPV'
        df.loc[pneum_hMPV_idx, 'ri_pneumonia_status'] = True

        random_draw4 = pd.Series(rng.random_sample(size=len(current_no_pneumonia)), index=current_no_pneumonia)
        pneumonia_by_parainfluenza = eff_prob_parainfluenza > random_draw4
        pneum_parainfluenza_idx = eff_prob_parainfluenza.index[pneumonia_by_parainfluenza]
        df.loc[pneum_parainfluenza_idx, 'ri_pneumonia_pathogen'] = 'parainfluenza'
        df.loc[pneum_parainfluenza_idx, 'ri_pneumonia_status'] = True

        random_draw5 = pd.Series(rng.random_sample(size=len(current_no_pneumonia)), index=current_no_pneumonia)
        pneumonia_by_streptococcus = eff_prob_streptococcus > random_draw5
        pneum_streptococcus_idx = eff_prob_streptococcus.index[pneumonia_by_streptococcus]
        df.loc[pneum_streptococcus_idx, 'ri_pneumonia_pathogen'] = 'streptococcus'
        df.loc[pneum_streptococcus_idx, 'ri_pneumonia_status'] = True

        random_draw6 = pd.Series(rng.random_sample(size=len(current_no_pneumonia)), index=current_no_pneumonia)
        pneumonia_by_hib = eff_prob_hib > random_draw6
        pneum_hib_idx = eff_prob_hib.index[pneumonia_by_hib]
        df.loc[pneum_hib_idx, 'ri_pneumonia_pathogen'] = 'hib'
        df.loc[pneum_hib_idx, 'ri_pneumonia_status'] = True

        random_draw7 = pd.Series(rng.random_sample(size=len(current_no_pneumonia)), index=current_no_pneumonia)
        pneumonia_by_TB = eff_prob_TB > random_draw7
        pneum_TB_idx = eff_prob_TB.index[pneumonia_by_TB]
        df.loc[pneum_TB_idx, 'ri_pneumonia_pathogen'] = 'TB'
        df.loc[pneum_TB_idx, 'ri_pneumonia_status'] = True

        random_draw8 = pd.Series(rng.random_sample(size=len(current_no_pneumonia)), index=current_no_pneumonia)
        pneumonia_by_staph = eff_prob_staph > random_draw8
        pneum_staph_idx = eff_prob_staph.index[pneumonia_by_staph]
        df.loc[pneum_staph_idx, 'ri_pneumonia_pathogen'] = 'staphylococcus aureus'
        df.loc[pneum_staph_idx, 'ri_pneumonia_status'] = True

        random_draw9 = pd.Series(rng.random_sample(size=len(current_no_pneumonia)), index=current_no_pneumonia)
        pneumonia_by_influenza = eff_prob_influenza > random_draw9
        pneum_influenza_idx = eff_prob_influenza.index[pneumonia_by_influenza]
        df.loc[pneum_influenza_idx, 'ri_pneumonia_pathogen'] = 'influenza'
        df.loc[pneum_influenza_idx, 'ri_pneumonia_status'] = True

        random_draw10 = pd.Series(rng.random_sample(size=len(current_no_pneumonia)), index=current_no_pneumonia)
        pneumonia_by_jirovecii = eff_prob_jirovecii > random_draw10
        pneum_jirovecii_idx = eff_prob_jirovecii.index[pneumonia_by_jirovecii]
        df.loc[pneum_jirovecii_idx, 'ri_pneumonia_pathogen'] = 'P. jirovecii'
        df.loc[pneum_jirovecii_idx, 'ri_pneumonia_status'] = True

