"""
Childhood diarrhoea module
Documentation: 04 - Methods Repository/Method_Child_EntericInfection.xlsx
"""
import logging

import numpy as np
import pandas as pd
from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent, Event, IndividualScopeEventMixin

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class NewDiarrhoea(Module):
    PARAMETERS = {
        'base_incidence_diarrhoea_by_rotavirus':
            Parameter(Types.LIST, 'incidence of diarrhoea caused by rotavirus in age groups 0-11, 12-23, 24-59 months '
                      ),
        'base_incidence_diarrhoea_by_shigella':
            Parameter(Types.LIST,
                      'incidence of diarrhoea caused by shigella spp in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_incidence_diarrhoea_by_adenovirus':
            Parameter(Types.LIST,
                      'incidence of diarrhoea caused by adenovirus 40/41 in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_incidence_diarrhoea_by_crypto':
            Parameter(Types.LIST,
                      'incidence of diarrhoea caused by cryptosporidium in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_incidence_diarrhoea_by_campylo':
            Parameter(Types.LIST,
                      'incidence of diarrhoea caused by campylobacter spp in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_incidence_diarrhoea_by_ETEC':
            Parameter(Types.LIST,
                      'incidence of diarrhoea caused by ST-ETEC in age groups 0-11, 12-23, 24-59 months'
                      ),
        'rr_gi_diarrhoea_HHhandwashing':
            Parameter(Types.REAL, 'relative rate of diarrhoea with household handwashing with soap'
                      ),
        'rr_gi_diarrhoea_improved_sanitation':
            Parameter(Types.REAL, 'relative rate of diarrhoea for improved sanitation'
                      ),
        'rr_gi_diarrhoea_clean_water':
            Parameter(Types.REAL, 'relative rate of diarrhoea for access to clean drinking water'
                      ),
        'rr_gi_diarrhoea_HIV':
            Parameter(Types.REAL, 'relative rate of diarrhoea for HIV positive status'
                      ),
        'rr_gi_diarrhoea_SAM':
            Parameter(Types.REAL, 'relative rate of diarrhoea for severe malnutrition'
                      ),
        'rr_gi_diarrhoea_excl_breast':
            Parameter(Types.REAL, 'relative rate of diarrhoea for exclusive breastfeeding upto 6 months'
                      ),
        'rr_gi_diarrhoea_cont_breast':
            Parameter(Types.REAL, 'relative rate of diarrhoea for continued breastfeeding 6 months to 2 years'
                      ),
        'rotavirus_AWD':
            Parameter(Types.REAL, 'acute diarrhoea type caused by rotavirus'
                      ),
        'shigella_AWD':
            Parameter(Types.REAL, 'acute diarrhoea type caused by shigella'
                      ),
        'adenovirus_AWD':
            Parameter(Types.REAL, 'acute diarrhoea type caused by adenovirus'
                      ),
        'crypto_AWD':
            Parameter(Types.REAL, 'acute diarrhoea type caused by cryptosporidium'
                      ),
        'campylo_AWD':
            Parameter(Types.REAL, 'acute diarrhoea type caused by campylobacter'
                      ),
        'ETEC_AWD':
            Parameter(Types.REAL, 'acute diarrhoea type caused by ST-ETEC'
                      ),
        'prob_dysentery_become_persistent':
            Parameter(Types.REAL, 'probability of dysentery becoming persistent diarrhoea'
                      ),
        'prob_watery_diarr_become_persistent':
            Parameter(Types.REAL, 'probability of acute watery diarrhoea becoming persistent diarrhoea, '
                                  'for children under 11 months, no SAM, no HIV'
                      ),
        'rr_bec_persistent_age12to23':
            Parameter(Types.REAL, 'relative rate of acute diarrhoea becoming persistent diarrhoea for age 12 to 23 months'
                      ),
        'rr_bec_persistent_age24to59':
            Parameter(Types.REAL,
                      'relative rate of acute diarrhoea becoming persistent diarrhoea for age 24 to 59 months'
                      ),
        'rr_bec_persistent_HIV':
            Parameter(Types.REAL,
                      'relative rate of acute diarrhoea becoming persistent diarrhoea for HIV positive'
                      ),
        'rr_bec_persistent_SAM':
            Parameter(Types.REAL,
                      'relative rate of acute diarrhoea becoming persistent diarrhoea for severely acute malnutrition'
                      ),
        'rr_bec_persistent_excl_breast':
            Parameter(Types.REAL,
                      'relative rate of acute diarrhoea becoming persistent diarrhoea for exclusive breastfeeding'
                      ),
        'rr_bec_persistent_cont_breast':
            Parameter(Types.REAL,
                      'relative rate of acute diarrhoea becoming persistent diarrhoea for continued breastfeeding'
                      ),
        'daly_mild_diarrhoea':
            Parameter(Types.REAL, 'DALY weight for diarrhoea with no dehydration'
                      ),
        'daly_moderate_diarrhoea':
            Parameter(Types.REAL, 'DALY weight for diarrhoea with some dehydration'
                      ),
        'daly_severe_diarrhoea':
            Parameter(Types.REAL, 'DALY weight for diarrhoea with severe dehydration'
                      ),
        # PARAMETERS FOR THE ICCM ALGORITHM PERFORMED BY HSA

        'prob_correct_id_danger_sign':
            Parameter(Types.REAL, 'probability of HSA correctly identified a general danger sign'
                      ),
        'prob_correct_id_diarrhoea_dehydration':
            Parameter(Types.REAL, 'probability of HSA correctly identified diarrhoea and dehydrayion'
                      ),
        'prob_correct_classified_diarrhoea_danger_sign':
            Parameter(Types.REAL, 'probability of HSA correctly identified diarrhoea with a danger sign'
                      ),
        'prob_correct_identified_persist_or_bloody_diarrhoea':
            Parameter(Types.REAL, 'probability of HSA correctly identified persistent diarrhoea or dysentery'
                      ),
        'prob_correct_classified_diarrhoea':
            Parameter(Types.REAL, 'probability of HSA correctly classified diarrhoea'
                      ),
        'prob_correct_referral_decision':
            Parameter(Types.REAL, 'probability of HSA correctly referred the case'
                      ),
        'prob_correct_treatment_advice_given':
            Parameter(Types.REAL, 'probability of HSA correctly treated and advised caretaker'
                      ),

    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'gi_diarrhoea_status': Property(Types.BOOL, 'symptomatic infection - diarrhoea disease'),
        'gi_diarrhoea_pathogen': Property(Types.CATEGORICAL, 'attributable pathogen for diarrhoea',
                                          categories=['rotavirus', 'shigella', 'adenovirus', 'cryptosporidium',
                                                      'campylobacter', 'ST-ETEC']),
        'gi_diarrhoea_acute_type': Property(Types.CATEGORICAL, 'clinical acute diarrhoea type',
                                            categories=['dysentery', 'acute watery diarrhoea']),
        'gi_dehydration_status': Property(Types.CATEGORICAL, 'dehydration status',
                                          categories=['no dehydration', 'some dehydration', 'severe dehydration']),
        'gi_persistent_diarrhoea': Property(Types.BOOL, 'diarrhoea episode longer than 14 days - persistent type'),
        'gi_diarrhoea_death': Property(Types.BOOL, 'death caused by diarrhoea'),
        'date_of_onset_diarrhoea': Property(Types.DATE, 'date of onset of diarrhoea'),
        'gi_recovered_date': Property(Types.DATE, 'date of recovery from enteric infection'),
        'gi_diarrhoea_death_date': Property(Types.DATE, 'date of death from enteric infection'),
        'gi_diarrhoea_count': Property(Types.REAL, 'number of diarrhoea episodes per individual'),
        'has_hiv': Property(Types.BOOL, 'temporary property - has hiv'),
        'malnutrition': Property(Types.BOOL, 'temporary property - malnutrition status'),
        'exclusive_breastfeeding': Property(Types.BOOL, 'temporary property - exclusive breastfeeding upto 6 mo'),
        'continued_breastfeeding': Property(Types.BOOL, 'temporary property - continued breastfeeding 6mo-2years'),
        # symptoms of diarrhoea for care seeking
        'di_diarrhoea_loose_watery_stools': Property(Types.BOOL, 'diarrhoea symptoms - loose or watery stools'),
        'di_blood_in_stools': Property(Types.BOOL, 'dysentery symptoms - blood in the stools'),
        'di_diarrhoea_over14days': Property(Types.BOOL, 'persistent diarrhoea - diarrhoea for 14 days or more'),
        'di_any_general_danger_sign': Property
        (Types.BOOL,
         'any danger sign - lethargic/uncounscious, not able to drink/breastfeed, convulsions and vomiting everything'),
        'correctly_identified_danger_signs': Property
        (Types.BOOL, 'HSA correctly identified at least one danger sign'
         ),
        'correctly_identified_diarrhoea_and_dehydration': Property
        (Types.BOOL, 'HSA correctly identified diarrhoea and dehydration'
         ),
        'correctly_classified_diarrhoea_with_danger_sign': Property
        (Types.BOOL, 'HSA correctly classified as diarrhoea with danger sign or with signs of severe dehydration'
         ),
        'correctly_classified_persistent_or_bloody_diarrhoea': Property
        (Types.BOOL, 'HSA correctly classified persistent diarrhoea or bloody diarrhoea'
         ),
        'correctly_classified_diarrhoea': Property
        (Types.BOOL, 'HSA correctly classified diarrhoea without blood and less than 14 days'
         ),
        'referral_options': Property
        (Types.CATEGORICAL,
         'Referral decisions', categories=['refer immediately', 'refer to health facility', 'do not refer']),
        'correct_referral_decision': Property
        (Types.BOOL,
         'HSA made the correct referral decision based on the assessment and classification process'
         ),
        'correct_treatment_and_advice_given': Property
        (Types.BOOL,
         'HSA has given the correct treatment for the classified condition'
         ),
    }

    def read_parameters(self, data_folder):
        """ Setup parameters values used by the module
        """
        p = self.parameters

        p['base_incidence_diarrhoea_by_rotavirus'] = [0.061, 0.02225, 0.00125]
        p['base_incidence_diarrhoea_by_shigella'] = [0.061, 0.02225, 0.00125]
        p['base_incidence_diarrhoea_by_adenovirus'] = [0.061, 0.02225, 0.00125]
        p['base_incidence_diarrhoea_by_crypto'] = [0.061, 0.02225, 0.00125]
        p['base_incidence_diarrhoea_by_campylo'] = [0.061, 0.02225, 0.00125]
        p['base_incidence_diarrhoea_by_ETEC'] = [0.061, 0.02225, 0.00125]
        p['rr_gi_diarrhoea_HHhandwashing'] = 0.5
        p['rr_gi_diarrhoea_improved_sanitation'] = 0.5
        p['rr_gi_diarrhoea_clean_water'] = 0.5
        p['rr_gi_diarrhoea_HIV'] = 1.4
        p['rr_gi_diarrhoea_SAM'] = 1.5
        p['rr_gi_diarrhoea_excl_breast'] = 0.5
        p['rr_gi_diarrhoea_cont_breast'] = 0.9
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
        p['prob_correct_id_diarrhoea_dehydration'] = 0.8
        p['prob_correct_id_danger_sign'] = 0.7
        p['prob_correct_id_persist_or_bloody_diarrhoea'] = 0.8
        p['prob_correctly_classified_diarrhoea_danger_sign'] = 0.8
        p['prob_correctly_classified_persist_or_bloody_diarrhoea'] = 0.8
        p['prob_correctly_classified_diarrhoea'] = 0.8
        p['prob_correct_referral_decision'] = 0.8
        p['prob_correct_treatment_advice_given'] = 0.8

        if 'HealthBurden' in self.sim.modules.keys():
            p['daly_mild_diarrhoea'] = self.sim.modules['HealthBurden'].get_daly_weight(32)
            p['daly_moderate_diarrhoea'] = self.sim.modules['HealthBurden'].get_daly_weight(35)
            p['daly_severe_diarrhoea'] = self.sim.modules['HealthBurden'].get_daly_weight(34)

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
        df['gi_diarrhoea_status'] = False
        df['gi_dehydration_status'] = 'no dehydration'
        df['date_of_onset_diarrhoea'] = pd.NaT
        df['gi_recovered_date'] = pd.NaT
        df['gi_diarrhoea_death_date'] = pd.NaT
        df['gi_diarrhoea_count'] = 0
        df['gi_diarrhoea_death'] = False
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
        sim.schedule_event(AcuteDiarrhoeaEvent(self), sim.date + DateOffset(months=3))

        # add an event to log to screen
        # sim.schedule_event(DysenteryLoggingEvent(self), sim.date + DateOffset(months=6))

        '''# death event
        death_all_diarrhoea = DeathDiarrhoeaEvent(self)
        sim.schedule_event(death_all_diarrhoea, sim.date)

        # add an event to log to screen
        sim.schedule_event(AcuteDiarrhoeaLoggingEvent(self), sim.date + DateOffset(months=6))

       '''

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

        logger.debug('This is Diarrhoea, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        logger.debug('This is diarrhoea reporting my health values')

        df = self.sim.population.props
        p = self.parameters

        health_values = df.loc[df.is_alive, 'gi_dehydration_status'].map({
            'none': 0,
            'no dehydration': p['daly_mild_diarrhoea'],
            'some dehydration': p['daly_moderate_diarrhoea'],
            'severe dehydration': p['daly_severe_diarrhoea']
        })
        health_values.name = 'Diarrhoea and dehydration symptoms'    # label the cause of this disability

        return health_values.loc[df.is_alive]   # returns the series


class AcuteDiarrhoeaEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=2))

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props
        m = self.module
        rng = m.rng

        no_diarrhoea0 = df.is_alive & (df.gi_diarrhoea_status == False) & (df.age_exact_years < 1)
        no_diarrhoea1 = df.is_alive & (df.gi_diarrhoea_status == False) &\
                        (df.age_exact_years >= 1) & (df.age_exact_years < 2)
        no_diarrhoea2 = df.is_alive & (df.gi_diarrhoea_status == False) & (df.age_years >= 2) & (df.age_years < 5)
        no_diarrhoea_under5 = df.is_alive & (df.gi_diarrhoea_status == False) & (df.age_years < 5)
        current_no_diarrhoea = df.index[no_diarrhoea_under5]

        diarrhoea_rotavirus0 = pd.Series(m.base_incidence_diarrhoea_by_rotavirus[0],
                                         index=df.index[no_diarrhoea0])
        diarrhoea_rotavirus1 = pd.Series(m.base_incidence_diarrhoea_by_rotavirus[1],
                                         index=df.index[no_diarrhoea1])
        diarrhoea_rotavirus2 = pd.Series(m.base_incidence_diarrhoea_by_rotavirus[2],
                                         index=df.index[no_diarrhoea2])

        diarrhoea_shigella0 = pd.Series(m.base_incidence_diarrhoea_by_shigella[0],
                                        index=df.index[no_diarrhoea0])
        diarrhoea_shigella1 = pd.Series(m.base_incidence_diarrhoea_by_shigella[1],
                                        index=df.index[no_diarrhoea1])
        diarrhoea_shigella2 = pd.Series(m.base_incidence_diarrhoea_by_shigella[2],
                                        index=df.index[no_diarrhoea2])

        diarrhoea_adenovirus0 = pd.Series(m.base_incidence_diarrhoea_by_adenovirus[0],
                                          index=df.index[no_diarrhoea0])
        diarrhoea_adenovirus1 = pd.Series(m.base_incidence_diarrhoea_by_adenovirus[1],
                                          index=df.index[no_diarrhoea1])
        diarrhoea_adenovirus2 = pd.Series(m.base_incidence_diarrhoea_by_adenovirus[2],
                                          index=df.index[no_diarrhoea2])

        diarrhoea_crypto0 = pd.Series(m.base_incidence_diarrhoea_by_crypto[0],
                                      index=df.index[no_diarrhoea0])
        diarrhoea_crypto1 = pd.Series(m.base_incidence_diarrhoea_by_crypto[1],
                                      index=df.index[no_diarrhoea1])
        diarrhoea_crypto2 = pd.Series(m.base_incidence_diarrhoea_by_crypto[2],
                                      index=df.index[no_diarrhoea2])

        diarrhoea_campylo0 = pd.Series(m.base_incidence_diarrhoea_by_campylo[0],
                                       index=df.index[no_diarrhoea0])
        diarrhoea_campylo1 = pd.Series(m.base_incidence_diarrhoea_by_campylo[1],
                                       index=df.index[no_diarrhoea1])
        diarrhoea_campylo2 = pd.Series(m.base_incidence_diarrhoea_by_campylo[2],
                                       index=df.index[no_diarrhoea2])

        diarrhoea_ETEC0 = pd.Series(m.base_incidence_diarrhoea_by_ETEC[0],
                                    index=df.index[no_diarrhoea0])
        diarrhoea_ETEC1 = pd.Series(m.base_incidence_diarrhoea_by_ETEC[1],
                                    index=df.index[no_diarrhoea1])
        diarrhoea_ETEC2 = pd.Series(m.base_incidence_diarrhoea_by_ETEC[2],
                                    index=df.index[no_diarrhoea2])

        # concatenating plus sorting
        eff_prob_rotavirus = pd.concat([diarrhoea_rotavirus0, diarrhoea_rotavirus1, diarrhoea_rotavirus2], axis=0).sort_index()
        eff_prob_shigella = pd.concat([diarrhoea_shigella0, diarrhoea_shigella1, diarrhoea_shigella2], axis=0).sort_index()
        eff_prob_adenovirus = pd.concat([diarrhoea_adenovirus0, diarrhoea_adenovirus1, diarrhoea_adenovirus2], axis=0).sort_index()
        eff_prob_crypto = pd.concat([diarrhoea_crypto0, diarrhoea_crypto1, diarrhoea_crypto2], axis=0).sort_index()
        eff_prob_campylo = pd.concat([diarrhoea_campylo0, diarrhoea_campylo1, diarrhoea_campylo2], axis=0).sort_index()
        eff_prob_ETEC = pd.concat([diarrhoea_ETEC0, diarrhoea_ETEC1, diarrhoea_ETEC2], axis=0).sort_index()

        eff_prob_all_pathogens = pd.concat([eff_prob_rotavirus, eff_prob_shigella, eff_prob_adenovirus,
                                            eff_prob_crypto, eff_prob_campylo, eff_prob_ETEC], axis=1)

        eff_prob_all_pathogens.loc[no_diarrhoea_under5 & df.li_no_access_handwashing == False] \
            *= m.rr_gi_diarrhoea_HHhandwashing
        eff_prob_all_pathogens.loc[no_diarrhoea_under5 & df.li_no_clean_drinking_water == False] \
            *= m.rr_gi_diarrhoea_clean_water
        eff_prob_all_pathogens.loc[no_diarrhoea_under5 & df.li_unimproved_sanitation == False] \
            *= m.rr_gi_diarrhoea_improved_sanitation
        eff_prob_all_pathogens.loc[no_diarrhoea_under5 & (df.has_hiv == True)] \
            *= m.rr_gi_diarrhoea_HIV
        eff_prob_all_pathogens.loc[no_diarrhoea_under5 & df.malnutrition == True] \
            *= m.rr_gi_diarrhoea_SAM
        eff_prob_all_pathogens.loc[no_diarrhoea_under5 & df.exclusive_breastfeeding == True & (df.age_exact_years <= 0.5)] \
            *= m.rr_gi_diarrhoea_excl_breast
        eff_prob_all_pathogens.loc[no_diarrhoea_under5 & df.continued_breastfeeding == True & (df.age_exact_years > 0.5) &
                                   (df.age_exact_years < 2)] *= m.rr_gi_diarrhoea_cont_breast

        random_draw1 = pd.Series(rng.random_sample(size=len(current_no_diarrhoea)), index=current_no_diarrhoea)
        diarrhoea_by_rotavirus = eff_prob_rotavirus > random_draw1
        diarr_rotavirus_idx = eff_prob_rotavirus.index[diarrhoea_by_rotavirus]
        df.loc[diarr_rotavirus_idx, 'gi_diarrhoea_pathogen'] = 'rotavirus'
        df.loc[diarr_rotavirus_idx, 'gi_diarrhoea_status'] = True

        random_draw2 = pd.Series(rng.random_sample(size=len(current_no_diarrhoea)), index=current_no_diarrhoea)
        diarrhoea_by_shigella = eff_prob_shigella > random_draw2
        diarr_shigella_idx = eff_prob_shigella.index[diarrhoea_by_shigella]
        df.loc[diarr_shigella_idx, 'gi_diarrhoea_pathogen'] = 'shigella'
        df.loc[diarr_shigella_idx, 'gi_diarrhoea_status'] = True

        random_draw3 = pd.Series(rng.random_sample(size=len(current_no_diarrhoea)), index=current_no_diarrhoea)
        diarrhoea_by_adenovirus = eff_prob_adenovirus > random_draw3
        diarr_adenovirus_idx = eff_prob_adenovirus.index[diarrhoea_by_adenovirus]
        df.loc[diarr_adenovirus_idx, 'gi_diarrhoea_pathogen'] = 'adenovirus'
        df.loc[diarr_adenovirus_idx, 'gi_diarrhoea_status'] = True

        random_draw4 = pd.Series(rng.random_sample(size=len(current_no_diarrhoea)), index=current_no_diarrhoea)
        diarrhoea_by_crypto = eff_prob_crypto > random_draw4
        diarr_crypto_idx = eff_prob_crypto.index[diarrhoea_by_crypto]
        df.loc[diarr_crypto_idx, 'gi_diarrhoea_pathogen'] = 'cryptosporidium'
        df.loc[diarr_crypto_idx, 'gi_diarrhoea_status'] = True

        random_draw5 = pd.Series(rng.random_sample(size=len(current_no_diarrhoea)), index=current_no_diarrhoea)
        diarrhoea_by_campylo = eff_prob_campylo > random_draw5
        diarr_campylo_idx = eff_prob_campylo.index[diarrhoea_by_campylo]
        df.loc[diarr_campylo_idx, 'gi_diarrhoea_pathogen'] = 'campylobacter'
        df.loc[diarr_campylo_idx, 'gi_diarrhoea_status'] = True

        random_draw6 = pd.Series(rng.random_sample(size=len(current_no_diarrhoea)), index=current_no_diarrhoea)
        diarrhoea_by_ETEC = eff_prob_ETEC > random_draw6
        diarr_ETEC_idx = eff_prob_ETEC.index[diarrhoea_by_ETEC]
        df.loc[diarr_ETEC_idx, 'gi_diarrhoea_pathogen'] = 'ST-ETEC'
        df.loc[diarr_ETEC_idx, 'gi_diarrhoea_status'] = True

        # ----------------- ASSIGN WHETHER IT IS DYSENTERY OR ACUTE WATERY DIARRHOEA ---------------------

        p_acute_watery_rotavirus = pd.Series(self.module.rotavirus_AWD, index=diarr_rotavirus_idx)
        random_draw = pd.Series(rng.random_sample(size=len(diarr_rotavirus_idx)), index=diarr_rotavirus_idx)
        diarr_rota_AWD = p_acute_watery_rotavirus >= random_draw
        diarr_rota_AWD_idx = p_acute_watery_rotavirus.index[diarr_rota_AWD]
        df.loc[diarr_rota_AWD_idx, 'gi_diarrhoea_acute_type'] = 'acute watery diarrhoea'
        diarr_rota_dysentery = p_acute_watery_rotavirus < random_draw
        diarr_rota_dysentery_idx = p_acute_watery_rotavirus.index[diarr_rota_dysentery]
        df.loc[diarr_rota_dysentery_idx, 'gi_diarrhoea_acute_type'] = 'dysentery'

        p_acute_watery_shigella = pd.Series(self.module.shigella_AWD, index=diarr_shigella_idx)
        random_draw = pd.Series(rng.random_sample(size=len(diarr_shigella_idx)), index=diarr_shigella_idx)
        diarr_shigella_AWD = p_acute_watery_shigella >= random_draw
        diarr_shigella_AWD_idx = p_acute_watery_shigella.index[diarr_shigella_AWD]
        df.loc[diarr_shigella_AWD_idx, 'gi_diarrhoea_acute_type'] = 'acute watery diarrhoea'
        diarr_shigella_dysentery = p_acute_watery_shigella < random_draw
        diarr_shigella_dysentery_idx = p_acute_watery_shigella.index[diarr_shigella_dysentery]
        df.loc[diarr_shigella_dysentery_idx, 'gi_diarrhoea_acute_type'] = 'dysentery'

        p_acute_watery_adeno = pd.Series(self.module.adenovirus_AWD, index=diarr_adenovirus_idx)
        random_draw = pd.Series(rng.random_sample(size=len(diarr_adenovirus_idx)), index=diarr_adenovirus_idx)
        diarr_adeno_AWD = p_acute_watery_adeno >= random_draw
        diarr_adeno_AWD_idx = p_acute_watery_adeno.index[diarr_adeno_AWD]
        df.loc[diarr_adeno_AWD_idx, 'gi_diarrhoea_acute_type'] = 'acute watery diarrhoea'
        diarr_adeno_dysentery = p_acute_watery_adeno < random_draw
        diarr_adeno_dysentery_idx = p_acute_watery_adeno.index[diarr_adeno_dysentery]
        df.loc[diarr_adeno_dysentery_idx, 'gi_diarrhoea_acute_type'] = 'dysentery'

        p_acute_watery_crypto = pd.Series(self.module.crypto_AWD, index=diarr_crypto_idx)
        random_draw = pd.Series(rng.random_sample(size=len(diarr_crypto_idx)), index=diarr_crypto_idx)
        diarr_crypto_AWD = p_acute_watery_crypto >= random_draw
        diarr_crypto_AWD_idx = p_acute_watery_crypto.index[diarr_crypto_AWD]
        df.loc[diarr_crypto_AWD_idx, 'gi_diarrhoea_acute_type'] = 'acute watery diarrhoea'
        diarr_crypto_dysentery = p_acute_watery_crypto < random_draw
        diarr_crypto_dysentery_idx = p_acute_watery_crypto.index[diarr_crypto_dysentery]
        df.loc[diarr_crypto_dysentery_idx, 'gi_diarrhoea_acute_type'] = 'dysentery'

        p_acute_watery_campylo = pd.Series(self.module.campylo_AWD, index=diarr_campylo_idx)
        random_draw = pd.Series(rng.random_sample(size=len(diarr_campylo_idx)), index=diarr_campylo_idx)
        diarr_campylo_AWD = p_acute_watery_campylo >= random_draw
        diarr_campylo_AWD_idx = p_acute_watery_campylo.index[diarr_campylo_AWD]
        df.loc[diarr_campylo_AWD_idx, 'gi_diarrhoea_acute_type'] = 'acute watery diarrhoea'
        diarr_campylo_dysentery = p_acute_watery_campylo < random_draw
        diarr_campylo_dysentery_idx = p_acute_watery_campylo.index[diarr_campylo_dysentery]
        df.loc[diarr_campylo_dysentery_idx, 'gi_diarrhoea_acute_type'] = 'dysentery'

        p_acute_watery_ETEC = pd.Series(self.module.ETEC_AWD, index=diarr_ETEC_idx)
        random_draw = pd.Series(rng.random_sample(size=len(diarr_ETEC_idx)), index=diarr_ETEC_idx)
        diarr_ETEC_AWD = p_acute_watery_ETEC >= random_draw
        diarr_ETEC_AWD_idx = p_acute_watery_ETEC.index[diarr_ETEC_AWD]
        df.loc[diarr_ETEC_AWD_idx, 'gi_diarrhoea_acute_type'] = 'acute watery diarrhoea'
        diarr_ETEC_dysentery = p_acute_watery_ETEC < random_draw
        diarr_ETEC_dysentery_idx = p_acute_watery_ETEC.index[diarr_ETEC_dysentery]
        df.loc[diarr_ETEC_dysentery_idx, 'gi_diarrhoea_acute_type'] = 'dysentery'

        # # # # # # WHEN THEY GET ACUTE DIARRHOEA - DATE # # # # # #
        incident_acute_diarrhoea = df.index[df.gi_diarrhoea_status == True]
        random_draw_days = np.random.randint(0, 60, size=len(incident_acute_diarrhoea))
        adding_days = pd.to_timedelta(random_draw_days, unit='d')
        date_of_aquisition = self.sim.date + adding_days
        df.loc[incident_acute_diarrhoea, 'date_of_onset_diarrhoea'] = date_of_aquisition

        # # # # # # ASSIGN DEHYDRATION LEVELS FOR ACUTE WATERY DIARRHOEA # # # # # #
        under5_watery_diarrhoea_idx = df.index[
            (df.age_years < 5) & df.is_alive & (df.gi_diarrhoea_acute_type == 'acute watery diarrhoea')]

        eff_prob_some_dehydration_acute_diarrhoea = pd.Series(0.5, index=under5_watery_diarrhoea_idx)
        eff_prob_severe_dehydration_acute_diarrhoea = pd.Series(0.2, index=under5_watery_diarrhoea_idx)
        random_draw_a = pd.Series(self.sim.rng.random_sample(size=len(under5_watery_diarrhoea_idx)),
                                  index=under5_watery_diarrhoea_idx)

        no_dehydration_acute_diarrhoea = \
            1 - (eff_prob_some_dehydration_acute_diarrhoea + eff_prob_severe_dehydration_acute_diarrhoea)
        some_dehydration_acute_diarrhoea = \
            (random_draw_a > no_dehydration_acute_diarrhoea) & \
            (random_draw_a < (no_dehydration_acute_diarrhoea + eff_prob_some_dehydration_acute_diarrhoea))
        severe_dehydration_acute_diarrhoea = \
            ((no_dehydration_acute_diarrhoea + eff_prob_some_dehydration_acute_diarrhoea) < random_draw_a) & \
            ((no_dehydration_acute_diarrhoea + eff_prob_some_dehydration_acute_diarrhoea +
              eff_prob_severe_dehydration_acute_diarrhoea) > random_draw_a)

        idx_some_dehydration_acute_diarrhoea = eff_prob_some_dehydration_acute_diarrhoea.index[
            some_dehydration_acute_diarrhoea]
        idx_severe_dehydration_acute_diarrhoea = eff_prob_severe_dehydration_acute_diarrhoea.index[
            severe_dehydration_acute_diarrhoea]

        df.loc[idx_some_dehydration_acute_diarrhoea, 'gi_dehydration_status'] = 'some dehydration'
        df.loc[idx_severe_dehydration_acute_diarrhoea, 'gi_dehydration_status'] = 'severe dehydration'

        # # # # # # ASSIGN DEHYDRATION LEVELS FOR ACUTE DYSENTERY # # # # # #
        under5_dysentery_idx = df.index[
            (df.age_years < 5) & df.is_alive & (df.gi_diarrhoea_acute_type == 'dysentery')]

        eff_prob_some_dehydration_dysentery = pd.Series(0.5, index=under5_dysentery_idx)
        eff_prob_severe_dehydration_dysentery = pd.Series(0.2, index=under5_dysentery_idx)
        random_draw_b = pd.Series(self.sim.rng.random_sample(size=len(under5_dysentery_idx)),
                                  index=under5_dysentery_idx)

        no_dehydration_dysentery = \
            1 - (eff_prob_some_dehydration_dysentery + eff_prob_severe_dehydration_dysentery)
        some_dehydration_dysentery = \
            (random_draw_b > no_dehydration_dysentery) & \
            (random_draw_b < (no_dehydration_dysentery + eff_prob_some_dehydration_dysentery))
        severe_dehydration_dysentery = \
            ((no_dehydration_dysentery + eff_prob_some_dehydration_dysentery) < random_draw_b) & \
            ((no_dehydration_dysentery + eff_prob_some_dehydration_dysentery +
              eff_prob_severe_dehydration_dysentery) > random_draw_b)

        idx_some_dehydration_dysentery = eff_prob_some_dehydration_dysentery.index[
            some_dehydration_dysentery]
        idx_severe_dehydration_dysentery = eff_prob_severe_dehydration_dysentery.index[
            severe_dehydration_dysentery]

        df.loc[idx_some_dehydration_dysentery, 'gi_dehydration_status'] = 'some dehydration'
        df.loc[idx_severe_dehydration_dysentery, 'gi_dehydration_status'] = 'severe dehydration'

        # # # # # # # # SYMPTOMS FROM ACUTE WATERY DIARRHOEA # # # # # # # # # # # # # # # # # # # # # # #
        df.loc[under5_watery_diarrhoea_idx, 'di_diarrhoea_loose_watery_stools'] = True
        df.loc[under5_watery_diarrhoea_idx, 'di_blood_in_stools'] = False
        df.loc[under5_watery_diarrhoea_idx, 'di_diarrhoea_over14days'] = False

        # # # # # # # # SYMPTOMS FROM DYSENTERY # # # # # # # # # # # # # # # # # # # # # # #
        df.loc[under5_dysentery_idx, 'di_diarrhoea_loose_watery_stools'] = True
        df.loc[under5_dysentery_idx, 'di_blood_in_stools'] = True
        df.loc[under5_dysentery_idx, 'di_diarrhoea_over14days'] = False

        # --------------------------------------------------------------------------------------------------------
        # SEEKING CARE FOR ACUTE WATERY DIARRHOEA
        # --------------------------------------------------------------------------------------------------------
        watery_diarrhoea_symptoms = \
            df.index[df.is_alive & (df.age_years < 5) & (df.di_diarrhoea_loose_watery_stools == True) &
                     (df.di_blood_in_stools == False) & df.di_diarrhoea_over14days == False]

        seeks_care = pd.Series(data=False, index=watery_diarrhoea_symptoms)
        for individual in watery_diarrhoea_symptoms:
            prob = self.sim.modules['HealthSystem'].get_prob_seek_care(individual, symptom_code=1)
            seeks_care[individual] = self.module.rng.rand() < prob
            date_seeking_care = df.date_of_onset_diarrhoea[individual] + pd.DateOffset(days=int(rng.uniform(0, 7)))
            event = HSI_Sick_Child_Seeks_Care_From_HSA(self.module, person_id=individual)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=2,
                                                                topen=self.sim.date,
                                                                tclose=None
                                                                )

        # --------------------------------------------------------------------------------------------------------
        # SEEKING CARE FOR ACUTE BLOODY DIARRHOEA
        # --------------------------------------------------------------------------------------------------------
        dysentery_symptoms = \
            df.index[df.is_alive & (df.age_years < 5) & (df.di_diarrhoea_loose_watery_stools == True) &
                     (df.di_blood_in_stools == True) & df.di_diarrhoea_over14days == False]

        seeks_care = pd.Series(data=False, index=dysentery_symptoms)
        for individual in dysentery_symptoms:
            prob = self.sim.modules['HealthSystem'].get_prob_seek_care(individual, symptom_code=1)
            seeks_care[individual] = self.module.rng.rand() < prob
            event = HSI_Sick_Child_Seeks_Care_From_HSA(self.module, person_id=individual)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=2,
                                                                topen=self.sim.date,
                                                                tclose=None
                                                                )

        # # # # # # # # # # # # # ACUTE DIARRHOEA BECOMING PERSISTENT # # # # # # # # # # # # #

        dysentery_bec_persistent = pd.Series(m.prob_dysentery_become_persistent, index=under5_dysentery_idx)
        watery_diarr_bec_persistent = pd.Series(m.prob_watery_diarr_become_persistent, index=under5_watery_diarrhoea_idx)
        becoming_persistent = pd.concat([dysentery_bec_persistent, watery_diarr_bec_persistent], axis=0).sort_index()

        becoming_persistent.loc[df.is_alive & (df.age_exact_years >= 1) & (df.age_exact_years < 2)] \
            *= m.rr_bec_persistent_age12to23
        becoming_persistent.loc[df.is_alive & (df.age_exact_years >= 2) & (df.age_exact_years < 5)] \
            *= m.rr_bec_persistent_age24to59
        becoming_persistent.loc[df.is_alive & (df.has_hiv == True)] \
            *= m.rr_bec_persistent_HIV
        becoming_persistent.loc[df.is_alive & df.malnutrition == True] \
            *= m.rr_bec_persistent_SAM
        becoming_persistent.loc[
            df.is_alive & df.exclusive_breastfeeding == True & (df.age_exact_years <= 0.5)] \
            *= m.rr_bec_persistent_excl_breast
        becoming_persistent.loc[
            df.is_alive & df.continued_breastfeeding == True & (df.age_exact_years > 0.5) &
            (df.age_exact_years < 2)] *= m.rr_bec_persistent_cont_breast

        random_draw_c = pd.Series(self.sim.rng.random_sample(size=len(incident_acute_diarrhoea)),
                                  index=incident_acute_diarrhoea)
        persistent_diarr = becoming_persistent > random_draw_c
        persistent_diarr_idx = becoming_persistent.index[persistent_diarr]
        df.loc[persistent_diarr_idx, 'gi_persistent_diarrhoea'] = True

        # SET THE TIMELINE FOR PERSISTENT DIARRHOEA
        date_become_persistent = date_of_aquisition + DateOffset(weeks=2)

        # # # # # # # # SYMPTOMS FROM PERSISTENT DIARRHOEA # # # # # # # # # # # # # # # # # # # # # # #
        df.loc[under5_dysentery_idx, 'di_diarrhoea_loose_watery_stools'] = True
        df.loc[under5_dysentery_idx, 'di_blood_in_stools'] = True | False
        df.loc[under5_dysentery_idx, 'di_diarrhoea_over14days'] = False

        # --------------------------------------------------------------------------------------------------------
        # SEEKING CARE FOR PERSISTENT DIARRHOEA
        # --------------------------------------------------------------------------------------------------------
        persistent_diarrhoea_symptoms = \
            df.index[df.is_alive & (df.age_years < 5) & (df.di_diarrhoea_loose_watery_stools == True | False)
                     & df.di_diarrhoea_over14days == True]

        seeks_care = pd.Series(data=False, index=persistent_diarrhoea_symptoms)
        for individual in persistent_diarrhoea_symptoms:
            prob = self.sim.modules['HealthSystem'].get_prob_seek_care(individual, symptom_code=1)
            # date_seeking_care = self.sim.date + pd.DateOffset(days=int(rng.uniform(0, 91)))
            seeks_care[individual] = self.module.rng.rand() < prob
            event = HSI_Sick_Child_Seeks_Care_From_HSA(self.module, person_id=individual)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=2,
                                                                topen=self.sim.date,
                                                                tclose=None
                                                                )


class HSI_Sick_Child_Seeks_Care_From_HSA(Event, IndividualScopeEventMixin):
    """ This is the first Health Systems Interaction event for diarrhoea module.
    A sick child presenting symptoms is taken to the HSA for assessment, referral and treatment. """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Under5OPD'] = 1  # This requires one out patient

        # Define the necessary information for an HSI
        # self.TREATMENT_ID = 'Sick_child_presents_for_care'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = self.sim.modules['HealthSystem'].get_blank_cons_footprint()
        self.ACCEPTED_FACILITY_LEVELS = [1]
        self.ALERT_OTHER_DISEASES = ['NewPneumonia']

    def apply(self, person_id):

        logger.debug('This is HSI_Sick_Child_Seeks_Care_From_HSA, a first appointment for person %d in the community',
                     person_id)

        df = self.sim.population.props
        p = self.module.parameters

        # ---------------- Work out if child with diarrhoea is correctly diagnosed by the HSA ----------------

        current_diarrhoea = df.index[df.is_alive & (df.gi_diarrhoea_status == True) & (df.age_years < 5)]

        for person_id in current_diarrhoea:
            diarrhoea_identified_by_HSA = \
                self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_diarrhoea_dehydration'],
                                                              (1 - p['prob_correct_id_diarrhoea_dehydration'])])
            if diarrhoea_identified_by_HSA[True]:
                df.at[person_id, 'correctly_identified_diarrhoea_and_dehydration'] = True
            if df.at[person_id, 'di_any_general_danger_sign']:
                danger_sign_identified_by_HSA = \
                    self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_danger_sign'],
                                                                  (1 - p['prob_correct_id_danger_sign'])])
                if diarrhoea_identified_by_HSA[True] & danger_sign_identified_by_HSA[True]:
                    df.at[person_id, 'correctly_classified_diarrhoea_with_danger_sign'] = True
                    HSA_referral_decision = \
                        self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_referral_decision'],
                                                                      (1 - p['prob_correct_referral_decision'])])
                    if HSA_referral_decision[True]:
                        df.at[person_id, 'correct_referral_decision'] = True
            if (df.at[person_id, 'gi_diarrhoea_acute_type'] == 'dysentery') | (
                df.at[person_id, 'gi_persistent_diarrhoea']):
                persistent_or_bloody_diarr_identified_by_HSA = self.sim.rng.choice([True, False], size=1, p=[
                    p['prob_correct_id_persist_or_bloody_diarrhoea'],
                    (1 - p['prob_correct_id_persist_or_bloody_diarrhoea'])])
                if diarrhoea_identified_by_HSA[True] & persistent_or_bloody_diarr_identified_by_HSA[True]:
                    df.at[person_id, 'correctly_classified_persistent_or_bloody_diarrhoea'] = True
                    HSA_referral_decision = \
                        self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_referral_decision'],
                                                                      (1 - p['prob_correct_referral_decision'])])
                    if HSA_referral_decision[True]:
                        df.at[person_id, 'correct_referral_decision'] = True
            if not (df.at[person_id, 'gi_persistent_diarrhoea'] & (
                (df.at[person_id, 'gi_diarrhoea_acute_type']) == 'dysentery') & (
                    df.at[person_id, 'di_any_general_danger_sign'])) & diarrhoea_identified_by_HSA[True]:
                df.at[person_id, 'correctly_classified_diarrhoea'] = True
                HSA_referral_decision = \
                    self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_referral_decision'],
                                                                  (1 - p['prob_correct_referral_decision'])])
                if HSA_referral_decision[True]:
                    df.at[person_id, 'correct_referral_decision'] = True
            if df.at[person_id, 'correct_referral_decision'] & \
                df.at[person_id, 'correctly_classified_diarrhoea_with_danger_sign']:
                df.at[person_id, 'referral_options'] = 'refer immediately'
                # Get the consumables required
                consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
                pkg_code1 = pd.unique(consumables.loc[consumables[
                                                          'Intervention_Pkg'] ==
                                                      'ORS',
                                                      'Intervention_Pkg_Code'])[37]
                event_referred_immediately = HSI_HSA_Diarrhoea_Referred_Immediately(self.module, person_id=person_id)
                self.sim.modules['HealthSystem'].schedule_event(event_referred_immediately,
                                                                priority=1,
                                                                topen=self.sim.date,
                                                                tclose=None)

            if df.at[person_id, 'correct_referral_decision'] & \
                df.at[person_id, 'correctly_classified_persistent_or_bloody_diarrhoea']:
                df.at[person_id, 'referral_options'] = 'refer to health facility'
                # Get the consumables required
                consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
                pkg_code1 = pd.unique(consumables.loc[consumables[
                                                          'Intervention_Pkg'] ==
                                                      'ORS',
                                                      'Intervention_Pkg_Code'])[37]
                event_referred_immediately = HSI_HSA_Diarrhoea_Referred_HealthFacility(self.module, person_id=person_id)
                self.sim.modules['HealthSystem'].schedule_event(event_referred_immediately,
                                                                priority=1,
                                                                topen=self.sim.date,
                                                                tclose=None)

            if df.at[person_id, 'correct_referral_decision'] & \
                df.at[person_id, 'correctly_classified_diarrhoea']:
                df.at[person_id, 'referral_options'] = 'do not refer'
                # Get the consumables required
                consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
                pkg_code1 = pd.unique(consumables.loc[consumables[
                                                          'Intervention_Pkg'] ==
                                                      'ORS',
                                                      'Intervention_Pkg_Code'])[37]
                pkg_code2 = pd.unique(consumables.loc[consumables[
                                                          'Intervention_Pkg'] ==
                                                      'Zinc for Children 0-6 months',
                                                      'Intervention_Pkg_Code'])[38]
                pkg_code3 = pd.unique(consumables.loc[consumables[
                                                          'Intervention_Pkg'] ==
                                                      'Zinc for Children 6-59 months',
                                                      'Intervention_Pkg_Code'])[39]

            if df.at[person_id, 'correct_referral_decision']:
                logger.debug(
                    '...This is HSI_Sick_Child_Seeks_Care_From_HSA: \
                    there should now be treatment for person %d',
                    person_id)
                event_treatment_decision = HSI_HSA_Diarrhoea_StartTreatment(self.module, person_id=person_id)
                self.sim.modules['HealthSystem'].schedule_event(event_treatment_decision)
            else:
                logger.debug(
                    '...This is HSI_Sick_Child_Seeks_Care_From_HSA: there will not be treatment for person %d',
                    person_id)


class HSI_HSA_Diarrhoea_StartTreatment(Event, IndividualScopeEventMixin):

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
                                              'ORS',
                                              'Intervention_Pkg_Code'])[37]
        pkg_code2 = pd.unique(consumables.loc[consumables[
                                                  'Intervention_Pkg'] ==
                                              'Zinc for Children 0-6 months',
                                              'Intervention_Pkg_Code'])[38]
        pkg_code3 = pd.unique(consumables.loc[consumables[
                                                  'Intervention_Pkg'] ==
                                              'Zinc for Children 6-59 months',
                                              'Intervention_Pkg_Code'])[39]

        item_code1 = \
            pd.unique(consumables.loc[consumables['Items'] == 'Ketamine hydrochloride 50mg/ml, 10ml', 'Item_Code'])[0]
        item_code2 = pd.unique(consumables.loc[consumables['Items'] == 'Underpants', 'Item_Code'])[0]

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1, pkg_code2],
            'Item_Code': [item_code1, item_code2]
        }

    def apply(self, person_id):
        logger.debug('This is HSI_Sick_Child_Seeks_Care_From_HSA, a first appointment for person %d in the community',
                     person_id)

        df = self.sim.population.props
        now = self.sim.date

        target_date_for_followup_appt = self.sim.date + DateOffset(days=5)

        logger.debug(
            '....This is HHSI_Sick_Child_Seeks_Care_From_HSA: scheduling a follow-up appointment for person %d on date %s',
            person_id, target_date_for_followup_appt)

        followup_appt = HSI_HSA_followup_care(self.module, person_id=person_id)

        # Request the heathsystem to have this follow-up appointment
        self.sim.modules['HealthSystem'].schedule_hsi_event(followup_appt,
                                                            priority=2,
                                                            topen=target_date_for_followup_appt,
                                                            tclose=target_date_for_followup_appt + DateOffset(weeks=2))

class DeathDiarrhoeaEvent(Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, population):

        df = population.props
        m = self.module
        rng = m.rng

        # ------------------------------------------------------------------------------------------------------
        # DEATH DUE TO ACUTE BLOODY DIARRHOEA - DYSENTERY
        # ------------------------------------------------------------------------------------------------------
        eff_prob_death_dysentery = \
            pd.Series(m.r_death_dysentery,
                      index=df.index[df.is_alive & (df.gi_diarrhoea_acute_type == 'dysentery') & (df.age_years < 5)])

        eff_prob_death_dysentery.loc[df.is_alive & (df.gi_diarrhoea_acute_type == 'dysentery') &
                                     (df.age_exact_years >= 1) & (df.age_exact_years < 2)] *= \
            m.rr_death_dysentery_age12to23mo
        eff_prob_death_dysentery.loc[df.is_alive & (df.gi_diarrhoea_acute_type == 'dysentery') &
                                     (df.age_exact_years >= 2) & (df.age_exact_years < 5)] *= \
            m.rr_death_dysentery_age24to59mo
        eff_prob_death_dysentery.loc[df.is_alive & (df.gi_diarrhoea_acute_type == 'dysentery') &
                                     (df.has_hiv == True) & (df.age_exact_years < 5)] *= m.rr_death_dysentery_HIV
        eff_prob_death_dysentery.loc[df.is_alive & (df.gi_diarrhoea_acute_type == 'dysentery') &
                                     df.malnutrition == True & (df.age_exact_years < 5)] *= m.rr_death_dysentery_SAM

        under5_dysentery_idx = df.index[(df.age_years < 5) & df.is_alive & (df.gi_diarrhoea_acute_type == 'dysentery')]

        random_draw = pd.Series(rng.random_sample(size=len(under5_dysentery_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive &
                                               (df.gi_diarrhoea_acute_type == 'dysentery')])
        dfx = pd.concat([eff_prob_death_dysentery, random_draw], axis=1)
        dfx.columns = ['eff_prob_death_dysentery', 'random_draw']

        for person_id in under5_dysentery_idx:
            if dfx.index[dfx.eff_prob_death_dysentery > dfx.random_draw]:
                df.at[person_id, 'gi_diarrhoea_death'] = True
            else:
                df.at[person_id, 'gi_diarrhoea_status'] = False

        # ------------------------------------------------------------------------------------------------------
        # DEATH DUE TO ACUTE WATERY DIARRHOEA
        # ------------------------------------------------------------------------------------------------------

        eff_prob_death_acute_diarrhoea = \
            pd.Series(m.r_death_acute_diarrhoea,
                      index=df.index[
                          df.is_alive & (df.gi_diarrhoea_acute_type == 'acute watery diarrhoea') & (df.age_years < 5)])
        eff_prob_death_acute_diarrhoea.loc[df.is_alive & (df.gi_diarrhoea_acute_type == 'acute watery diarrhoea') &
                                           (df.age_exact_years >= 1) & (df.age_exact_years < 2)] *= \
            m.rr_death_acute_diar_age12to23mo
        eff_prob_death_acute_diarrhoea.loc[df.is_alive & (df.gi_diarrhoea_acute_type == 'acute watery diarrhoea') &
                                           (df.age_exact_years >= 2) & (df.age_exact_years < 5)] *= \
            m.rr_death_acute_diar_age24to59mo
        eff_prob_death_acute_diarrhoea.loc[df.is_alive & (df.gi_diarrhoea_acute_type == 'acute watery diarrhoea') &
                                           (df.has_hiv == True) & (df.age_exact_years < 5)] *= \
            m.rr_death_acute_diar_HIV
        eff_prob_death_acute_diarrhoea.loc[df.is_alive & (df.gi_diarrhoea_acute_type == 'acute watery diarrhoea') &
                                           df.malnutrition == True & (df.age_exact_years < 5)] *= \
            m.rr_death_acute_diar_SAM

        under5_acute_diarrhoea_idx = df.index[(df.age_years < 5) & df.is_alive &
                                              (df.gi_diarrhoea_acute_type == 'acute watery diarrhoea')]

        random_draw = pd.Series(rng.random_sample(size=len(under5_acute_diarrhoea_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive &
                                               (df.gi_diarrhoea_acute_type == 'acute watery diarrhoea')])
        dfx = pd.concat([eff_prob_death_acute_diarrhoea, random_draw], axis=1)
        dfx.columns = ['eff_prob_death_acute_diarrhoea', 'random_draw']

        for person_id in under5_acute_diarrhoea_idx:
            if dfx.index[dfx.eff_prob_death_acute_diarrhoea > dfx.random_draw]:
                df.at[person_id, 'gi_diarrhoea_death'] = True
            else:
                df.at[person_id, 'gi_diarrhoea_status'] = False

        # ------------------------------------------------------------------------------------------------------
        # DEATH DUE TO PERSISTENT DIARRHOEA
        # ------------------------------------------------------------------------------------------------------

        eff_prob_death_persistent_diarrhoea = \
            pd.Series(m.r_death_persistent_diarrhoea,
                      index=df.index[
                          df.is_alive & (df.gi_persistent_diarrhoea == True) & (df.age_years < 5)])
        eff_prob_death_persistent_diarrhoea.loc[df.is_alive & (df.gi_persistent_diarrhoea == True) &
                                                (df.age_exact_years >= 1) & (df.age_exact_years < 2)] *= \
            m.rr_death_persistent_diar_age12to23mo
        eff_prob_death_persistent_diarrhoea.loc[df.is_alive & (df.gi_persistent_diarrhoea == True) &
                                                (df.age_exact_years >= 2) & (df.age_exact_years < 5)] *= \
            m.rr_death_persistent_diar_age24to59mo
        eff_prob_death_persistent_diarrhoea.loc[df.is_alive & (df.gi_persistent_diarrhoea == True) &
                                                (df.has_hiv == True) & (df.age_exact_years < 5)] *= \
            m.rr_death_persistent_diar_HIV
        eff_prob_death_persistent_diarrhoea.loc[df.is_alive & (df.gi_persistent_diarrhoea == True) &
                                                df.malnutrition == True & (df.age_exact_years < 5)] *= \
            m.rr_death_persistent_diar_SAM

        under5_persistent_diarrhoea_idx = df.index[(df.age_years < 5) & df.is_alive &
                                                   (df.gi_persistent_diarrhoea == True)]

        random_draw = pd.Series(rng.random_sample(size=len(under5_persistent_diarrhoea_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive &
                                               (df.gi_persistent_diarrhoea == True)])
        dfx = pd.concat([eff_prob_death_persistent_diarrhoea, random_draw], axis=1)
        dfx.columns = ['eff_prob_death_persistent_diarrhoea', 'random_draw']

        for person_id in under5_persistent_diarrhoea_idx:
            if dfx.index[dfx.eff_prob_death_persistent_diarrhoea > dfx.random_draw]:
                df.at[person_id, 'gi_diarrhoea_death'] = True
            else:
                df.at[person_id, 'ei_diarrhoea_status'] = False

        death_this_period = df.index[(df.gi_diarrhoea_death == True)]
        for individual_id in death_this_period:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id, 'ChildhoodDiarrhoea'),
                                    self.sim.date)
