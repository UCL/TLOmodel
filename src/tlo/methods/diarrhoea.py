"""
Childhood diarrhoea module
Documentation: 04 - Methods Repository/Method_Child_EntericInfection.xlsx
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent, Event, IndividualScopeEventMixin
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import demography

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Diarrhoea(Module):
    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

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
        'base_incidence_diarrhoea_by_sapovirus':
            Parameter(Types.LIST,
                      'incidence of diarrhoea caused by sapovirus in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_incidence_diarrhoea_by_norovirus':
            Parameter(Types.LIST,
                      'incidence of diarrhoea caused by norovirus in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_incidence_diarrhoea_by_astrovirus':
            Parameter(Types.LIST,
                      'incidence of diarrhoea caused by astrovirus in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_incidence_diarrhoea_by_tEPEC':
            Parameter(Types.LIST,
                      'incidence of diarrhoea caused by tEPEC in age groups 0-11, 12-23, 24-59 months'
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
        'rr_diarr_death_age12to23mo':
            Parameter(Types.REAL,
                      'relative rate of diarrhoea death for ages 12 to 23 months'
                      ),
        'rr_diarr_death_age24to59mo':
            Parameter(Types.REAL,
                      'relative rate of diarrhoea death for ages 24 to 59 months'
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
        'r_recovery_AWD':
            Parameter(Types.REAL, 'baseline recovery rate for acute water diarrhoea'
                      ),
        'r_recovery_dysentery':
            Parameter(Types.REAL, 'baseline recovery rate for acute bloody diarrhoea'
                      ),
        'rr_recovery_dehydration':
            Parameter(Types.REAL, 'relative rate of recovery for diarrhoea with any dehydration'
                      ),

    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    # TODO: why do some have gi_ and some have di_ as the prefix?
    PROPERTIES = {
        'gi_diarrhoea_status': Property(Types.BOOL, 'symptomatic infection - diarrhoea disease'),
        'gi_diarrhoea_pathogen': Property(Types.CATEGORICAL, 'attributable pathogen for diarrhoea',
                                          categories=['rotavirus', 'shigella', 'adenovirus', 'cryptosporidium',
                                                      'campylobacter', 'ST-ETEC', 'sapovirus', 'norovirus',
                                                      'astrovirus', 'tEPEC']),
        'gi_diarrhoea_type': Property(Types.CATEGORICAL, 'progression of diarrhoea type',
                                      categories=['acute', 'prolonged', 'persistent']),
        'gi_diarrhoea_acute_type': Property(Types.CATEGORICAL, 'clinical acute diarrhoea type',
                                            categories=['dysentery', 'acute watery diarrhoea']),
        'gi_dehydration_status': Property(Types.CATEGORICAL, 'dehydration status',
                                          categories=['no dehydration', 'some dehydration', 'severe dehydration']),
        'gi_persistent_diarrhoea': Property(Types.CATEGORICAL,
                                            'diarrhoea episode longer than 14 days with or without dehydration',
                                            categories=['persistent diarrhoea', 'severe persistent diarrhoea']),
        'gi_diarrhoea_death': Property(Types.BOOL, 'death caused by diarrhoea'),
        'date_of_onset_diarrhoea': Property(Types.DATE, 'date of onset of diarrhoea'),
        'diarrhoea_ep_duration': Property(Types.REAL, 'duration of diarrhoea episode'),
        'gi_recovered_date': Property(Types.DATE, 'date of recovery from enteric infection'),
        'gi_diarrhoea_death_date': Property(Types.DATE, 'date of death from enteric infection'),
        'gi_diarrhoea_count': Property(Types.INT, 'annual counter for diarrhoea episodes'),
        'has_hiv': Property(Types.BOOL, 'temporary property - has hiv'),   # TODO: you could use the real one.
        'malnutrition': Property(Types.BOOL, 'temporary property - malnutrition status'),
        'exclusive_breastfeeding': Property(Types.BOOL, 'temporary property - exclusive breastfeeding upto 6 mo'),
        'continued_breastfeeding': Property(Types.BOOL, 'temporary property - continued breastfeeding 6mo-2years'),
        # symptoms of diarrhoea for care seeking
        'di_diarrhoea_loose_watery_stools': Property(Types.BOOL, 'diarrhoea symptoms - loose or watery stools'),
        'di_blood_in_stools': Property(Types.BOOL, 'dysentery symptoms - blood in the stools'),
        'di_dehydration_present': Property(Types.BOOL, 'diarrhoea symptoms - dehydration'),
        'di_sympt_fever': Property(Types.BOOL, 'diarrhoea symptoms - associated fever'),
        'di_sympt_vomiting': Property(Types.BOOL, 'diarrhoea symptoms - associated vomoting'),
        'di_diarrhoea_over14days': Property(Types.BOOL, 'persistent diarrhoea - diarrhoea for 14 days or more'),
        'di_any_general_danger_sign': Property
        (Types.BOOL,
         'any danger sign - lethargic/uncounscious, not able to drink/breastfeed, convulsions and vomiting everything'),
        'correctly_identified_danger_signs': Property
        (Types.BOOL, 'HSA correctly identified at least one danger sign'
         ),
        'correctly_assessed_diarrhoea_and_dehydration': Property
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
        """ Setup parameters values used by the module """

        p = self.parameters
        dfd = pd.read_excel(
            Path(self.resourcefilepath) / 'ResourceFile_Childhood_Diarrhoea.xlsx', sheet_name='Parameter_values')
        dfd.set_index("Parameter_name", inplace=True)

        # all diarrhoea prevalence values
        p['rp_acute_diarr_age12to23mo'] = dfd.loc['rp_acute_diarr_age12to23mo', 'value1']
        p['rp_acute_diarr_age24to59mo'] = dfd.loc['rp_acute_diarr_age24to59mo', 'value1']
        p['rp_acute_diarr_HIV'] = dfd.loc['rp_acute_diarr_HIV', 'value1']
        p['rp_acute_diarr_SAM'] = dfd.loc['rp_acute_diarr_SAM', 'value1']
        p['rp_acute_diarr_excl_breast'] = dfd.loc['rp_acute_diarr_excl_breast', 'value1']
        p['rp_acute_diarr_cont_breast'] = dfd.loc['rp_acute_diarr_cont_breast', 'value1']
        p['rp_acute_diarr_HHhandwashing'] = dfd.loc['rp_acute_diarr_HHhandwashing', 'value1']
        p['rp_acute_diarr_clean_water'] = dfd.loc['rp_acute_diarr_clean_water', 'value1']
        p['rp_acute_diarr_improved_sanitation'] = dfd.loc['rp_acute_diarr_improved_sanitation', 'value1']
        p['base_prev_persistent_diarrhoea'] = dfd.loc['base_prev_persistent_diarrhoea', 'value1']
        p['rp_persistent_diarrhoea_age12to23mo'] = dfd.loc['rp_persistent_diarrhoea_age12to23mo', 'value1']
        p['rp_persistent_diarrhoea_age24to59mo'] = dfd.loc['rp_persistent_diarrhoea_age24to59mo', 'value1']
        p['rp_persistent_diarrhoea_HIV'] = dfd.loc['rp_persistent_diarrhoea_HIV', 'value1']
        p['rp_persistent_diarrhoea_SAM'] = dfd.loc['rp_persistent_diarrhoea_SAM', 'value1']
        p['rp_persistent_diarrhoea_excl_breast'] = dfd.loc['rp_persistent_diarrhoea_excl_breast', 'value1']
        p['rp_persistent_diarrhoea_cont_breast'] = dfd.loc['rp_persistent_diarrhoea_cont_breast', 'value1']
        p['rp_persistent_diarrhoea_HHhandwashing'] = dfd.loc['rp_persistent_diarrhoea_HHhandwashing', 'value1']
        p['rp_persistent_diarrhoea_clean_water'] = dfd.loc['rp_persistent_diarrhoea_clean_water', 'value1']
        p['rp_persistent_diarrhoea_improved_sanitation'] = dfd.loc['rp_persistent_diarrhoea_improved_sanitation', 'value1']
        p['init_prop_diarrhoea_status'] = [
            dfd.loc['init_prop_diarrhoea_status', 'value1'],
            dfd.loc['init_prop_diarrhoea_status', 'value2'],
            dfd.loc['init_prop_diarrhoea_status', 'value3']
        ]
        # diarrhoea incidence by pathogen and relative rates
        p['base_incidence_diarrhoea_by_rotavirus'] = [
            dfd.loc['base_incidence_diarrhoea_by_rotavirus', 'value1'],
            dfd.loc['base_incidence_diarrhoea_by_rotavirus', 'value2'],
            dfd.loc['base_incidence_diarrhoea_by_rotavirus', 'value3']
        ]
        p['base_incidence_diarrhoea_by_shigella'] = [
            dfd.loc['base_incidence_diarrhoea_by_shigella', 'value1'],
            dfd.loc['base_incidence_diarrhoea_by_shigella', 'value2'],
            dfd.loc['base_incidence_diarrhoea_by_shigella', 'value3']
            ]
        p['base_incidence_diarrhoea_by_adenovirus'] = [
            dfd.loc['base_incidence_diarrhoea_by_adenovirus', 'value1'],
            dfd.loc['base_incidence_diarrhoea_by_adenovirus', 'value2'],
            dfd.loc['base_incidence_diarrhoea_by_adenovirus', 'value3']
            ]
        p['base_incidence_diarrhoea_by_crypto'] = [
            dfd.loc['base_incidence_diarrhoea_by_crypto', 'value1'],
            dfd.loc['base_incidence_diarrhoea_by_crypto', 'value2'],
            dfd.loc['base_incidence_diarrhoea_by_crypto', 'value3']
        ]
        p['base_incidence_diarrhoea_by_campylo'] = [
            dfd.loc['base_incidence_diarrhoea_by_campylo', 'value1'],
            dfd.loc['base_incidence_diarrhoea_by_campylo', 'value2'],
            dfd.loc['base_incidence_diarrhoea_by_campylo', 'value3']
            ]
        p['base_incidence_diarrhoea_by_ETEC'] = [
            dfd.loc['base_incidence_diarrhoea_by_ETEC', 'value1'],
            dfd.loc['base_incidence_diarrhoea_by_ETEC', 'value2'],
            dfd.loc['base_incidence_diarrhoea_by_ETEC', 'value3']
            ]
        p['base_incidence_diarrhoea_by_sapovirus'] = [0.005, 0.005, 0.005]
        p['base_incidence_diarrhoea_by_norovirus'] = [0.005, 0.005, 0.005]
        p['base_incidence_diarrhoea_by_astrovirus'] = [0.005, 0.005, 0.005]
        p['base_incidence_diarrhoea_by_EPEC'] = [0.005, 0.005, 0.005]

        p['rr_gi_diarrhoea_HHhandwashing'] = dfd.loc['rr_gi_diarrhoea_HHhandwashing', 'value1']
        p['rr_gi_diarrhoea_improved_sanitation'] = dfd.loc['rr_gi_diarrhoea_improved_sanitation', 'value1']
        p['rr_gi_diarrhoea_clean_water'] = dfd.loc['rr_gi_diarrhoea_improved_sanitation', 'value1']
        p['rr_gi_diarrhoea_HIV'] = dfd.loc['rr_gi_diarrhoea_HIV', 'value1']
        p['rr_gi_diarrhoea_SAM'] = dfd.loc['rr_gi_diarrhoea_malnutrition', 'value1']
        p['rr_gi_diarrhoea_excl_breast'] = dfd.loc['rr_gi_diarrhoea_excl_breastfeeding', 'value1']
        p['rr_gi_diarrhoea_cont_breast'] = dfd.loc['rr_gi_diarrhoea_excl_conti_breast', 'value1']
        # proportion of acute watery and dysentery by pathogen
        p['rotavirus_AWD'] = dfd.loc['proportion_AWD_by_rotavirus', 'value1']
        p['shigella_AWD'] = dfd.loc['proportion_AWD_by_shigella', 'value1']
        p['adenovirus_AWD'] = dfd.loc['proportion_AWD_by_adenovirus', 'value1']
        p['crypto_AWD'] = dfd.loc['proportion_AWD_by_crypto', 'value1']
        p['campylo_AWD'] = dfd.loc['proportion_AWD_by_campylo', 'value1']
        p['ETEC_AWD'] = dfd.loc['proportion_AWD_by_ETEC', 'value1']
        p['sapovirus_AWD'] = dfd.loc['proportion_AWD_by_sapovirus', 'value1']
        p['norovirus_AWD'] = dfd.loc['proportion_AWD_by_norovirus', 'value1']
        p['astrovirus_AWD'] = dfd.loc['proportion_AWD_by_astrovirus', 'value1']
        p['EPEC_AWD'] = dfd.loc['proportion_AWD_by_EPEC', 'value1']
        # pathogens causing fever
        p['rotavirus_fever'] = dfd.loc['fever_by_rotavirus', 'value1']
        p['shigella_fever'] = dfd.loc['fever_by_shigella', 'value1']
        p['adenovirus_fever'] = dfd.loc['fever_by_adenovirus', 'value1']
        p['crypto_fever'] = dfd.loc['fever_by_crypto', 'value1']
        p['campylo_fever'] = dfd.loc['fever_by_campylo', 'value1']
        p['ETEC_fever'] = dfd.loc['fever_by_ETEC', 'value1']
        p['sapovirus_fever'] = dfd.loc['fever_by_sapovirus', 'value1']
        p['norovirus_fever'] = dfd.loc['fever_by_norovirus', 'value1']
        p['astrovirus_fever'] = dfd.loc['fever_by_astrovirus', 'value1']
        p['EPEC_fever'] = dfd.loc['fever_by_EPEC', 'value1']
        # pathogens causing vomiting
        p['rotavirus_vomiting'] = dfd.loc['vomiting_by_rotavirus', 'value1']
        p['shigella_vomiting'] = dfd.loc['vomiting_by_shigella', 'value1']
        p['adenovirus_vomiting'] = dfd.loc['vomiting_by_adenovirus', 'value1']
        p['crypto_vomiting'] = dfd.loc['vomiting_by_crypto', 'value1']
        p['campylo_vomiting'] = dfd.loc['vomiting_by_campylo', 'value1']
        p['ETEC_vomiting'] = dfd.loc['vomiting_by_ETEC', 'value1']
        p['sapovirus_vomiting'] = dfd.loc['vomiting_by_sapovirus', 'value1']
        p['norovirus_vomiting'] = dfd.loc['vomiting_by_norovirus', 'value1']
        p['astrovirus_vomiting'] = dfd.loc['vomiting_by_astrovirus', 'value1']
        p['EPEC_vomiting'] = dfd.loc['vomiting_by_EPEC', 'value1']
        # pathogens causing dehydration
        p['rotavirus_dehydration'] = dfd.loc['dehydration_by_rotavirus', 'value1']
        p['shigella_dehydration'] = dfd.loc['dehydration_by_shigella', 'value1']
        p['adenovirus_dehydration'] = dfd.loc['dehydration_by_adenovirus', 'value1']
        p['crypto_dehydration'] = dfd.loc['dehydration_by_crypto', 'value1']
        p['campylo_dehydration'] = dfd.loc['dehydration_by_campylo', 'value1']
        p['ETEC_dehydration'] = dfd.loc['dehydration_by_ETEC', 'value1']
        p['sapovirus_dehydration'] = dfd.loc['dehydration_by_sapovirus', 'value1']
        p['norovirus_dehydration'] = dfd.loc['dehydration_by_norovirus', 'value1']
        p['astrovirus_dehydration'] = dfd.loc['dehydration_by_astrovirus', 'value1']
        p['EPEC_dehydration'] = dfd.loc['dehydration_by_EPEC', 'value1']
        # prolonged diarrhoea by pathogen
        p['rotavirus_prolonged_diarr'] = dfd.loc['prolonged_diarr_rotavirus', 'value1']
        p['shigella_prolonged_diarr'] = dfd.loc['prolonged_diarr_shigella', 'value1']
        p['adenovirus_prolonged_diarr'] = dfd.loc['prolonged_diarr_adenovirus', 'value1']
        p['crypto_prolonged_diarr'] = dfd.loc['prolonged_diarr_crypto', 'value1']
        p['campylo_prolonged_diarr'] = dfd.loc['prolonged_diarr_campylo', 'value1']
        p['ETEC_prolonged_diarr'] = dfd.loc['prolonged_diarr_ETEC', 'value1']
        p['sapovirus_prolonged_diarr'] = dfd.loc['prolonged_diarr_sapovirus', 'value1']
        p['norovirus_prolonged_diarr'] = dfd.loc['prolonged_diarr_norovirus', 'value1']
        p['astrovirus_prolonged_diarr'] = dfd.loc['prolonged_diarr_astrovirus', 'value1']
        p['EPEC_prolonged_diarr'] = dfd.loc['prolonged_diarr_EPEC', 'value1']
        # parameters for acute diarrhoea becoming persistent
        p['prob_dysentery_become_persistent'] = dfd.loc['prob_dysentery_become_persistent', 'value1']
        p['prob_watery_diarr_become_persistent'] = dfd.loc['prob_watery_diarr_become_persistent', 'value1']
        p['rr_bec_persistent_age12to23'] = dfd.loc['rr_bec_persistent_age12to23', 'value1']
        p['rr_bec_persistent_age24to59'] = dfd.loc['rr_bec_persistent_age24to59', 'value1']
        p['rr_bec_persistent_HIV'] = dfd.loc['prob_dysentery_become_persistent', 'value1']
        p['rr_bec_persistent_SAM'] = dfd.loc['rr_bec_persistent_HIV', 'value1']
        p['rr_bec_persistent_excl_breast'] = dfd.loc['rr_bec_persistent_excl_breast', 'value1']
        p['rr_bec_persistent_cont_breast'] = dfd.loc['rr_bec_persistent_cont_breast', 'value1']
        # health system iccm stuff
        p['prob_correct_id_diarrhoea_dehydration'] = 0.8
        p['prob_correct_id_danger_sign'] = 0.7
        p['prob_correct_id_persist_or_bloody_diarrhoea'] = 0.8
        p['prob_correctly_classified_diarrhoea_danger_sign'] = 0.8
        p['prob_correctly_classified_persist_or_bloody_diarrhoea'] = 0.8
        p['prob_correctly_classified_diarrhoea'] = 0.8
        p['prob_correct_referral_decision'] = 0.8
        p['prob_correct_treatment_advice_given'] = 0.8
        p['dhs_care_seeking_2010'] = 0.58
        p['IMCI_effectiveness_2010'] = 0.6
        p['r_death_diarrhoea'] = 0.3
        p['prob_prolonged_to_persistent_diarr'] = 0.2866
        p['case_fatality_rate_AWD'] = dfd.loc['case_fatality_rate_AWD', 'value1']
        p['case_fatality_rate_dysentery'] = dfd.loc['case_fatality_rate_dysentery', 'value1']
        p['case_fatality_rate_persistent'] = dfd.loc['case_fatality_rate_persistent', 'value1']
        p['rr_diarr_death_age12to23mo'] = dfd.loc['rr_diarr_death_age12to23mo', 'value1']
        p['rr_diarr_death_age24to59mo'] = dfd.loc['rr_diarr_death_age24to59mo', 'value1']
        p['rr_diarr_death_dehydration'] = dfd.loc['rr_diarr_death_dehydration', 'value1']
        p['rr_diarr_death_HIV'] = dfd.loc['rr_diarr_death_HIV', 'value1']
        p['rr_diarr_death_SAM'] = dfd.loc['rr_diarr_death_SAM', 'value1']
        p['r_recovery_AWD'] = 0.8
        p['r_recovery_dysentery'] = 0.5
        p['rr_recovery_dehydration'] = 0.81


        # DALY weights  # Could load these
        if 'HealthBurden' in self.sim.modules.keys():
            p['daly_mild_diarrhoea'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=32)
            p['daly_moderate_diarrhoea'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=35)
            p['daly_severe_diarrhoea'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=34)

    def initialise_population(self, population):
        """Set our property values for the initial population.
        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.
        :param population: the population of individuals
        """
        df = population.props  # a shortcut to the data-frame storing data for individuals
        rng = self.rng

        # DEFAULTS
        df['gi_diarrhoea_status'] = False
        df['gi_diarrhoea_acute_type'] = np.nan
        df['gi_diarrhoea_type'] = np.nan
        df['gi_persistent_diarrhoea'] = np.nan
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

        # TODO: looks like some properties not defined here?

        # # # # # # # # # # # # # PREVALENCE OF DIARRHOEA AT THE START OF SIMULATION 2010 # # # # # # # # # # # # #
        '''

        # # # # # # # # # DIAGNOSED AND TREATED BASED ON CARE SEEKING AND IMCI EFFECTIVENESS # # # # # # # # #
        # currently no init_diarrhoea_idx is empty
        init_diarrhoea_idx = df.index[df.is_alive & df.age_exact_years < 5 & (df.gi_diarrhoea_status is True)]
        random_draw = self.sim.rng.random_sample(size=len(init_diarrhoea_idx))
        prob_sought_care = pd.Series(self.dhs_care_seeking_2010, index=init_diarrhoea_idx)
        sought_care = prob_sought_care > random_draw
        sought_care_idx = prob_sought_care.index[sought_care]

        for i in sought_care_idx:
            random_draw1 = self.sim.rng.random_sample(size=len(sought_care_idx))
            diagnosed_and_treated = df.index[
                df.is_alive & (random_draw1 < self.parameters['IMCI_effectiveness_2010'])
                & (df.age_years < 5)]
            df.at[diagnosed_and_treated[i], 'gi_diarrhoea_status'] = False

        # # # # # # # # # # ASSIGN RECOVERY AND DEATH TO BASELINE DIARRHOEA CASES # # # # # # # # # #

        not_treated_diarrhoea_idx = df.index[df.is_alive & df.age_exact_years < 5 & (df.gi_diarrhoea_status is True)]
        for i in not_treated_diarrhoea_idx:
            random_draw2 = self.sim.rng.random_sample(size=len(not_treated_diarrhoea_idx))
            death_diarrhoea = df.index[
                df.is_alive & (random_draw2 < self.parameters['r_death_diarrhoea'])
                & (df.age_years < 5)]
            if death_diarrhoea[i]:
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, i, 'Diarrhoea'), self.sim.date)
                df.at[i, 'gi_diarrhoea_status'] = False
            else:
                df.at[i, 'gi_diarrhoea_status'] = False
                '''

    def initialise_simulation(self, sim):
        """
        Get ready for simulation start.
        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        # add the basic event for acute diarrhoea ---------------------------------------------------
        sim.schedule_event(AcuteDiarrhoeaEvent(self), sim.date + DateOffset(months=0))

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

        # add an event to log to screen
        sim.schedule_event(DiarrhoeaLoggingEvent(self), sim.date + DateOffset(months=12))

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        This is called by the simulation whenever a new person is born.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        df = self.sim.population.props

        df.at[child_id, 'gi_recovered_date'] = pd.NaT
        df.at[child_id, 'gi_diarrhoea_status'] = False
        df.at[child_id, 'gi_diarrhoea_acute_type'] = np.nan
        df.at[child_id, 'gi_diarrhoea_type'] = np.nan
        df.at[child_id, 'gi_persistent_diarrhoea'] = np.nan
        df.at[child_id, 'gi_dehydration_status'] = 'no dehydration'
        df.at[child_id, 'date_of_onset_diarrhoea'] = pd.NaT
        df.at[child_id, 'gi_recovered_date'] = pd.NaT
        df.at[child_id, 'gi_diarrhoea_death_date'] = pd.NaT
        df.at[child_id, 'gi_diarrhoea_death'] = False
        df.at[child_id, 'gi_diarrhoea_pathogen'] = np.nan
        df.at[child_id, 'gi_diarrhoea_count'] = 0

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug('This is Diarrhoea, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)
        pass

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
        super().__init__(module, frequency=DateOffset(months=3))

    def apply(self, population):
        # TODO: Say what this event is for and what it will do.
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props
        m = self.module
        rng = m.rng
        now = self.sim.date
        p = self.module.parameters

        # SET DIARRHOEAL PROPERTIES BACK TO FALSE FOR THE EVENT
        '''df['gi_diarrhoea_status'] = False
        df['gi_diarrhoea_acute_type'] = np.nan
        df['gi_persistent_diarrhoea'] = np.nan
        df['gi_dehydration_status'] = 'no dehydration'
        df['di_dehydration_present'] = False
        df['date_of_onset_diarrhoea'] = pd.NaT
        df['gi_recovered_date'] = pd.NaT
        df['gi_diarrhoea_death_date'] = pd.NaT
        df['gi_diarrhoea_death'] = False
        df['gi_diarrhoea_type'] = np.nan
        df['gi_diarrhoea_pathogen'] = np.nan '''

        # indexes  TODO: Can you make these names more descriptive? [btw- these things are sometimes called 'masks']
        # These applications of probabilities of onset of something could be made easier with the LM
        no_diarrhoea0 = df.is_alive & (df.gi_diarrhoea_status == False) & (df.age_exact_years < 1)
        no_diarrhoea1 = \
            df.is_alive & (df.gi_diarrhoea_status == False) & (df.age_exact_years >= 1) & (df.age_exact_years < 2)
        no_diarrhoea2 = \
            df.is_alive & (df.gi_diarrhoea_status == False) & (df.age_exact_years >= 2) & (df.age_exact_years < 5)
        no_diarrhoea_under5 = df.is_alive & (df.gi_diarrhoea_status == False) & (df.age_exact_years < 5)
        current_no_diarrhoea = df.index[no_diarrhoea_under5]

        # assign baseline incidence of diarrhoea by pathogen
        diarrhoea_rotavirus0 = pd.Series(m.base_incidence_diarrhoea_by_rotavirus[0], index=df.index[no_diarrhoea0])
        diarrhoea_rotavirus1 = pd.Series(m.base_incidence_diarrhoea_by_rotavirus[1], index=df.index[no_diarrhoea1])
        diarrhoea_rotavirus2 = pd.Series(m.base_incidence_diarrhoea_by_rotavirus[2], index=df.index[no_diarrhoea2])

        diarrhoea_shigella0 = pd.Series(m.base_incidence_diarrhoea_by_shigella[0], index=df.index[no_diarrhoea0])
        diarrhoea_shigella1 = pd.Series(m.base_incidence_diarrhoea_by_shigella[1], index=df.index[no_diarrhoea1])
        diarrhoea_shigella2 = pd.Series(m.base_incidence_diarrhoea_by_shigella[2], index=df.index[no_diarrhoea2])

        diarrhoea_adenovirus0 = pd.Series(m.base_incidence_diarrhoea_by_adenovirus[0], index=df.index[no_diarrhoea0])
        diarrhoea_adenovirus1 = pd.Series(m.base_incidence_diarrhoea_by_adenovirus[1], index=df.index[no_diarrhoea1])
        diarrhoea_adenovirus2 = pd.Series(m.base_incidence_diarrhoea_by_adenovirus[2], index=df.index[no_diarrhoea2])

        diarrhoea_crypto0 = pd.Series(m.base_incidence_diarrhoea_by_crypto[0], index=df.index[no_diarrhoea0])
        diarrhoea_crypto1 = pd.Series(m.base_incidence_diarrhoea_by_crypto[1], index=df.index[no_diarrhoea1])
        diarrhoea_crypto2 = pd.Series(m.base_incidence_diarrhoea_by_crypto[2], index=df.index[no_diarrhoea2])

        diarrhoea_campylo0 = pd.Series(m.base_incidence_diarrhoea_by_campylo[0], index=df.index[no_diarrhoea0])
        diarrhoea_campylo1 = pd.Series(m.base_incidence_diarrhoea_by_campylo[1], index=df.index[no_diarrhoea1])
        diarrhoea_campylo2 = pd.Series(m.base_incidence_diarrhoea_by_campylo[2], index=df.index[no_diarrhoea2])

        diarrhoea_ETEC0 = pd.Series(m.base_incidence_diarrhoea_by_ETEC[0], index=df.index[no_diarrhoea0])
        diarrhoea_ETEC1 = pd.Series(m.base_incidence_diarrhoea_by_ETEC[1], index=df.index[no_diarrhoea1])
        diarrhoea_ETEC2 = pd.Series(m.base_incidence_diarrhoea_by_ETEC[2], index=df.index[no_diarrhoea2])

        diarrhoea_sapovirus0 = pd.Series(m.base_incidence_diarrhoea_by_sapovirus[0], index=df.index[no_diarrhoea0])
        diarrhoea_sapovirus1 = pd.Series(m.base_incidence_diarrhoea_by_sapovirus[1], index=df.index[no_diarrhoea1])
        diarrhoea_sapovirus2 = pd.Series(m.base_incidence_diarrhoea_by_sapovirus[2], index=df.index[no_diarrhoea2])

        diarrhoea_norovirus0 = pd.Series(m.base_incidence_diarrhoea_by_norovirus[0], index=df.index[no_diarrhoea0])
        diarrhoea_norovirus1 = pd.Series(m.base_incidence_diarrhoea_by_norovirus[1], index=df.index[no_diarrhoea1])
        diarrhoea_norovirus2 = pd.Series(m.base_incidence_diarrhoea_by_norovirus[2], index=df.index[no_diarrhoea2])

        diarrhoea_astrovirus0 = pd.Series(m.base_incidence_diarrhoea_by_astrovirus[0], index=df.index[no_diarrhoea0])
        diarrhoea_astrovirus1 = pd.Series(m.base_incidence_diarrhoea_by_astrovirus[1], index=df.index[no_diarrhoea1])
        diarrhoea_astrovirus2 = pd.Series(m.base_incidence_diarrhoea_by_astrovirus[2], index=df.index[no_diarrhoea2])

        diarrhoea_EPEC0 = pd.Series(m.base_incidence_diarrhoea_by_EPEC[0], index=df.index[no_diarrhoea0])
        diarrhoea_EPEC1 = pd.Series(m.base_incidence_diarrhoea_by_EPEC[1], index=df.index[no_diarrhoea1])
        diarrhoea_EPEC2 = pd.Series(m.base_incidence_diarrhoea_by_EPEC[2], index=df.index[no_diarrhoea2])

        # concatenating plus sorting
        eff_prob_rotavirus = pd.concat([diarrhoea_rotavirus0, diarrhoea_rotavirus1, diarrhoea_rotavirus2], axis=0).sort_index()
        eff_prob_shigella = pd.concat([diarrhoea_shigella0, diarrhoea_shigella1, diarrhoea_shigella2], axis=0).sort_index()
        eff_prob_adenovirus = pd.concat([diarrhoea_adenovirus0, diarrhoea_adenovirus1, diarrhoea_adenovirus2], axis=0).sort_index()
        eff_prob_crypto = pd.concat([diarrhoea_crypto0, diarrhoea_crypto1, diarrhoea_crypto2], axis=0).sort_index()
        eff_prob_campylo = pd.concat([diarrhoea_campylo0, diarrhoea_campylo1, diarrhoea_campylo2], axis=0).sort_index()
        eff_prob_ETEC = pd.concat([diarrhoea_ETEC0, diarrhoea_ETEC1, diarrhoea_ETEC2], axis=0).sort_index()
        eff_prob_sapovirus = pd.concat([diarrhoea_sapovirus0, diarrhoea_sapovirus1, diarrhoea_sapovirus2], axis=0).sort_index()
        eff_prob_norovirus = pd.concat([diarrhoea_norovirus0, diarrhoea_norovirus1, diarrhoea_norovirus2], axis=0).sort_index()
        eff_prob_astrovirus = pd.concat([diarrhoea_astrovirus0, diarrhoea_astrovirus1, diarrhoea_astrovirus2], axis=0).sort_index()
        eff_prob_EPEC = pd.concat([diarrhoea_EPEC0, diarrhoea_EPEC1, diarrhoea_EPEC2], axis=0).sort_index()

        eff_prob_all_pathogens = pd.concat([eff_prob_rotavirus, eff_prob_shigella, eff_prob_adenovirus,
                                            eff_prob_crypto, eff_prob_campylo, eff_prob_ETEC, eff_prob_sapovirus,
                                            eff_prob_norovirus, eff_prob_astrovirus, eff_prob_EPEC], axis=1)

        eff_prob_all_pathogens.loc[no_diarrhoea_under5 & df.li_no_access_handwashing == False] \
            *= m.rr_gi_diarrhoea_HHhandwashing
        eff_prob_all_pathogens.loc[no_diarrhoea_under5 & df.li_no_clean_drinking_water == False] \
            *= m.rr_gi_diarrhoea_clean_water
        eff_prob_all_pathogens.loc[no_diarrhoea_under5 & df.li_unimproved_sanitation == False] \
            *= m.rr_gi_diarrhoea_improved_sanitation
        eff_prob_all_pathogens.loc[no_diarrhoea_under5 & df.has_hiv] \
            *= m.rr_gi_diarrhoea_HIV
        eff_prob_all_pathogens.loc[no_diarrhoea_under5 & df.malnutrition] \
            *= m.rr_gi_diarrhoea_SAM
        eff_prob_all_pathogens.loc[no_diarrhoea_under5 & df.exclusive_breastfeeding & (df.age_exact_years <= 0.5)] \
            *= m.rr_gi_diarrhoea_excl_breast
        eff_prob_all_pathogens.loc[no_diarrhoea_under5 & df.continued_breastfeeding & (df.age_exact_years > 0.5) &
                                   (df.age_exact_years < 2)] *= m.rr_gi_diarrhoea_cont_breast # # # # remove this

        # cumulative sum to determine which pathogen is the cause of diarrhoea
        random_draw_all = pd.Series(rng.random_sample(size=len(current_no_diarrhoea)), index=current_no_diarrhoea)
        eff_prob_none = 1 - eff_prob_all_pathogens.sum(axis=1)
        dfx = pd.concat([eff_prob_none, eff_prob_all_pathogens], axis=1)
        dfx = dfx.cumsum(axis=1)
        dfx.columns = ['prob_none', 'rotavirus', 'shigella', 'adenovirus', 'cryptosporidium',
                       'campylobacter', 'ST-ETEC', 'sapovirus', 'norovirus', 'astrovirus', 'tEPEC']
        dfx['random_draw_all'] = random_draw_all

        # running counts
        pathogen_episodes = {'rotavirus': np.nan, 'shigella': np.nan, 'adenovirus': np.nan, 'cryptosporidium': np.nan,
                             'campylobacter': np.nan, 'ST-ETEC': np.nan, 'sapovirus': np.nan, 'norovirus': np.nan,
                             'astrovirus': np.nan, 'tEPEC': np.nan}

        for i, column in enumerate(dfx.columns):
            # go through each pathogen and assign the pathogen and status
            if column in ('prob_none', 'random_draw_all'):
                # skip probability of none and random draw columns
                continue

            idx_to_infect = dfx.index[
                ((dfx.iloc[:, i - 1] < dfx.random_draw_all)
                 & (dfx.loc[:, column] >= dfx.random_draw_all))]
            df.loc[idx_to_infect, 'gi_diarrhoea_pathogen'] = column
            df.loc[idx_to_infect, 'gi_diarrhoea_status'] = True
            df.loc[idx_to_infect, 'gi_diarrhoea_type'] = 'acute'
            df.loc[idx_to_infect, 'gi_diarrhoea_count'] += 1
            pathogen_episodes.update(
                df.loc[df.is_alive & (df.age_exact_years < 5), 'gi_diarrhoea_pathogen'].value_counts().to_dict())

        '''pop_under5 = len(df[df.is_alive & (df.age_exact_years < 5)])
        rota_inc = pathogen_episodes.get('rotavirus')
        shig_inc = pathogen_episodes.get('shigella')
        adeno_inc = pathogen_episodes.get('adenovirus')
        crypto_inc = pathogen_episodes.get('cryptosporidium')
        campy_inc = pathogen_episodes.get('campylobacter')
        etec_inc = pathogen_episodes.get('ST-ETEC')
        sapo_inc = pathogen_episodes.get('sapovirus')
        noro_inc = pathogen_episodes.get('norovirus')
        astro_inc = pathogen_episodes.get('astrovirus')
        epec_inc = pathogen_episodes.get('tEPEC')

        logger.info('%s|number_pathogen_diarrhoea|%s', self.sim.date,
                    {'rotavirus': rota_inc,
                     'shigella': shig_inc,
                     'adenovirus': adeno_inc,
                     'cryptosporidium': crypto_inc,
                     'campylobacter': campy_inc,
                     'ETEC': etec_inc,
                     'sapovirus': sapo_inc,
                     'norovirus': noro_inc,
                     'astrovirus': astro_inc,
                     'tEPEC': epec_inc
                     })'''

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Here's an example of another way to organise this job of working out who gets which (if any) pathogen:

        # Make a dict to hold the equations that govern the probability that a person gets a pathogen
        prob_eqs = dict()
        prob_eqs.update({
            # Define the equation using LinearModel (note that this stage could be done in read_parms)
            'rota':  LinearModel(LinearModelType.MULTIPLICATIVE,
                                1,
                                Predictor('age_years')
                                        .when('<1', m.base_incidence_diarrhoea_by_rotavirus[0])
                                        .when('<2', m.base_incidence_diarrhoea_by_rotavirus[1])
                                        .when('<3', m.base_incidence_diarrhoea_by_rotavirus[2])
                                        .otherwise(0.0),
                                Predictor('li_no_access_handwashing')
                                        .when('False', m.rr_gi_diarrhoea_HHhandwashing),
                                Predictor().
                                        when('continued_breastfeeding & age_exact_years > 0.5', m.rr_gi_diarrhoea_cont_breast)
                                  )
        })

        prob_eqs.update({
            'shigella':  LinearModel(LinearModelType.MULTIPLICATIVE,
                                1,
                                Predictor('age_years')
                                        .when('<1', m.base_incidence_diarrhoea_by_shigella[0])
                                        .when('<2', m.base_incidence_diarrhoea_by_shigella[1])
                                        .when('<3', m.base_incidence_diarrhoea_by_shigella[2])
                                        .otherwise(0.0),
                                Predictor('li_no_access_handwashing')
                                        .when('False', m.rr_gi_diarrhoea_HHhandwashing),
                                Predictor().
                                        when('continued_breastfeeding & age_exact_years > 0.5', m.rr_gi_diarrhoea_cont_breast)
                                  )
        })

        # Compute probabilities and organise in a dataframe
        probs = pd.DataFrame()
        for k,v in prob_eqs.items():
            probs[k] = v.predict(df.loc[df['age_years']<5])

        # Declare that pathogens are mutally exclusive and do the random choice for each person
        probs['none'] = 1 - probs.sum(axis=1)
        outcome = pd.Series(data='none', index = probs.index)

        # then we work out if the people who got the diarahhea, get a symptom:
        prob_symptoms = {
            'rota': {'fever': 0.1, 'dehyration': 0.2},
            'shigella': {'fever': 0.1, 'dehydration': 0.2}
        }

        for i in outcome.index:
            outcome[i] = rng.choice(probs.columns, p=probs.loc[i].values)
            # (NB. there is probably a faster way to do this using apply than the for loop)


            # this person will get the following symptoms:
            if outcome[i] != 'none':
                symp = prob_symptoms[outcome[i]]

                # print([str(s) + '_' for s in symp.keys() if rng.rand()<symp[s]])

        # check out what people got:
        outcome.value_counts()



        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


        # ----------------------- ALLOCATE A RANDOM DATE OF ONSET OF ACUTE DIARRHOEA -----------------------
        incident_acute_diarrhoea = df.index[df.is_alive & df.gi_diarrhoea_status & (df.age_exact_years < 5)]
        random_draw_days = np.random.randint(0, 90, size=len(incident_acute_diarrhoea))  # runs every 3 months
        adding_days = pd.to_timedelta(random_draw_days, unit='d')
        df.loc[incident_acute_diarrhoea, 'date_of_onset_diarrhoea'] = self.sim.date + adding_days
        # -------------------------------------------------------------------------------------------------

        # ------------------- ASSIGN DIARRHOEA TYPE - DYSENTERY OR ACUTE WATERY DIARRHOEA ----------------------
        # # # ROTAVIRUS # # #
        diarr_rotavirus_idx = df.index[df.gi_diarrhoea_pathogen == 'rotavirus'] & incident_acute_diarrhoea
        p_acute_watery_rotavirus = pd.Series(self.module.rotavirus_AWD, index=diarr_rotavirus_idx)
        random_draw = pd.Series(rng.random_sample(size=len(diarr_rotavirus_idx)), index=diarr_rotavirus_idx)
        diarr_rota_AWD = p_acute_watery_rotavirus >= random_draw
        diarr_rota_AWD_idx = p_acute_watery_rotavirus.index[diarr_rota_AWD]
        df.loc[diarr_rota_AWD_idx, 'gi_diarrhoea_acute_type'] = 'acute watery diarrhoea'
        diarr_rota_dysentery = p_acute_watery_rotavirus < random_draw
        diarr_rota_dysentery_idx = p_acute_watery_rotavirus.index[diarr_rota_dysentery]
        df.loc[diarr_rota_dysentery_idx, 'gi_diarrhoea_acute_type'] = 'dysentery'

        # # # SHIGELLA # # #
        diarr_shigella_idx = \
            df.index[df.gi_diarrhoea_pathogen == 'shigella'] & incident_acute_diarrhoea
        p_acute_watery_shigella = pd.Series(self.module.shigella_AWD, index=diarr_shigella_idx)
        random_draw1 = pd.Series(rng.random_sample(size=len(diarr_shigella_idx)), index=diarr_shigella_idx)
        diarr_shigella_AWD = p_acute_watery_shigella >= random_draw1
        diarr_shigella_AWD_idx = p_acute_watery_shigella.index[diarr_shigella_AWD]
        df.loc[diarr_shigella_AWD_idx, 'gi_diarrhoea_acute_type'] = 'acute watery diarrhoea'
        diarr_shigella_dysentery = p_acute_watery_shigella < random_draw1
        diarr_shigella_dysentery_idx = p_acute_watery_shigella.index[diarr_shigella_dysentery]
        df.loc[diarr_shigella_dysentery_idx, 'gi_diarrhoea_acute_type'] = 'dysentery'

        # # # ADENOVIRUS # # #
        diarr_adenovirus_idx = df.index[df.gi_diarrhoea_pathogen == 'adenovirus'] & incident_acute_diarrhoea
        p_acute_watery_adeno = pd.Series(self.module.adenovirus_AWD, index=diarr_adenovirus_idx)
        random_draw2 = pd.Series(rng.random_sample(size=len(diarr_adenovirus_idx)), index=diarr_adenovirus_idx)
        diarr_adeno_AWD = p_acute_watery_adeno >= random_draw2
        diarr_adeno_AWD_idx = p_acute_watery_adeno.index[diarr_adeno_AWD]
        df.loc[diarr_adeno_AWD_idx, 'gi_diarrhoea_acute_type'] = 'acute watery diarrhoea'
        diarr_adeno_dysentery = p_acute_watery_adeno < random_draw2
        diarr_adeno_dysentery_idx = p_acute_watery_adeno.index[diarr_adeno_dysentery]
        df.loc[diarr_adeno_dysentery_idx, 'gi_diarrhoea_acute_type'] = 'dysentery'

        # # # CRYPTOSPORIDIUM # # #
        diarr_crypto_idx = df.index[df.gi_diarrhoea_pathogen == 'cryptosporidium'] & incident_acute_diarrhoea
        p_acute_watery_crypto = pd.Series(self.module.crypto_AWD, index=diarr_crypto_idx)
        random_draw3 = pd.Series(rng.random_sample(size=len(diarr_crypto_idx)), index=diarr_crypto_idx)
        diarr_crypto_AWD = p_acute_watery_crypto >= random_draw3
        diarr_crypto_AWD_idx = p_acute_watery_crypto.index[diarr_crypto_AWD]
        df.loc[diarr_crypto_AWD_idx, 'gi_diarrhoea_acute_type'] = 'acute watery diarrhoea'
        diarr_crypto_dysentery = p_acute_watery_crypto < random_draw3
        diarr_crypto_dysentery_idx = p_acute_watery_crypto.index[diarr_crypto_dysentery]
        df.loc[diarr_crypto_dysentery_idx, 'gi_diarrhoea_acute_type'] = 'dysentery'

        # # # CAMPYLOBACTER # # #
        diarr_campylo_idx = \
            df.index[df.gi_diarrhoea_pathogen == 'campylobacter'] & incident_acute_diarrhoea
        p_acute_watery_campylo = pd.Series(self.module.campylo_AWD, index=diarr_campylo_idx)
        random_draw4 = pd.Series(rng.random_sample(size=len(diarr_campylo_idx)), index=diarr_campylo_idx)
        diarr_campylo_AWD = p_acute_watery_campylo >= random_draw4
        diarr_campylo_AWD_idx = p_acute_watery_campylo.index[diarr_campylo_AWD]
        df.loc[diarr_campylo_AWD_idx, 'gi_diarrhoea_acute_type'] = 'acute watery diarrhoea'
        diarr_campylo_dysentery = p_acute_watery_campylo < random_draw4
        diarr_campylo_dysentery_idx = p_acute_watery_campylo.index[diarr_campylo_dysentery]
        df.loc[diarr_campylo_dysentery_idx, 'gi_diarrhoea_acute_type'] = 'dysentery'

        # # # ST-ETEC # # #
        diarr_ETEC_idx = df.index[df.gi_diarrhoea_pathogen == 'ST-ETEC'] & incident_acute_diarrhoea
        p_acute_watery_ETEC = pd.Series(self.module.ETEC_AWD, index=diarr_ETEC_idx)
        random_draw5 = pd.Series(rng.random_sample(size=len(diarr_ETEC_idx)), index=diarr_ETEC_idx)
        diarr_ETEC_AWD = p_acute_watery_ETEC >= random_draw5
        diarr_ETEC_AWD_idx = p_acute_watery_ETEC.index[diarr_ETEC_AWD]
        df.loc[diarr_ETEC_AWD_idx, 'gi_diarrhoea_acute_type'] = 'acute watery diarrhoea'
        diarr_ETEC_dysentery = p_acute_watery_ETEC < random_draw5
        diarr_ETEC_dysentery_idx = p_acute_watery_ETEC.index[diarr_ETEC_dysentery]
        df.loc[diarr_ETEC_dysentery_idx, 'gi_diarrhoea_acute_type'] = 'dysentery'

        # # # SAPOVIRUS # # #
        diarr_sapovirus_idx = df.index[df.gi_diarrhoea_pathogen == 'sapovirus'] & incident_acute_diarrhoea
        p_acute_watery_sapovirus = pd.Series(self.module.sapovirus_AWD, index=diarr_sapovirus_idx)
        random_draw6 = pd.Series(rng.random_sample(size=len(diarr_sapovirus_idx)), index=diarr_sapovirus_idx)
        diarr_sapovirus_AWD = p_acute_watery_sapovirus >= random_draw6
        diarr_sapovirus_AWD_idx = p_acute_watery_sapovirus.index[diarr_sapovirus_AWD]
        df.loc[diarr_sapovirus_AWD_idx, 'gi_diarrhoea_acute_type'] = 'acute watery diarrhoea'
        diarr_sapovirus_dysentery = p_acute_watery_sapovirus < random_draw6
        diarr_sapovirus_dysentery_idx = p_acute_watery_sapovirus.index[diarr_sapovirus_dysentery]
        df.loc[diarr_sapovirus_dysentery_idx, 'gi_diarrhoea_acute_type'] = 'dysentery'

        # # # NOROVIRUS # # #
        diarr_norovirus_idx = df.index[df.gi_diarrhoea_pathogen == 'norovirus'] & incident_acute_diarrhoea
        p_acute_watery_norovirus = pd.Series(self.module.norovirus_AWD, index=diarr_norovirus_idx)
        random_draw7 = pd.Series(rng.random_sample(size=len(diarr_norovirus_idx)), index=diarr_norovirus_idx)
        diarr_norovirus_AWD = p_acute_watery_norovirus >= random_draw7
        diarr_norovirus_AWD_idx = p_acute_watery_norovirus.index[diarr_norovirus_AWD]
        df.loc[diarr_norovirus_AWD_idx, 'gi_diarrhoea_acute_type'] = 'acute watery diarrhoea'
        diarr_norovirus_dysentery = p_acute_watery_norovirus < random_draw7
        diarr_norovirus_dysentery_idx = p_acute_watery_norovirus.index[diarr_norovirus_dysentery]
        df.loc[diarr_norovirus_dysentery_idx, 'gi_diarrhoea_acute_type'] = 'dysentery'

        # # # ASTROVIRUS # # #
        diarr_astrovirus_idx = df.index[df.gi_diarrhoea_pathogen == 'astrovirus'] & incident_acute_diarrhoea
        p_acute_watery_astrovirus = pd.Series(self.module.astrovirus_AWD, index=diarr_astrovirus_idx)
        random_draw8 = pd.Series(rng.random_sample(size=len(diarr_astrovirus_idx)), index=diarr_astrovirus_idx)
        diarr_astrovirus_AWD = p_acute_watery_astrovirus >= random_draw8
        diarr_astrovirus_AWD_idx = p_acute_watery_astrovirus.index[diarr_astrovirus_AWD]
        df.loc[diarr_astrovirus_AWD_idx, 'gi_diarrhoea_acute_type'] = 'acute watery diarrhoea'
        diarr_astrovirus_dysentery = p_acute_watery_astrovirus < random_draw8
        diarr_astrovirus_dysentery_idx = p_acute_watery_astrovirus.index[diarr_astrovirus_dysentery]
        df.loc[diarr_astrovirus_dysentery_idx, 'gi_diarrhoea_acute_type'] = 'dysentery'

        # # # tEPEC # # #
        diarr_EPEC_idx = df.index[df.gi_diarrhoea_pathogen == 'tEPEC'] & incident_acute_diarrhoea
        p_acute_watery_EPEC = pd.Series(self.module.EPEC_AWD, index=diarr_EPEC_idx)
        random_draw9 = pd.Series(rng.random_sample(size=len(diarr_EPEC_idx)), index=diarr_EPEC_idx)
        diarr_EPEC_AWD = p_acute_watery_EPEC >= random_draw9
        diarr_EPEC_AWD_idx = p_acute_watery_EPEC.index[diarr_EPEC_AWD]
        df.loc[diarr_EPEC_AWD_idx, 'gi_diarrhoea_acute_type'] = 'acute watery diarrhoea'
        diarr_EPEC_dysentery = p_acute_watery_EPEC < random_draw9
        diarr_EPEC_dysentery_idx = p_acute_watery_EPEC.index[diarr_EPEC_dysentery]
        df.loc[diarr_EPEC_dysentery_idx, 'gi_diarrhoea_acute_type'] = 'dysentery'

        # -------------------------------------------------------------------------------------------
        # # # # # # # # # # # # # # # # # ASSIGN DEHYDRATION LEVELS # # # # # # # # # # # # # # # # #
        # ANY DEHYDRATION CAUSED BY PATHOGEN
        # ROTAVIRUS
        # TODO: should this be a call to self.module.parameters['rotavirus_dehyration']?
        p_dehydration_rotavirus = pd.Series(self.module.rotavirus_dehydration, index=diarr_rotavirus_idx)
        random_draw_c = pd.Series(rng.random_sample(size=len(diarr_rotavirus_idx)), index=diarr_rotavirus_idx)
        diarr_rota_dehydration = p_dehydration_rotavirus >= random_draw_c
        diarr_rota_dehydration_idx = p_dehydration_rotavirus.index[diarr_rota_dehydration]
        df.loc[diarr_rota_dehydration_idx, 'di_dehydration_present'] = True

        # SHIGELLA
        p_dehydration_shigella = pd.Series(self.module.shigella_dehydration, index=diarr_shigella_idx)
        random_draw_c = pd.Series(rng.random_sample(size=len(diarr_shigella_idx)), index=diarr_shigella_idx)
        diarr_shigella_dehydration = p_dehydration_shigella >= random_draw_c
        diarr_shigella_dehydration_idx = p_dehydration_shigella.index[diarr_shigella_dehydration]
        df.loc[diarr_shigella_dehydration_idx, 'di_dehydration_present'] = True
        # ADENOVIRUS
        p_dehydration_adenovirus = pd.Series(self.module.adenovirus_dehydration, index=diarr_adenovirus_idx)
        random_draw_c = pd.Series(rng.random_sample(size=len(diarr_adenovirus_idx)), index=diarr_adenovirus_idx)
        diarr_adeno_dehydration = p_dehydration_adenovirus >= random_draw_c
        diarr_adeno_dehydration_idx = p_dehydration_adenovirus.index[diarr_adeno_dehydration]
        df.loc[diarr_adeno_dehydration_idx, 'di_dehydration_present'] = True
        # CRYPTOSPORIDIUM
        p_dehydration_crypto = pd.Series(self.module.crypto_dehydration, index=diarr_crypto_idx)
        random_draw_c = pd.Series(rng.random_sample(size=len(diarr_crypto_idx)), index=diarr_crypto_idx)
        diarr_crypto_dehydration = p_dehydration_crypto >= random_draw_c
        diarr_crypto_dehydration_idx = p_dehydration_crypto.index[diarr_crypto_dehydration]
        df.loc[diarr_crypto_dehydration_idx, 'di_dehydration_present'] = True
        # CAMPYLOBACTER
        p_dehydration_campylo = pd.Series(self.module.campylo_dehydration, index=diarr_campylo_idx)
        random_draw_c = pd.Series(rng.random_sample(size=len(diarr_campylo_idx)), index=diarr_campylo_idx)
        diarr_campylo_dehydration = p_dehydration_campylo >= random_draw_c
        diarr_campylo_dehydration_idx = p_dehydration_campylo.index[diarr_campylo_dehydration]
        df.loc[diarr_campylo_dehydration_idx, 'di_dehydration_present'] = True
        # ST-ETEC
        p_dehydration_ETEC = pd.Series(self.module.ETEC_dehydration, index=diarr_ETEC_idx)
        random_draw_c = pd.Series(rng.random_sample(size=len(diarr_ETEC_idx)), index=diarr_ETEC_idx)
        diarr_ETEC_dehydration = p_dehydration_ETEC >= random_draw_c
        diarr_ETEC_dehydration_idx = p_dehydration_ETEC.index[diarr_ETEC_dehydration]
        df.loc[diarr_ETEC_dehydration_idx, 'di_dehydration_present'] = True
        # SAPOVIRUS
        p_dehydration_sapovirus = pd.Series(self.module.sapovirus_dehydration, index=diarr_sapovirus_idx)
        random_draw_c = pd.Series(rng.random_sample(size=len(diarr_sapovirus_idx)), index=diarr_sapovirus_idx)
        diarr_sapo_dehydration = p_dehydration_sapovirus >= random_draw_c
        diarr_sapo_dehydration_idx = p_dehydration_sapovirus.index[diarr_sapo_dehydration]
        df.loc[diarr_sapo_dehydration_idx, 'di_dehydration_present'] = True
        # NOROVIRUS
        p_dehydration_norovirus = pd.Series(self.module.norovirus_dehydration, index=diarr_norovirus_idx)
        random_draw_c = pd.Series(rng.random_sample(size=len(diarr_norovirus_idx)), index=diarr_norovirus_idx)
        diarr_noro_dehydration = p_dehydration_norovirus >= random_draw_c
        diarr_noro_dehydration_idx = p_dehydration_norovirus.index[diarr_noro_dehydration]
        df.loc[diarr_noro_dehydration_idx, 'di_dehydration_present'] = True
        # ASTROVIRUS
        p_dehydration_astrovirus = pd.Series(self.module.astrovirus_dehydration, index=diarr_astrovirus_idx)
        random_draw_c = pd.Series(rng.random_sample(size=len(diarr_astrovirus_idx)), index=diarr_astrovirus_idx)
        diarr_astro_dehydration = p_dehydration_astrovirus >= random_draw_c
        diarr_astro_dehydration_idx = p_dehydration_astrovirus.index[diarr_astro_dehydration]
        df.loc[diarr_astro_dehydration_idx, 'di_dehydration_present'] = True
        # tEPEC
        p_dehydration_EPEC = pd.Series(self.module.EPEC_dehydration, index=diarr_EPEC_idx)
        random_draw_c = pd.Series(rng.random_sample(size=len(diarr_EPEC_idx)), index=diarr_EPEC_idx)
        diarr_EPEC_dehydration = p_dehydration_EPEC >= random_draw_c
        diarr_EPEC_dehydration_idx = p_dehydration_EPEC.index[diarr_EPEC_dehydration]
        df.loc[diarr_EPEC_dehydration_idx, 'di_dehydration_present'] = True

        # # # ASSIGN SOME OR SEVERE DEHYDRATION LEVELS FOR DIARRHOEA EPISODE # # #
        di_with_dehydration_idx = df.index[df.di_dehydration_present] & incident_acute_diarrhoea
        prob_some_dehydration = pd.Series(0.7, index=di_with_dehydration_idx)
        prob_severe_dehydration = pd.Series(0.3, index=di_with_dehydration_idx)
        random_draw = pd.Series(self.sim.rng.random_sample(size=len(di_with_dehydration_idx)),
                                index=di_with_dehydration_idx)
        dfx = pd.concat([prob_some_dehydration, prob_severe_dehydration, random_draw], axis=1)
        dfx.columns = ['p_some_dehydration', 'p_severe_dehydration', 'random_draw']
        diarr_some_dehydration = dfx.index[dfx.p_some_dehydration > dfx.random_draw]
        diarr_severe_dehydration = \
            dfx.index[(dfx.p_some_dehydration < dfx.random_draw) & (dfx.p_some_dehydration + dfx.p_severe_dehydration)
                      > dfx.random_draw]
        df.loc[diarr_some_dehydration, 'gi_dehydration_status'] = 'some dehydration'
        df.loc[diarr_severe_dehydration, 'gi_dehydration_status'] = 'severe dehydration'
        # ----------------------------------------------------------------------------------------------------------

        # # # # # # # # # # # # # # ASSIGN SYMPTOMS ASSOCIATED WITH EACH PATHOGENS # # # # # # # # # # # # # #
        # ROTAVIRUS
        # ----------------------- FEVER -----------------------
        p_fever_rotavirus = pd.Series(self.module.rotavirus_fever, index=diarr_rotavirus_idx)
        random_draw_a = pd.Series(rng.random_sample(size=len(diarr_rotavirus_idx)), index=diarr_rotavirus_idx)
        diarr_rota_fever = p_fever_rotavirus >= random_draw_a
        diarr_rota_fever_idx = p_fever_rotavirus.index[diarr_rota_fever]
        df.loc[diarr_rota_fever_idx, 'di_sympt_fever'] = True
        # ----------------------- VOMITING -----------------------
        p_vomiting_rotavirus = pd.Series(self.module.rotavirus_vomiting, index=diarr_rotavirus_idx)
        random_draw_b = pd.Series(rng.random_sample(size=len(diarr_rotavirus_idx)), index=diarr_rotavirus_idx)
        diarr_rota_vomiting = p_vomiting_rotavirus >= random_draw_b
        diarr_rota_vomiting_idx = p_vomiting_rotavirus.index[diarr_rota_vomiting]
        df.loc[diarr_rota_vomiting_idx, 'di_sympt_vomiting'] = True
        # ----------------------- BLOODY STOOLS -----------------------
        df.loc[diarr_rota_dysentery_idx, 'di_blood_in_stools'] = True

        # SHIGELLA
        # ----------------------- FEVER -----------------------
        p_fever_shigella = pd.Series(self.module.shigella_fever, index=diarr_shigella_idx)
        random_draw_a = pd.Series(rng.random_sample(size=len(diarr_shigella_idx)), index=diarr_shigella_idx)
        diarr_shigella_fever = p_fever_shigella >= random_draw_a
        diarr_shigella_fever_idx = p_fever_shigella.index[diarr_shigella_fever]
        df.loc[diarr_shigella_fever_idx, 'di_sympt_fever'] = True
        # ----------------------- VOMITING -----------------------
        p_vomiting_shigella = pd.Series(self.module.shigella_vomiting, index=diarr_shigella_idx)
        random_draw_b = pd.Series(rng.random_sample(size=len(diarr_shigella_idx)), index=diarr_shigella_idx)
        diarr_shigella_vomiting = p_vomiting_shigella >= random_draw_b
        diarr_shigella_vomiting_idx = p_vomiting_shigella.index[diarr_shigella_vomiting]
        df.loc[diarr_shigella_vomiting_idx, 'di_sympt_vomiting'] = True
        # ----------------------- BLOODY STOOLS -----------------------
        df.loc[diarr_shigella_dysentery_idx, 'di_blood_in_stools'] = True

        # ADENOVIRUS
        # ----------------------- FEVER -----------------------
        p_fever_adenovirus = pd.Series(self.module.adenovirus_fever, index=diarr_adenovirus_idx)
        random_draw_a = pd.Series(rng.random_sample(size=len(diarr_adenovirus_idx)), index=diarr_adenovirus_idx)
        diarr_adeno_fever = p_fever_adenovirus >= random_draw_a
        diarr_adeno_fever_idx = p_fever_adenovirus.index[diarr_adeno_fever]
        df.loc[diarr_adeno_fever_idx, 'di_sympt_fever'] = True
        # ----------------------- VOMITING -----------------------
        p_vomiting_adenovirus = pd.Series(self.module.adenovirus_vomiting, index=diarr_adenovirus_idx)
        random_draw_b = pd.Series(rng.random_sample(size=len(diarr_adenovirus_idx)), index=diarr_adenovirus_idx)
        diarr_adeno_vomiting = p_vomiting_adenovirus >= random_draw_b
        diarr_adeno_vomiting_idx = p_vomiting_adenovirus.index[diarr_adeno_vomiting]
        df.loc[diarr_adeno_vomiting_idx, 'di_sympt_vomiting'] = True
        # ----------------------- BLOODY STOOLS -----------------------
        df.loc[diarr_adeno_dysentery_idx, 'di_blood_in_stools'] = True

        # CRYPTOSPORIDIUM
        # ----------------------- FEVER -----------------------
        p_fever_crypto = pd.Series(self.module.crypto_fever, index=diarr_crypto_idx)
        random_draw_a = pd.Series(rng.random_sample(size=len(diarr_crypto_idx)), index=diarr_crypto_idx)
        diarr_crypto_fever = p_fever_crypto >= random_draw_a
        diarr_crypto_fever_idx = p_fever_crypto.index[diarr_crypto_fever]
        df.loc[diarr_crypto_fever_idx, 'di_sympt_fever'] = True
        # ----------------------- VOMITING -----------------------
        p_vomiting_crypto = pd.Series(self.module.crypto_vomiting, index=diarr_crypto_idx)
        random_draw_b = pd.Series(rng.random_sample(size=len(diarr_crypto_idx)), index=diarr_crypto_idx)
        diarr_crypto_vomiting = p_vomiting_crypto >= random_draw_b
        diarr_crypto_vomiting_idx = p_vomiting_crypto.index[diarr_crypto_vomiting]
        df.loc[diarr_crypto_vomiting_idx, 'di_sympt_vomiting'] = True
        # ----------------------- BLOODY STOOLS -----------------------
        df.loc[diarr_crypto_dysentery_idx, 'di_blood_in_stools'] = True

        # CAMPYLOBACTER
        # ----------------------- FEVER -----------------------
        p_fever_campylo = pd.Series(self.module.campylo_fever, index=diarr_campylo_idx)
        random_draw_a = pd.Series(rng.random_sample(size=len(diarr_campylo_idx)), index=diarr_campylo_idx)
        diarr_campylo_fever = p_fever_campylo >= random_draw_a
        diarr_campylo_fever_idx = p_fever_campylo.index[diarr_campylo_fever]
        df.loc[diarr_campylo_fever_idx, 'di_sympt_fever'] = True
        # ----------------------- VOMITING -----------------------
        p_vomiting_campylo = pd.Series(self.module.campylo_vomiting, index=diarr_campylo_idx)
        random_draw_b = pd.Series(rng.random_sample(size=len(diarr_campylo_idx)), index=diarr_campylo_idx)
        diarr_campylo_vomiting = p_vomiting_campylo >= random_draw_b
        diarr_campylo_vomiting_idx = p_vomiting_campylo.index[diarr_campylo_vomiting]
        df.loc[diarr_campylo_vomiting_idx, 'di_sympt_vomiting'] = True
        # ----------------------- BLOODY STOOLS -----------------------
        df.loc[diarr_campylo_dysentery_idx, 'di_blood_in_stools'] = True

        # ST-ETEC
        # ----------------------- FEVER -----------------------
        p_fever_ETEC = pd.Series(self.module.ETEC_fever, index=diarr_ETEC_idx)
        random_draw_a = pd.Series(rng.random_sample(size=len(diarr_ETEC_idx)), index=diarr_ETEC_idx)
        diarr_ETEC_fever = p_fever_ETEC >= random_draw_a
        diarr_ETEC_fever_idx = p_fever_ETEC.index[diarr_ETEC_fever]
        df.loc[diarr_ETEC_fever_idx, 'di_sympt_fever'] = True
        # ----------------------- VOMITING -----------------------
        p_vomiting_ETEC = pd.Series(self.module.ETEC_vomiting, index=diarr_ETEC_idx)
        random_draw_b = pd.Series(rng.random_sample(size=len(diarr_ETEC_idx)), index=diarr_ETEC_idx)
        diarr_ETEC_vomiting = p_vomiting_ETEC >= random_draw_b
        diarr_ETEC_vomiting_idx = p_vomiting_ETEC.index[diarr_ETEC_vomiting]
        df.loc[diarr_ETEC_vomiting_idx, 'di_sympt_vomiting'] = True
        # ----------------------- BLOODY STOOLS -----------------------
        df.loc[diarr_ETEC_dysentery_idx, 'di_blood_in_stools'] = True

        # SAPOVIRUS
        # ----------------------- FEVER -----------------------
        p_fever_sapovirus = pd.Series(self.module.sapovirus_fever, index=diarr_sapovirus_idx)
        random_draw_a = pd.Series(rng.random_sample(size=len(diarr_sapovirus_idx)), index=diarr_sapovirus_idx)
        diarr_sapo_fever = p_fever_sapovirus >= random_draw_a
        diarr_sapo_fever_idx = p_fever_sapovirus.index[diarr_sapo_fever]
        df.loc[diarr_sapo_fever_idx, 'di_sympt_fever'] = True
        # ----------------------- VOMITING -----------------------
        p_vomiting_sapovirus = pd.Series(self.module.sapovirus_vomiting, index=diarr_sapovirus_idx)
        random_draw_b = pd.Series(rng.random_sample(size=len(diarr_sapovirus_idx)), index=diarr_sapovirus_idx)
        diarr_sapo_vomiting = p_vomiting_sapovirus >= random_draw_b
        diarr_sapo_vomiting_idx = p_vomiting_sapovirus.index[diarr_sapo_vomiting]
        df.loc[diarr_sapo_vomiting_idx, 'di_sympt_vomiting'] = True
        # ----------------------- BLOODY STOOLS -----------------------
        df.loc[diarr_sapovirus_dysentery_idx, 'di_blood_in_stools'] = True

        # NOROVIRUS
        # ----------------------- FEVER -----------------------
        p_fever_norovirus = pd.Series(self.module.norovirus_fever, index=diarr_norovirus_idx)
        random_draw_a = pd.Series(rng.random_sample(size=len(diarr_norovirus_idx)), index=diarr_norovirus_idx)
        diarr_noro_fever = p_fever_norovirus >= random_draw_a
        diarr_noro_fever_idx = p_fever_norovirus.index[diarr_noro_fever]
        df.loc[diarr_noro_fever_idx, 'di_sympt_fever'] = True
        # ----------------------- VOMITING -----------------------
        p_vomiting_norovirus = pd.Series(self.module.norovirus_vomiting, index=diarr_norovirus_idx)
        random_draw_b = pd.Series(rng.random_sample(size=len(diarr_norovirus_idx)), index=diarr_norovirus_idx)
        diarr_noro_vomiting = p_vomiting_norovirus >= random_draw_b
        diarr_noro_vomiting_idx = p_vomiting_norovirus.index[diarr_noro_vomiting]
        df.loc[diarr_noro_vomiting_idx, 'di_sympt_vomiting'] = True
        # ----------------------- BLOODY STOOLS -----------------------
        df.loc[diarr_norovirus_dysentery_idx, 'di_blood_in_stools'] = True

        # ASTROVIRUS
        # ----------------------- FEVER -----------------------
        p_fever_astrovirus = pd.Series(self.module.astrovirus_fever, index=diarr_astrovirus_idx)
        random_draw_a = pd.Series(rng.random_sample(size=len(diarr_astrovirus_idx)), index=diarr_astrovirus_idx)
        diarr_astro_fever = p_fever_astrovirus >= random_draw_a
        diarr_astro_fever_idx = p_fever_astrovirus.index[diarr_astro_fever]
        df.loc[diarr_astro_fever_idx, 'di_sympt_fever'] = True
        # ----------------------- VOMITING -----------------------
        p_vomiting_astrovirus = pd.Series(self.module.astrovirus_vomiting, index=diarr_astrovirus_idx)
        random_draw_b = pd.Series(rng.random_sample(size=len(diarr_astrovirus_idx)), index=diarr_astrovirus_idx)
        diarr_astro_vomiting = p_vomiting_astrovirus >= random_draw_b
        diarr_astro_vomiting_idx = p_vomiting_astrovirus.index[diarr_astro_vomiting]
        df.loc[diarr_astro_vomiting_idx, 'di_sympt_vomiting'] = True
        # ----------------------- BLOODY STOOLS -----------------------
        df.loc[diarr_astrovirus_dysentery_idx, 'di_blood_in_stools'] = True

        # tEPEC
        # ----------------------- FEVER -----------------------
        p_fever_EPEC = pd.Series(self.module.EPEC_fever, index=diarr_EPEC_idx)
        random_draw_a = pd.Series(rng.random_sample(size=len(diarr_EPEC_idx)), index=diarr_EPEC_idx)
        diarr_EPEC_fever = p_fever_EPEC >= random_draw_a
        diarr_EPEC_fever_idx = p_fever_EPEC.index[diarr_EPEC_fever]
        df.loc[diarr_EPEC_fever_idx, 'di_sympt_fever'] = True
        # ----------------------- VOMITING -----------------------
        p_vomiting_EPEC = pd.Series(self.module.EPEC_vomiting, index=diarr_EPEC_idx)
        random_draw_b = pd.Series(rng.random_sample(size=len(diarr_EPEC_idx)), index=diarr_EPEC_idx)
        diarr_EPEC_vomiting = p_vomiting_EPEC >= random_draw_b
        diarr_EPEC_vomiting_idx = p_vomiting_EPEC.index[diarr_EPEC_vomiting]
        df.loc[diarr_EPEC_vomiting_idx, 'di_sympt_vomiting'] = True
        # ----------------------- BLOODY STOOLS -----------------------
        df.loc[diarr_EPEC_dysentery_idx, 'di_blood_in_stools'] = True

        # ---------------------------------------------------------------------------------------------------
        # # # # # # # # # # # # # # # # ACUTE DIARRHOEA BECOMING PERSISTENT # # # # # # # # # # # # # # # # #

        # # # # # # FIRST ASSIGN THE PROBABILITY OF PROLONGED DIARRHOEA (over 7 days) BY PATHOGEN # # # # # #
        # # # ROTAVIRUS # # #
        p_prolonged_diarr_rota = pd.Series(self.module.rotavirus_prolonged_diarr, index=diarr_rotavirus_idx)
        random_draw = pd.Series(rng.random_sample(size=len(diarr_rotavirus_idx)), index=diarr_rotavirus_idx)
        ProD_rota = p_prolonged_diarr_rota > random_draw
        ProD_rota_idx = p_prolonged_diarr_rota.index[ProD_rota]
        df.loc[ProD_rota_idx, 'gi_diarrhoea_type'] = 'prolonged'
        # # # SHIGELLA # # #
        p_prolonged_diarr_shigella = pd.Series(self.module.shigella_prolonged_diarr, index=diarr_shigella_idx)
        random_draw = pd.Series(rng.random_sample(size=len(diarr_shigella_idx)), index=diarr_shigella_idx)
        ProD_shigella = p_prolonged_diarr_shigella > random_draw
        ProD_shigella_idx = p_prolonged_diarr_shigella.index[ProD_shigella]
        df.loc[ProD_shigella_idx, 'gi_diarrhoea_type'] = 'prolonged'
        # # # ADENOVIRUS # # #
        p_prolonged_diarr_adeno = pd.Series(self.module.adenovirus_prolonged_diarr, index=diarr_adenovirus_idx)
        random_draw = pd.Series(rng.random_sample(size=len(diarr_adenovirus_idx)), index=diarr_adenovirus_idx)
        ProD_adeno = p_prolonged_diarr_adeno > random_draw
        ProD_adeno_idx = p_prolonged_diarr_adeno.index[ProD_adeno]
        df.loc[ProD_adeno_idx, 'gi_diarrhoea_type'] = 'prolonged'
        # # # CRYPTOSPORIDIUM # # #
        p_prolonged_diarr_crypto = pd.Series(self.module.crypto_prolonged_diarr, index=diarr_crypto_idx)
        random_draw = pd.Series(rng.random_sample(size=len(diarr_crypto_idx)), index=diarr_crypto_idx)
        ProD_crypto = p_prolonged_diarr_crypto > random_draw
        ProD_crypto_idx = p_prolonged_diarr_crypto.index[ProD_crypto]
        df.loc[ProD_crypto_idx, 'gi_diarrhoea_type'] = 'prolonged'
        # # # CAMPYLOBACTER # # #
        p_prolonged_diarr_campylo = pd.Series(self.module.campylo_prolonged_diarr, index=diarr_campylo_idx)
        random_draw = pd.Series(rng.random_sample(size=len(diarr_campylo_idx)), index=diarr_campylo_idx)
        ProD_campylo = p_prolonged_diarr_campylo > random_draw
        ProD_campylo_idx = p_prolonged_diarr_campylo.index[ProD_campylo]
        df.loc[ProD_campylo_idx, 'gi_diarrhoea_type'] = 'prolonged'
        # # # ST-ETEC # # #
        p_prolonged_diarr_ETEC = pd.Series(self.module.ETEC_prolonged_diarr, index=diarr_ETEC_idx)
        random_draw = pd.Series(rng.random_sample(size=len(diarr_ETEC_idx)), index=diarr_ETEC_idx)
        ProD_ETEC = p_prolonged_diarr_ETEC > random_draw
        ProD_ETEC_idx = p_prolonged_diarr_ETEC.index[ProD_ETEC]
        df.loc[ProD_ETEC_idx, 'gi_diarrhoea_type'] = 'prolonged'
        # # # SAPOVIRUS # # #
        p_prolonged_diarr_sapo = pd.Series(self.module.sapovirus_prolonged_diarr, index=diarr_sapovirus_idx)
        random_draw = pd.Series(rng.random_sample(size=len(diarr_sapovirus_idx)), index=diarr_sapovirus_idx)
        ProD_sapo = p_prolonged_diarr_sapo > random_draw
        ProD_sapo_idx = p_prolonged_diarr_sapo.index[ProD_sapo]
        df.loc[ProD_sapo_idx, 'gi_diarrhoea_type'] = 'prolonged'
        # # # NOROVIRUS # # #
        p_prolonged_diarr_noro = pd.Series(self.module.norovirus_prolonged_diarr, index=diarr_norovirus_idx)
        random_draw = pd.Series(rng.random_sample(size=len(diarr_norovirus_idx)), index=diarr_norovirus_idx)
        ProD_noro = p_prolonged_diarr_noro > random_draw
        ProD_noro_idx = p_prolonged_diarr_noro.index[ProD_noro]
        df.loc[ProD_noro_idx, 'gi_diarrhoea_type'] = 'prolonged'
        # # # ASTROVIRUS # # #
        p_prolonged_diarr_astro = pd.Series(self.module.astrovirus_prolonged_diarr, index=diarr_astrovirus_idx)
        random_draw = pd.Series(rng.random_sample(size=len(diarr_astrovirus_idx)), index=diarr_astrovirus_idx)
        ProD_astro = p_prolonged_diarr_astro > random_draw
        ProD_astro_idx = p_prolonged_diarr_astro.index[ProD_astro]
        df.loc[ProD_astro_idx, 'gi_diarrhoea_type'] = 'prolonged'
        # # # EPEC # # #
        p_prolonged_diarr_EPEC = pd.Series(self.module.EPEC_prolonged_diarr, index=diarr_EPEC_idx)
        random_draw = pd.Series(rng.random_sample(size=len(diarr_EPEC_idx)), index=diarr_EPEC_idx)
        ProD_EPEC = p_prolonged_diarr_EPEC > random_draw
        ProD_EPEC_idx = p_prolonged_diarr_EPEC.index[ProD_EPEC]
        df.loc[ProD_EPEC_idx, 'gi_diarrhoea_type'] = 'prolonged'

        # --------------------------------------------------------------------------------------------------------
        # SEEKING CARE FOR ACUTE WATERY DIARRHOEA
        # --------------------------------------------------------------------------------------------------------

        # TODO: when you declare the symptoms in the symptom manager, the health care seeking will follow automatically

        '''watery_diarrhoea_symptoms = \
            df.index[df.is_alive & (df.age_years < 5) & (df.di_diarrhoea_loose_watery_stools == True) &
                     (df.di_blood_in_stools == False) & df.di_diarrhoea_over14days == False]

        seeks_care = pd.Series(data=False, index=watery_diarrhoea_symptoms)
        for individual in watery_diarrhoea_symptoms:
            prob = self.sim.modules['HealthSystem'].get_prob_seek_care(individual, symptom_code=1)
            seeks_care[individual] = self.module.rng.rand() < prob
            date_seeking_care = df.date_of_onset_diarrhoea[individual] + pd.DateOffset(days=int(rng.uniform(0, 7)))
            event = HSI_ICCM(self.module, person_id=individual)
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
            event = HSI_ICCM(self.module, person_id=individual)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=2,
                                                                topen=self.sim.date,
                                                                tclose=None
                                                                )
                                                                '''

        # ---------------------------------------------------------------------------------------
        # # # # # # NEXT ASSIGN THE PROBABILITY OF BECOMING PERSISTENT (over 14 days) # # # # # #
        ProD_idx = df.index[df.gi_diarrhoea_status & (df.gi_diarrhoea_type == 'prolonged') & df.is_alive &
                            (df.age_exact_years < 5)]
        becoming_persistent = pd.Series(self.module.prob_prolonged_to_persistent_diarr, index=ProD_idx)

        becoming_persistent.loc[df.is_alive & (df.age_exact_years >= 1) & (df.age_exact_years < 2)] \
            *= m.rr_bec_persistent_age12to23
        becoming_persistent.loc[df.is_alive & (df.age_exact_years >= 2) & (df.age_exact_years < 5)] \
            *= m.rr_bec_persistent_age24to59
        becoming_persistent.loc[df.is_alive & (df.age_exact_years < 5) & df.has_hiv] \
            *= m.rr_bec_persistent_HIV
        becoming_persistent.loc[df.is_alive & (df.age_exact_years < 5) & df.malnutrition == True] \
            *= m.rr_bec_persistent_SAM
        becoming_persistent.loc[
            df.is_alive & df.exclusive_breastfeeding == True & (df.age_exact_years <= 0.5)] \
            *= m.rr_bec_persistent_excl_breast
        becoming_persistent.loc[
            df.is_alive & df.continued_breastfeeding == True & (df.age_exact_years > 0.5) &
            (df.age_exact_years < 2)] *= m.rr_bec_persistent_cont_breast

        random_draw = pd.Series(self.sim.rng.random_sample(size=len(becoming_persistent)),
                                index=becoming_persistent.index)
        persistent_diarr = becoming_persistent > random_draw
        persistent_diarr_idx = becoming_persistent.index[persistent_diarr]
        df.loc[persistent_diarr_idx, 'gi_diarrhoea_type'] = 'persistent'
        df.loc[persistent_diarr_idx, 'di_diarrhoea_over14days'] = True

        # # # # # # PERSISTENT DIARRHOEA OR SEVERE PERSISTENT DIARRHOEA # # # # # #
        severe_persistent_diarr = \
            df.index[df.gi_diarrhoea_status & (df.gi_diarrhoea_type == 'persistent') &
                     (df.gi_dehydration_status != 'no dehydration')]
        df.loc[severe_persistent_diarr, 'gi_persistent_diarrhoea'] = 'severe persistent diarrhoea'

        just_persistent_diarr = \
            df.index[df.gi_diarrhoea_status & (df.gi_diarrhoea_type == 'persistent') &
                     (df.gi_dehydration_status == 'no dehydration')]
        df.loc[just_persistent_diarr, 'gi_persistent_diarrhoea'] = 'persistent diarrhoea'


        # TODO: so the symptom in the symptom manager are 'diarh' and 'persistent_diarh'

        '''
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
                 event = HSI_ICCM(self.module, person_id=individual)
                 self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                     priority=2,
                                                                     topen=self.sim.date,
                                                                     tclose=None
                                                                     )
                                                                     '''

        # -----------------------------------------------------------------------------------------------------
        # log the clinical cases
        AWD_cases = \
            df.loc[df.is_alive & (df.age_exact_years < 5) & df.gi_diarrhoea_status &
                   (df.gi_diarrhoea_acute_type == 'acute watery diarrhoea') & (df.gi_diarrhoea_type != 'persistent')]
        dysentery_cases = \
            df.loc[df.is_alive & (df.age_exact_years < 5) & df.gi_diarrhoea_status &
                   (df.gi_diarrhoea_acute_type == 'dysentery') & (df.gi_diarrhoea_type != 'persistent')]
        persistent_diarr_cases = \
            df.loc[df.is_alive & (df.age_exact_years < 5) & df.gi_diarrhoea_status &
                   (df.gi_diarrhoea_type == 'persistent')]

        # clinical_types = pd.concat([AWD_cases, dysentery_cases, persistent_diarr_cases], axis=0).sort_index()

        logger.info('%s|clinical_diarrhoea_type|%s', self.sim.date,
                    {# 'total': len(clinical_types),
                     'AWD': len(AWD_cases),
                     'dysentery': len(dysentery_cases),
                     'persistent': len(persistent_diarr_cases)
                     })

        # --------------------------------------------------------------------------------------------------------
        # # # # # # ASSIGN DEATH PROBABILITIES BASED ON DEHYDRATION, CO-MORBIDITIES AND DIARRHOEA TYPE # # # # # #
        # mortality rates by diarrhoea clinical type
        cfr_AWD = \
            pd.Series(0.0056, index=AWD_cases)
        cfr_dysentery = \
            pd.Series(0.0427, index=dysentery_cases)
        cfr_persistent_diarr = \
            pd.Series(0.1395, index=persistent_diarr_cases)

        # added effects of risk factors for death
        eff_prob_death_diarr = pd.concat([cfr_AWD, cfr_dysentery, cfr_persistent_diarr], axis=0).sort_index()
        # eff_prob_death_diarr.loc[df.is_alive & df.gi_diarrhoea_status & (df.age_exact_years >= 1) &
        #                          (df.age_exact_years < 2)] *= m.rr_diarr_death_age12to23mo
        # eff_prob_death_diarr.loc[df.is_alive & df.gi_diarrhoea_status & (df.age_exact_years >= 2) &
        #                          (df.age_exact_years < 5)] *= m.rr_diarr_death_age24to59mo
        # eff_prob_death_diarr.loc[df.is_alive & (df.gi_diarrhoea_status == True) & (df.age_exact_years < 5) &
         #                        (df.has_hiv == True)] *= m.rr_diarr_death_HIV
        # eff_prob_death_diarr.loc[df.is_alive & (df.gi_diarrhoea_status == True) & (df.age_exact_years < 5) &
          #                       (df.malnutrition == True)] *= m.rr_diarr_death_SAM
        # TODO:add dehydration, add other co-morbidities

        random_draw_death = \
            pd.Series(self.sim.rng.random_sample(size=len(eff_prob_death_diarr)), index=eff_prob_death_diarr.index)
        death_from_diarr = eff_prob_death_diarr > random_draw_death
        death_from_diarr_idx = eff_prob_death_diarr.index[death_from_diarr]

        # acute diarrhoea death
        for child in death_from_diarr_idx & df.index[df.gi_diarrhoea_type == 'acute']:
            random_date = rng.randint(low=4, high=6)
            random_days = pd.to_timedelta(random_date, unit='d')
            death_event = DeathDiarrhoeaEvent(self.module, person_id=child,
                                              cause='diarrhoea')  # make that death event
            self.sim.schedule_event(death_event, df.at[child, 'date_of_onset_diarrhoea'] + random_days)  # schedule the death for acute cases
        for child in death_from_diarr_idx & df.index[df.gi_diarrhoea_type == 'prolonged']:
            random_date1 = rng.randint(low=7, high=13)
            random_days1 = pd.to_timedelta(random_date1, unit='d')
            death_event = DeathDiarrhoeaEvent(self.module, person_id=child,
                                              cause='diarrhoea')  # make that death event
            self.sim.schedule_event(death_event, df.at[child, 'date_of_onset_diarrhoea'] + random_days1)  # schedule the death for prolonged cases
        # persistent diarrhoea death
        for child in death_from_diarr_idx & df.index[df.gi_diarrhoea_type == 'persistent']:
            random_date2 = rng.randint(low=14, high=30)
            random_days2 = pd.to_timedelta(random_date2, unit='d')
            death_event = DeathDiarrhoeaEvent(self.module, person_id=child,
                                              cause='diarrhoea')  # make that death event
            self.sim.schedule_event(death_event, df.at[child, 'date_of_onset_diarrhoea'] + random_days2)  # schedule the death for persistent cases

        # TODO: Nice!
        # schedule recovery for those who didn't die
        recovery_from_diarr = eff_prob_death_diarr <= random_draw_death
        recovery_from_diarr_idx = eff_prob_death_diarr.index[recovery_from_diarr]
        # acute diarrhoea
        for child in recovery_from_diarr_idx & df.index[df.gi_diarrhoea_type == 'acute']:
            random_date = rng.randint(low=4, high=6)
            random_days = pd.to_timedelta(random_date, unit='d')
            self.sim.schedule_event(SelfRecoverEvent(self.module, person_id=child),
                                    df.at[child, 'date_of_onset_diarrhoea'] + random_days)

        for child in recovery_from_diarr_idx & df.index[df.gi_diarrhoea_type == 'prolongued']:
            random_date1 = rng.randint(low=7, high=13)
            random_days1 = pd.to_timedelta(random_date1, unit='d')
            self.sim.schedule_event(SelfRecoverEvent(self.module, person_id=child),
                                    df.at[child, 'date_of_onset_diarrhoea'] + random_days1)
        # persistent diarrhoea
        for child in recovery_from_diarr_idx & df.index[df.gi_diarrhoea_type == 'persistent']:
            random_date2 = rng.randint(low=14, high=21)
            random_days2 = pd.to_timedelta(random_date2, unit='d')
            self.sim.schedule_event(SelfRecoverEvent(self.module, person_id=child),
                                    df.at[child, 'date_of_onset_diarrhoea'] + random_days2)


class SelfRecoverEvent(Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        # set everything back to default
        df.at[person_id, 'gi_recovered_date'] = pd.NaT
        df.at[person_id, 'gi_diarrhoea_status'] = False
        df.at[person_id, 'gi_diarrhoea_acute_type'] = np.nan
        df.at[person_id, 'gi_diarrhoea_type'] = np.nan
        df.at[person_id, 'gi_persistent_diarrhoea'] = np.nan
        df.at[person_id, 'gi_dehydration_status'] = 'no dehydration'
        df.at[person_id, 'date_of_onset_diarrhoea'] = pd.NaT
        df.at[person_id, 'gi_recovered_date'] = pd.NaT
        df.at[person_id, 'gi_diarrhoea_death_date'] = pd.NaT
        df.at[person_id, 'gi_diarrhoea_death'] = False
        df.at[person_id, 'gi_diarrhoea_pathogen'] = np.nan


class DeathDiarrhoeaEvent(Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id, cause):
        super().__init__(module, person_id=person_id)
        self.cause = cause

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe

        if df.at[person_id, 'is_alive']:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, person_id, cause='diarrhoea'),
                                    self.sim.date)
            df.at[person_id, 'gi_diarrhoea_death_date'] = self.sim.date
            df.at[person_id, 'gi_diarrhoea_death'] = True
            # logger.info('This is DeathDiarrhoeaEvent determining if person %d on the date %s will die '
            #             'from their disease', person_id, self.sim.date)
            # death_count = sum(person_id)
            # # Log the diarrhoea death information
            # logger.info('%s|death_diarrhoea|%s', self.sim.date,
            #             {'death': sum(death_count)
            #              })


class DiarrhoeaLoggingEvent (RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props
        now = self.sim.date

        # -----------------------------------------------------------------------------------------------------
        # sum all the counters for previous year
        count_episodes = df['gi_diarrhoea_count'].sum()
        pop_under5 = len(df[df.is_alive & (df.age_exact_years < 5)])
        # overall incidence rate in under 5
        inc_100cy = (count_episodes / pop_under5) * 100
        logger.info('%s|episodes_counts|%s', now, {'incidence_per100cy': inc_100cy})
        # logger.info('%s|pop_counts|%s', now, {'pop_len': pop_under5})

        # TODO: I think it's easier to output the number of events in the logger and work out the incidence afterwards.
        # So, I would propose just logging value counts.


        # log information on attributable pathogens
        pathogen_count = df[df.is_alive & df.age_years.between(0, 5)].groupby('gi_diarrhoea_pathogen').size()
        under5 = df[df.is_alive & df.age_years.between(0, 5)]
        # all_patho_counts = sum(pathogen_count)
        length_under5 = len(under5)
        # total_inc = all_patho_counts * 4 * 100 / length_under5
        rota_inc = (pathogen_count['rotavirus'] * 4 / length_under5) * 100
        shigella_inc = (pathogen_count['shigella'] * 4 / length_under5) * 100
        adeno_inc = (pathogen_count['adenovirus'] * 4 / length_under5) * 100
        crypto_inc = (pathogen_count['cryptosporidium'] * 4 / length_under5) * 100
        campylo_inc = (pathogen_count['campylobacter'] * 4 / length_under5) * 100
        ETEC_inc = (pathogen_count['ST-ETEC'] * 4 / length_under5) * 100
        sapo_inc = (pathogen_count['sapovirus'] * 4 / length_under5) * 100
        noro_inc = (pathogen_count['norovirus'] * 4 / length_under5) * 100
        astro_inc = (pathogen_count['astrovirus'] * 4 / length_under5) * 100
        tEPEC_inc = (pathogen_count['tEPEC'] * 4 / length_under5) * 100

        # incidence rate by pathogen
        logger.info('%s|diarr_incidence_by_patho|%s', self.sim.date,
                    {#'total': total_inc,
                     'rotavirus': rota_inc,
                     'shigella': shigella_inc,
                     'adenovirus': adeno_inc,
                     'cryptosporidium': crypto_inc,
                     'campylobacter': campylo_inc,
                     'ETEC': ETEC_inc,
                     'sapovirus': sapo_inc,
                     'norovirus': noro_inc,
                     'astrovirus': astro_inc,
                     'tEPEC': tEPEC_inc
                     })

        # incidence rate per age group by pathogen
        pathogen_0to11mo = df[df.is_alive & (df.age_years < 1)].groupby('gi_diarrhoea_pathogen').size()
        len_under12mo = df[df.is_alive & df.age_years.between(0, 1)]
        pathogen_12to23mo = df[df.is_alive & (df.age_years >= 1) & (df.age_years < 2)].groupby(
            'gi_diarrhoea_pathogen').size()
        len_11to23mo = df[df.is_alive & df.age_years.between(1, 2)]
        pathogen_24to59mo = df[df.is_alive & (df.age_years >= 2) & (df.age_years < 5)].groupby(
            'gi_diarrhoea_pathogen').size()
        len_24to59mo = df[df.is_alive & df.age_years.between(2, 5)]

        rota_inc_by_age = [((pathogen_0to11mo['rotavirus'] * 4 * 100) / len(len_under12mo)),
                           ((pathogen_12to23mo['rotavirus'] * 4 * 100) / len(len_11to23mo)),
                           ((pathogen_24to59mo['rotavirus'] * 4 * 100) / len(len_24to59mo))]
        shig_inc_by_age = [(pathogen_0to11mo['shigella'] * 4 * 100) / len(len_under12mo),
                           (pathogen_12to23mo['shigella'] * 4 * 100) / len(len_11to23mo),
                           (pathogen_24to59mo['shigella'] * 4 * 100) / len(len_24to59mo)]
        adeno_inc_by_age = [(pathogen_0to11mo['adenovirus'] * 4 * 100) / len(len_under12mo),
                            (pathogen_12to23mo['adenovirus'] * 4 * 100) / len(len_11to23mo),
                            (pathogen_24to59mo['adenovirus'] * 4 * 100) / len(len_24to59mo)]
        crypto_inc_by_age = [(pathogen_0to11mo['cryptosporidium'] * 4 * 100) / len(len_under12mo),
                             (pathogen_12to23mo['cryptosporidium'] * 4 * 100) / len(len_11to23mo),
                             (pathogen_24to59mo['cryptosporidium'] * 4 * 100) / len(len_24to59mo)]
        campylo_inc_by_age = [(pathogen_0to11mo['campylobacter'] * 4 * 100) / len(len_under12mo),
                              (pathogen_12to23mo['campylobacter'] * 4 * 100) / len(len_11to23mo),
                              (pathogen_24to59mo['campylobacter'] * 4 * 100) / len(len_24to59mo)]
        etec_inc_by_age = [(pathogen_0to11mo['ST-ETEC'] * 4 * 100) / len(len_under12mo),
                           (pathogen_12to23mo['ST-ETEC'] * 4 * 100) / len(len_11to23mo),
                           (pathogen_24to59mo['ST-ETEC'] * 4 * 100) / len(len_24to59mo)]
        sapo_inc_by_age = [(pathogen_0to11mo['sapovirus'] * 4 * 100) / len(len_under12mo),
                           (pathogen_12to23mo['sapovirus'] * 4 * 100) / len(len_11to23mo),
                           (pathogen_24to59mo['sapovirus'] * 4 * 100) / len(len_24to59mo)]
        noro_inc_by_age = [(pathogen_0to11mo['norovirus'] * 4 * 100) / len(len_under12mo),
                           (pathogen_12to23mo['norovirus'] * 4 * 100) / len(len_11to23mo),
                           (pathogen_24to59mo['norovirus'] * 4 * 100) / len(len_24to59mo)]
        astro_inc_by_age = [(pathogen_0to11mo['astrovirus'] * 4 * 100) / len(len_under12mo),
                            (pathogen_12to23mo['astrovirus'] * 4 * 100) / len(len_11to23mo),
                            (pathogen_24to59mo['astrovirus'] * 4 * 100) / len(len_24to59mo)]
        epec_inc_by_age = [(pathogen_0to11mo['tEPEC'] * 4 * 100) / len(len_under12mo),
                           (pathogen_12to23mo['tEPEC'] * 4 * 100) / len(len_11to23mo),
                           (pathogen_24to59mo['tEPEC'] * 4 * 100) / len(len_24to59mo)]

        logger.info('%s|diarr_incidence_age0_11|%s', self.sim.date,
                    {'total': (sum(pathogen_0to11mo) * 4 * 100) / len_under12mo.size,
                     'rotavirus': rota_inc_by_age[0],
                     'shigella': shig_inc_by_age[0],
                     'adenovirus': adeno_inc_by_age[0],
                     'cryptosporidium': crypto_inc_by_age[0],
                     'campylobacter': campylo_inc_by_age[0],
                     'ETEC': etec_inc_by_age[0],
                     'sapovirus': sapo_inc_by_age[0],
                     'norovirus': noro_inc_by_age[0],
                     'astrovirus': astro_inc_by_age[0],
                     'tEPEC': epec_inc_by_age[0]
                     })
        logger.info('%s|diarr_incidence_age12_23|%s', self.sim.date,
                    {'total': (sum(pathogen_0to11mo) * 4 * 100) / len_11to23mo.size,
                     'rotavirus': rota_inc_by_age[1],
                     'shigella': shig_inc_by_age[1],
                     'adenovirus': adeno_inc_by_age[1],
                     'cryptosporidium': crypto_inc_by_age[1],
                     'campylobacter': campylo_inc_by_age[1],
                     'ETEC': etec_inc_by_age[1],
                     'sapovirus': sapo_inc_by_age[1],
                     'norovirus': noro_inc_by_age[1],
                     'astrovirus': astro_inc_by_age[1],
                     'tEPEC': epec_inc_by_age[1]
                     })
        logger.info('%s|diarr_incidence_age24_59|%s', self.sim.date,
                    {'total': (sum(pathogen_0to11mo) * 4 * 100) / pathogen_24to59mo.size,
                     'rotavirus': rota_inc_by_age[2],
                     'shigella': shig_inc_by_age[2],
                     'adenovirus': adeno_inc_by_age[2],
                     'cryptosporidium': crypto_inc_by_age[2],
                     'campylobacter': campylo_inc_by_age[2],
                     'ETEC': etec_inc_by_age[2],
                     'sapovirus': sapo_inc_by_age[2],
                     'norovirus': noro_inc_by_age[2],
                     'astrovirus': astro_inc_by_age[2],
                     'tEPEC': epec_inc_by_age[2]
                     })
        '''
      # Log the acute diarrhoea information
        diarrhoea_count = df[df.is_alive & df.age_years.between(0, 5)].groupby('gi_diarrhoea_acute_type').size()
        logger.info('%s|acute_diarrhoea|%s', self.sim.date,
                    {'total': sum(diarrhoea_count),
                     'acute_dysentery': diarrhoea_count['dysentery'],
                     'AWD': diarrhoea_count['acute watery diarrhoea'],
                     })
                   
        any_dehydration = df[df.is_alive & (df.age_exact_years < 5) & df.gi_diarrhoea_status &
                             (df.gi_dehydration_status != 'no dehydration')]
        logger.info('%s|dehydration_levels|%s', self.sim.date,
                    {'total': len(any_dehydration),
                     })
        
        # # # # # # # # # # rate of recovery for acute diarrhoea episodes # # # # # # # # # #
        AWD_idx = df.index[df.is_alive & (df.age_exact_years < 5) & df.gi_diarrhoea_status &
                           (df.gi_diarrhoea_type != 'persistent') & (
                                   df.gi_diarrhoea_acute_type == 'acute watery diarrhoea')]
        AWD_to_recover = pd.Series(self.module.r_recovery_AWD, index=AWD_idx)

        dysentery_idx = df.index[df.is_alive & (df.age_exact_years < 5) & df.gi_diarrhoea_status &
                                 (df.gi_diarrhoea_type != 'persistent') & (df.gi_diarrhoea_acute_type == 'dysentery')]
        dysentery_to_recover = pd.Series(self.module.r_recovery_dysentery, index=dysentery_idx)
        acute_episodes_idx = AWD_idx.union(dysentery_idx)

        eff_prob_recovery = pd.concat([AWD_to_recover, dysentery_to_recover], axis=0).sort_index()
        eff_prob_recovery.loc[acute_episodes_idx & df.gi_dehydration_status == 'some dehydration'] \
            *= m.rr_recovery_dehydration
        eff_prob_recovery.loc[acute_episodes_idx & df.gi_dehydration_status == 'severe dehydration'] \
            *= m.rr_recovery_dehydration

        random_draw_r = pd.Series(rng.random_sample(size=len(acute_episodes_idx)), index=acute_episodes_idx)
        acute_cases_to_recover = eff_prob_recovery > random_draw_r
        acute7_to_recover_idx = eff_prob_recovery.index[acute_cases_to_recover] & \
                                df.index[df.gi_diarrhoea_type == 'acute']
        prolonged_to_recover_idx = eff_prob_recovery.index[acute_cases_to_recover] & \
                                   df.index[df.gi_diarrhoea_type == 'prolonged']

        # Schedule recovery for acute diarrhoea episode
        for person_id in acute7_to_recover_idx:
            random_date = rng.randint(low=3, high=6)
            random_days = pd.to_timedelta(random_date, unit='d')
            self.sim.schedule_event(SelfRecoverEvent(self.module, person_id=person_id),
                                    df.at[person_id, 'date_of_onset_diarrhoea'] + random_days)
            df.at[person_id, 'gi_recovered_date'] = df.at[person_id, 'date_of_onset_diarrhoea'] + random_days

        # Schedule recovery for prolonged diarrhoea episode
        for person_id in prolonged_to_recover_idx:
            random_date = rng.randint(low=7, high=13)
            random_days = pd.to_timedelta(random_date, unit='d')
            self.sim.schedule_event(SelfRecoverEvent(self.module, person_id=person_id),
                                    df.at[person_id, 'date_of_onset_diarrhoea'] + random_days)
            df.at[person_id, 'gi_recovered_date'] = df.at[person_id, 'date_of_onset_diarrhoea'] + random_days            
                     
                     
        
        
        tmp = len(df.loc[(df.age_years.between(0, 5)) & (
                df.date_of_onset_diarrhoea > (now - DateOffset(months=self.repeat)))])
        '''
