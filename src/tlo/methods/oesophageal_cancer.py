"""
Oesophageal Cancer - module

Documentation: 04 - Methods Repository/Method_Oesophageal_Cancer.xlsx
"""
import logging
from collections import defaultdict
from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent, IndividualScopeEventMixin, Event
from tlo.methods import demography
import numpy as np
import pandas as pd
import random
from pathlib import Path


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Oesophageal_Cancer(Module):

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    # todo: consider adding palliative care ;

    PARAMETERS = {
        'r_low_grade_dysplasia_none': Parameter(Types.REAL, 'probabilty per 3 months of incident low grade '
                                                            'oesophageal dysplasia, amongst people with no '
                                                            'oesophageal dysplasia (men, age20, no excess alcohol, '
                                                            'no tobacco)'),
        'rr_low_grade_dysplasia_none_female': Parameter(Types.REAL, 'rate ratio for low grade oesophageal dysplasia '
                                                                    'for females'),
        'rr_low_grade_dysplasia_none_per_year_older': Parameter(Types.REAL, 'rate ratio for low grade oesophageal '
                                                                            'dysplasia per year older from age 20'),
        'rr_low_grade_dysplasia_none_tobacco': Parameter(Types.REAL, 'rate ratio for low grade oesophageal '
                                                                     'dysplasia for tobacco smokers'),
        'rr_low_grade_dysplasia_none_ex_alc': Parameter(Types.REAL, 'rate ratio for low grade oesophageal dysplasia '
                                                                    'for no excess alcohol'),
        'r_high_grade_dysplasia_low_grade_dysp': Parameter(Types.REAL, 'probabilty per 3 months of high grade '
                                                                       'oesophageal dysplasia, amongst people with '
                                                                       'low grade dysplasia'),
        'rr_high_grade_dysp_undergone_curative_treatment': Parameter(Types.REAL, 'rate ratio for high grade dysplasia '
                                                                                 'for people with low grade dysplasia '
                                                                                 'if had curative treatment at low '
                                                                                 'grade dysplasia stage'),
        'r_stage1_high_grade_dysp': Parameter(Types.REAL, 'probabilty per 3 months of stage 1 oesophageal cancer '
                                                          'amongst people with high grade dysplasia'),
        'rr_stage1_undergone_curative_treatment': Parameter(Types.REAL, 'rate ratio for stage 1 oesophageal '
                                                                                 'cancer for people with high grade '
                                                                                 'dysplasia if had curative treatment '
                                                                                 'at high grade dysplasia stage'),
        'r_stage2_stage1': Parameter(Types.REAL, 'probabilty per 3 months of stage 2 oesophageal cancer '
                                                          'amongst people with stage 1'),
        'rr_stage2_undergone_curative_treatment': Parameter(Types.REAL, 'rate ratio for stage 2 oesophageal '
                                                                                 'cancer for people with stage 1 '
                                                                                 'oesophageal cancer if had curative '
                                                                                 'treatment at stage 1'),
        'r_stage3_stage2': Parameter(Types.REAL, 'probabilty per 3 months of stage 3 oesophageal cancer '
                                                                             'amongst people with stage 2'),
        'rr_stage3_undergone_curative_treatment': Parameter(Types.REAL, 'rate ratio for stage 3 oesophageal '
                                                                        'cancer for people with stage 2 '
                                                                        'oesophageal cancer if had curative '
                                                                        'treatment at stage 2'),
        'r_stage4_stage3': Parameter(Types.REAL, 'probabilty per 3 months of stage 4 oesophageal cancer '
                                                 'amongst people with stage 3'),
        'rr_stage4_undergone_curative_treatment': Parameter(Types.REAL, 'rate ratio for stage 4 oesophageal '
                                                                        'cancer for people with stage 3 '
                                                                        'oesophageal cancer if had curative '
                                                                        'treatment at stage 3'),
        'r_death_oesoph_cancer': Parameter(Types.REAL, 'probabilty per 3 months of death from oesophageal cancer '
                                                       'mongst people with stage 4 oesophageal cancer'),
        'r_curative_treatment_low_grade_dysp': Parameter(Types.REAL, 'probabilty per 3 months of receiving medical '
                                                                     'treatment aimed at cure if have low grade '
                                                                     'dysplasia, given diagnosis (surgery, '
                                                                     'radiotherapy and/or chemotherapy'),
        'rr_curative_treatment_high_grade_dysp': Parameter(Types.REAL, 'relative rate of receiving medical '
                                                                     'treatment aimed at cure if have high grade '
                                                                     'dysplasia, given diagnosis (surgery, '
                                                                     'radiotherapy and/or chemotherapy'),
        'rr_curative_treatment_stage1': Parameter(Types.REAL, 'relative rate of receiving medical '
                                                                      'treatment aimed at cure if have stage1, '
                                                                      'given diagnosis (surgery, '
                                                                      'radiotherapy and/or chemotherapy'),
        'rr_curative_treatment_stage2': Parameter(Types.REAL, 'relative rate of receiving medical '
                                                             'treatment aimed at cure if have stage2, '
                                                             'given diagnosis (surgery, '
                                                             'radiotherapy and/or chemotherapy'),
        'rr_curative_treatment_stage3': Parameter(Types.REAL, 'relative rate of receiving medical '
                                                             'treatment aimed at cure if have stage3, '
                                                             'given diagnosis (surgery, '
                                                             'radiotherapy and/or chemotherapy'),
        'r_diagnosis_low_grade_dysp': Parameter(Types.REAL, 'probability per 3 months of diagnosis in a person with '
                                                            'low grade oesophageal dysplasia'),
        'rr_diagnosis_high_grade_dysp': Parameter(Types.REAL, 'rate ratio for diagnosis if have high grade oesophageal '
                                                              'dysplasia'),
        'rr_diagnosis_stage1': Parameter(Types.REAL, 'rate ratio for diagnosis if have high stage 1 oesophageal '
                                                     'cancer'),
        'rr_diagnosis_stage2': Parameter(Types.REAL, 'rate ratio for diagnosis if have high stage 2 oesophageal '
                                                     'cancer'),
        'rr_diagnosis_stage3': Parameter(Types.REAL, 'rate ratio for diagnosis if have high stage 3 oesophageal '
                                                     'cancer'),
        'rr_diagnosis_stage4': Parameter(Types.REAL, 'rate ratio for diagnosis if have high stage 4 oesophageal '
                                                     'cancer'),
        'init_prop_oes_cancer_stage': Parameter(Types.REAL, 'initial proportions in ca_oesophagus categories for '
                                                     'man aged 20 with no excess alcohol and no tobacco'),
        'rp_oes_cancer_female': Parameter(Types.REAL, 'relative prevalence at baseline of oesophageal dysplasia/cancer '
                                                      'if female '),
        'rp_oes_cancer_per_year_older': Parameter(Types.REAL, 'relative prevalence at baseline of oesophageal '
                                                              'dysplasia/cancer per year older than 20 '),
        'rp_oes_cancer_tobacco': Parameter(Types.REAL, 'relative prevalence at baseline of oesophageal dysplasia/cancer '
                                                      'if tobacco '),
        'rp_oes_cancer_ex_alc': Parameter(Types.REAL,
                                           'relative prevalence at baseline of oesophageal dysplasia/cancer '),
        'init_prop_diagnosed_oes_cancer_by_stage': Parameter(Types.LIST, 'initial proportions of people with'
                                                                         'oesophageal dysplasia/cancer diagnosed'),
        'init_prop_treatment_status_oes_cancer': Parameter(Types.LIST,'initial proportions of people with'
                                                                     'oesophageal dysplasia/cancer treated'),
        # these definitions for disability weights are the ones in the global burden of disease list (Salomon)
        'daly_wt_oes_cancer_controlled': Parameter(Types.REAL, 'disability weight for oesophageal cancer '
                                                               'controlled phase - code 547'),
        'daly_wt_oes_cancer_terminal': Parameter(Types.REAL, 'disability weight for oesophageal cancer '
                                                               'terminal - code 548'),
        'daly_wt_oes_cancer_metastatic': Parameter(Types.REAL, 'disability weight for oesophageal cancer '
                                                               'metastatic - code 549'),
        'daly_wt_oes_cancer_primary_therapy': Parameter(Types.REAL, 'disability weight for oesophageal cancer '
                                                               'primary therapy - code 550')
    }

    """
    547, Controlled phase of esophageal cancer, Generic uncomplicated disease: worry and daily
    medication, has a chronic disease that requires medication every day and causes some
    worry but minimal interference with daily activities., 0.049, 0.031, 0.072

    548, Terminal phase of esophageal cancer, "Terminal phase, with medication (for cancers, 
    end-stage kidney/liver disease)", "has lost a lot of weight and regularly uses strong 
    medication to avoid constant pain. The person has no appetite, feels nauseous, and needs 
    to spend most of the day in bed.", 0.54, 0.377, 0.687

    549, Metastatic phase of esophageal cancer, "Cancer, metastatic", "has severe pain, extreme 
    fatigue, weight loss and high anxiety.", 0.451, 0.307, 0.6

    550, Diagnosis and primary therapy phase of esophageal cancer, "Cancer, diagnosis and 
    primary therapy ", "has pain, nausea, fatigue, weight loss and high anxiety.", 0.288, 0.193, 
    0.399
    """


    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'ca_oesophagus': Property(Types.CATEGORICAL, 'oesophageal dysplasia / cancer stage: none, low_grade_dysplasia'
                                                     'high_grade_dysplasia, stage1, stage2, stage3, stage4',
                                  categories=['none', 'low_grade_dysplasia', 'high_grade_dysplasia', 'stage1',
                                              'stage2', 'stage3', 'stage4']),
        'ca_oesophagus_curative_treatment_requested': Property(Types.BOOL,
                                                               'curative treatment requested of health care system '
                                                               'this 3 month period'),
        'ca_oesophagus_curative_treatment': Property(Types.CATEGORICAL, 'oesophageal dysplasia / cancer stage at'
                                                     'time of attempted curative treatment: never had treatment'
                                                     'low grade dysplasia'
                                                     'high grade dysplasia, stage 1, stage 2, stage 3',
                                                     categories=['never', 'low_grade_dysplasia', 'high_grade_dysplasia',
                                                                 'stage1', 'stage2', 'stage3']),
        'ca_oesophagus_diagnosed': Property(Types.BOOL, 'diagnosed with oesophageal dysplasia / cancer'),
        'ca_oesophageal_cancer_death': Property(Types.BOOL, 'death from oesophageal cancer'),
        'ca_incident_oes_cancer_diagnosis_this_3_month_period': Property(Types.BOOL, 'incident oesophageal cancer'
                                                                'diagnosis this 3 month period'),
        'ca_date_treatment_oesophageal_cancer': Property(Types.DATE, 'date of receiving attempted curative treatment'),
        'ca_disability': Property(Types.REAL, 'disability weight this three month period')
    }

    TREATMENT_ID = 'attempted curative treatment for oesophageal cancer'

    def read_parameters(self, data_folder):
        """Setup parameters used by the module, now including disability weights
        """


        p = self.parameters

        dfd = pd.read_excel(
                            Path(self.resourcefilepath) / 'ResourceFile_Oesophageal_Cancer.xlsx',
                            sheet_name='parameter_values')
        dfd.set_index('parameter_name', inplace=True)

        p['r_low_grade_dysplasia_none'] = dfd.loc['r_low_grade_dysplasia_none', 'value']
        p['rr_low_grade_dysplasia_none_female'] = dfd.loc['rr_low_grade_dysplasia_none_female', 'value']
        p['rr_low_grade_dysplasia_none_per_year_older'] = dfd.loc['rr_low_grade_dysplasia_none_per_year_older', 'value']
        p['rr_low_grade_dysplasia_none_tobacco'] = dfd.loc['rr_low_grade_dysplasia_none_tobacco', 'value']
        p['rr_low_grade_dysplasia_none_ex_alc'] = dfd.loc['rr_low_grade_dysplasia_none_ex_alc', 'value']
        p['r_high_grade_dysplasia_low_grade_dysp'] = dfd.loc['r_high_grade_dysplasia_low_grade_dysp', 'value']
        p['rr_high_grade_dysp_undergone_curative_treatment'] = dfd.loc['rr_high_grade_dysp_undergone_curative_treatment', 'value']
        p['r_stage1_high_grade_dysp'] = dfd.loc['r_stage1_high_grade_dysp', 'value']
        p['rr_stage1_undergone_curative_treatment'] = dfd.loc['rr_stage1_undergone_curative_treatment', 'value']
        p['r_stage2_stage1'] = dfd.loc['r_stage2_stage1', 'value']
        p['rr_stage2_undergone_curative_treatment'] = dfd.loc['rr_stage2_undergone_curative_treatment', 'value']
        p['r_stage3_stage2'] = dfd.loc['r_stage3_stage2', 'value']
        p['rr_stage3_undergone_curative_treatment'] = dfd.loc['rr_stage3_undergone_curative_treatment', 'value']
        p['r_stage4_stage3'] = dfd.loc['r_stage4_stage3', 'value']
        p['rr_stage4_undergone_curative_treatment'] = dfd.loc['rr_stage4_undergone_curative_treatment', 'value']
        p['r_death_oesoph_cancer'] = dfd.loc['r_death_oesoph_cancer', 'value']
        p['r_curative_treatment_low_grade_dysp'] = dfd.loc['r_curative_treatment_low_grade_dysp', 'value']
        p['rr_curative_treatment_high_grade_dysp'] = dfd.loc['rr_curative_treatment_high_grade_dysp', 'value']
        p['rr_curative_treatment_stage1'] = dfd.loc['rr_curative_treatment_stage1', 'value']
        p['rr_curative_treatment_stage2'] = dfd.loc['rr_curative_treatment_stage2', 'value']
        p['rr_curative_treatment_stage3'] = dfd.loc['rr_curative_treatment_stage3', 'value']
        p['r_diagnosis_stage1'] = dfd.loc['r_diagnosis_stage1', 'value']
        p['rr_diagnosis_low_grade_dysp'] = dfd.loc['rr_diagnosis_low_grade_dysp', 'value']
        p['rr_diagnosis_high_grade_dysp'] = dfd.loc['rr_diagnosis_high_grade_dysp', 'value']
        p['rr_diagnosis_stage2'] = dfd.loc['rr_diagnosis_stage2', 'value']
        p['rr_diagnosis_stage3'] = dfd.loc['rr_diagnosis_stage3', 'value']
        p['rr_diagnosis_stage4'] = dfd.loc['rr_diagnosis_stage4', 'value']
        p['init_prop_oes_cancer_stage'] = \
                    [dfd.loc['init_prop_oes_cancer_stage', 'value'], dfd.loc['init_prop_oes_cancer_stage', 'value2'],
                     dfd.loc['init_prop_oes_cancer_stage', 'value3'], dfd.loc['init_prop_oes_cancer_stage', 'value4'],
                     dfd.loc['init_prop_oes_cancer_stage', 'value5'], dfd.loc['init_prop_oes_cancer_stage', 'value6']]
        p['rp_oes_cancer_female'] = dfd.loc['rp_oes_cancer_female', 'value']
        p['rp_oes_cancer_per_year_older'] = dfd.loc['rp_oes_cancer_per_year_older', 'value']
        p['rp_oes_cancer_tobacco'] = dfd.loc['rp_oes_cancer_tobacco', 'value']
        p['rp_oes_cancer_ex_alc'] = dfd.loc['rp_oes_cancer_ex_alc', 'value']
        p['init_prop_diagnosed_oes_cancer_by_stage'] = \
            [dfd.loc['init_prop_diagnosed_oes_cancer_by_stage', 'value'], dfd.loc['init_prop_diagnosed_oes_cancer_by_stage', 'value2'],
             dfd.loc['init_prop_diagnosed_oes_cancer_by_stage', 'value3'], dfd.loc['init_prop_diagnosed_oes_cancer_by_stage', 'value4'],
             dfd.loc['init_prop_diagnosed_oes_cancer_by_stage', 'value5'], dfd.loc['init_prop_diagnosed_oes_cancer_by_stage', 'value6']]
        p['init_prop_treatment_status_oes_cancer'] = \
            [dfd.loc['init_prop_treatment_status_oes_cancer', 'value'],
            dfd.loc['init_prop_treatment_status_oes_cancer', 'value2'],
            dfd.loc['init_prop_treatment_status_oes_cancer', 'value3'],
            dfd.loc['init_prop_treatment_status_oes_cancer', 'value4'],
            dfd.loc['init_prop_treatment_status_oes_cancer', 'value5'],
            dfd.loc['init_prop_treatment_status_oes_cancer', 'value6']]

        if 'HealthBurden' in self.sim.modules.keys():
            # get the DALY weight - 547-550 are the sequale codes for oesophageal cancer
            self.parameters['daly_wt_oes_cancer_controlled'] = \
                self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=547)
            self.parameters['daly_wt_oes_cancer_terminal'] = \
                self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=548)
            self.parameters['daly_wt_oes_cancer_metastatic'] = \
                self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=549)
            self.parameters['daly_wt_oes_cancer_primary_therapy'] = \
                self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=550)

    def initialise_population(self, population):
        """Set our property values for the initial population.
        :param population: the population of individuals
        """
        df = population.props  # a shortcut to the data-frame storing data for individuals
        m = self
        rng = m.rng

        # -------------------- DEFAULTS ------------------------------------------------------------

        df['ca_oesophagus'] = 'none'
        df['ca_oesophagus_diagnosed'] = False
        df['ca_oesophagus_curative_treatment'] = 'never'
        df['ca_oesophageal_cancer_death'] = False
        df['ca_incident_oes_cancer_diagnosis_this_3_month_period'] = False
        df['ca_disability'] = 0
        df['ca_oesophagus_curative_treatment_requested'] = False
        df['ca_date_treatment_oesophageal_cancer'] = pd.NaT

        # -------------------- ASSIGN VALUES OF OESOPHAGEAL DYSPLASIA/CANCER STATUS AT BASELINE -----------

        agege20_idx = df.index[(df.age_years >= 20) & df.is_alive]

        # create dataframe of the probabilities of ca_oesophagus status for 20 year old males, no ex alcohol, no tobacco
        p_oes_dys_can = pd.DataFrame(data=[m.init_prop_oes_cancer_stage],
                                     columns=['low grade dysplasia', 'high grade dysplasia', 'stage 1',
                                              'stage 2', 'stage 3', 'stage 4'], index=agege20_idx)

        # create probabilities of oes dysplasia and oe cancer for all over age 20
        p_oes_dys_can.loc[(df.sex == 'F') & (df.age_years >= 20) & df.is_alive] *= m.rp_oes_cancer_female
        p_oes_dys_can.loc[df.li_ex_alc & (df.age_years >= 20) & df.is_alive] *= m.rp_oes_cancer_ex_alc
        p_oes_dys_can.loc[df.li_tob & (df.age_years >= 20) & df.is_alive] *= m.rp_oes_cancer_tobacco

        p_oes_dys_can_age_muliplier = pd.Series(m.rp_oes_cancer_per_year_older ** (df.age_years - 20),
                                                index=agege20_idx)

        random_draw = pd.Series(rng.random_sample(size=len(agege20_idx)),
                                   index=df.index[(df.age_years >= 20) & df.is_alive])

        # create a temporary dataframe called dfx to hold values of probabilities and random draw
        dfx = pd.concat([p_oes_dys_can, p_oes_dys_can_age_muliplier, random_draw], axis=1)
        dfx.columns = ['p_low_grade_dysplasia', 'p_high_grade_dysplasia', 'p_stage1', 'p_stage2', 'p_stage3',
                       'p_stage4', 'p_oes_dys_can_age_muliplier', 'random_draw']

        dfx.p_low_grade_dysplasia *= dfx.p_oes_dys_can_age_muliplier
        dfx.p_high_grade_dysplasia *= dfx.p_oes_dys_can_age_muliplier
        dfx.p_stage1 *= dfx.p_oes_dys_can_age_muliplier
        dfx.p_stage2 *= dfx.p_oes_dys_can_age_muliplier
        dfx.p_stage3 *= dfx.p_oes_dys_can_age_muliplier
        dfx.p_stage4 *= dfx.p_oes_dys_can_age_muliplier

        # based on probabilities of being in each category, define cut-offs to determine status from
        # random draw uniform(0,1)

        # assign baseline values of ca_oesophagus based on probabilities and value of random draw
        idx_low_grade_dysplasia = dfx.index[dfx.p_low_grade_dysplasia > dfx.random_draw]
        idx_high_grade_dysplasia = dfx.index[(dfx.p_low_grade_dysplasia < dfx.random_draw) &
                   ((dfx.p_low_grade_dysplasia + dfx.p_high_grade_dysplasia) > dfx.random_draw)]
        idx_stage1 = dfx.index[((dfx.p_low_grade_dysplasia + dfx.p_high_grade_dysplasia) < dfx.random_draw) &
                   ((dfx.p_low_grade_dysplasia + dfx.p_high_grade_dysplasia + dfx.p_stage1) > dfx.random_draw)]
        idx_stage2 = dfx.index[((dfx.p_low_grade_dysplasia + dfx.p_high_grade_dysplasia + dfx.p_stage1)
                               < dfx.random_draw) & ((dfx.p_low_grade_dysplasia +
                                dfx.p_high_grade_dysplasia + dfx.p_stage1 + dfx.p_stage2) > dfx.random_draw)]
        idx_stage3 = dfx.index[((dfx.p_low_grade_dysplasia +
                                dfx.p_high_grade_dysplasia + dfx.p_stage1 + dfx.p_stage2) < dfx.random_draw)
        & ((dfx.p_low_grade_dysplasia +
                                dfx.p_high_grade_dysplasia + dfx.p_stage1 + dfx.p_stage2 + dfx.p_stage3)
                               > dfx.random_draw)]
        idx_stage4 = dfx.index[((dfx.p_low_grade_dysplasia +
                                dfx.p_high_grade_dysplasia + dfx.p_stage1 + dfx.p_stage2 + dfx.p_stage3)
                               < dfx.random_draw) & ((dfx.p_low_grade_dysplasia +
                                dfx.p_high_grade_dysplasia + dfx.p_stage1 + dfx.p_stage2 + dfx.p_stage3
                                                     + dfx.p_stage4) > dfx.random_draw)]

        df.loc[idx_low_grade_dysplasia, 'ca_oesophagus'] = 'low_grade_dysplasia'
        df.loc[idx_high_grade_dysplasia, 'ca_oesophagus'] = 'high_grade_dysplasia'
        df.loc[idx_stage1, 'ca_oesophagus'] = 'stage1'
        df.loc[idx_stage2, 'ca_oesophagus'] = 'stage2'
        df.loc[idx_stage3, 'ca_oesophagus'] = 'stage3'
        df.loc[idx_stage4, 'ca_oesophagus'] = 'stage4'

        # -------------------- ASSIGN VALUES CA_OESOPHAGUS DIAGNOSED AT BASELINE --------------------------------

        low_grade_dys_idx = df.index[df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia')]
        high_grade_dys_idx = df.index[df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia')]
        stage1_oes_can_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage1')]
        stage2_oes_can_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage2')]
        stage3_oes_can_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage3')]
        stage4_oes_can_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage4')]

        # create a series of random uniform(0,1) for those with low grade dys (m. is self.)
        random_draw = pd.Series(rng.random_sample(size=len(low_grade_dys_idx)),
                                index=df.index[df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia')])
        # allocate who is diagnosed this period based on random draw and parameter with prop diagnosed
        df.loc[low_grade_dys_idx, 'ca_oesophagus_diagnosed'] = \
            random_draw < m.init_prop_diagnosed_oes_cancer_by_stage[0]
        # now do as above for each oesophageal cancer stage
        random_draw = pd.Series(rng.random_sample(size=len(high_grade_dys_idx)),
                                index=df.index[df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia')])
        df.loc[high_grade_dys_idx, 'ca_oesophagus_diagnosed'] = \
            random_draw < m.init_prop_diagnosed_oes_cancer_by_stage[1]
        random_draw = pd.Series(rng.random_sample(size=len(stage1_oes_can_idx)),
                                index=df.index[df.is_alive & (df.ca_oesophagus == 'stage1')])
        df.loc[stage1_oes_can_idx, 'ca_oesophagus_diagnosed'] = \
            random_draw < m.init_prop_diagnosed_oes_cancer_by_stage[2]
        random_draw = pd.Series(rng.random_sample(size=len(stage2_oes_can_idx)),
                                index=df.index[df.is_alive & (df.ca_oesophagus == 'stage2')])
        df.loc[stage2_oes_can_idx, 'ca_oesophagus_diagnosed'] = \
            random_draw < m.init_prop_diagnosed_oes_cancer_by_stage[3]
        random_draw = pd.Series(rng.random_sample(size=len(stage3_oes_can_idx)),
                                index=df.index[df.is_alive & (df.ca_oesophagus == 'stage3')])
        df.loc[stage3_oes_can_idx, 'ca_oesophagus_diagnosed'] = \
            random_draw < m.init_prop_diagnosed_oes_cancer_by_stage[4]
        random_draw = pd.Series(rng.random_sample(size=len(stage4_oes_can_idx)),
                                  index=df.index[df.is_alive & (df.ca_oesophagus == 'stage4')])
        df.loc[stage4_oes_can_idx, 'ca_oesophagus_diagnosed'] = \
            random_draw < m.init_prop_diagnosed_oes_cancer_by_stage[5]

        # -------------------- ASSIGN VALUES CA_OESOPHAGUS_CURATIVE_TREATMENT AT BASELINE -------------------

        # create indexes for people in each stage
        low_grade_dys_diagnosed_idx = df.index[df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia')
                                            & df.ca_oesophagus_diagnosed]
        high_grade_dys_diagnosed_idx = df.index[df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia')
                                            & df.ca_oesophagus_diagnosed]
        stage1_oes_can_diagnosed_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage1')
                                            & df.ca_oesophagus_diagnosed]
        stage2_oes_can_diagnosed_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage2')
                                            & df.ca_oesophagus_diagnosed]
        stage3_oes_can_diagnosed_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage3')
                                            & df.ca_oesophagus_diagnosed]
        stage4_oes_can_diagnosed_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage4')
                                            & df.ca_oesophagus_diagnosed]
        # for those with low grade dysplasia, create a series with random uniform(0,1)
        random_draw = pd.Series(rng.random_sample(size=len(low_grade_dys_diagnosed_idx)),
                                index=df.index[df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia')
                                               & df.ca_oesophagus_diagnosed])
        # create series with initial proportion with low grade dysplasia, based on parameter
        p_treatment = pd.Series(m.init_prop_treatment_status_oes_cancer[0], index=df.index[df.is_alive &
                                            (df.ca_oesophagus == 'low_grade_dysplasia')
                                            & df.ca_oesophagus_diagnosed])
        # create a temporary dataframe dfx that has columns probability of treatment and random draw
        dfx = pd.concat([p_treatment, random_draw], axis=1)
        dfx.columns = ['p_treatment', 'random_draw']
        # create an index for those who are diagnosed and set ca_oesophagus_curative_treatment to
        # low_grade_dysplasia
        idx_low_grade_dysplasia_treatment = dfx.index[dfx.p_treatment > dfx.random_draw]
        df.loc[idx_low_grade_dysplasia_treatment, 'ca_oesophagus_curative_treatment'] = 'low_grade_dysplasia'

        # do as above for high grade dysplasia
        random_draw = pd.Series(rng.random_sample(size=len(high_grade_dys_diagnosed_idx)),
                                index=df.index[df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia')
                                               & df.ca_oesophagus_diagnosed])
        p_treatment = pd.Series(m.init_prop_treatment_status_oes_cancer[1], index=df.index[df.is_alive &
                                            (df.ca_oesophagus == 'high_grade_dysplasia')
                                            & df.ca_oesophagus_diagnosed])
        dfx = pd.concat([p_treatment, random_draw], axis=1)
        dfx.columns = ['p_treatment', 'random_draw']
        idx_high_grade_dysplasia_treatment = dfx.index[dfx.p_treatment > dfx.random_draw]
        df.loc[idx_high_grade_dysplasia_treatment, 'ca_oesophagus_curative_treatment'] = 'high_grade_dysplasia'

        # do as above for stage1 oes cancer
        random_draw = pd.Series(rng.random_sample(size=len(stage1_oes_can_diagnosed_idx)),
                                index=df.index[df.is_alive & (df.ca_oesophagus == 'stage1')
                                               & df.ca_oesophagus_diagnosed])
        p_treatment = pd.Series(m.init_prop_treatment_status_oes_cancer[2], index=df.index[df.is_alive &
                                            (df.ca_oesophagus == 'stage1')
                                            & df.ca_oesophagus_diagnosed])
        dfx = pd.concat([p_treatment, random_draw], axis=1)
        dfx.columns = ['p_treatment', 'random_draw']
        idx_stage1_oes_can_treatment = dfx.index[dfx.p_treatment > dfx.random_draw]
        df.loc[idx_stage1_oes_can_treatment, 'ca_oesophagus_curative_treatment'] = 'stage1'

        # do as above for stage2 oes cancer
        random_draw = pd.Series(rng.random_sample(size=len(stage2_oes_can_diagnosed_idx)),
                                index=df.index[df.is_alive & (df.ca_oesophagus == 'stage2')
                                               & df.ca_oesophagus_diagnosed])
        p_treatment = pd.Series(m.init_prop_treatment_status_oes_cancer[3], index=df.index[df.is_alive &
                                            (df.ca_oesophagus == 'stage2')
                                            & df.ca_oesophagus_diagnosed])
        dfx = pd.concat([p_treatment, random_draw], axis=1)
        dfx.columns = ['p_treatment', 'random_draw']
        idx_stage2_oes_can_treatment = dfx.index[dfx.p_treatment > dfx.random_draw]
        df.loc[idx_stage2_oes_can_treatment, 'ca_oesophagus_curative_treatment'] = 'stage2'

        # do as above for stage3 oes cancer
        random_draw = pd.Series(rng.random_sample(size=len(stage3_oes_can_diagnosed_idx)),
                                index=df.index[df.is_alive & (df.ca_oesophagus == 'stage3')
                                               & df.ca_oesophagus_diagnosed])
        p_treatment = pd.Series(m.init_prop_treatment_status_oes_cancer[4], index=df.index[df.is_alive &
                                            (df.ca_oesophagus == 'stage3')
                                            & df.ca_oesophagus_diagnosed])
        dfx = pd.concat([p_treatment, random_draw], axis=1)
        dfx.columns = ['p_treatment', 'random_draw']
        idx_stage3_oes_can_treatment = dfx.index[dfx.p_treatment > dfx.random_draw]
        df.loc[idx_stage3_oes_can_treatment, 'ca_oesophagus_curative_treatment'] = 'stage3'

    def initialise_simulation(self, sim):
        """Add lifestyle events to the simulation
        """
        # start simulation immediately - so above values are updated immediately
        event = OesCancerEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=0))

        event = OesCancerLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=0))

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """

        df = self.sim.population.props

        df.at[child_id, 'ca_oesophagus'] = 'none'
        df.at[child_id, 'ca_oesophagus_diagnosed'] = False
        df.at[child_id, 'ca_oesophagus_curative_treatment'] = 'never'
        df.at[child_id, 'ca_oesophageal_cancer_death'] = 'never'
        df.at[child_id, 'ca_incident_oes_cancer_diagnosis_this_3_month_period'] = False
        df.at[child_id, 'ca_disability'] = 0

    def query_symptoms_now(self):
        # This is called by the health-care seeking module
        # All modules refresh the symptomology of persons at this time
        # And report it on the unified symptomology scale
        # Map the specific symptoms for this disease onto the unified coding scheme
        df = self.sim.population.props  # shortcut to population properties dataframe

        # todo: currently diagnosis can just occur at some point - need to create properties for
        # todo: symptoms such as dysphagia and then symptoms that are caused by later stage cancers
        # todo: and let these determine presentation to the health system (and hence diagnosis)

        return pd.Series('1', index=df.index[df.is_alive])

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug('This is oesophageal_cancer, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)
        pass

    def report_daly_values(self):
        # This must send back a dataframe that reports on the HealthStates for all individuals over
        # the past month

        #       logger.debug('This is Oesophageal Cancer reporting my health values')

        df = self.sim.population.props  # shortcut to population properties dataframe

        disability_series_for_alive_persons = df.loc[df['is_alive'], 'ca_disability']

        return disability_series_for_alive_persons


class OesCancerEvent(RegularEvent, PopulationScopeEventMixin):
    """
    Regular event that updates all oesophagealcancer properties for population
    """
    def __init__(self, module):
        """schedule to run every 3 months
        note: if change this offset from 3 months need to consider code conditioning on age.years_exact
        :param module: the module that created this event
        """
        super().__init__(module, frequency=DateOffset(months=3))

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """

        TREATMENT_ID = 'attempted curative treatment for oesophageal cancer'

        df = population.props
        m = self.module
        rng = m.rng

        # set ca_oesophageal_cancer_death back to False after death
        df['ca_oesophageal_cancer_death'] = False
        df['ca_disability'] = 0
#       df['ca_oesophagus_curative_treatment_requested'] = False
        df['ca_incident_oes_cancer_diagnosis_this_3_month_period'] = False

        # -------------------- UPDATING of CA-OESOPHAGUS OVER TIME -----------------------------------

        # updating for people aged over 20 with current status 'none'

        # create indexes of subgroups of people with no current oes cancer
        ca_oes_current_none_idx = df.index[df.is_alive & (df.ca_oesophagus == 'none') & (df.age_years >= 20)]
        ca_oes_current_none_f_idx = df.index[df.is_alive & (df.ca_oesophagus == 'none') & (df.age_years >= 20) &
                                           (df.sex == 'F')]
        ca_oes_current_none_tob_idx = df.index[df.is_alive & (df.ca_oesophagus == 'none') & (df.age_years >= 20) &
                                           df.li_tob]
        ca_oes_current_none_ex_alc_idx = df.index[df.is_alive & (df.ca_oesophagus == 'none') & (df.age_years >= 20) &
                                           df.li_ex_alc]

        # create series of the parameter r_low_grade_dysplasia_none which is the probability of low grade dysplasia for
        # men, age20, no excess alcohol, no tobacco
        eff_prob_low_grade_dysp = pd.Series(m.r_low_grade_dysplasia_none,
                                    index=df.index[df.is_alive & (df.ca_oesophagus == 'none') & (df.age_years >= 20)])
        # update this "effective" probability by multiplyng by the rate ratio being female for the females
        eff_prob_low_grade_dysp.loc[ca_oes_current_none_f_idx] *= m.rr_low_grade_dysplasia_none_female
        # update this "effective" probability by multiplyng by the rate ratio being a tobacco user for the tobacco users
        eff_prob_low_grade_dysp.loc[ca_oes_current_none_tob_idx] *= m.rr_low_grade_dysplasia_none_tobacco
        # as above for excess alcohol users
        eff_prob_low_grade_dysp.loc[ca_oes_current_none_ex_alc_idx] *= m.rr_low_grade_dysplasia_none_ex_alc
        # create series which is rate ratio to be applied for each persons age - rate rate by age is continuous
        p_oes_dys_can_age_muliplier = pd.Series(m.rr_low_grade_dysplasia_none_per_year_older ** (df.age_years - 20),
                                                index=ca_oes_current_none_idx)
        # create a random draw for each in the data frame of all aged > 20 without oes cancer
        random_draw = pd.Series(rng.random_sample(size=len(ca_oes_current_none_idx)),
                                   index=df.index[(df.age_years >= 20) & df.is_alive & (df.ca_oesophagus == 'none')])
        # create a temporary dataframe with the effective probability, the age multiplier and the random draw
        dfx = pd.concat([eff_prob_low_grade_dysp, p_oes_dys_can_age_muliplier, random_draw], axis=1)
        dfx.columns = ['eff_prob_low_grade_dysp', 'p_oes_dys_can_age_muliplier', 'random_draw']
        # update the effective probability so it now includes the effect of age
        dfx.eff_prob_low_grade_dysp *= p_oes_dys_can_age_muliplier
        # based on the random draw determine who develops low grade dysplasia in this update
        idx_incident_low_grade_dysp = dfx.index[dfx.eff_prob_low_grade_dysp > dfx.random_draw]
        df.loc[idx_incident_low_grade_dysp, 'ca_oesophagus'] = 'low_grade_dysplasia'

        # updating for people aged over 20 with current status 'low grade dysplasia'
        # this uses a very similar approach to that described in detail above

        ca_oes_current_low_grade_dysp_idx = df.index[df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia') &
                                                     (df.age_years >= 20)]
        ca_oes_current_low_grade_dysp_treated_idx = df.index[df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia') &
                                                     (df.age_years >= 20)
                                                     & (df.ca_oesophagus_curative_treatment == 'low_grade_dysplasia')]
        eff_prob_high_grade_dysp = pd.Series(m.r_high_grade_dysplasia_low_grade_dysp,
                                             index=df.index[df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia')
                                                            & (df.age_years >= 20)])
        eff_prob_high_grade_dysp.loc[ca_oes_current_low_grade_dysp_treated_idx] \
            *= m.rr_high_grade_dysp_undergone_curative_treatment
        random_draw = pd.Series(rng.random_sample(size=len(ca_oes_current_low_grade_dysp_idx)),
                                index=df.index[(df.age_years >= 20) & df.is_alive &
                                               (df.ca_oesophagus == 'low_grade_dysplasia')])
        dfx = pd.concat([eff_prob_high_grade_dysp, random_draw], axis=1)
        dfx.columns = ['eff_prob_high_grade_dysp', 'random_draw']
        idx_incident_high_grade_dysp = dfx.index[dfx.eff_prob_high_grade_dysp > dfx.random_draw]
        df.loc[idx_incident_high_grade_dysp, 'ca_oesophagus'] = 'high_grade_dysplasia'

        # updating for people aged over 20 with current status 'high grade dysplasia'
        # this uses a very similar approach to that described in detail above

        ca_oes_current_high_grade_dysp_idx = df.index[df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia') &
                                                     (df.age_years >= 20)]
        ca_oes_current_high_grade_dysp_treated_idx = df.index[df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia') &
                                                     (df.age_years >= 20)
                                                     & (df.ca_oesophagus_curative_treatment == 'high_grade_dysplasia')]
        eff_prob_stage1 = pd.Series(m.r_stage1_high_grade_dysp,
                                             index=df.index[df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia')
                                                            & (df.age_years >= 20)])
        eff_prob_stage1.loc[ca_oes_current_high_grade_dysp_treated_idx] \
            *= m.rr_stage1_undergone_curative_treatment
        random_draw = pd.Series(rng.random_sample(size=len(ca_oes_current_high_grade_dysp_idx)),
                                index=df.index[(df.age_years >= 20) & df.is_alive &
                                               (df.ca_oesophagus == 'high_grade_dysplasia')])
        dfx = pd.concat([eff_prob_stage1, random_draw], axis=1)
        dfx.columns = ['eff_prob_stage1', 'random_draw']
        idx_incident_stage1 = dfx.index[dfx.eff_prob_stage1 > dfx.random_draw]
        df.loc[idx_incident_stage1, 'ca_oesophagus'] = 'stage1'

        # updating for people aged over 20 with current status stage 1 oes cancer
        # this uses a very similar approach to that described in detail above

        ca_oes_current_stage1_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage1') &
                                             (df.age_years >= 20)]
        ca_oes_current_stage1_treated_idx = df.index[
            df.is_alive & (df.ca_oesophagus == 'stage1') &
            (df.age_years >= 20)
            & (df.ca_oesophagus_curative_treatment == 'stage1')]
        eff_prob_stage2 = pd.Series(m.r_stage2_stage1,
                                    index=df.index[df.is_alive & (df.ca_oesophagus == 'stage1')
                                                   & (df.age_years >= 20)])
        eff_prob_stage2.loc[ca_oes_current_stage1_treated_idx] \
            *= m.rr_stage2_undergone_curative_treatment
        random_draw = pd.Series(rng.random_sample(size=len(ca_oes_current_stage1_idx)),
                                index=df.index[(df.age_years >= 20) & df.is_alive &
                                               (df.ca_oesophagus == 'stage1')])
        dfx = pd.concat([eff_prob_stage2, random_draw], axis=1)
        dfx.columns = ['eff_prob_stage2', 'random_draw']
        idx_incident_stage2 = dfx.index[dfx.eff_prob_stage2 > dfx.random_draw]
        df.loc[idx_incident_stage2, 'ca_oesophagus'] = 'stage2'

        # updating for people aged over 20 with current status stage 2 oes cancer
        # this uses a very similar approach to that described in detail above

        ca_oes_current_stage2_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage2') &
                                             (df.age_years >= 20)]
        ca_oes_current_stage2_treated_idx = df.index[
            df.is_alive & (df.ca_oesophagus == 'stage2') &
            (df.age_years >= 20)
            & (df.ca_oesophagus_curative_treatment == 'stage2')]
        eff_prob_stage3 = pd.Series(m.r_stage3_stage2,
                                    index=df.index[df.is_alive & (df.ca_oesophagus == 'stage2')
                                                   & (df.age_years >= 20)])
        eff_prob_stage3.loc[ca_oes_current_stage2_treated_idx] \
            *= m.rr_stage3_undergone_curative_treatment
        random_draw = pd.Series(rng.random_sample(size=len(ca_oes_current_stage2_idx)),
                                index=df.index[(df.age_years >= 20) & df.is_alive &
                                               (df.ca_oesophagus == 'stage2')])
        dfx = pd.concat([eff_prob_stage3, random_draw], axis=1)
        dfx.columns = ['eff_prob_stage3', 'random_draw']
        idx_incident_stage3 = dfx.index[dfx.eff_prob_stage3 > dfx.random_draw]
        df.loc[idx_incident_stage3, 'ca_oesophagus'] = 'stage3'

        # updating for people aged over 20 with current status stage 3 oes cancer
        # this uses a very similar approach to that described in detail above

        ca_oes_current_stage3_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage3') &
                                             (df.age_years >= 20)]
        ca_oes_current_stage3_treated_idx = df.index[
            df.is_alive & (df.ca_oesophagus == 'stage3') &
            (df.age_years >= 20)
            & (df.ca_oesophagus_curative_treatment == 'stage3')]
        eff_prob_stage4 = pd.Series(m.r_stage4_stage3,
                                    index=df.index[df.is_alive & (df.ca_oesophagus == 'stage3')
                                                   & (df.age_years >= 20)])
        eff_prob_stage4.loc[ca_oes_current_stage3_treated_idx] \
            *= m.rr_stage4_undergone_curative_treatment
        random_draw = pd.Series(rng.random_sample(size=len(ca_oes_current_stage3_idx)),
                                index=df.index[(df.age_years >= 20) & df.is_alive &
                                               (df.ca_oesophagus == 'stage3')])
        dfx = pd.concat([eff_prob_stage4, random_draw], axis=1)
        dfx.columns = ['eff_prob_stage4', 'random_draw']
        idx_incident_stage4 = dfx.index[dfx.eff_prob_stage4 > dfx.random_draw]
        df.loc[idx_incident_stage4, 'ca_oesophagus'] = 'stage4'

        # -------------------- UPDATING OF CA_OESOPHAGUS DIAGNOSED OVER TIME --------------------------------

        # todo: make diagnosis an hsi event (and model symptoms (dysphagia) leading to presentation
        # todo: for diagnosis

        df['ca_incident_oes_cancer_diagnosis_this_3_month_period'] = False

        # update diagnosis status for undiagnosed people with low grade dysplasia

        # create index of people with undiagnosed low grade dysplasia
        ca_oes_current_low_grade_dysp_not_diag_idx = df.index[
            df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia') &
            (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed]
        # create series which is the probability of them being diagnosed, based on probability of being
        # diagnosed if in stage 1 and the rate ratio for diagnosis if have low grade dysplasis rather than
        # stage 1 (remember "stage" 1 is the third stage (low grade dysplasia, high grade dysplasia, stage 1, etc)
        eff_prob_diag = pd.Series(m.r_diagnosis_stage1 * m.rr_diagnosis_low_grade_dysp,
                                  index=df.index[df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia') &
                                                 (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed])
        random_draw = rng.random_sample(size=len(ca_oes_current_low_grade_dysp_not_diag_idx))
        df.loc[ca_oes_current_low_grade_dysp_not_diag_idx, 'ca_oesophagus_diagnosed'] = (random_draw < eff_prob_diag)
        # based on a random draw from a uniform 0,1 determine if diagnosis occurs this period
        df.loc[ca_oes_current_low_grade_dysp_not_diag_idx, 'ca_incident_oes_cancer_diagnosis_this_3_month_period'] \
            = (random_draw < eff_prob_diag)

        # update diagnosis status for undiagnosed people with high grade dysplasia
        # uses the same approach as above

        ca_oes_current_high_grade_dysp_not_diag_idx = df.index[df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia') &
                                                          (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed]
        eff_prob_diag = pd.Series(m.r_diagnosis_stage1 * m.rr_diagnosis_high_grade_dysp,
                              index=df.index[df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia') &
                                             (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed])
        random_draw = rng.random_sample(size=len(ca_oes_current_high_grade_dysp_not_diag_idx))
        df.loc[ca_oes_current_high_grade_dysp_not_diag_idx, 'ca_oesophagus_diagnosed'] = (random_draw < eff_prob_diag)
        df.loc[ca_oes_current_high_grade_dysp_not_diag_idx, 'ca_incident_oes_cancer_diagnosis_this_3_month_period'] \
            = (random_draw < eff_prob_diag)

        # update diagnosis status for undiagnosed people with stage 1 oes cancer
        # uses the same approach as above

        ca_oes_current_stage1_not_diag_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage1') &
                                                      (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed]
        eff_prob_diag = pd.Series(m.r_diagnosis_stage1,
                                  index=df.index[df.is_alive & (df.ca_oesophagus == 'stage1') &
                                                 (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed])
        random_draw = rng.random_sample(size=len(ca_oes_current_stage1_not_diag_idx))
        df.loc[ca_oes_current_stage1_not_diag_idx, 'ca_oesophagus_diagnosed'] = (random_draw < eff_prob_diag)
        df.loc[ca_oes_current_stage1_not_diag_idx, 'ca_incident_oes_cancer_diagnosis_this_3_month_period'] \
            = (random_draw < eff_prob_diag)

        # update diagnosis status for undiagnosed people with stage 2 oes cancer
        # uses the same approach as above

        ca_oes_current_stage2_not_diag_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage2') &
                                                          (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed]
        eff_prob_diag = pd.Series(m.r_diagnosis_stage1 * m.rr_diagnosis_stage2,
                              index=df.index[df.is_alive & (df.ca_oesophagus == 'stage2') &
                                             (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed])
        random_draw = rng.random_sample(size=len(ca_oes_current_stage2_not_diag_idx))
        df.loc[ca_oes_current_stage2_not_diag_idx, 'ca_oesophagus_diagnosed'] = (random_draw < eff_prob_diag)
        df.loc[ca_oes_current_stage2_not_diag_idx, 'ca_incident_oes_cancer_diagnosis_this_3_month_period'] \
            = (random_draw < eff_prob_diag)

        # update diagnosis status for undiagnosed people with stage 3 oes cancer
        # uses the same approach as above

        ca_oes_current_stage3_not_diag_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage3') &
                                                          (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed]
        eff_prob_diag = pd.Series(m.r_diagnosis_stage1 * m.rr_diagnosis_stage3,
                              index=df.index[df.is_alive & (df.ca_oesophagus == 'stage3') &
                                             (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed])
        random_draw = rng.random_sample(size=len(ca_oes_current_stage3_not_diag_idx))
        df.loc[ca_oes_current_stage3_not_diag_idx, 'ca_oesophagus_diagnosed'] = (random_draw < eff_prob_diag)
        df.loc[ca_oes_current_stage3_not_diag_idx, 'ca_incident_oes_cancer_diagnosis_this_3_month_period'] \
            = (random_draw < eff_prob_diag)

        # update diagnosis status for undiagnosed people with stage 4 oes cancer
        # uses the same approach as above

        ca_oes_current_stage4_not_diag_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage4') &
                                                          (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed]
        eff_prob_diag = pd.Series(m.r_diagnosis_stage1 * m.rr_diagnosis_stage4,
                              index=df.index[df.is_alive & (df.ca_oesophagus == 'stage4') &
                                             (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed])
        random_draw = rng.random_sample(size=len(ca_oes_current_stage4_not_diag_idx))
        df.loc[ca_oes_current_stage4_not_diag_idx, 'ca_oesophagus_diagnosed'] = (random_draw < eff_prob_diag)
        df.loc[ca_oes_current_stage4_not_diag_idx, 'ca_incident_oes_cancer_diagnosis_this_3_month_period'] \
            = (random_draw < eff_prob_diag)

        # -------------------- UPDATING VALUES OF CA_OESOPHAGUS_CURATIVE_TREATMENT -------------------

        # update ca_oesophagus_curative_treatment for diagnosed, untreated people with low grade dysplasia w
        # this uses the approach descibed in detail above for updating diagosis status
        ca_oes_diag_low_grade_dysp_not_treated_idx = df.index[
            df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia') &
            (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (df.ca_oesophagus_curative_treatment == 'never')]
        eff_prob_treatment = pd.Series(m.r_curative_treatment_low_grade_dysp,
                                       index=df.index[df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia') &
                                                      (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (
                                                              df.ca_oesophagus_curative_treatment == 'never')])
        random_draw = pd.Series(rng.random_sample(size=len(ca_oes_diag_low_grade_dysp_not_treated_idx)),
                                index=df.index[df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia') &
                                               (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (
                                                       df.ca_oesophagus_curative_treatment == 'never')])
        dfx = pd.concat([eff_prob_treatment, random_draw], axis=1)
        dfx.columns = ['eff_prob_treatment', 'random_draw']
        requested_treatment_low_grade_dysplasia_idx = dfx.index[dfx.eff_prob_treatment > dfx.random_draw]
        df.loc[requested_treatment_low_grade_dysplasia_idx, 'ca_oesophagus_curative_treatment_requested'] = True

        # generate the HSI Events whereby persons present for care and get treatment_low_grade_dysplasia

        for person_id in requested_treatment_low_grade_dysplasia_idx:
            # For this person, determine when they will seek care (uniform distibition [0,91]days from now)
            date_seeking_care = self.sim.date + pd.DateOffset(days=int(self.module.rng.uniform(0,91)))

            # For this person, create the HSI Event for their presentation for care
            hsi_present_for_care = HSIoStartTreatmentLowGradeOesDysplasia(self.module,person_id)

            # Enter this event to the HealthSystem
            self.sim.modules['HealthSystem'].schedule_hsi_event(hsi_present_for_care,
                                                                priority=0,
                                                                topen=date_seeking_care,
                                                                tclose=None)

        # update ca_oesophagus_curative_treatment for diagnosed, untreated people with high grade dysplasia w
        # this follows the same approach as above

        ca_oes_diag_high_grade_dysp_not_treated_idx = df.index[
            df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia') &
            (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (df.ca_oesophagus_curative_treatment == 'never')]
        eff_prob_treatment = pd.Series(m.r_curative_treatment_low_grade_dysp*m.rr_curative_treatment_high_grade_dysp,
                                       index=df.index[df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia') &
                                                      (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (
                                                              df.ca_oesophagus_curative_treatment == 'never')])
        random_draw = pd.Series(rng.random_sample(size=len(ca_oes_diag_high_grade_dysp_not_treated_idx)),
                                index=df.index[df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia') &
                                               (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (
                                                       df.ca_oesophagus_curative_treatment == 'never')])
        dfx = pd.concat([eff_prob_treatment, random_draw], axis=1)
        dfx.columns = ['eff_prob_treatment', 'random_draw']
        requested_treatment_high_grade_dysplasia_idx = dfx.index[dfx.eff_prob_treatment > dfx.random_draw]
        df.loc[requested_treatment_high_grade_dysplasia_idx, 'ca_oesophagus_curative_treatment_requested'] = True

        # generate the HSI Events whereby persons present for care and get treatment_high_grade_dysplasia
        # this follows the same approach as above

        for person_id in requested_treatment_high_grade_dysplasia_idx:
            # For this person, determine when they will seek care (uniform distibition [0,91]days from now)
            date_seeking_care = self.sim.date + pd.DateOffset(days=int(self.module.rng.uniform(0,91)))

            # For this person, create the HSI Event for their presentation for care
            hsi_present_for_care = HSIoStartTreatmentHighGradeOesDysplasia(self.module, person_id)

            # Enter this event to the HealthSystem
            self.sim.modules['HealthSystem'].schedule_hsi_event(hsi_present_for_care,
                                                                priority=0,
                                                                topen=date_seeking_care,
                                                                tclose=None)

        # update ca_oesophagus_curative_treatment for diagnosed, untreated people with stage 1
        # this follows the same approach as above

        ca_oes_diag_stage1_not_treated_idx = df.index[
            df.is_alive & (df.ca_oesophagus == 'stage1') &
            (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (df.ca_oesophagus_curative_treatment == 'never')]
        eff_prob_treatment = pd.Series(m.r_curative_treatment_low_grade_dysp * m.rr_curative_treatment_stage1,
                                       index=df.index[df.is_alive & (df.ca_oesophagus == 'stage1') &
                                                      (df.age_years >= 20) & df.ca_oesophagus_diagnosed &
                                                      (df.ca_oesophagus_curative_treatment == 'never')])
        random_draw = pd.Series(rng.random_sample(size=len(ca_oes_diag_stage1_not_treated_idx)),
                                index=df.index[df.is_alive & (df.ca_oesophagus == 'stage1') &
                                               (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (
                                                   df.ca_oesophagus_curative_treatment == 'never')])
        dfx = pd.concat([eff_prob_treatment, random_draw], axis=1)
        dfx.columns = ['eff_prob_treatment', 'random_draw']
        requested_treatment_stage1_idx = dfx.index[dfx.eff_prob_treatment > dfx.random_draw]
        df.loc[requested_treatment_stage1_idx, 'ca_oesophagus_curative_treatment_requested'] = True

        # generate the HSI Events whereby persons present for care and get treatment atsge 1
        # this follows the same approach as above

        for person_id in requested_treatment_stage1_idx:
            # For this person, determine when they will seek care (uniform distibition [0,91]days from now)
            date_seeking_care = self.sim.date + pd.DateOffset(days=int(self.module.rng.uniform(0,91)))

            # For this person, create the HSI Event for their presentation for care
            hsi_present_for_care = HSIoStartTreatmentStage1OesCancer(self.module, person_id)

            # Enter this event to the HealthSystem
            self.sim.modules['HealthSystem'].schedule_hsi_event(hsi_present_for_care,
                                                                priority=0,
                                                                topen=date_seeking_care,
                                                                tclose=None)

        # update ca_oesophagus_curative_treatment for diagnosed, untreated people with stage 2
        # this follows the same approach as above

        ca_oes_diag_stage2_not_treated_idx = df.index[
            df.is_alive & (df.ca_oesophagus == 'stage2') &
            (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (df.ca_oesophagus_curative_treatment == 'never')]
        eff_prob_treatment = pd.Series(m.r_curative_treatment_low_grade_dysp*m.rr_curative_treatment_stage2,
                                       index=df.index[df.is_alive & (df.ca_oesophagus == 'stage2') &
                                                      (df.age_years >= 20) & df.ca_oesophagus_diagnosed &
                                                      (df.ca_oesophagus_curative_treatment == 'never') ] )
        random_draw = pd.Series(rng.random_sample(size=len(ca_oes_diag_stage2_not_treated_idx)),
                                index=df.index[df.is_alive & (df.ca_oesophagus == 'stage2') &
                                               (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (
                                                       df.ca_oesophagus_curative_treatment == 'never')])
        dfx = pd.concat([eff_prob_treatment, random_draw], axis=1)
        dfx.columns = ['eff_prob_treatment', 'random_draw']
        requested_treatment_stage2_idx = dfx.index[dfx.eff_prob_treatment > dfx.random_draw]
        df.loc[requested_treatment_stage2_idx, 'ca_oesophagus_curative_treatment_requested'] = True

        # generate the HSI Events whereby persons present for care and get treatment for stage 2 oes cancer
        # this follows the same approach as above

        for person_id in requested_treatment_stage2_idx:
            # For this person, determine when they will seek care (uniform distibition [0,91]days from now)
            date_seeking_care = self.sim.date + pd.DateOffset(days=int(self.module.rng.uniform(0,91)))

            # For this person, create the HSI Event for their presentation for care
            hsi_present_for_care = HSIoStartTreatmentStage2OesCancer(self.module, person_id)

            # Enter this event to the HealthSystem
            self.sim.modules['HealthSystem'].schedule_hsi_event(hsi_present_for_care,
                                                                priority=0,
                                                                topen=date_seeking_care,
                                                                tclose=None)

        # update ca_oesophagus_curative_treatment for diagnosed, untreated people with stage 3
        # this follows the same approach as above

        ca_oes_diag_stage3_not_treated_idx = df.index[
            df.is_alive & (df.ca_oesophagus == 'stage3') &
            (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (df.ca_oesophagus_curative_treatment == 'never')]
        eff_prob_treatment = pd.Series(m.r_curative_treatment_low_grade_dysp*m.rr_curative_treatment_stage3,
                                       index=df.index[df.is_alive & (df.ca_oesophagus == 'stage3') &
                                                      (df.age_years >= 20) & df.ca_oesophagus_diagnosed &
                                                      (df.ca_oesophagus_curative_treatment == 'never') ] )
        random_draw = pd.Series(rng.random_sample(size=len(ca_oes_diag_stage3_not_treated_idx)),
                                index=df.index[df.is_alive & (df.ca_oesophagus == 'stage3') &
                                               (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (
                                                       df.ca_oesophagus_curative_treatment == 'never')])
        dfx = pd.concat([eff_prob_treatment, random_draw], axis=1)
        dfx.columns = ['eff_prob_treatment', 'random_draw']
        requested_treatment_stage3_idx = dfx.index[dfx.eff_prob_treatment > dfx.random_draw]
        df.loc[requested_treatment_stage3_idx, 'ca_oesophagus_curative_treatment_requested'] = True

        # generate the HSI Events whereby persons present for care and get treatment for stage 2 oes cancer

        for person_id in requested_treatment_stage3_idx:
            # For this person, determine when they will seek care (uniform distibition [0,91]days from now)
            date_seeking_care = self.sim.date + pd.DateOffset(days=int(self.module.rng.uniform(0, 91)))

            # For this person, create the HSI Event for their presentation for care
            hsi_present_for_care = HSIoStartTreatmentStage3OesCancer(self.module, person_id)

            # Enter this event to the HealthSystem
            self.sim.modules['HealthSystem'].schedule_hsi_event(hsi_present_for_care,
                                                                priority=0,
                                                                topen=date_seeking_care,
                                                                tclose=None)

        # -------------------- DISABLITY -----------------------------------------------------------

        """
        547, Controlled phase of esophageal cancer, Generic uncomplicated disease: worry and daily
        medication, has a chronic disease that requires medication every day and causes some
        worry but minimal interference with daily activities., 0.049, 0.031, 0.072

        548, Terminal phase of esophageal cancer, "Terminal phase, with medication (for cancers, 
        end-stage kidney/liver disease)", "has lost a lot of weight and regularly uses strong 
        medication to avoid constant pain. The person has no appetite, feels nauseous, and needs 
        to spend most of the day in bed.", 0.54, 0.377, 0.687

        549, Metastatic phase of esophageal cancer, "Cancer, metastatic", "has severe pain, extreme 
        fatigue, weight loss and high anxiety.", 0.451, 0.307, 0.6
        
        550, Diagnosis and primary therapy phase of esophageal cancer, "Cancer, diagnosis and 
        primary therapy ", "has pain, nausea, fatigue, weight loss and high anxiety.", 0.288, 0.193, 
        0.399
        """

        # assume disability does not depend on whether diagnosed but may want to change in future
        ca_oes_low_grade_dysplasia_idx = df.index[df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia')]
        ca_oes_high_grade_dysplasia_idx = df.index[df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia')]
        ca_oes_stage1_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage1')]
        ca_oes_stage2_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage2')]
        ca_oes_stage3_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage3')]
        ca_oes_stage4_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage4')]

        # todo: note these disability weights don't map fully to cancer stages - may need to re-visit these
        # todo: choices below at some point

        # todo: I think the m.daly_wt_oes_cancer_primary_therapy etc being read in are not being
        # todo: recognised as REAL - need to covert to REAL so can take linear combinations of them below

        df.loc[ca_oes_low_grade_dysplasia_idx, 'ca_disability'] = 0.01
        df.loc[ca_oes_high_grade_dysplasia_idx, 'ca_disability'] = 0.01
        df.loc[ca_oes_stage1_idx, 'ca_disability'] = m.daly_wt_oes_cancer_controlled
        df.loc[ca_oes_stage2_idx, 'ca_disability'] = m.daly_wt_oes_cancer_primary_therapy
        df.loc[ca_oes_stage3_idx, 'ca_disability'] = m.daly_wt_oes_cancer_primary_therapy
        df.loc[ca_oes_stage4_idx, 'ca_disability'] = m.daly_wt_oes_cancer_metastatic

    # -------------------- DEATH FROM OESOPHAGEAL CANCER ---------------------------------------

        stage4_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage4')]
        random_draw = m.rng.random_sample(size=len(stage4_idx))
        df.loc[stage4_idx, 'ca_oesophageal_cancer_death'] = (random_draw < m.r_death_oesoph_cancer)

        death_this_period = df.index[df.ca_oesophageal_cancer_death]
        for individual_id in death_this_period:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id, 'Oesophageal_cancer'),
                                    self.sim.date)


class HSIoStartTreatmentLowGradeOesDysplasia(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    It is appointment at which someone with low grade oes dysplasia is treated.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        # todo this below will change to another type of appointment
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient appt

        # Get the consumables required
        the_cons_footprint = self.sim.modules['HealthSystem'].get_blank_cons_footprint()
        # TODO: Here adjust the cons footprint so that it includes oes cancer treatment

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'start_treatment_low_grade_oes_dysplasia'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = [0]  # Enforces that this apppointment must happen at level 0
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):

        df = self.sim.population.props

        df.ca_oesophagus_curative_treatment[person_id] = 'low_grade_dysplasia'
        df.ca_date_treatment_oesophageal_cancer[person_id] = self.sim.date


class HSIoStartTreatmentHighGradeOesDysplasia(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        # todo this below will change to another type of appointment
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient appt

        the_cons_footprint = self.sim.modules['HealthSystem'].get_blank_cons_footprint()
        # TODO: Here adjust the cons footprint so that it includes oes cancer treatment

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'start_treatment_high_grade_oes_dysplasia'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = [0]  # Enforces that this apppointment must happen at level 0
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):

        df = self.sim.population.props
        df.ca_oesophagus_curative_treatment[person_id] = 'high_grade_dysplasia'
        df.ca_date_treatment_oesophageal_cancer[person_id] = self.sim.date


class HSIoStartTreatmentStage1OesCancer(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        # todo this below will change to another type of appointment
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient appt

        the_cons_footprint = self.sim.modules['HealthSystem'].get_blank_cons_footprint()
        # TODO: Here adjust the cons footprint so that it includes oes cancer treatment

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'start_treatment_stage1_oes_cancer'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = [0]  # Enforces that this apppointment must happen at level 0
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):

        df = self.sim.population.props
        df.ca_oesophagus_curative_treatment[person_id] = 'stage1'
        df.ca_date_treatment_oesophageal_cancer[person_id] = self.sim.date


class HSIoStartTreatmentStage2OesCancer(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        # todo this below will change to another type of appointment
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient appt

        the_cons_footprint = self.sim.modules['HealthSystem'].get_blank_cons_footprint()
        # TODO: Here adjust the cons footprint so that it includes oes cancer treatment

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'start_treatment_stage2_oes_cancer'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = [0]  # Enforces that this apppointment must happen at level 0
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):

        df = self.sim.population.props
        df.ca_oesophagus_curative_treatment[person_id] = 'stage2'
        df.ca_date_treatment_oesophageal_cancer[person_id] = self.sim.date


class HSIoStartTreatmentStage3OesCancer(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        # todo this below will change to another type of appointment
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient appt

        the_cons_footprint = self.sim.modules['HealthSystem'].get_blank_cons_footprint()
        # TODO: Here adjust the cons footprint so that it includes oes cancer treatment

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'start_treatment_stage3_oes_cancer'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = [0]  # Enforces that this apppointment must happen at level 0
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):

        df = self.sim.population.props
        df.ca_oesophagus_curative_treatment[person_id] = 'stage3'
        df.ca_date_treatment_oesophageal_cancer[person_id] = self.sim.date

class OesCancerLoggingEvent(RegularEvent, PopulationScopeEventMixin):
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

        """

        n_alive = df.is_alive.sum()
        n_alive_ge20 = (df.is_alive & (df.age_years >= 20)).sum()

        n_incident_low_grade_dys_diag = (df.is_alive & df.ca_incident_oes_cancer_diagnosis_this_3_month_period
                                    & (df.ca_oesophagus == 'low_grade_dysplasia')).sum()
        n_incident_high_grade_dys_diag = (df.is_alive & df.ca_incident_oes_cancer_diagnosis_this_3_month_period
                                    & (df.ca_oesophagus == 'high_grade_dysplasia')).sum()
        n_incident_oc_stage1_diag = (df.is_alive & df.ca_incident_oes_cancer_diagnosis_this_3_month_period
                                    & (df.ca_oesophagus == 'stage1')).sum()
        n_incident_oc_stage2_diag = (df.is_alive & df.ca_incident_oes_cancer_diagnosis_this_3_month_period
                                    & (df.ca_oesophagus == 'stage2')).sum()
        n_incident_oc_stage3_diag = (df.is_alive & df.ca_incident_oes_cancer_diagnosis_this_3_month_period
                                    & (df.ca_oesophagus == 'stage3')).sum()
        n_incident_oc_stage4_diag = (df.is_alive & df.ca_incident_oes_cancer_diagnosis_this_3_month_period
                                    & (df.ca_oesophagus == 'stage4')).sum()

        n_incident_oes_cancer_diagnosis = n_incident_oc_stage1_diag \
                                          + n_incident_oc_stage2_diag + n_incident_oc_stage3_diag + \
                                          n_incident_oc_stage4_diag

        n_low_grade_dysplasia = (df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia')).sum()
        n_high_grade_dysplasia = (df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia')).sum()
        n_stage1_oc = (df.is_alive & (df.ca_oesophagus == 'stage1')).sum()
        n_stage2_oc = (df.is_alive & (df.ca_oesophagus == 'stage2')).sum()
        n_stage3_oc = (df.is_alive & (df.ca_oesophagus == 'stage3')).sum()
        n_stage4_oc = (df.is_alive & (df.ca_oesophagus == 'stage4')).sum()

        n_stage4_undiagnosed_oc = (df.is_alive & (df.ca_oesophagus == 'stage4') & ~df.ca_oesophagus_diagnosed).sum()

        n_low_grade_dysplasia_diag = (df.is_alive & df.ca_oesophagus_diagnosed & (df.ca_oesophagus == 'low_grade_dysplasia')).sum()
        n_high_grade_dysplasia_diag = (df.is_alive & df.ca_oesophagus_diagnosed & (df.ca_oesophagus == 'high_grade_dysplasia')).sum()
        n_stage1_oc_diag = (df.is_alive & df.ca_oesophagus_diagnosed & (df.ca_oesophagus == 'stage1')).sum()
        n_stage2_oc_diag = (df.is_alive & df.ca_oesophagus_diagnosed & (df.ca_oesophagus == 'stage2')).sum()
        n_stage3_oc_diag = (df.is_alive & df.ca_oesophagus_diagnosed & (df.ca_oesophagus == 'stage3')).sum()
        n_stage4_oc_diag = (df.is_alive & df.ca_oesophagus_diagnosed & (df.ca_oesophagus == 'stage4')).sum()

        n_received_trt_this_period_low_grade_dysplasia = (df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia')
                                                          & df.ca_date_treatment_oesophageal_cancer == self.sim.date).sum()
        n_received_trt_this_period_high_grade_dysplasia = (df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia')
                                                          & df.ca_date_treatment_oesophageal_cancer == self.sim.date).sum()
        n_received_trt_this_period_stage1 = (df.is_alive & (df.ca_oesophagus == 'stage1')
                                                          & df.ca_date_treatment_oesophageal_cancer == self.sim.date).sum()
        n_received_trt_this_period_stage2 = (df.is_alive & (df.ca_oesophagus == 'stage2')
                                                          & df.ca_date_treatment_oesophageal_cancer == self.sim.date).sum()
        n_received_trt_this_period_stage3 = (df.is_alive & (df.ca_oesophagus == 'stage3')
                                                          & df.ca_date_treatment_oesophageal_cancer == self.sim.date).sum()

        n_oc_death = df.ca_oesophageal_cancer_death.sum()

        cum_deaths = (~df.is_alive).sum()

        logger.info('%s| n_alive_ge20 |%s| n_incident_oes_cancer_diagnosis|%s| n_incident_low_grade_dys_diag|%s| '
                    'n_incident_high_grade_dys_diag|%s| n_incident_oc_stage1_diag|%s| '
                    'n_incident_oc_stage2_diag|%s| n_incident_oc_stage3_diag|%s|n_incident_oc_stage4_diag|%s| '
                    'n_low_grade_dysplasia|%s| n_high_grade_dysplasia|%s|  n_stage1_oc|%s| n_stage2_oc|%s| '
                    'n_stage3_oc|%s| n_stage4_oc|%s| n_low_grade_dysplasia_diag|%s| n_high_grade_dysplasia_diag'
                    '|%s| n_stage1_oc_diag|%s| n_stage2_oc_diag|%s| n_stage3_oc_diag|%s| n_stage4_oc_diag |%s|'
                    'n_received_trt_this_period_low_grade_dysplasia |%s| n_received_trt_this_period_high_grade_dysplasia'
                    '|%s| n_received_trt_this_period_stage1|%s| n_received_trt_this_period_stage2|%s|'
                    'n_received_trt_this_period_stage3|%s|n_stage4_undiagnosed_oc|%s| cum_deaths |%s |n_alive|%s|'
                    'n_oc_death|%s',
                    self.sim.date, n_alive_ge20, n_incident_oes_cancer_diagnosis, n_incident_low_grade_dys_diag,
                    n_incident_high_grade_dys_diag, n_incident_oc_stage1_diag,
                    n_incident_oc_stage2_diag, n_incident_oc_stage3_diag,n_incident_oc_stage4_diag,
                    n_low_grade_dysplasia, n_high_grade_dysplasia,  n_stage1_oc, n_stage2_oc,
                    n_stage3_oc, n_stage4_oc, n_low_grade_dysplasia_diag, n_high_grade_dysplasia_diag,
                    n_stage1_oc_diag, n_stage2_oc_diag, n_stage3_oc_diag, n_stage4_oc_diag,
                    n_received_trt_this_period_low_grade_dysplasia, n_received_trt_this_period_high_grade_dysplasia,
                    n_received_trt_this_period_stage1, n_received_trt_this_period_stage2,
                    n_received_trt_this_period_stage3, n_stage4_undiagnosed_oc, cum_deaths, n_alive, n_oc_death
                    )

        """
        logger.info('%s|person_one|%s',
                     self.sim.date,
                     df.loc[0].to_dict())
         


        


