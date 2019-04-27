"""
Oesophageal Cancer - module

Documentation: 04 - Methods Repository/Method_Oesophageal_Cancer.xlsx
"""
import logging
from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import demography
import numpy as np
import pandas as pd
import random


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Oesophageal_Cancer(Module):

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath


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
    }

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
        'ca_disability': Property(Types.REAL, 'disability weight this three month period')
    }

    TREATMENT_ID = 'attempted curative treatment for oesophageal cancer'

    def read_parameters(self, data_folder):
        """Setup parameters used by the module
        """
        p = self.parameters

        p['r_low_grade_dysplasia_none'] = 0.00001
        p['rr_low_grade_dysplasia_none_female'] = 1.3
        p['rr_low_grade_dysplasia_none_per_year_older'] = 1.1
        p['rr_low_grade_dysplasia_none_tobacco'] = 2.0
        p['rr_low_grade_dysplasia_none_ex_alc'] = 1.0
        p['r_high_grade_dysplasia_low_grade_dysp'] = 0.03
        p['rr_high_grade_dysp_undergone_curative_treatment'] = 0.1
        p['r_stage1_high_grade_dysp'] = 0.01
        p['rr_stage1_undergone_curative_treatment'] = 0.1
        p['r_stage2_stage1'] = 0.05
        p['rr_stage2_undergone_curative_treatment'] = 0.1
        p['r_stage3_stage2'] = 0.05
        p['rr_stage3_undergone_curative_treatment'] = 0.1
        p['r_stage4_stage3'] = 0.05
        p['rr_stage4_undergone_curative_treatment'] = 0.3
        p['r_death_oesoph_cancer'] = 0.4
        p['r_curative_treatment_low_grade_dysp'] = 0.01
        p['rr_curative_treatment_high_grade_dysp'] = 1.0
        p['rr_curative_treatment_stage1'] = 1.0
        p['rr_curative_treatment_stage2'] = 1.0
        p['rr_curative_treatment_stage3'] = 1.0
        p['r_diagnosis_stage1'] = 0.1
        p['rr_diagnosis_low_grade_dysp'] = 0.1
        p['rr_diagnosis_high_grade_dysp'] = 0.1
        p['rr_diagnosis_stage2'] = 3
        p['rr_diagnosis_stage3'] = 4
        p['rr_diagnosis_stage4'] = 5
        p['init_prop_oes_cancer_stage'] = [0.0003, 0.0001, 0.00005, 0.00003, 0.000005, 0.000001]
        p['rp_oes_cancer_female'] = 1.3
        p['rp_oes_cancer_per_year_older'] = 1.1
        p['rp_oes_cancer_tobacco'] = 2.0
        p['rp_oes_cancer_ex_alc'] = 1.0
        p['init_prop_diagnosed_oes_cancer_by_stage'] = [0.01, 0.03, 0.10, 0.20, 0.30, 0.8]
        p['init_prop_treatment_status_oes_cancer'] = [0.01, 0.01, 0.05, 0.05, 0.05, 0.05]

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

        random_draw = pd.Series(rng.random_sample(size=len(low_grade_dys_idx)),
                                index=df.index[df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia')])
        df.loc[low_grade_dys_idx, 'ca_oesophagus_diagnosed'] = \
            random_draw < m.init_prop_diagnosed_oes_cancer_by_stage[0]
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

        random_draw = pd.Series(rng.random_sample(size=len(low_grade_dys_diagnosed_idx)),
                                index=df.index[df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia')
                                               & df.ca_oesophagus_diagnosed])
        p_treatment = pd.Series(m.init_prop_treatment_status_oes_cancer[0], index=df.index[df.is_alive &
                                            (df.ca_oesophagus == 'low_grade_dysplasia')
                                            & df.ca_oesophagus_diagnosed])
        dfx = pd.concat([p_treatment, random_draw], axis=1)
        dfx.columns = ['p_treatment', 'random_draw']
        idx_low_grade_dysplasia_treatment = dfx.index[dfx.p_treatment > dfx.random_draw]
        df.loc[idx_low_grade_dysplasia_treatment, 'ca_oesophagus_curative_treatment'] = 'low_grade_dysplasia'

        random_draw = pd.Series(rng.random_sample(size=len(high_grade_dys_diagnosed_idx)),
                                index=df.index[df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia')
                                               & df.ca_oesophagus_diagnosed])
        p_treatment = pd.Series(m.init_prop_treatment_status_oes_cancer[1], index=df.index[df.is_alive &
                                            (df.ca_oesophagus == 'high_grade_dysplasia')
                                            & df.ca_oesophagus_diagnosed])
        dfx = pd.concat([p_treatment, random_draw], axis=1)
        dfx.columns = ['p_treatment', 'random_draw']
        idx_high_grade_dysplasia_treatment = dfx.index[dfx.p_treatment > dfx.random_draw]
        df.loc[idx_high_grade_dysplasia_treatment, 'ca_oesophagus_curative_treatment'] = 'low_grade_dysplasia'

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

        random_draw = pd.Series(rng.random_sample(size=len(stage4_oes_can_diagnosed_idx)),
                                index=df.index[df.is_alive & (df.ca_oesophagus == 'stage4')
                                               & df.ca_oesophagus_diagnosed])
        p_treatment = pd.Series(m.init_prop_treatment_status_oes_cancer[5], index=df.index[df.is_alive &
                                            (df.ca_oesophagus == 'stage4')
                                            & df.ca_oesophagus_diagnosed])
        dfx = pd.concat([p_treatment, random_draw], axis=1)
        dfx.columns = ['p_treatment', 'random_draw']
        idx_stage4_oes_can_treatment = dfx.index[dfx.p_treatment > dfx.random_draw]
        df.loc[idx_stage4_oes_can_treatment, 'ca_oesophagus_curative_treatment'] = 'stage4'

    def initialise_simulation(self, sim):
        """Add lifestyle events to the simulation
        """
        event = OesCancerEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=3))

        event = OesCancerLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=0))

        # todo: amend this below when identifid data
        # Define the footprint for the intervention on the common resources
        footprint_for_treatment = pd.DataFrame(index=np.arange(1), data={
            'Name': Oesophageal_Cancer.TREATMENT_ID,
            'Nurse_Time': 15,
            'Doctor_Time': 15,
            'Electricity': False,
            'Water': False})

        self.sim.modules['HealthSystem'].register_interventions(footprint_for_treatment)

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

#       logger.debug("This is Epilepsy being asked to report unified symptomology")

        # Map the specific symptoms for this disease onto the unified coding scheme
        df = self.sim.population.props  # shortcut to population properties dataframe

        return pd.Series('1', index=df.index[df.is_alive])

    def on_healthsystem_interaction(self, person_id, cue_type=None, disease_specific=None):

        #       logger.debug('This is epilepsy, being alerted about a health system interaction '
        #                    'person %d triggered by %s : %s', person_id, cue_type, disease_specific)

        pass

    def report_qaly_values(self):
        # This must send back a dataframe that reports on the HealthStates for all individuals over
        # the past year

        #       logger.debug('This is oesophageal cancer reporting my health values')

        df = self.sim.population.props  # shortcut to population properties dataframe

        disability_series = df.ca_disability

        return disability_series


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
        df.loc[~df.is_alive, 'ca_oesophageal_cancer_death'] = False
        df['ca_disability'] = 0
        df['ca_oesophagus_curative_treatment_requested'] = False
        df['ca_incident_oes_cancer_diagnosis_this_3_month_period'] = False

        # -------------------- UPDATING of CA-OESOPHAGUS OVER TIME -----------------------------------

        # updating for peopl aged over 20 with current status 'none'

        ca_oes_current_none_idx = df.index[df.is_alive & (df.ca_oesophagus == 'none') & (df.age_years >= 20)]
        ca_oes_current_none_f_idx = df.index[df.is_alive & (df.ca_oesophagus == 'none') & (df.age_years >= 20) &
                                           (df.sex == 'F')]
        ca_oes_current_none_tob_idx = df.index[df.is_alive & (df.ca_oesophagus == 'none') & (df.age_years >= 20) &
                                           df.li_tob]
        ca_oes_current_none_ex_alc_idx = df.index[df.is_alive & (df.ca_oesophagus == 'none') & (df.age_years >= 20) &
                                           df.li_ex_alc]

        eff_prob_low_grade_dysp = pd.Series(m.r_low_grade_dysplasia_none,
                                    index=df.index[df.is_alive & (df.ca_oesophagus == 'none') & (df.age_years >= 20)])

        eff_prob_low_grade_dysp.loc[ca_oes_current_none_f_idx] *= m.rr_low_grade_dysplasia_none_female
        eff_prob_low_grade_dysp.loc[ca_oes_current_none_tob_idx] *= m.rr_low_grade_dysplasia_none_tobacco
        eff_prob_low_grade_dysp.loc[ca_oes_current_none_ex_alc_idx] *= m.rr_low_grade_dysplasia_none_ex_alc

        p_oes_dys_can_age_muliplier = pd.Series(m.rr_low_grade_dysplasia_none_per_year_older ** (df.age_years - 20),
                                                index=ca_oes_current_none_idx)

        random_draw = pd.Series(rng.random_sample(size=len(ca_oes_current_none_idx)),
                                   index=df.index[(df.age_years >= 20) & df.is_alive & (df.ca_oesophagus == 'none')])

        dfx = pd.concat([eff_prob_low_grade_dysp, p_oes_dys_can_age_muliplier, random_draw], axis=1)
        dfx.columns = ['eff_prob_low_grade_dysp', 'p_oes_dys_can_age_muliplier', 'random_draw']
        dfx.eff_prob_low_grade_dysp *= p_oes_dys_can_age_muliplier
        idx_incident_low_grade_dysp = dfx.index[dfx.eff_prob_low_grade_dysp > dfx.random_draw]
        df.loc[idx_incident_low_grade_dysp, 'ca_oesophagus'] = 'low_grade_dysplasia'

        # updating for people aged over 20 with current status 'low grade dysplasia'

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

        df['ca_incident_oes_cancer_diagnosis_this_3_month_period'] = False

        # update diagnosis status for undiagnosed people with low grade dysplasia

        ca_oes_current_low_grade_dysp_not_diag_idx = df.index[
            df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia') &
            (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed]
        eff_prob_diag = pd.Series(m.r_diagnosis_stage1 * m.rr_diagnosis_low_grade_dysp,
                                  index=df.index[df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia') &
                                                 (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed])
        random_draw = rng.random_sample(size=len(ca_oes_current_low_grade_dysp_not_diag_idx))
        df.loc[ca_oes_current_low_grade_dysp_not_diag_idx, 'ca_oesophagus_diagnosed'] = (random_draw < eff_prob_diag)
        df.loc[ca_oes_current_low_grade_dysp_not_diag_idx, 'ca_incident_oes_cancer_diagnosis_this_3_month_period'] \
            = (random_draw < eff_prob_diag)

        # update diagnosis status for undiagnosed people with high grade dysplasia

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

        # update ca_oesophagus_curative_treatment for diagnosed, untreated people with high grade dysplasia w

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

        # update ca_oesophagus_curative_treatment for diagnosed, untreated people with stage 1

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

        # update ca_oesophagus_curative_treatment for diagnosed, untreated people with stage 2

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

        # update ca_oesophagus_curative_treatment for diagnosed, untreated people with stage 3

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

        # receive an attempt at curative treatment if health system has capacity

        requested_treatment_idx = df.index[df.is_alive & df.ca_oesophagus_curative_treatment_requested]

        requested_treatment = pd.Series(True,
                                        index=df.index[df.is_alive & df.ca_oesophagus_curative_treatment_requested])
        stage_at_trt_request = pd.Series(df.ca_oesophagus,
                                        index=df.index[df.is_alive & df.ca_oesophagus_curative_treatment_requested])

        dfxx = pd.concat([requested_treatment, stage_at_trt_request], axis=1)
        dfxx.columns = ['requested_treatment', 'stage_at_trt_request']

        # note that in future may need to distinguish treatment by stage
        # note that this line seems to apply to all in dfxx so had to restrict it to those needing to be treated
        for index in dfxx:
            dfxx['gets_trt'] = self.sim.modules['HealthSystem'].query_access_to_service(index, TREATMENT_ID)

        df.loc[requested_treatment_idx, 'ca_oesophagus_curative_treatment'] = dfxx['stage_at_trt_request']

        # -------------------- DISABLITY -----------------------------------------------------------

        ca_oes_low_grade_dysplasia_idx = df.index[df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasa')]
        ca_oes_high_grade_dysplasia_idx = df.index[df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasa')]
        ca_oes_stage1_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage1')]
        ca_oes_stage2_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage2')]
        ca_oes_stage3_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage3')]
        ca_oes_stage4_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage4')]

        # todo
        df.loc[ca_oes_low_grade_dysplasia_idx, 'ca_disability'] = 0.xx
        df.loc[ca_oes_high_grade_dysplasia_idx, 'ca_disability'] = 0.xx
        df.loc[ca_oes_stage1_idx, 'ca_disability'] = 0.xx
        df.loc[ca_oes_stage2_idx, 'ca_disability'] = 0.xx
        df.loc[ca_oes_stage3_idx, 'ca_disability'] = 0.xx
        df.loc[ca_oes_stage4_idx, 'ca_disability'] = 0.xx


    # -------------------- DEATH FROM OESOPHAGEAL CANCER ---------------------------------------

        stage4_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage4')]
        random_draw = m.rng.random_sample(size=len(stage4_idx))
        df.loc[stage4_idx, 'ca_oesophageal_cancer_death'] = (random_draw < m.r_death_oesoph_cancer)

        death_this_period = df.index[df.ca_oesophageal_cancer_death]
        for individual_id in death_this_period:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id, 'Oesophageal_cancer'),
                                    self.sim.date)

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

        n_alive = df.is_alive.sum()
        n_alive_ge20 = (df.is_alive & (df.age_years >= 20)).sum()

        n_incident_oes_cancer_diagnosis = (df.is_alive & df.ca_incident_oes_cancer_diagnosis_this_3_month_period).sum()
        n_incident_low_grade_dys_diag = (df.is_alive & df.ca_incident_oes_cancer_diagnosis_this_3_month_period
                                    & (df.ca_oesophasus == 'low_grade_dysplasia')).sum()
        n_incident_high_grade_dys_diag = (df.is_alive & df.ca_incident_oes_cancer_diagnosis_this_3_month_period
                                    & (df.ca_oesophasus == 'high_grade_dysplasia')).sum()
        n_incident_oc_stage1_diag = (df.is_alive & df.ca_incident_oes_cancer_diagnosis_this_3_month_period
                                    & (df.ca_oesophasus == 'stage1')).sum()
        n_incident_oc_stage2_diag = (df.is_alive & df.ca_incident_oes_cancer_diagnosis_this_3_month_period
                                    & (df.ca_oesophasus == 'stage2')).sum()
        n_incident_oc_stage3_diag = (df.is_alive & df.ca_incident_oes_cancer_diagnosis_this_3_month_period
                                    & (df.ca_oesophasus == 'stage3')).sum()
        n_incident_oc_stage4_diag = (df.is_alive & df.ca_incident_oes_cancer_diagnosis_this_3_month_period
                                    & (df.ca_oesophasus == 'stage4')).sum()




        cum_deaths = (~df.is_alive).sum()

        #       logger.info('%s,%s,', self.sim.date, n_epi_death)

        logger.info('%s|prop_seiz_stat_0|%s|prop_seiz_stat_1|%s|prop_seiz_stat_2|%s|'
                    'prop_seiz_stat_3|%s|prop_antiepilep_seiz_stat_0|%s|prop_antiepilep_seiz_stat_1|%s|'
                    'prop_antiepilep_seiz_stat_2|%s|prop_antiepilep_seiz_stat_3|%s|n_epi_death|%s|'
                    'cum_deaths|%s|epi_death_rate |%s',
                    self.sim.date, prop_seiz_stat_0, prop_seiz_stat_1, prop_seiz_stat_2, prop_seiz_stat_3,
                    prop_antiepilep_seiz_stat_0, prop_antiepilep_seiz_stat_1, prop_antiepilep_seiz_stat_2,
                    prop_antiepilep_seiz_stat_3, n_epi_death, cum_deaths, epi_death_rate
                    )
