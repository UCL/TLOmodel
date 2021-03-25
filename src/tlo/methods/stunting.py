"""
Childhood stunting module
Documentation: '04 - Methods Repository/Undernutrition module - Description.docx'

Overview
=======
This module applies the prevalence of stunting at the population-level, and schedules new incidences of stunting

"""
import copy
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata, demography
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.symptommanager import Symptom

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------


class Stunting(Module):
    """
    This module applies the prevalence of stunting at the population-level,
    based on the Malawi DHS Survey 2015-2016.
    The definitions:
    - moderate stunting: height-for-age Z-score (HAZ) <-2 SD from the reference mean
    - severe stunting: height-for-age Z-score (HAZ) <-3 SD from the reference mean

    """

    # Declare Metadata
    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
    }

    PARAMETERS = {
        # prevalence of stunting by age group
        'prev_HAZ_distribution_age_0_5mo': Parameter(
            Types.LIST, 'distribution of HAZ among less than 6 months of age in 2015'),
        'prev_HAZ_distribution_age_6_11mo': Parameter(
            Types.LIST, 'distribution of HAZ among 6 months and 1 year of age in 2015'),
        'prev_HAZ_distribution_age_12_23mo': Parameter(
            Types.LIST, 'distribution of HAZ among 1 year olds in 2015'),
        'prev_HAZ_distribution_age_24_35mo': Parameter(
            Types.LIST, 'distribution of HAZ among 2 year olds in 2015'),
        'prev_HAZ_distribution_age_36_47mo': Parameter(
            Types.LIST, 'distribution of HAZ among 3 year olds in 2015'),
        'prev_HAZ_distribution_age_48_59mo': Parameter(
            Types.LIST, 'distribution of HAZ among 4 year olds  in 2015'),
        # effect of risk factors on stunting prevalence
        'or_stunting_male': Parameter(
            Types.REAL, 'odds ratio of stunting if male gender'),
        'or_stunting_no_recent_diarrhoea': Parameter(
            Types.REAL, 'odds ratio of stunting if no recent diarrhoea in past 2 weeks, compared to recent episode'),
        'or_stunting_single_birth': Parameter(
            Types.REAL, 'odds ratio of stunting if single birth, ref group multiple birth (twins)'),
        'or_stunting_mother_no_education': Parameter(
            Types.REAL, 'odds ratio of stunting if mother has no formal education, ref group secondary education'),
        'or_stunting_mother_primary_education': Parameter(
            Types.REAL, 'odds ratio of stunting if mother has primary education, ref group secondary education'),
        'or_stunting_motherBMI_underweight': Parameter(
            Types.REAL, 'odds ratio of stunting if mother has low BMI, ref group high BMI (overweight)'),
        'or_stunting_motherBMI_normal': Parameter(
            Types.REAL, 'odds ratio of stunting if mother has normal BMI, ref group high BMI (overweight)'),
        'or_stunting_hhwealth_Q1': Parameter(
            Types.REAL, 'odds ratio of stunting if household wealth is poorest Q1, ref group Q5'),
        'or_stunting_hhwealth_Q2': Parameter(
            Types.REAL, 'odds ratio of stunting if household wealth is poorer Q2, ref group Q5'),
        'or_stunting_hhwealth_Q3': Parameter(
            Types.REAL, 'odds ratio of stunting if household wealth is middle Q3, ref group Q5'),
        'or_stunting_hhwealth_Q4': Parameter(
            Types.REAL, 'odds ratio of stunting if household wealth is richer Q4, ref group Q5'),
        'base_inc_rate_stunting': Parameter(
            Types.REAL, 'baseline incidence of stunting'),
        'rr_stunting_preterm_and_AGA': Parameter(
            Types.REAL, 'relative risk of stunting if born preterm and adequate for gestational age'),
        'rr_stunting_SGA_and_term': Parameter(
            Types.REAL, 'relative risk of stunting if born term and small for geatational age'),
        'rr_stunting_SGA_and_preterm': Parameter(
            Types.REAL, 'relative risk of stunting if born preterm and small for gestational age'),

    }

    PROPERTIES = {
        'un_HAZ_score': Property(Types.REAL, 'height-for-age z-score')

    }

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    def read_parameters(self, data_folder):
        """
        :param data_folder: path of a folder supplied to the Simulation containing data files.
              Typically modules would read a particular file within here.
        :return:
        """
        # Update parameters from the resource dataframe
        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_Undernutrition.xlsx', sheet_name='Parameter_values_CM')
        self.load_parameters_from_dataframe(dfd)

        p = self.parameters

        # if 'HealthBurden' in self.sim.modules.keys():
        #     #get the DALY weight - 860-862 are the sequale codes for epilepsy
            # p['daly_wt_epilepsy_severe'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=860)
            # p['daly_wt_epilepsy_less_severe'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=861)
            # p['daly_wt_epilepsy_seizure_free'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=862)

    def initialise_population(self, population):
        """
        Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population:
        :return:
        """
        df = population.props
        p = self.parameters
        rng = self.rng

        df.loc[df.is_alive, 'ep_seiz_stat'] = '0'
        df.loc[df.is_alive, 'ep_antiep'] = False
        df.loc[df.is_alive, 'ep_epi_death'] = False
        df.loc[df.is_alive, 'ep_disability'] = 0

        mean_HAZ_age_0_5mo = p['prev_HAZ_distribution_age_0_5mo'][0]
        mean_HAZ_age_6_11mo = p['prev_HAZ_distribution_age_6_11mo'][0]
        mean_HAZ_age_12_23mo = p['prev_HAZ_distribution_age_12_23mo'][0]
        mean_HAZ_age_24_35mo = p['prev_HAZ_distribution_age_24_35mo'][0]
        mean_HAZ_age_36_47mo = p['prev_HAZ_distribution_age_36_47mo'][0]
        mean_HAZ_age_48_59mo = p['prev_HAZ_distribution_age_48_59mo'][0]

        sd_HAZ_age_0_5mo = p['prev_HAZ_distribution_age_0_5mo'][1]
        sd_HAZ_age_6_11mo = p['prev_HAZ_distribution_age_6_11mo'][1]
        sd_HAZ_age_12_23mo = p['prev_HAZ_distribution_age_12_23mo'][1]
        sd_HAZ_age_24_35mo = p['prev_HAZ_distribution_age_24_35mo'][1]
        sd_HAZ_age_36_47mo = p['prev_HAZ_distribution_age_36_47mo'][1]
        sd_HAZ_age_48_59mo = p['prev_HAZ_distribution_age_48_59mo'][1]

        # allocate initial prevalence of stunting at the start of the simulation
        # apply by age group
        index_children_aged_0_5mo = df.index[df.is_alive & df.age_exact_years < 0.5]
        index_children_aged_6_11mo = df.index[df.is_alive & ((df.age_exact_years >= 0.5) & (df.age_exact_years < 1))]
        index_children_aged_12_23mo = df.index[df.is_alive & ((df.age_exact_years >= 1) & (df.age_exact_years < 2))]
        index_children_aged_24_35mo = df.index[df.is_alive & ((df.age_exact_years >= 2) & (df.age_exact_years < 3))]
        index_children_aged_36_47mo = df.index[df.is_alive & ((df.age_exact_years >= 3) & (df.age_exact_years < 4))]
        index_children_aged_48_59mo = df.index[df.is_alive & ((df.age_exact_years >= 4) & (df.age_exact_years < 5))]

        ### Random draw of HAZ scores from a normal distribution ###
        # random draw of HAZ scores from a normal distribution for under 6 months old
        HAZ_distribution_under_6mo = np.random.normal(loc=p['prev_HAZ_distribution_age_0_5mo'][0],
                                                      scale=p['prev_HAZ_distribution_age_0_5mo'][1])

        HAZ_score_under6mo = pd.Series(HAZ_distribution_under_6mo, index=index_children_aged_0_5mo)
        # update df
        df.loc['un_HAZ_score'] = HAZ_score_under6mo

        # for 6 to 11 months
        for i in index_children_aged_6_11mo:
            HAZ_distribution_among_6_11mo = np.random.normal(loc=p['prev_HAZ_distribution_age_6_11mo'][0],
                                                             scale=p['prev_HAZ_distribution_age_6_11mo'][1])
            df.at[i, 'un_HAZ_score'] = HAZ_distribution_among_6_11mo
        # HAZ_score_among_6_11mo = pd.Series(HAZ_distribution_among_6_11mo, index=index_children_aged_6_11mo)


        HAZ_distribution_among_1yo = np.random.normal(loc=p['prev_HAZ_distribution_age_12_23mo'][0],
                                                      scale=p['prev_HAZ_distribution_age_12_23mo'][1])
        df.loc[df.is_alive, 'un_HAZ_score'] = HAZ_distribution_among_1yo
        if index_children_aged_24_35mo:
            HAZ_distribution_among_2yo = np.random.normal(loc=p['prev_HAZ_distribution_age_24_35mo'][0],
                                                          scale=p['prev_HAZ_distribution_age_24_35mo'][1])
            df.loc[df.is_alive, 'un_HAZ_score'] = HAZ_distribution_among_2yo
        if index_children_aged_36_47mo:
            HAZ_distribution_among_3yo = np.random.normal(loc=p['prev_HAZ_distribution_age_36_47mo'][0],
                                                          scale=p['prev_HAZ_distribution_age_36_47mo'][1])
            df.loc[df.is_alive, 'un_HAZ_score'] = HAZ_distribution_among_3yo
        if index_children_aged_48_59mo:
            HAZ_distribution_among_4yo = np.random.normal(loc=p['prev_HAZ_distribution_age_48_59mo'][0],
                                                          scale=p['prev_HAZ_distribution_age_48_59mo'][1])
            df.loc[df.is_alive, 'un_HAZ_score'] = HAZ_distribution_among_4yo


    def initialise_simulation(self, sim):
        pass

    def on_birth(self, mother_id, child_id):
        pass
