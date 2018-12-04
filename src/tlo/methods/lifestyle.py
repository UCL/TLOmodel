"""
A skeleton template for disease methods.
"""

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent
import numpy as np
import pandas as pd


class Lifestyle(Module):
    """
    One line summary goes here...
    All disease modules need to be implemented as a class inheriting from Module.
    They need to provide several methods which will be called by the simulation
    framework:
    * `read_parameters(data_folder)`
    * `initialise_population(population)`
    * `initialise_simulation(sim)`
    * `on_birth(mother, child)`
    """

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    PARAMETERS = {
        'r_urban': Parameter(Types.REAL, 'probability per 3 mths of change from rural to urban'),
        'r_rural': Parameter(Types.REAL, 'probability per 3 mths of change from urban to rural'),
        'r_overwt': Parameter(Types.REAL, 'probability per 3 mths of change from not_overwt to overwt if male'),
        'r_not_overwt': Parameter(Types.REAL, 'probability per 3 mths of change from overwt to not overwt'),
        'rr_overwt_f': Parameter(Types.REAL, 'risk ratio for becoming overwt if female rather than male'),
        'rr_overwt_urban': Parameter(Types.REAL, 'risk ratio for becoming overwt if urban rather than rural'),
        'r_low_ex': Parameter(Types.REAL, 'probability per 3 mths of change from not low ex to low ex'),
        'r_not_low_ex': Parameter(Types.REAL, 'probability per 3 mths of change from low ex to not low ex'),
        'rr_low_ex_f': Parameter(Types.REAL, 'risk ratio for becoming low ex if female rather than male'),
        'rr_low_ex_urban': Parameter(Types.REAL, 'risk ratio for becoming low ex if urban rather than rural'),
        'r_tob': Parameter(Types.REAL, 'probability per 3 mths of change from not tob to tob'),
        'r_not_tob': Parameter(Types.REAL, 'probability per 3 mths of change from tob to not tob'),
        'rr_tob_age2039': Parameter(Types.REAL, 'risk ratio for tob if age 20-39 compared with 15-19'),
        'rr_tob_agege40': Parameter(Types.REAL, 'risk ratio for tob if age ge 40 compared with 15-19'),
        'rr_tob_f': Parameter(Types.REAL, 'risk ratio for tob if female'),
        'rr_tob_wealth': Parameter(Types.REAL, 'risk ratio for tob per 1 higher wealth level (higher wealth level = lower wealth)'),
        'r_ex_alc': Parameter(Types.REAL, 'probability per 3 mths of change from not ex alc to ex alc'),
        'r_not_ex_alc': Parameter(Types.REAL, 'probability per 3 mths of change from excess alc to not excess alc'),
        'rr_ex_alc_f': Parameter(Types.REAL, 'risk ratio for becoming ex alc if female rather than male'),
        'init_p_urban': Parameter(Types.REAL, 'proportion urban at baseline'),
        'init_p_wealth_urban': Parameter(Types.LIST, 'List of probabilities of category given urban'),
        'init_p_wealth_rural': Parameter(Types.LIST, 'List of probabilities of category given rural'),
        'init_dist_mar_stat_age1520': Parameter(Types.LIST, 'proportions never, current, div_wid age 15-20 baseline'),
        'init_dist_mar_stat_age2030': Parameter(Types.LIST, 'proportions never, current, div_wid age 20-30 baseline'),
        'init_dist_mar_stat_age3040': Parameter(Types.LIST, 'proportions never, current, div_wid age 30-40 baseline'),
        'init_dist_mar_stat_age4050': Parameter(Types.LIST, 'proportions never, current, div_wid age 40-50 baseline'),
        'init_dist_mar_stat_age5060': Parameter(Types.LIST, 'proportions never, current, div_wid age 50-60 baseline'),
        'init_dist_mar_stat_agege60': Parameter(Types.LIST, 'proportions never, current, div_wid age 60+ baseline'),
        'r_mar': Parameter(Types.REAL, 'prob per 3 months of marriage when age 15-30'),
        'r_div_wid': Parameter(Types.REAL, 'prob per 3 months of becoming divorced or widowed, amongst those married'),
        'init_p_on_contrac': Parameter(Types.REAL, 'initial proportion of women age 15-49 on contraceptives'),
        'init_dist_con_t': Parameter(Types.LIST, 'initial proportions on different contraceptive types'),
        'r_contrac': Parameter(Types.REAL, 'prob per 3 months of starting contraceptive if age 15-50'),
        'r_contrac_int': Parameter(Types.REAL, 'prob per 3 months of interrupting or stopping contraception (note current method of contrac is a different propeerty'),
        'r_con_from_1': Parameter(Types.LIST, 'probs per 3 months of moving from contraception method 1'),
        'r_con_from_2': Parameter(Types.LIST, 'probs per 3 months of moving from contraception method 2'),
        'r_con_from_3': Parameter(Types.LIST, 'probs per 3 months of moving from contraception method 3'),
        'r_con_from_4': Parameter(Types.LIST, 'probs per 3 months of moving from contraception method 4'),
        'r_con_from_5': Parameter(Types.LIST, 'probs per 3 months of moving from contraception method 5'),
        'r_con_from_6': Parameter(Types.LIST, 'probs per 3 months of moving from contraception method 6'),
        'r_stop_ed': Parameter(Types.REAL, 'prob per 3 months of stopping education if male'),
        'rr_stop_ed_lower_wealth': Parameter(Types.REAL, 'relative rate of stopping education per 1 lower wealth quintile'),
        'p_ed_primary': Parameter(Types.REAL, 'probability at age 5 that start primary education if male'),
        'rp_ed_primary_higher_wealth': Parameter(Types.REAL, 'relative probability of starting school per 1 higher wealth level' ),
        'p_ed_secondar': Parameter(Types.REAL, 'probability at age 11 that start secondary education at 11 if male and in primary education'),
        'rp_ed_secondary_higher_wealth': Parameter(Types.REAL, 'relative probability of starting secondary school per 1 higher wealth level'),
        'init_age2030_w5_some_ed': Parameter(Types.REAL, 'proportions of low wealth 20-30 year olds with some education at baseline'),
        'init_rp_some_ed_age1520': Parameter(Types.REAL, 'relative prevalence of some education at baseline if age 1520'),
        'init_rp_some_ed_age2030': Parameter(Types.REAL, 'relative prevalence of some education at baseline if age 2030'),
        'init_rp_some_ed_age3040': Parameter(Types.REAL, 'relative prevalence of some education at baseline if age 3040'),
        'init_rp_some_ed_age4050': Parameter(Types.REAL, 'relative prevalence of some education at baseline if age 4050'),
        'init_rp_some_ed_age5060': Parameter(Types.REAL, 'relative prevalence of some education at baseline if age 5060'),
        'init_rp_some_ed_per_higher_wealth': Parameter(Types.REAL, 'relative prevalence of some education at baseline per higher wealth level'),
        'init_prop_age2030_w5_some_ed_sec': Parameter(Types.REAL, 'proportion of low wealth aged 20-30 with some education who have secondary education at baseline'),
        'init_rp_some_ed_sec_age1520': Parameter(Types.REAL, 'relative prevalence of sec_ed for age 15-20'),
        'init_rp_some_ed_sec_age3040': Parameter(Types.REAL, 'relative prevalence of sec_ed for age 30-40'),
        'init_rp_some_ed_sec_age4050': Parameter(Types.REAL, 'relative prevalence of sec_ed for age 40-50'),
        'init_rp_some_ed_sec_age5060': Parameter(Types.REAL, 'relative prevalence of sec_ed for age 50-60'),
        'init_rp_some_ed_sec_agege60': Parameter(Types.REAL, 'relative prevalence of sec_ed for age 60+'),
        'init_rp_some_ed_sec_per_higher_wealth': Parameter(Types.REAL, 'relative prevalence of sec_ed per higher wealth level'),
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'li_urban': Property(Types.BOOL, 'Currently urban'),
        'li_date_trans_to_urban': Property(Types.DATE, 'date of transition to urban'),
        'li_wealth': Property(Types.CATEGORICAL, 'wealth level', categories=[1, 2, 3, 4, 5]),
        'li_overwt': Property(Types.BOOL, 'currently overweight'),
        'li_low_ex': Property(Types.BOOL, 'currently low ex'),
        'li_tob': Property(Types.BOOL, 'current using tobacco'),
        'li_ex_alc': Property(Types.BOOL, 'current excess alcohol'),
        'li_mar_stat': Property(Types.CATEGORICAL, 'marital status', categories=[1, 2, 3]),
        'li_on_con': Property(Types.BOOL, 'on contraceptive'),
        'li_con_t': Property(Types.CATEGORICAL, 'contraceptive type', categories=[1, 2, 3, 4, 5, 6]),
        'li_in_ed': Property(Types.BOOL, 'currently in education'),
        'li_ed_lev': Property(Types.CATEGORICAL, 'education level achieved as of now', categories=[1, 2, 3]),
    }

    def __init__(self):
        super().__init__()
        self.store = {'alive': []}
        self.o_prop_urban = {'prop_urban': []}
        self.o_prop_m_urban_overwt = {'prop_m_urban_overwt': []}
        self.o_prop_f_urban_overwt = {'prop_f_urban_overwt': []}
        self.o_prop_m_rural_overwt = {'prop_m_rural_overwt': []}
        self.o_prop_f_rural_overwt = {'prop_f_rural_overwt': []}
        self.o_prop_m_urban_low_ex = {'prop_m_urban_low_ex': []}
        self.o_prop_f_urban_low_ex = {'prop_f_urban_low_ex': []}
        self.o_prop_m_rural_low_ex = {'prop_m_rural_low_ex': []}
        self.o_prop_f_rural_low_ex = {'prop_f_rural_low_ex': []}
        self.o_prop_m_ex_alc = {'prop_m_ex_alc': []}
        self.o_prop_f_ex_alc = {'prop_f_ex_alc': []}
        self.o_prop_wealth1 = {'prop_wealth1': []}
        self.o_prop_tob = {'prop_tob': []}
        self.o_prop_m_age1519_w1_tob = {'prop_m_age1519_w1_tob': []}
        self.o_prop_m_age2039_w1_tob = {'prop_m_age2039_w1_tob': []}
        self.o_prop_m_agege40_w1_tob = {'prop_m_agege40_w1_tob': []}
        self.o_prop_m_age1519_w2_tob = {'prop_m_age1519_w2_tob': []}
        self.o_prop_m_age2039_w2_tob = {'prop_m_age2039_w2_tob': []}
        self.o_prop_m_agege40_w2_tob = {'prop_m_agege40_w2_tob': []}
        self.o_prop_m_age1519_w3_tob = {'prop_m_age1519_w3_tob': []}
        self.o_prop_m_age2039_w3_tob = {'prop_m_age2039_w3_tob': []}
        self.o_prop_m_agege40_w3_tob = {'prop_m_agege40_w3_tob': []}
        self.o_prop_m_age1519_w4_tob = {'prop_m_age1519_w4_tob': []}
        self.o_prop_m_age2039_w4_tob = {'prop_m_age2039_w4_tob': []}
        self.o_prop_m_agege40_w4_tob = {'prop_m_agege40_w4_tob': []}
        self.o_prop_m_age1519_w5_tob = {'prop_m_age1519_w5_tob': []}
        self.o_prop_m_age2039_w5_tob = {'prop_m_age2039_w5_tob': []}
        self.o_prop_m_agege40_w5_tob = {'prop_m_agege40_w5_tob': []}
        self.o_prop_f_age1519_w1_tob = {'prop_f_age1519_w1_tob': []}
        self.o_prop_f_age2039_w1_tob = {'prop_f_age2039_w1_tob': []}
        self.o_prop_f_agege40_w1_tob = {'prop_f_agege40_w1_tob': []}
        self.o_prop_f_age1519_w2_tob = {'prop_f_age1519_w2_tob': []}
        self.o_prop_f_age2039_w2_tob = {'prop_f_age2039_w2_tob': []}
        self.o_prop_f_agege40_w2_tob = {'prop_f_agege40_w2_tob': []}
        self.o_prop_f_age1519_w3_tob = {'prop_f_age1519_w3_tob': []}
        self.o_prop_f_age2039_w3_tob = {'prop_f_age2039_w3_tob': []}
        self.o_prop_f_agege40_w3_tob = {'prop_f_agege40_w3_tob': []}
        self.o_prop_f_age1519_w4_tob = {'prop_f_age1519_w4_tob': []}
        self.o_prop_f_age2039_w4_tob = {'prop_f_age2039_w4_tob': []}
        self.o_prop_f_agege40_w4_tob = {'prop_f_agege40_w4_tob': []}
        self.o_prop_f_age1519_w5_tob = {'prop_f_age1519_w5_tob': []}
        self.o_prop_f_age2039_w5_tob = {'prop_f_age2039_w5_tob': []}
        self.o_prop_f_agege40_w5_tob = {'prop_f_agege40_w5_tob': []}
        self.o_prop_mar_stat_1 = {'prop_mar_stat_1': []}
        self.o_prop_mar_stat_2 = {'prop_mar_stat_2': []}
        self.o_prop_mar_stat_3 = {'prop_mar_stat_3': []}
        self.o_prop_mar_stat_1_agege60 = {'prop_mar_stat_1_agege60': []}
        self.o_prop_mar_stat_2_agege60 = {'prop_mar_stat_2_agege60': []}
        self.o_prop_mar_stat_3_agege60 = {'prop_mar_stat_3_agege60': []}
        self.o_prop_f_1550_on_con = {'prop_f_1550_on_con': []}

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.
        Here we do nothing.
        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        self.parameters['r_urban'] = 0.002
        self.parameters['r_rural'] = 0.0001
        self.parameters['r_overwt'] = 0.0025
        self.parameters['r_not_overwt'] = 0.001
        self.parameters['rr_overwt_f'] = 0.8
        self.parameters['rr_overwt_urban'] = 1.5
        self.parameters['r_low_ex'] = 0.001
        self.parameters['r_not_low_ex'] = 0.0001
        self.parameters['rr_low_ex_f'] = 0.6
        self.parameters['rr_low_ex_urban'] = 2.0
        self.parameters['r_tob'] = 0.0004
        self.parameters['r_not_tob'] = 0.000
        self.parameters['rr_tob_f'] = 0.1
        self.parameters['rr_tob_age2039'] = 1.2
        self.parameters['rr_tob_agege40'] = 1.5
        self.parameters['rr_tob_wealth'] = 1.3
        self.parameters['r_ex_alc'] = 0.003
        self.parameters['r_not_ex_alc'] = 0.000
        self.parameters['rr_ex_alc_f'] = 0.07
        self.parameters['init_p_urban'] = 0.17
        self.parameters['init_p_wealth_urban'] = [0.75, 0.16, 0.05, 0.02, 0.02]
        self.parameters['init_p_wealth_rural'] = [0.11, 0.21, 0.22, 0.23, 0.23]
        self.parameters['init_p_overwt_agelt15'] = 0.0
        self.parameters['init_p_ex_alc_m'] = 0.15
        self.parameters['init_p_ex_alc_f'] = 0.01
        self.parameters['init_dist_mar_stat_age1520'] = [0.70, 0.30, 0.00]
        self.parameters['init_dist_mar_stat_age2030'] = [0.15, 0.80, 0.05]
        self.parameters['init_dist_mar_stat_age3040'] = [0.05, 0.70, 0.25]
        self.parameters['init_dist_mar_stat_age4050'] = [0.03, 0.50, 0.47]
        self.parameters['init_dist_mar_stat_age5060'] = [0.03, 0.30, 0.67]
        self.parameters['init_dist_mar_stat_agege60'] = [0.03, 0.20, 0.77]
        self.parameters['r_mar'] = 0.03
        self.parameters['r_div_wid'] = 0.01
        self.parameters['init_p_on_contrac'] = 0.30
        self.parameters['init_dist_con_t'] = [0.17, 0.17, 0.17, 0.17, 0.17, 0.15]
        self.parameters['r_contrac'] = 0.05
        self.parameters['r_contrac_int'] = 0.1
        self.parameters['r_con_from_1'] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        self.parameters['r_con_from_2'] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        self.parameters['r_con_from_3'] = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        self.parameters['r_con_from_4'] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        self.parameters['r_con_from_5'] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.parameters['r_con_from_6'] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        self.parameters['r_stop_ed'] = 0.01
        self.parameters['p_ed_primary'] = 0.75
        self.parameters['rp_ed_primary_higher_wealth'] = 1.05
        self.parameters['p_ed_secondary'] = 0.5
        self.parameters['rp_ed_secondary_higher_wealth'] = 1.15
        self.parameters['init_age2030_w5_some_ed'] = 0.3
        self.parameters['init_rp_some_ed_age1520'] = 0.35
        self.parameters['init_rp_some_ed_age3040'] = 0.25
        self.parameters['init_rp_some_ed_age4050'] = 0.2
        self.parameters['init_rp_some_ed_age5060'] = 0.15
        self.parameters['init_rp_some_ed_agege60'] = 0.1
        self.parameters['init_rp_some_ed_per_higher_wealth'] = 1.3
        self.parameters['init_prop_age2030_w5_some_ed_sec'] = 0.3
        self.parameters['init_rp_some_ed_sec_age1520'] = 0.35
        self.parameters['init_rp_some_ed_sec_age3040'] = 0.25
        self.parameters['init_rp_some_ed_sec_age4050'] = 0.2
        self.parameters['init_rp_some_ed_sec_age5060'] = 0.15
        self.parameters['init_rp_some_ed_sec_agege60'] = 0.1
        self.parameters['init_rp_some_ed_sec_per_higher_wealth'] = 1.3

    def initialise_population(self, population):
        """Set our property values for the initial population.
        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.
        :param population: the population of individuals
        """
        df = population.props  # a shortcut to the data-frame storing data for individuals
        df['li_urban'] = False  # default: all individuals rural
        df['li_date_trans_to_urban'] = pd.NaT
        df['li_wealth'].values[:] = 3  # default: all individuals wealth 3
        df['li_overwt'] = False  # default: all not overwt
        df['li_low_ex'] = False  # default all not low ex
        df['li_tob'] = False  # default all not tob
        df['li_ex_alc'] = False  # default all not ex alc
        df['li_mar_stat'] = 1  # default: all individuals never married
        df['li_on_con'] = False # default: all not on contraceptives
        df['li_con_t'] = 1  # default: call contraceptive type 1, but when li_on_con = False this property becomes most recent contraceptive used
        df['li_in_ed'] = False   # default: not in education
        df['li_ed_lev'].values[:] = 1   # default: education level = 1 - no education

        #  this below calls the age dataframe / call age.years to get age in years
        age = population.age

        agelt15_index = df.index[age.years < 15]

        # todo: allocate wealth level at baseline

        # urban
        # randomly selected some individuals as urban
        initial_urban = self.parameters['init_p_urban']
        initial_rural = 1 - initial_urban
        df['li_urban'] = np.random.choice([True, False], size=len(df), p=[initial_urban, initial_rural])

        # get the indices of all individuals who are urban
        urban_index = df.index[df.li_urban]
        # randomly sample wealth category according to urban wealth probs and assign to urban ind.
        df.loc[urban_index, 'li_wealth'] = np.random.choice([1, 2, 3, 4, 5],
                                                            size=len(urban_index),
                                                            p=self.parameters['init_p_wealth_urban'])

        # get the indicies of all individual who are rural (i.e. not urban)
        rural_index = df.index[~df.li_urban]
        df.loc[rural_index, 'li_wealth'] = np.random.choice([1, 2, 3, 4, 5],
                                                            size=len(rural_index),
                                                            p=self.parameters['init_p_wealth_rural'])

        # get indices of all individuals over 15 years
        gte_15 = df.index[age.years >= 15]

        # overwt;
        overweight_lookup = pd.DataFrame(data=[('M', True, 0.46),
                                               ('M', False, 0.27),
                                               ('F', True, 0.32),
                                               ('F', False, 0.17)],
                                         columns=['sex', 'is_urban', 'p_ow'])

        overweight_probs = df.loc[gte_15, ['sex', 'li_urban']].merge(overweight_lookup,
                                                                     left_on=['sex', 'li_urban'],
                                                                     right_on=['sex', 'is_urban'],
                                                                     how='left')['p_ow']

        random_draw = self.rng.random_sample(size=len(gte_15))
        df.loc[gte_15, 'li_overwt'] = (random_draw < overweight_probs.values)

        # low_ex;
        low_ex_lookup = pd.DataFrame(data=[('M', True, 0.32),
                                           ('M', False, 0.11),
                                           ('F', True, 0.18),
                                           ('F', False, 0.07)],
                                     columns=['sex', 'is_urban', 'p_low_ex'])

        low_ex_probs = df.loc[gte_15, ['sex', 'li_urban']].merge(low_ex_lookup,
                                                                 left_on=['sex', 'li_urban'],
                                                                 right_on=['sex', 'is_urban'],
                                                                 how='left')['p_low_ex']

        random_draw = self.rng.random_sample(size=len(gte_15))
        df.loc[gte_15, 'li_low_ex'] = (random_draw < low_ex_probs.values)

        # tob ;
        tob_lookup = pd.DataFrame([('M', '15-19', 0.01),
                                   ('M', '20-24', 0.04),
                                   ('M', '25-29', 0.04),
                                   ('M', '30-34', 0.04),
                                   ('M', '35-39', 0.04),
                                   ('M', '40-44', 0.06),
                                   ('M', '45-49', 0.06),
                                   ('M', '50-54', 0.06),
                                   ('M', '55-59', 0.06),
                                   ('M', '60-64', 0.06),
                                   ('M', '65-69', 0.06),
                                   ('M', '70-74', 0.06),
                                   ('M', '75-79', 0.06),
                                   ('M', '80-84', 0.06),
                                   ('M', '85-89', 0.06),
                                   ('M', '90-94', 0.06),
                                   ('M', '95-99', 0.06),
                                   ('M', '100+',  0.06),

                                   ('F', '15-19', 0.002),
                                   ('F', '20-24', 0.002),
                                   ('F', '25-29', 0.002),
                                   ('F', '30-34', 0.002),
                                   ('F', '35-39', 0.002),
                                   ('F', '40-44', 0.002),
                                   ('F', '45-49', 0.002),
                                   ('F', '50-54', 0.002),
                                   ('F', '55-59', 0.002),
                                   ('F', '60-64', 0.002),
                                   ('F', '65-69', 0.002),
                                   ('F', '70-74', 0.002),
                                   ('F', '75-79', 0.002),
                                   ('F', '80-84', 0.002),
                                   ('F', '85-89', 0.002),
                                   ('F', '90-94', 0.002),
                                   ('F', '95-99', 0.002),
                                   ('F', '100+',  0.002)],
                                  columns=['sex', 'age_range', 'p_tob'])

        # join the population dataframe with age information (we need them both together)
        df_with_age = df.loc[gte_15, ['sex', 'li_wealth']].merge(age, left_index=True, right_index=True, how='inner')
        assert len(df_with_age) == len(gte_15)  # check we have the same number of individuals after the merge

        # join the population-with-age dataframe with the tobacco use lookup table (join on sex and age_range)
        tob_probs = df_with_age.merge(tob_lookup, left_on=['sex', 'age_range'], right_on=['sex', 'age_range'],
                                      how='left')

        assert np.array_equal(tob_probs.years_exact, df_with_age.years_exact)
        # check the order of individuals is the same by comparing exact ages
        assert tob_probs.p_tob.isna().sum() == 0  # ensure we found a p_tob for every individual

        # each individual has a baseline probability
        # multiply this probability by the wealth level. wealth is a category, so convert to integer
        tob_probs = tob_probs['li_wealth'].astype(int) * tob_probs['p_tob']

        # we now have the probability of tobacco use for each individual where age >= 15
        # draw a random number between 0 and 1 for all of them
        random_draw = self.rng.random_sample(size=len(gte_15))

        # decide on tobacco use based on the individual probability is greater than random draw
        # this is a list of True/False. assign to li_tob
        df.loc[gte_15, 'li_tob'] = (random_draw < tob_probs.values)

        # ex alc;
        df.loc[agelt15_index, 'li_ex_alc'] = False

        i_p_ex_alc_m = self.parameters['init_p_ex_alc_m']
        i_p_not_ex_alc_m = 1 - i_p_ex_alc_m
        i_p_ex_alc_f = self.parameters['init_p_ex_alc_f']
        i_p_not_ex_alc_f = 1 - i_p_ex_alc_f

        m_agege15_index = df.index[(age.years >= 15) & (df.sex == 'M')]
        f_agege15_index = df.index[(age.years >= 15) & (df.sex == 'F')]

        df.loc[m_agege15_index, 'li_ex_alc'] = np.random.choice([True, False], size=len(m_agege15_index),
                                                                p=[i_p_ex_alc_m, i_p_not_ex_alc_m])
        df.loc[f_agege15_index, 'li_ex_alc'] = np.random.choice([True, False], size=len(f_agege15_index),
                                                                p=[i_p_ex_alc_f, i_p_not_ex_alc_f])

        # mar stat (marital status)

        age1520_index = df.index[(age.years >= 15) & (age.years < 20) & df.is_alive]
        age2030_index = df.index[(age.years >= 20) & (age.years < 30) & df.is_alive]
        age3040_index = df.index[(age.years >= 30) & (age.years < 40) & df.is_alive]
        age4050_index = df.index[(age.years >= 40) & (age.years < 50) & df.is_alive]
        age5060_index = df.index[(age.years >= 50) & (age.years < 60) & df.is_alive]
        agege60_index = df.index[(age.years >= 60) & df.is_alive]

        df.loc[age1520_index, 'li_mar_stat'] = np.random.choice([1, 2, 3], size=len(age1520_index), p=self.parameters['init_dist_mar_stat_age1520'])
        df.loc[age2030_index, 'li_mar_stat'] = np.random.choice([1, 2, 3], size=len(age2030_index), p=self.parameters['init_dist_mar_stat_age2030'])
        df.loc[age3040_index, 'li_mar_stat'] = np.random.choice([1, 2, 3], size=len(age3040_index), p=self.parameters['init_dist_mar_stat_age3040'])
        df.loc[age4050_index, 'li_mar_stat'] = np.random.choice([1, 2, 3], size=len(age4050_index), p=self.parameters['init_dist_mar_stat_age4050'])
        df.loc[age5060_index, 'li_mar_stat'] = np.random.choice([1, 2, 3], size=len(age5060_index), p=self.parameters['init_dist_mar_stat_age5060'])
        df.loc[agege60_index, 'li_mar_stat'] = np.random.choice([1, 2, 3], size=len(agege60_index), p=self.parameters['init_dist_mar_stat_agege60'])

        # li_on_con (contraception)

        f_age1550_idx = df.index[(age.years >= 15) & (age.years < 50) & df.is_alive & (df.sex == 'F')]
        df.loc[f_age1550_idx, 'li_on_con'] = np.random.choice([True, False], size=len(f_age1550_idx), p=[self.parameters['init_p_on_contrac'], 1 - self.parameters['init_p_on_contrac']])

        f_age1550_on_con_idx = df.index[(age.years >= 15) & (age.years < 50) & df.is_alive & (df.sex == 'F') & (df.li_on_con == True)]
        df.loc[f_age1550_on_con_idx, 'li_con_t'] = np.random.choice([1, 2, 3, 4, 5, 6], size=len(f_age1550_on_con_idx), p=self.parameters['init_dist_con_t'])

        # education (li_in_ed and li_ed_lev)

        ed_lev_1_ = 1 - self.parameters['init_age2030_w5_some_ed']
        ed_lev_3_ = self.parameters['init_age2030_w5_some_ed'] * self.parameters['init_prop_age2030_w5_some_ed_sec']
        ed_lev_2_ = 1 - ed_lev_1_ - ed_lev_3_
        age2030_w5_idx = df.index[(age.years >= 20) & (age.years < 30) & (df.li_wealth == 5) & df.is_alive]
        df.loc[age2030_w5_idx, 'li_ed_lev'] = np.random.choice([1, 2, 3], size=len(age2030_w5_idx),
                                                               p=[ed_lev_1_, ed_lev_2_, ed_lev_3_])

        ed_lev_1_ = 1 - (self.parameters['init_age2030_w5_some_ed'] * self.parameters['init_rp_some_ed_age1520'])
        ed_lev_3_ = self.parameters['init_age2030_w5_some_ed'] * self.parameters['init_rp_some_ed_age1520'] \
                    * self.parameters['init_prop_age2030_w5_some_ed_sec'] * self.parameters['init_rp_some_ed_sec_age1520']
        ed_lev_2_ = 1 - ed_lev_1_ - ed_lev_3_
        age1520_w5_idx = df.index[(age.years >= 15) & (age.years < 20) & (df.li_wealth == 5) & df.is_alive]
        df.loc[age1520_w5_idx, 'li_ed_lev'] = np.random.choice([1, 2, 3], size=len(age1520_w5_idx),
                                                               p=[ed_lev_1_, ed_lev_2_, ed_lev_3_])


#       self.parameters['init_rp_some_ed_age1520'] = 0.35
#       self.parameters['init_rp_some_ed_age3040'] = 0.25
#       self.parameters['init_rp_some_ed_age4050'] = 0.2
#       self.parameters['init_rp_some_ed_age5060'] = 0.15
#       self.parameters['init_rp_some_ed_agege60'] = 0.1
#       self.parameters['init_rp_some_ed_per_higher_wealth'] = 1.3
#
#       self.parameters['init_rp_some_ed_sec_age1520'] = 0.35
#       self.parameters['init_rp_some_ed_sec_age3040'] = 0.25
#       self.parameters['init_rp_some_ed_sec_age4050'] = 0.2
#       self.parameters['init_rp_some_ed_sec_age5060'] = 0.15
#       self.parameters['init_rp_some_ed_sec_agege60'] = 0.1
#       self.parameters['init_rp_some_ed_sec_per_higher_wealth'] = 1.3

    def initialise_simulation(self, sim):
        """Get ready for simulation start.
        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """
        event = LifestyleEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=3))

        event = LifestylesLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=3))

    def on_birth(self, mother, child):
        """Initialise our properties for a newborn individual.
        This is called by the simulation whenever a new person is born.
        :param mother: the mother for this child
        :param child: the new child
        """

        child.li_urban = mother.li_urban
        child.li_wealth = mother.li_wealth

 #      child.date_of_birth
 #      child.sex
 #      child.mother_id
 #      child.is_alive


class LifestyleEvent(RegularEvent, PopulationScopeEventMixin):
    """A skeleton class for an event
    Regular events automatically reschedule themselves at a fixed frequency,
    and thus implement discrete timestep type behaviour. The frequency is
    specified when calling the base class constructor in our __init__ method.
    """
    def __init__(self, module):
        """One line summary here
        We need to pass the frequency at which we want to occur to the base class
        constructor using super(). We also pass the module that created this event,
        so that random number generators can be scoped per-module.
        :param module: the module that created this event
        """
        super().__init__(module, frequency=DateOffset(months=3))
        self.r_urban = module.parameters['r_urban']
        self.r_rural = module.parameters['r_rural']
        self.r_overwt = module.parameters['r_overwt']
        self.r_not_overwt = module.parameters['r_not_overwt']
        self.rr_overwt_f = module.parameters['rr_overwt_f']
        self.rr_overwt_urban = module.parameters['rr_overwt_urban']
        self.r_low_ex = module.parameters['r_low_ex']
        self.r_not_low_ex = module.parameters['r_not_low_ex']
        self.rr_low_ex_f = module.parameters['rr_low_ex_f']
        self.rr_low_ex_urban = module.parameters['rr_low_ex_urban']
        self.r_tob = module.parameters['r_tob']
        self.r_not_tob = module.parameters['r_not_tob']
        self.rr_tob_f = module.parameters['rr_tob_f']
        self.rr_tob_age2039 = module.parameters['rr_tob_age2039']
        self.rr_tob_agege40 = module.parameters['rr_tob_agege40']
        self.rr_tob_wealth = module.parameters['rr_tob_wealth']
        self.r_ex_alc = module.parameters['r_ex_alc']
        self.r_not_ex_alc = module.parameters['r_not_ex_alc']
        self.rr_ex_alc_f = module.parameters['rr_ex_alc_f']
        self.r_mar = module.parameters['r_mar']
        self.r_div_wid = module.parameters['r_div_wid']
        self.r_contrac = module.parameters['r_contrac']
        self.r_contrac_int = module.parameters['r_contrac_int']
        self.r_con_from_1 = module.parameters['r_con_from_1']
        self.r_con_from_2 = module.parameters['r_con_from_2']
        self.r_con_from_3 = module.parameters['r_con_from_3']
        self.r_con_from_4 = module.parameters['r_con_from_4']
        self.r_con_from_5 = module.parameters['r_con_from_5']
        self.r_con_from_6 = module.parameters['r_con_from_6']
        self.p_ed_primary = module.parameters['p_ed_primary']

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props

        age = population.age

        # TODO: remove in live code!
        currently_alive = df[df.is_alive]
        people_to_die = currently_alive.sample(n=int(len(currently_alive) * 0.005)).index
        if len(people_to_die):
            df.loc[people_to_die, 'is_alive'] = False

        # 1. get (and hold) index of current urban rural status
        currently_rural = df.index[~df.li_urban & df.is_alive]
        currently_urban = df.index[df.li_urban & df.is_alive]

        # 2. handle new transitions
        now_urban = np.random.choice([True, False],
                                     size=len(currently_rural),
                                     p=[self.r_urban, 1 - self.r_urban])
        # if any have transitioned to urban
        if now_urban.sum():
            urban_idx = currently_rural[now_urban]
            df.loc[urban_idx, 'li_urban'] = True
            df.loc[urban_idx, 'li_date_trans_to_urban'] = self.sim.date

        # 3. handle new transitions to rural
        now_rural = np.random.choice([True, False], size=len(currently_urban), p=[self.r_rural, 1 - self.r_rural])
        # if any have transitioned to rural
        if now_rural.sum():
            rural_idx = currently_urban[now_rural]
            df.loc[rural_idx, 'li_urban'] = False

        # as above - transition between overwt and not overwt
        # transition to ovrwt depends on sex

        currently_not_overwt_f_urban = df.index[~df.li_overwt & df.is_alive & (df.sex == 'F') & df.li_urban
                                                & (age.years >= 15)]
        currently_not_overwt_m_urban = df.index[~df.li_overwt & df.is_alive & (df.sex == 'M') & df.li_urban
                                                & (age.years >= 15)]
        currently_not_overwt_f_rural = df.index[~df.li_overwt & df.is_alive & (df.sex == 'F') & ~df.li_urban
                                                & (age.years >= 15)]
        currently_not_overwt_m_rural = df.index[~df.li_overwt & df.is_alive & (df.sex == 'M') & ~df.li_urban
                                                & (age.years >= 15)]
        currently_overwt = df.index[df.li_overwt & df.is_alive]

        ri_overwt_f_urban = self.r_overwt * self.rr_overwt_f * self.rr_overwt_urban
        ri_overwt_f_rural = self.r_overwt * self.rr_overwt_f
        ri_overwt_m_urban = self.r_overwt * self.rr_overwt_urban
        ri_overwt_m_rural = self.r_overwt

        now_overwt_f_urban = np.random.choice([True, False],
                                              size=len(currently_not_overwt_f_urban),
                                              p=[ri_overwt_f_urban, 1 - ri_overwt_f_urban])

        if now_overwt_f_urban.sum():
            overwt_f_urban_idx = currently_not_overwt_f_urban[now_overwt_f_urban]
            df.loc[overwt_f_urban_idx, 'li_overwt'] = True

        now_overwt_m_urban = np.random.choice([True, False],
                                              size=len(currently_not_overwt_m_urban),
                                              p=[ri_overwt_m_urban, 1 - ri_overwt_m_urban])

        if now_overwt_m_urban.sum():
            overwt_m_urban_idx = currently_not_overwt_m_urban[now_overwt_m_urban]
            df.loc[overwt_m_urban_idx, 'li_overwt'] = True

        now_not_overwt = np.random.choice([True, False], size=len(currently_overwt),
                                          p=[self.r_not_overwt, 1 - self.r_not_overwt])

        now_overwt_f_rural = np.random.choice([True, False],
                                              size=len(currently_not_overwt_f_rural),
                                              p=[ri_overwt_f_rural, 1 - ri_overwt_f_rural])
        if now_overwt_f_rural.sum():
            overwt_f_rural_idx = currently_not_overwt_f_rural[now_overwt_f_rural]
            df.loc[overwt_f_rural_idx, 'li_overwt'] = True

        now_overwt_m_rural = np.random.choice([True, False],
                                              size=len(currently_not_overwt_m_rural),
                                              p=[ri_overwt_m_rural, 1 - ri_overwt_m_rural])
        if now_overwt_m_rural.sum():
            overwt_m_rural_idx = currently_not_overwt_m_rural[now_overwt_m_rural]
            df.loc[overwt_m_rural_idx, 'li_overwt'] = True

        if now_not_overwt.sum():
            not_overwt_idx = currently_overwt[now_not_overwt]
            df.loc[not_overwt_idx, 'li_overwt'] = False

        # transition between low ex and not low ex
        currently_not_low_ex_f_urban = df.index[~df.li_low_ex & df.is_alive & (df.sex == 'F') & df.li_urban
                                                & (age.years >= 15)]
        currently_not_low_ex_m_urban = df.index[~df.li_low_ex & df.is_alive & (df.sex == 'M') & df.li_urban
                                                & (age.years >= 15)]
        currently_not_low_ex_f_rural = df.index[~df.li_low_ex & df.is_alive & (df.sex == 'F') & ~df.li_urban
                                                & (age.years >= 15)]
        currently_not_low_ex_m_rural = df.index[~df.li_low_ex & df.is_alive & (df.sex == 'M') & ~df.li_urban
                                                & (age.years >= 15)]
        currently_low_ex = df.index[df.li_low_ex & df.is_alive]

        ri_low_ex_f_urban = self.r_low_ex * self.rr_low_ex_f * self.rr_low_ex_urban
        ri_low_ex_f_rural = self.r_low_ex * self.rr_low_ex_f
        ri_low_ex_m_urban = self.r_low_ex * self.rr_low_ex_urban
        ri_low_ex_m_rural = self.r_low_ex

        now_low_ex_f_urban = np.random.choice([True, False],
                                              size=len(currently_not_low_ex_f_urban),
                                              p=[ri_low_ex_f_urban, 1 - ri_low_ex_f_urban])

        if now_low_ex_f_urban.sum():
            low_ex_f_urban_idx = currently_not_low_ex_f_urban[now_low_ex_f_urban]
            df.loc[low_ex_f_urban_idx, 'li_low_ex'] = True

        now_low_ex_m_urban = np.random.choice([True, False],
                                              size=len(currently_not_low_ex_m_urban),
                                              p=[ri_low_ex_m_urban, 1 - ri_low_ex_m_urban])

        if now_low_ex_m_urban.sum():
            low_ex_m_urban_idx = currently_not_low_ex_m_urban[now_low_ex_m_urban]
            df.loc[low_ex_m_urban_idx, 'li_low_ex'] = True

        now_not_low_ex = np.random.choice([True, False], size=len(currently_low_ex),
                                          p=[self.r_not_low_ex, 1 - self.r_not_low_ex])

        now_low_ex_f_rural = np.random.choice([True, False],
                                              size=len(currently_not_low_ex_f_rural),
                                              p=[ri_low_ex_f_rural, 1 - ri_low_ex_f_rural])
        if now_low_ex_f_rural.sum():
            low_ex_f_rural_idx = currently_not_low_ex_f_rural[now_low_ex_f_rural]
            df.loc[low_ex_f_rural_idx, 'li_low_ex'] = True

        now_low_ex_m_rural = np.random.choice([True, False],
                                              size=len(currently_not_low_ex_m_rural),
                                              p=[ri_low_ex_m_rural, 1 - ri_low_ex_m_rural])
        if now_low_ex_m_rural.sum():
            low_ex_m_rural_idx = currently_not_low_ex_m_rural[now_low_ex_m_rural]
            df.loc[low_ex_m_rural_idx, 'li_low_ex'] = True

        if now_not_low_ex.sum():
            not_low_ex_idx = currently_low_ex[now_not_low_ex]
            df.loc[not_low_ex_idx, 'li_low_ex'] = False

        # transition between not tob and tob
        currently_not_tob_f_age1519_w1 = df.index[~df.li_tob & df.is_alive & (df.sex == 'F') & (age.years >= 15)
                                                  & (age.years < 20) & (df.li_wealth == 1)]
        currently_not_tob_m_age1519_w1 = df.index[~df.li_tob & df.is_alive & (df.sex == 'M') & (age.years >= 15)
                                                  & (age.years < 20) & (df.li_wealth == 1)]
        currently_not_tob_f_age2039_w1 = df.index[~df.li_tob & df.is_alive & (df.sex == 'F') & (age.years >= 20)
                                                  & (age.years < 40) & (df.li_wealth == 1)]
        currently_not_tob_m_age2039_w1 = df.index[~df.li_tob & df.is_alive & (df.sex == 'M') & (age.years >= 20)
                                                  & (age.years < 40) & (df.li_wealth == 1)]
        currently_not_tob_f_agege40_w1 = df.index[~df.li_tob & df.is_alive & (df.sex == 'F') & (age.years >= 40)
                                                  & (df.li_wealth == 1)]
        currently_not_tob_m_agege40_w1 = df.index[~df.li_tob & df.is_alive & (df.sex == 'M') & (age.years >= 40)
                                                  & (df.li_wealth == 1)]
        currently_not_tob_f_age1519_w2 = df.index[~df.li_tob & df.is_alive & (df.sex == 'F') & (age.years >= 15)
                                                  & (age.years < 20) & (df.li_wealth == 2)]
        currently_not_tob_m_age1519_w2 = df.index[~df.li_tob & df.is_alive & (df.sex == 'M') & (age.years >= 15)
                                                  & (age.years < 20) & (df.li_wealth == 2)]
        currently_not_tob_f_age2039_w2 = df.index[~df.li_tob & df.is_alive & (df.sex == 'F') & (age.years >= 20)
                                                  & (age.years < 40) & (df.li_wealth == 2)]
        currently_not_tob_m_age2039_w2 = df.index[~df.li_tob & df.is_alive & (df.sex == 'M') & (age.years >= 20)
                                                  & (age.years < 40) & (df.li_wealth == 2)]
        currently_not_tob_f_agege40_w2 = df.index[~df.li_tob & df.is_alive & (df.sex == 'F') & (age.years >= 40)
                                                  & (df.li_wealth == 2)]
        currently_not_tob_m_agege40_w2 = df.index[~df.li_tob & df.is_alive & (df.sex == 'M') & (age.years >= 40)
                                                  & (df.li_wealth == 2)]
        currently_not_tob_f_age1519_w3 = df.index[~df.li_tob & df.is_alive & (df.sex == 'F') & (age.years >= 15)
                                                  & (age.years < 20) & (df.li_wealth == 3)]
        currently_not_tob_m_age1519_w3 = df.index[~df.li_tob & df.is_alive & (df.sex == 'M') & (age.years >= 15)
                                                  & (age.years < 20) & (df.li_wealth == 3)]
        currently_not_tob_f_age2039_w3 = df.index[~df.li_tob & df.is_alive & (df.sex == 'F') & (age.years >= 20)
                                                  & (age.years < 40) & (df.li_wealth == 3)]
        currently_not_tob_m_age2039_w3 = df.index[~df.li_tob & df.is_alive & (df.sex == 'M') & (age.years >= 20)
                                                  & (age.years < 40) & (df.li_wealth == 3)]
        currently_not_tob_f_agege40_w3 = df.index[~df.li_tob & df.is_alive & (df.sex == 'F') & (age.years >= 40)
                                                  & (df.li_wealth == 3)]
        currently_not_tob_m_agege40_w3 = df.index[~df.li_tob & df.is_alive & (df.sex == 'M') & (age.years >= 40)
                                                  & (df.li_wealth == 3)]
        currently_not_tob_f_age1519_w4 = df.index[~df.li_tob & df.is_alive & (df.sex == 'F') & (age.years >= 15)
                                                  & (age.years < 20) & (df.li_wealth == 4)]
        currently_not_tob_m_age1519_w4 = df.index[~df.li_tob & df.is_alive & (df.sex == 'M') & (age.years >= 15)
                                                  & (age.years < 20) & (df.li_wealth == 4)]
        currently_not_tob_f_age2039_w4 = df.index[~df.li_tob & df.is_alive & (df.sex == 'F') & (age.years >= 20)
                                                  & (age.years < 40) & (df.li_wealth == 4)]
        currently_not_tob_m_age2039_w4 = df.index[~df.li_tob & df.is_alive & (df.sex == 'M') & (age.years >= 20)
                                                  & (age.years < 40) & (df.li_wealth == 4)]
        currently_not_tob_f_agege40_w4 = df.index[~df.li_tob & df.is_alive & (df.sex == 'F') & (age.years >= 40)
                                                  & (df.li_wealth == 4)]
        currently_not_tob_m_agege40_w4 = df.index[~df.li_tob & df.is_alive & (df.sex == 'M') & (age.years >= 40)
                                                  & (df.li_wealth == 4)]
        currently_not_tob_f_age1519_w5 = df.index[~df.li_tob & df.is_alive & (df.sex == 'F') & (age.years >= 15)
                                                  & (age.years < 20) & (df.li_wealth == 5)]
        currently_not_tob_m_age1519_w5 = df.index[~df.li_tob & df.is_alive & (df.sex == 'M') & (age.years >= 15)
                                                  & (age.years < 20) & (df.li_wealth == 5)]
        currently_not_tob_f_age2039_w5 = df.index[~df.li_tob & df.is_alive & (df.sex == 'F') & (age.years >= 20)
                                                  & (age.years < 40) & (df.li_wealth == 5)]
        currently_not_tob_m_age2039_w5 = df.index[~df.li_tob & df.is_alive & (df.sex == 'M') & (age.years >= 20)
                                                  & (age.years < 40) & (df.li_wealth == 5)]
        currently_not_tob_f_agege40_w5 = df.index[~df.li_tob & df.is_alive & (df.sex == 'F') & (age.years >= 40)
                                                  & (df.li_wealth == 5)]
        currently_not_tob_m_agege40_w5 = df.index[~df.li_tob & df.is_alive & (df.sex == 'M') & (age.years >= 40)
                                                  & (df.li_wealth == 5)]

        currently_tob = df.index[df.li_tob & df.is_alive]

        ri_tob_f_age1519_w1 = self.r_tob * self.rr_tob_f
        ri_tob_f_age2039_w1 = self.r_tob * self.rr_tob_f * self.rr_tob_age2039
        ri_tob_f_agege40_w1 = self.r_tob * self.rr_tob_f * self.rr_tob_agege40
        ri_tob_m_age1519_w1 = self.r_tob
        ri_tob_m_age2039_w1 = self.r_tob * self.rr_tob_age2039
        ri_tob_m_agege40_w1 = self.r_tob * self.rr_tob_agege40

        ri_tob_f_age1519_w2 = self.r_tob * self.rr_tob_f * self.rr_tob_wealth  
        ri_tob_f_age2039_w2 = self.r_tob * self.rr_tob_f * self.rr_tob_age2039 * self.rr_tob_wealth 
        ri_tob_f_agege40_w2 = self.r_tob * self.rr_tob_f * self.rr_tob_agege40 * self.rr_tob_wealth 
        ri_tob_m_age1519_w2 = self.r_tob * self.rr_tob_wealth
        ri_tob_m_age2039_w2 = self.r_tob * self.rr_tob_age2039 * self.rr_tob_wealth
        ri_tob_m_agege40_w2 = self.r_tob * self.rr_tob_agege40 * self.rr_tob_wealth

        ri_tob_f_age1519_w3 = self.r_tob * self.rr_tob_f * self.rr_tob_wealth * self.rr_tob_wealth
        ri_tob_f_age2039_w3 = self.r_tob * self.rr_tob_f * self.rr_tob_age2039 * self.rr_tob_wealth * self.rr_tob_wealth
        ri_tob_f_agege40_w3 = self.r_tob * self.rr_tob_f * self.rr_tob_agege40 * self.rr_tob_wealth * self.rr_tob_wealth
        ri_tob_m_age1519_w3 = self.r_tob * self.rr_tob_wealth * self.rr_tob_wealth
        ri_tob_m_age2039_w3 = self.r_tob * self.rr_tob_age2039 * self.rr_tob_wealth * self.rr_tob_wealth
        ri_tob_m_agege40_w3 = self.r_tob * self.rr_tob_agege40 * self.rr_tob_wealth * self.rr_tob_wealth

        ri_tob_f_age1519_w4 = self.r_tob * self.rr_tob_f * self.rr_tob_wealth * self.rr_tob_wealth * self.rr_tob_wealth
        ri_tob_f_age2039_w4 = self.r_tob * self.rr_tob_f * self.rr_tob_age2039 * self.rr_tob_wealth * self.rr_tob_wealth * self.rr_tob_wealth
        ri_tob_f_agege40_w4 = self.r_tob * self.rr_tob_f * self.rr_tob_agege40 * self.rr_tob_wealth * self.rr_tob_wealth * self.rr_tob_wealth
        ri_tob_m_age1519_w4 = self.r_tob * self.rr_tob_wealth * self.rr_tob_wealth * self.rr_tob_wealth
        ri_tob_m_age2039_w4 = self.r_tob * self.rr_tob_age2039 * self.rr_tob_wealth * self.rr_tob_wealth * self.rr_tob_wealth
        ri_tob_m_agege40_w4 = self.r_tob * self.rr_tob_agege40 * self.rr_tob_wealth * self.rr_tob_wealth * self.rr_tob_wealth

        ri_tob_f_age1519_w5 = self.r_tob * self.rr_tob_f * self.rr_tob_wealth * self.rr_tob_wealth * self.rr_tob_wealth * self.rr_tob_wealth
        ri_tob_f_age2039_w5 = self.r_tob * self.rr_tob_f * self.rr_tob_age2039 * self.rr_tob_wealth * self.rr_tob_wealth * self.rr_tob_wealth * self.rr_tob_wealth
        ri_tob_f_agege40_w5 = self.r_tob * self.rr_tob_f * self.rr_tob_agege40 * self.rr_tob_wealth * self.rr_tob_wealth * self.rr_tob_wealth * self.rr_tob_wealth
        ri_tob_m_age1519_w5 = self.r_tob * self.rr_tob_wealth * self.rr_tob_wealth * self.rr_tob_wealth * self.rr_tob_wealth
        ri_tob_m_age2039_w5 = self.r_tob * self.rr_tob_age2039 * self.rr_tob_wealth * self.rr_tob_wealth * self.rr_tob_wealth * self.rr_tob_wealth
        ri_tob_m_agege40_w5 = self.r_tob * self.rr_tob_agege40 * self.rr_tob_wealth * self.rr_tob_wealth * self.rr_tob_wealth * self.rr_tob_wealth

        now_tob_f_age1519_w1 = np.random.choice([True, False],
                                                size=len(currently_not_tob_f_age1519_w1),
                                                p=[ri_tob_f_age1519_w1, 1 - ri_tob_f_age1519_w1])

        if now_tob_f_age1519_w1.sum():
            tob_f_age1519_w1_idx = currently_not_tob_f_age1519_w1[now_tob_f_age1519_w1]
            df.loc[tob_f_age1519_w1_idx, 'li_tob'] = True

        now_tob_m_age1519_w1 = np.random.choice([True, False],
                                                size=len(currently_not_tob_m_age1519_w1),
                                                p=[ri_tob_m_age1519_w1, 1 - ri_tob_m_age1519_w1])

        if now_tob_m_age1519_w1.sum():
            tob_m_age1519_w1_idx = currently_not_tob_m_age1519_w1[now_tob_m_age1519_w1]
            df.loc[tob_m_age1519_w1_idx, 'li_tob'] = True

        now_tob_f_age1519_w2 = np.random.choice([True, False],
                                                size=len(currently_not_tob_f_age1519_w2),
                                                p=[ri_tob_f_age1519_w2, 1 - ri_tob_f_age1519_w2])

        if now_tob_f_age1519_w2.sum():
            tob_f_age1519_w2_idx = currently_not_tob_f_age1519_w2[now_tob_f_age1519_w2]
            df.loc[tob_f_age1519_w2_idx, 'li_tob'] = True

        now_tob_m_age1519_w2 = np.random.choice([True, False],
                                                size=len(currently_not_tob_m_age1519_w2),
                                                p=[ri_tob_m_age1519_w2, 1 - ri_tob_m_age1519_w2])

        if now_tob_m_age1519_w2.sum():
            tob_m_age1519_w2_idx = currently_not_tob_m_age1519_w2[now_tob_m_age1519_w2]
            df.loc[tob_m_age1519_w2_idx, 'li_tob'] = True

        now_tob_f_age1519_w3 = np.random.choice([True, False],
                                                size=len(currently_not_tob_f_age1519_w3),
                                                p=[ri_tob_f_age1519_w3, 1 - ri_tob_f_age1519_w3])

        if now_tob_f_age1519_w3.sum():
            tob_f_age1519_w3_idx = currently_not_tob_f_age1519_w3[now_tob_f_age1519_w3]
            df.loc[tob_f_age1519_w3_idx, 'li_tob'] = True

        now_tob_m_age1519_w3 = np.random.choice([True, False],
                                                size=len(currently_not_tob_m_age1519_w3),
                                                p=[ri_tob_m_age1519_w3, 1 - ri_tob_m_age1519_w3])

        if now_tob_m_age1519_w3.sum():
            tob_m_age1519_w3_idx = currently_not_tob_m_age1519_w3[now_tob_m_age1519_w3]
            df.loc[tob_m_age1519_w3_idx, 'li_tob'] = True

        now_tob_f_age1519_w4 = np.random.choice([True, False],
                                               size=len(currently_not_tob_f_age1519_w4),
                                               p=[ri_tob_f_age1519_w4, 1 - ri_tob_f_age1519_w4])

        if now_tob_f_age1519_w4.sum():
           tob_f_age1519_w4_idx = currently_not_tob_f_age1519_w4[now_tob_f_age1519_w4]
           df.loc[tob_f_age1519_w4_idx, 'li_tob'] = True

        now_tob_m_age1519_w4 = np.random.choice([True, False],
                                                size=len(currently_not_tob_m_age1519_w4),
                                                p=[ri_tob_m_age1519_w4, 1 - ri_tob_m_age1519_w4])

        if now_tob_m_age1519_w4.sum():
            tob_m_age1519_w4_idx = currently_not_tob_m_age1519_w4[now_tob_m_age1519_w4]
            df.loc[tob_m_age1519_w4_idx, 'li_tob'] = True

        now_tob_f_age1519_w5 = np.random.choice([True, False],
                                                size=len(currently_not_tob_f_age1519_w5),
                                                p=[ri_tob_f_age1519_w5, 1 - ri_tob_f_age1519_w5])

        if now_tob_f_age1519_w5.sum():
            tob_f_age1519_w5_idx = currently_not_tob_f_age1519_w5[now_tob_f_age1519_w5]
            df.loc[tob_f_age1519_w5_idx, 'li_tob'] = True

        now_tob_m_age1519_w5 = np.random.choice([True, False],
                                                size=len(currently_not_tob_m_age1519_w5),
                                                p=[ri_tob_m_age1519_w5, 1 - ri_tob_m_age1519_w5])

        if now_tob_m_age1519_w5.sum():
            tob_m_age1519_w5_idx = currently_not_tob_m_age1519_w5[now_tob_m_age1519_w5]
            df.loc[tob_m_age1519_w5_idx, 'li_tob'] = True

        now_tob_f_age2039_w1 = np.random.choice([True, False],
                                                size=len(currently_not_tob_f_age2039_w1),
                                                p=[ri_tob_f_age2039_w1, 1 - ri_tob_f_age2039_w1])

        if now_tob_f_age2039_w1.sum():
            tob_f_age2039_w1_idx = currently_not_tob_f_age2039_w1[now_tob_f_age2039_w1]
            df.loc[tob_f_age2039_w1_idx, 'li_tob'] = True

        now_tob_m_age2039_w1 = np.random.choice([True, False],
                                                size=len(currently_not_tob_m_age2039_w1),
                                                p=[ri_tob_m_age2039_w1, 1 - ri_tob_m_age2039_w1])

        if now_tob_m_age2039_w1.sum():
            tob_m_age2039_w1_idx = currently_not_tob_m_age2039_w1[now_tob_m_age2039_w1]
            df.loc[tob_m_age2039_w1_idx, 'li_tob'] = True

        now_tob_f_age2039_w2 = np.random.choice([True, False],
                                                size=len(currently_not_tob_f_age2039_w2),
                                                p=[ri_tob_f_age2039_w2, 1 - ri_tob_f_age2039_w2])

        if now_tob_f_age2039_w2.sum():
            tob_f_age2039_w2_idx = currently_not_tob_f_age2039_w2[now_tob_f_age2039_w2]
            df.loc[tob_f_age2039_w2_idx, 'li_tob'] = True

        now_tob_m_age2039_w2 = np.random.choice([True, False],
                                                size=len(currently_not_tob_m_age2039_w2),
                                                p=[ri_tob_m_age2039_w2, 1 - ri_tob_m_age2039_w2])

        if now_tob_m_age2039_w2.sum():
            tob_m_age2039_w2_idx = currently_not_tob_m_age2039_w2[now_tob_m_age2039_w2]
            df.loc[tob_m_age2039_w2_idx, 'li_tob'] = True

        now_tob_f_age2039_w3 = np.random.choice([True, False],
                                                size=len(currently_not_tob_f_age2039_w3),
                                                p=[ri_tob_f_age2039_w3, 1 - ri_tob_f_age2039_w3])

        if now_tob_f_age2039_w3.sum():
            tob_f_age2039_w3_idx = currently_not_tob_f_age2039_w3[now_tob_f_age2039_w3]
            df.loc[tob_f_age2039_w3_idx, 'li_tob'] = True

        now_tob_m_age2039_w3 = np.random.choice([True, False],
                                                size=len(currently_not_tob_m_age2039_w3),
                                                p=[ri_tob_m_age2039_w3, 1 - ri_tob_m_age2039_w3])

        if now_tob_m_age2039_w3.sum():
            tob_m_age2039_w3_idx = currently_not_tob_m_age2039_w3[now_tob_m_age2039_w3]
            df.loc[tob_m_age2039_w3_idx, 'li_tob'] = True

        now_tob_f_age2039_w4 = np.random.choice([True, False],
                                                size=len(currently_not_tob_f_age2039_w4),
                                                p=[ri_tob_f_age2039_w4, 1 - ri_tob_f_age2039_w4])

        if now_tob_f_age2039_w4.sum():
            tob_f_age2039_w4_idx = currently_not_tob_f_age2039_w4[now_tob_f_age2039_w4]
            df.loc[tob_f_age2039_w4_idx, 'li_tob'] = True

        now_tob_m_age2039_w4 = np.random.choice([True, False],
                                                size=len(currently_not_tob_m_age2039_w4),
                                                p=[ri_tob_m_age2039_w4, 1 - ri_tob_m_age2039_w4])

        if now_tob_m_age2039_w4.sum():
            tob_m_age2039_w4_idx = currently_not_tob_m_age2039_w4[now_tob_m_age2039_w4]
            df.loc[tob_m_age2039_w4_idx, 'li_tob'] = True

        now_tob_f_age2039_w5 = np.random.choice([True, False],
                                                size=len(currently_not_tob_f_age2039_w5),
                                                p=[ri_tob_f_age2039_w5, 1 - ri_tob_f_age2039_w5])

        if now_tob_f_age2039_w5.sum():
            tob_f_age2039_w5_idx = currently_not_tob_f_age2039_w5[now_tob_f_age2039_w5]
            df.loc[tob_f_age2039_w5_idx, 'li_tob'] = True

        now_tob_m_age2039_w5 = np.random.choice([True, False],
                                                size=len(currently_not_tob_m_age2039_w5),
                                                p=[ri_tob_m_age2039_w5, 1 - ri_tob_m_age2039_w5])

        if now_tob_m_age2039_w5.sum():
            tob_m_age2039_w5_idx = currently_not_tob_m_age2039_w5[now_tob_m_age2039_w5]
            df.loc[tob_m_age2039_w5_idx, 'li_tob'] = True

        now_tob_f_agege40_w1 = np.random.choice([True, False],
                                                size=len(currently_not_tob_f_agege40_w1),
                                                p=[ri_tob_f_agege40_w1, 1 - ri_tob_f_agege40_w1])

        if now_tob_f_agege40_w1.sum():
            tob_f_agege40_w1_idx = currently_not_tob_f_agege40_w1[now_tob_f_agege40_w1]
            df.loc[tob_f_agege40_w1_idx, 'li_tob'] = True

        now_tob_m_agege40_w1 = np.random.choice([True, False],
                                                size=len(currently_not_tob_m_agege40_w1),
                                                p=[ri_tob_m_agege40_w1, 1 - ri_tob_m_agege40_w1])

        if now_tob_m_agege40_w1.sum():
            tob_m_agege40_w1_idx = currently_not_tob_m_agege40_w1[now_tob_m_agege40_w1]
            df.loc[tob_m_agege40_w1_idx, 'li_tob'] = True

        now_tob_f_agege40_w2 = np.random.choice([True, False],
                                                size=len(currently_not_tob_f_agege40_w2),
                                                p=[ri_tob_f_agege40_w2, 1 - ri_tob_f_agege40_w2])

        if now_tob_f_agege40_w2.sum():
            tob_f_agege40_w2_idx = currently_not_tob_f_agege40_w2[now_tob_f_agege40_w2]
            df.loc[tob_f_agege40_w2_idx, 'li_tob'] = True

        now_tob_m_agege40_w2 = np.random.choice([True, False],
                                                size=len(currently_not_tob_m_agege40_w2),
                                                p=[ri_tob_m_agege40_w2, 1 - ri_tob_m_agege40_w2])

        if now_tob_m_agege40_w2.sum():
            tob_m_agege40_w2_idx = currently_not_tob_m_agege40_w2[now_tob_m_agege40_w2]
            df.loc[tob_m_agege40_w2_idx, 'li_tob'] = True

        now_tob_f_agege40_w3 = np.random.choice([True, False],
                                                size=len(currently_not_tob_f_agege40_w3),
                                                p=[ri_tob_f_agege40_w3, 1 - ri_tob_f_agege40_w3])

        if now_tob_f_agege40_w3.sum():
            tob_f_agege40_w3_idx = currently_not_tob_f_agege40_w3[now_tob_f_agege40_w3]
            df.loc[tob_f_agege40_w3_idx, 'li_tob'] = True

        now_tob_m_agege40_w3 = np.random.choice([True, False],
                                                size=len(currently_not_tob_m_agege40_w3),
                                                p=[ri_tob_m_agege40_w3, 1 - ri_tob_m_agege40_w3])

        if now_tob_m_agege40_w3.sum():
            tob_m_agege40_w3_idx = currently_not_tob_m_agege40_w3[now_tob_m_agege40_w3]
            df.loc[tob_m_agege40_w3_idx, 'li_tob'] = True

        now_tob_f_agege40_w4 = np.random.choice([True, False],
                                                size=len(currently_not_tob_f_agege40_w4),
                                                p=[ri_tob_f_agege40_w4, 1 - ri_tob_f_agege40_w4])

        if now_tob_f_agege40_w4.sum():
            tob_f_agege40_w4_idx = currently_not_tob_f_agege40_w4[now_tob_f_agege40_w4]
            df.loc[tob_f_agege40_w4_idx, 'li_tob'] = True

        now_tob_m_agege40_w4 = np.random.choice([True, False],
                                                size=len(currently_not_tob_m_agege40_w4),
                                                p=[ri_tob_m_agege40_w4, 1 - ri_tob_m_agege40_w4])

        if now_tob_m_agege40_w4.sum():
            tob_m_agege40_w4_idx = currently_not_tob_m_agege40_w4[now_tob_m_agege40_w4]
            df.loc[tob_m_agege40_w4_idx, 'li_tob'] = True

        now_tob_f_agege40_w5 = np.random.choice([True, False],
                                                size=len(currently_not_tob_f_agege40_w5),
                                                p=[ri_tob_f_agege40_w5, 1 - ri_tob_f_agege40_w5])

        if now_tob_f_agege40_w5.sum():
            tob_f_agege40_w5_idx = currently_not_tob_f_agege40_w5[now_tob_f_agege40_w5]
            df.loc[tob_f_agege40_w5_idx, 'li_tob'] = True

        now_tob_m_agege40_w5 = np.random.choice([True, False],
                                                size=len(currently_not_tob_m_agege40_w5),
                                                p=[ri_tob_m_agege40_w5, 1 - ri_tob_m_agege40_w5])

        if now_tob_m_agege40_w5.sum():
            tob_m_agege40_w5_idx = currently_not_tob_m_agege40_w5[now_tob_m_agege40_w5]
            df.loc[tob_m_agege40_w5_idx, 'li_tob'] = True

        now_not_tob = np.random.choice([True, False], size=len(currently_tob),
                                       p=[self.r_not_tob, 1 - self.r_not_tob])

        if now_not_tob.sum():
            not_tob_idx = currently_tob[now_not_tob]
            df.loc[not_tob_idx, 'li_tob'] = False

    # transition to ex alc depends on sex

        currently_not_ex_alc_f = df.index[~df.li_ex_alc & df.is_alive & (df.sex == 'F') & (age.years >= 15)]
        currently_not_ex_alc_m = df.index[~df.li_ex_alc & df.is_alive & (df.sex == 'M') & (age.years >= 15)]
        currently_ex_alc = df.index[df.li_ex_alc & df.is_alive]

        ri_ex_alc_f = self.r_ex_alc*self.rr_ex_alc_f
        ri_ex_alc_m = self.r_ex_alc

        now_ex_alc_f = np.random.choice([True, False],
                                        size=len(currently_not_ex_alc_f),
                                        p=[ri_ex_alc_f, 1 - ri_ex_alc_f])
        if now_ex_alc_f.sum():
            ex_alc_f_idx = currently_not_ex_alc_f[now_ex_alc_f]
            df.loc[ex_alc_f_idx, 'li_ex_alc'] = True

        now_ex_alc_m = np.random.choice([True, False],
                                        size=len(currently_not_ex_alc_m),
                                        p=[ri_ex_alc_m, 1 - ri_ex_alc_m])
        if now_ex_alc_m.sum():
            ex_alc_m_idx = currently_not_ex_alc_m[now_ex_alc_m]
            df.loc[ex_alc_m_idx, 'li_ex_alc'] = True

        now_not_ex_alc = np.random.choice([True, False], size=len(currently_ex_alc),
                                          p=[self.r_not_ex_alc, 1 - self.r_not_ex_alc])
        if now_not_ex_alc.sum():
            not_ex_alc_idx = currently_ex_alc[now_not_ex_alc]
            df.loc[not_ex_alc_idx, 'li_ex_alc'] = False

    # transitions in mar stat

        curr_never_mar_index = df.index[df.is_alive & (age.years >= 15) & (age.years < 30) & (df.li_mar_stat == 1)]
        now_mar = np.random.choice([True, False], size=len(curr_never_mar_index), p=[self.r_mar, 1 - self.r_mar])
        if now_mar.sum():
            now_mar_index = curr_never_mar_index[now_mar]
            df.loc[now_mar_index, 'li_mar_stat'] = 2

        curr_mar_index = df.index[df.is_alive & (df.li_mar_stat == 2)]
        now_div_wid = np.random.choice([True, False], size=len(curr_mar_index), p=[self.r_div_wid, 1 - self.r_div_wid])
        if now_div_wid.sum():
            now_div_wid_index = curr_mar_index[now_div_wid]
            df.loc[now_div_wid_index, 'li_mar_stat'] = 3

        # updating of contraceptive status

        curr_not_on_con_idx = df.index[df.is_alive & (age.years >= 15) & (age.years < 50) & (df.sex == 'F') & ~df.li_on_con]
        now_on_con = np.random.choice([True, False], size=len(curr_not_on_con_idx), p=[self.r_contrac, 1 - self.r_contrac])
        if now_on_con.sum():
            now_on_con_index = curr_not_on_con_idx[now_on_con]
            df.loc[now_on_con_index, 'li_on_con'] = True

        curr_on_con_idx = df.index[df.is_alive & (age.years >= 15) & (age.years < 50) & (df.sex == 'F') & df.li_on_con]
        now_not_on_con = np.random.choice([True, False], size=len(curr_on_con_idx), p=[self.r_contrac_int, 1 - self.r_contrac_int])
        if now_not_on_con.sum():
            now_not_on_con_index = curr_on_con_idx[now_not_on_con]
            df.loc[now_not_on_con_index, 'li_on_con'] = False

        f_age50_idx = df.index[df.is_alive & (age.years == 50) & (df.sex == 'F') & df.li_on_con]
        df.loc[f_age50_idx, 'li_on_con'] = False

        curr_on_con_t_1_idx = df.index[df.is_alive & (age.years >= 15) & (age.years < 50) & (df.sex == 'F') & df.li_on_con & (df.li_con_t == 1)]
        df.loc[curr_on_con_t_1_idx, 'li_con_t'] = np.random.choice([1, 2, 3, 4, 5, 6], size=len(curr_on_con_t_1_idx), p=self.r_con_from_1)
        curr_on_con_t_2_idx = df.index[df.is_alive & (age.years >= 15) & (age.years < 50) & (df.sex == 'F') & df.li_on_con & (df.li_con_t == 2)]
        df.loc[curr_on_con_t_2_idx, 'li_con_t'] = np.random.choice([1, 2, 3, 4, 5, 6], size=len(curr_on_con_t_2_idx), p=self.r_con_from_2)
        curr_on_con_t_3_idx = df.index[df.is_alive & (age.years >= 15) & (age.years < 50) & (df.sex == 'F') & df.li_on_con & (df.li_con_t == 3)]
        df.loc[curr_on_con_t_3_idx, 'li_con_t'] = np.random.choice([1, 2, 3, 4, 5, 6], size=len(curr_on_con_t_3_idx), p=self.r_con_from_3)
        curr_on_con_t_4_idx = df.index[df.is_alive & (age.years >= 15) & (age.years < 50) & (df.sex == 'F') & df.li_on_con & (df.li_con_t == 4)]
        df.loc[curr_on_con_t_4_idx, 'li_con_t'] = np.random.choice([1, 2, 3, 4, 5, 6], size=len(curr_on_con_t_4_idx), p=self.r_con_from_4)
        curr_on_con_t_5_idx = df.index[df.is_alive & (age.years >= 15) & (age.years < 50) & (df.sex == 'F') & df.li_on_con & (df.li_con_t == 5)]
        df.loc[curr_on_con_t_5_idx, 'li_con_t'] = np.random.choice([1, 2, 3, 4, 5, 6], size=len(curr_on_con_t_5_idx), p=self.r_con_from_5)
        curr_on_con_t_6_idx = df.index[df.is_alive & (age.years >= 15) & (age.years < 50) & (df.sex == 'F') & df.li_on_con & (df.li_con_t == 6)]
        df.loc[curr_on_con_t_6_idx, 'li_con_t'] = np.random.choice([1, 2, 3, 4, 5, 6], size=len(curr_on_con_t_6_idx), p=self.r_con_from_6)

        # update education

#       m_age5_idx = df.index[(age.years == 5) & df.is_alive]
#       df.loc[m_age5_idx, 'li_ed_lev'] = np.random.choice([1, 2, 3], size=len(m_age5_idx), p=[1 - self.p_ed_primary, self.p_ed_primary, 0])
#       m_age5_in_ed_idx = df.index[(age.years == 5) & df.is_alive & (df.li_ed_lev == 2)]
#       df.loc[m_age5_in_ed_idx, 'li_in_ed'] = True


class LifestylesLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """comments...
        """
        # run this event every 3 month
        self.repeat = 3
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props

        age = population.age

        urban_alive = (df.is_alive & df.li_urban).sum()
        alive = df.is_alive.sum()

        ex_alc = (df.is_alive & (age.years >= 15) & df.li_ex_alc).sum()

        prop_urban = urban_alive / alive

        wealth1 = df.index[(df.li_wealth == 1) & df.is_alive]

        mar_stat_1_idx = df.index[df.is_alive & (df.li_mar_stat == 1) & (age.years >= 15)]
        mar_stat_2_idx = df.index[df.is_alive & (df.li_mar_stat == 2) & (age.years >= 15)]
        mar_stat_3_idx = df.index[df.is_alive & (df.li_mar_stat == 3) & (age.years >= 15)]

        m_idx = df.index[df.is_alive & (df.sex == 'M') & (age.years >= 15)]
        f_idx = df.index[df.is_alive & (df.sex == 'F') & (age.years >= 15)]
        ge15_idx = df.index[df.is_alive & (age.years >= 15)]

        mar_stat_1_agege60_idx = df.index[df.is_alive & (df.li_mar_stat == 1) & (age.years >= 60)]
        mar_stat_2_agege60_idx = df.index[df.is_alive & (df.li_mar_stat == 2) & (age.years >= 60)]
        mar_stat_3_agege60_idx = df.index[df.is_alive & (df.li_mar_stat == 3) & (age.years >= 60)]
        ge60_idx = df.index[df.is_alive & (age.years >= 60)]

        m_urban_ge15_overwt = df.index[(age.years >= 15) & (df.sex == 'M') & df.li_overwt & df.is_alive & df.li_urban]
        f_urban_ge15_overwt = df.index[(age.years >= 15) & (df.sex == 'F') & df.li_overwt & df.is_alive & df.li_urban]
        m_rural_ge15_overwt = df.index[(age.years >= 15) & (df.sex == 'M') & df.li_overwt & df.is_alive & ~df.li_urban]
        f_rural_ge15_overwt = df.index[(age.years >= 15) & (df.sex == 'F') & df.li_overwt & df.is_alive & ~df.li_urban]

        m_urban_ge15_low_ex = df.index[(age.years >= 15) & (df.sex == 'M') & df.li_low_ex & df.is_alive & df.li_urban]
        f_urban_ge15_low_ex = df.index[(age.years >= 15) & (df.sex == 'F') & df.li_low_ex & df.is_alive & df.li_urban]
        m_rural_ge15_low_ex = df.index[(age.years >= 15) & (df.sex == 'M') & df.li_low_ex & df.is_alive & ~df.li_urban]
        f_rural_ge15_low_ex = df.index[(age.years >= 15) & (df.sex == 'F') & df.li_low_ex & df.is_alive & ~df.li_urban]

        m_urban_ge15 = df.index[(age.years >= 15) & (df.sex == 'M') & df.li_urban & df.is_alive]
        f_urban_ge15 = df.index[(age.years >= 15) & (df.sex == 'F') & df.li_urban & df.is_alive]
        m_rural_ge15 = df.index[(age.years >= 15) & (df.sex == 'M') & ~df.li_urban & df.is_alive]
        f_rural_ge15 = df.index[(age.years >= 15) & (df.sex == 'F') & ~df.li_urban & df.is_alive]

        tob = df.index[df.li_tob & df.is_alive & (age.years >= 15)]

        m_age1519_w1_tob = df.index[df.li_tob & df.is_alive & (age.years >= 15) & (age.years < 20) & (df.sex == 'M')
                                    & (df.li_wealth == 1)]
        m_age2039_w1_tob = df.index[df.li_tob & df.is_alive & (age.years >= 20) & (age.years < 39) & (df.sex == 'M')
                                    & (df.li_wealth == 1)]
        m_agege40_w1_tob = df.index[df.li_tob & df.is_alive & (age.years >= 40) & (df.sex == 'M') & (df.li_wealth == 1)]
        f_age1519_w1_tob = df.index[df.li_tob & df.is_alive & (age.years >= 15) & (age.years < 20) & (df.sex == 'F')
                                    & (df.li_wealth == 1)]
        f_age2039_w1_tob = df.index[df.li_tob & df.is_alive & (age.years >= 20) & (age.years < 39) & (df.sex == 'F')
                                    & (df.li_wealth == 1)]
        f_agege40_w1_tob = df.index[df.li_tob & df.is_alive & (age.years >= 40) & (df.sex == 'F') & (df.li_wealth == 1)]
        m_age1519_w2_tob = df.index[df.li_tob & df.is_alive & (age.years >= 15) & (age.years < 20) & (df.sex == 'M')
                                    & (df.li_wealth == 2)]
        m_age2039_w2_tob = df.index[df.li_tob & df.is_alive & (age.years >= 20) & (age.years < 39) & (df.sex == 'M')
                                    & (df.li_wealth == 2)]
        m_agege40_w2_tob = df.index[df.li_tob & df.is_alive & (age.years >= 40) & (df.sex == 'M') & (df.li_wealth == 2)]
        f_age1519_w2_tob = df.index[df.li_tob & df.is_alive & (age.years >= 15) & (age.years < 20) & (df.sex == 'F')
                                    & (df.li_wealth == 2)]
        f_age2039_w2_tob = df.index[df.li_tob & df.is_alive & (age.years >= 20) & (age.years < 39) & (df.sex == 'F')
                                    & (df.li_wealth == 2)]
        f_agege40_w2_tob = df.index[df.li_tob & df.is_alive & (age.years >= 40) & (df.sex == 'F') & (df.li_wealth == 2)]
        m_age1519_w3_tob = df.index[df.li_tob & df.is_alive & (age.years >= 15) & (age.years < 20) & (df.sex == 'M')
                                    & (df.li_wealth == 3)]
        m_age2039_w3_tob = df.index[df.li_tob & df.is_alive & (age.years >= 20) & (age.years < 39) & (df.sex == 'M')
                                    & (df.li_wealth == 3)]
        m_agege40_w3_tob = df.index[df.li_tob & df.is_alive & (age.years >= 40) & (df.sex == 'M') & (df.li_wealth == 3)]
        f_age1519_w3_tob = df.index[df.li_tob & df.is_alive & (age.years >= 15) & (age.years < 20) & (df.sex == 'F')
                                    & (df.li_wealth == 3)]
        f_age2039_w3_tob = df.index[df.li_tob & df.is_alive & (age.years >= 20) & (age.years < 39) & (df.sex == 'F')
                                    & (df.li_wealth == 3)]
        f_agege40_w3_tob = df.index[df.li_tob & df.is_alive & (age.years >= 40) & (df.sex == 'F') & (df.li_wealth == 3)]
        m_age1519_w4_tob = df.index[df.li_tob & df.is_alive & (age.years >= 15) & (age.years < 20) & (df.sex == 'M')
                                    & (df.li_wealth == 4)]
        m_age2039_w4_tob = df.index[df.li_tob & df.is_alive & (age.years >= 20) & (age.years < 39) & (df.sex == 'M')
                                    & (df.li_wealth == 4)]
        m_agege40_w4_tob = df.index[df.li_tob & df.is_alive & (age.years >= 40) & (df.sex == 'M') & (df.li_wealth == 4)]
        f_age1519_w4_tob = df.index[df.li_tob & df.is_alive & (age.years >= 15) & (age.years < 20) & (df.sex == 'F')
                                    & (df.li_wealth == 4)]
        f_age2039_w4_tob = df.index[df.li_tob & df.is_alive & (age.years >= 20) & (age.years < 39) & (df.sex == 'F')
                                    & (df.li_wealth == 4)]
        f_agege40_w4_tob = df.index[df.li_tob & df.is_alive & (age.years >= 40) & (df.sex == 'F') & (df.li_wealth == 4)]
        m_age1519_w5_tob = df.index[df.li_tob & df.is_alive & (age.years >= 15) & (age.years < 20) & (df.sex == 'M')
                                    & (df.li_wealth == 5)]
        m_age2039_w5_tob = df.index[df.li_tob & df.is_alive & (age.years >= 20) & (age.years < 39) & (df.sex == 'M')
                                    & (df.li_wealth == 5)]
        m_agege40_w5_tob = df.index[df.li_tob & df.is_alive & (age.years >= 40) & (df.sex == 'M') & (df.li_wealth == 5)]
        f_age1519_w5_tob = df.index[df.li_tob & df.is_alive & (age.years >= 15) & (age.years < 20) & (df.sex == 'F')
                                    & (df.li_wealth == 5)]
        f_age2039_w5_tob = df.index[df.li_tob & df.is_alive & (age.years >= 20) & (age.years < 39) & (df.sex == 'F')
                                    & (df.li_wealth == 5)]
        f_agege40_w5_tob = df.index[df.li_tob & df.is_alive & (age.years >= 40) & (df.sex == 'F') & (df.li_wealth == 5)]
        m_age1519_w1 = df.index[df.is_alive & (age.years >= 15) & (age.years < 20) & (df.sex == 'M')
                                & (df.li_wealth == 1)]
        m_age2039_w1 = df.index[df.is_alive & (age.years >= 20) & (age.years < 39) & (df.sex == 'M')
                                & (df.li_wealth == 1)]
        m_agege40_w1 = df.index[df.is_alive & (age.years >= 40) & (df.sex == 'M') & (df.li_wealth == 1)]
        f_age1519_w1 = df.index[df.is_alive & (age.years >= 15) & (age.years < 20) & (df.sex == 'F')
                                & (df.li_wealth == 1)]
        f_age2039_w1 = df.index[df.is_alive & (age.years >= 20) & (age.years < 39) & (df.sex == 'F')
                                & (df.li_wealth == 1)]
        f_agege40_w1 = df.index[df.is_alive & (age.years >= 40) & (df.sex == 'F') & (df.li_wealth == 1)]
        m_age1519_w2 = df.index[df.is_alive & (age.years >= 15) & (age.years < 20) & (df.sex == 'M')
                                & (df.li_wealth == 2)]
        m_age2039_w2 = df.index[df.is_alive & (age.years >= 20) & (age.years < 39) & (df.sex == 'M')
                                & (df.li_wealth == 2)]
        m_agege40_w2 = df.index[df.is_alive & (age.years >= 40) & (df.sex == 'M') & (df.li_wealth == 2)]
        f_age1519_w2 = df.index[df.is_alive & (age.years >= 15) & (age.years < 20) & (df.sex == 'F')
                                & (df.li_wealth == 2)]
        f_age2039_w2 = df.index[df.is_alive & (age.years >= 20) & (age.years < 39) & (df.sex == 'F')
                                & (df.li_wealth == 2)]
        f_agege40_w2 = df.index[df.is_alive & (age.years >= 40) & (df.sex == 'F') & (df.li_wealth == 2)]
        m_age1519_w3 = df.index[df.is_alive & (age.years >= 15) & (age.years < 20) & (df.sex == 'M')
                                & (df.li_wealth == 3)]
        m_age2039_w3 = df.index[df.is_alive & (age.years >= 20) & (age.years < 39) & (df.sex == 'M')
                                & (df.li_wealth == 3)]
        m_agege40_w3 = df.index[df.is_alive & (age.years >= 40) & (df.sex == 'M') & (df.li_wealth == 3)]
        f_age1519_w3 = df.index[df.is_alive & (age.years >= 15) & (age.years < 20) & (df.sex == 'F')
                                & (df.li_wealth == 3)]
        f_age2039_w3 = df.index[df.is_alive & (age.years >= 20) & (age.years < 39) & (df.sex == 'F')
                                & (df.li_wealth == 3)]
        f_agege40_w3 = df.index[df.is_alive & (age.years >= 40) & (df.sex == 'F') & (df.li_wealth == 3)]
        m_age1519_w4 = df.index[df.is_alive & (age.years >= 15) & (age.years < 20) & (df.sex == 'M')
                                & (df.li_wealth == 4)]
        m_age2039_w4 = df.index[df.is_alive & (age.years >= 20) & (age.years < 39) & (df.sex == 'M')
                                & (df.li_wealth == 4)]
        m_agege40_w4 = df.index[df.is_alive & (age.years >= 40) & (df.sex == 'M') & (df.li_wealth == 4)]
        f_age1519_w4 = df.index[df.is_alive & (age.years >= 15) & (age.years < 20) & (df.sex == 'F')
                                & (df.li_wealth == 4)]
        f_age2039_w4 = df.index[df.is_alive & (age.years >= 20) & (age.years < 39) & (df.sex == 'F')
                                & (df.li_wealth == 4)]
        f_agege40_w4 = df.index[df.is_alive & (age.years >= 40) & (df.sex == 'F') & (df.li_wealth == 4)]
        m_age1519_w5 = df.index[df.is_alive & (age.years >= 15) & (age.years < 20) & (df.sex == 'M')
                                & (df.li_wealth == 5)]
        m_age2039_w5 = df.index[df.is_alive & (age.years >= 20) & (age.years < 39) & (df.sex == 'M')
                                & (df.li_wealth == 5)]
        m_agege40_w5 = df.index[df.is_alive & (age.years >= 40) & (df.sex == 'M') & (df.li_wealth == 5)]
        f_age1519_w5 = df.index[df.is_alive & (age.years >= 15) & (age.years < 20) & (df.sex == 'F')
                                & (df.li_wealth == 5)]
        f_age2039_w5 = df.index[df.is_alive & (age.years >= 20) & (age.years < 39) & (df.sex == 'F')
                                & (df.li_wealth == 5)]
        f_agege40_w5 = df.index[df.is_alive & (age.years >= 40) & (df.sex == 'F') & (df.li_wealth == 5)]

        f_ex_alc = (df.is_alive & (age.years >= 15) & (df.sex == 'F') & df.li_ex_alc).sum()
        m_ex_alc = (df.is_alive & (age.years >= 15) & (df.sex == 'M') & df.li_ex_alc).sum()

        n_m_ge15 = (df.is_alive & (age.years >= 15) & (df.sex == 'M')).sum()
        n_f_ge15 = (df.is_alive & (age.years >= 15) & (df.sex == 'F')).sum()
        n_ge15 = (df.is_alive & (age.years >= 15)).sum()

        n_f_age1550 = (df.is_alive & (age.years >= 15) & (age.years < 50) & (df.sex == 'F')).sum()
        n_f_age1550_on_con = (df.is_alive & (age.years >= 15) & (age.years < 50) & (df.sex == 'F') & df.li_on_con).sum()

        prop_f_1550_on_con = n_f_age1550_on_con / n_f_age1550

        self.module.store['alive'].append(alive)

        proportion_urban = urban_alive / (df.is_alive.sum())
        rural_alive = (df.is_alive & (~df.li_urban)).sum()

        mask = (df['li_date_trans_to_urban'] > self.sim.date - DateOffset(months=self.repeat))
        newly_urban_in_last_3mths = mask.sum()

        prop_m_urban_overwt = len(m_urban_ge15_overwt) / len(m_urban_ge15)
        prop_f_urban_overwt = len(f_urban_ge15_overwt) / len(f_urban_ge15)
        prop_m_rural_overwt = len(m_rural_ge15_overwt) / len(m_rural_ge15)
        prop_f_rural_overwt = len(f_rural_ge15_overwt) / len(f_rural_ge15)

        prop_m_urban_low_ex = len(m_urban_ge15_low_ex) / len(m_urban_ge15)
        prop_f_urban_low_ex = len(f_urban_ge15_low_ex) / len(f_urban_ge15)
        prop_m_rural_low_ex = len(m_rural_ge15_low_ex) / len(m_rural_ge15)
        prop_f_rural_low_ex = len(f_rural_ge15_low_ex) / len(f_rural_ge15)

        prop_wealth1 = len(wealth1) / alive
        prop_tob = len(tob) / n_ge15

        prop_f_ex_alc = f_ex_alc / n_f_ge15
        prop_m_ex_alc = m_ex_alc / n_m_ge15

        prop_mar_stat_1 = len(mar_stat_1_idx) / len(ge15_idx)
        prop_mar_stat_2 = len(mar_stat_2_idx) / len(ge15_idx)
        prop_mar_stat_3 = len(mar_stat_3_idx) / len(ge15_idx)

        prop_mar_stat_1_agege60 = len(mar_stat_1_agege60_idx) / len(ge60_idx)
        prop_mar_stat_2_agege60 = len(mar_stat_2_agege60_idx) / len(ge60_idx)
        prop_mar_stat_3_agege60 = len(mar_stat_3_agege60_idx) / len(ge60_idx)

        prop_m_age1519_w1_tob = len(m_age1519_w1_tob) / len(m_age1519_w1)
        prop_f_age1519_w1_tob = len(f_age1519_w1_tob) / len(f_age1519_w1)
        prop_m_age2039_w1_tob = len(m_age2039_w1_tob) / len(m_age2039_w1)
        prop_f_age2039_w1_tob = len(f_age2039_w1_tob) / len(f_age2039_w1)
        prop_m_agege40_w1_tob = len(m_agege40_w1_tob) / len(m_agege40_w1)
        prop_f_agege40_w1_tob = len(f_agege40_w1_tob) / len(f_agege40_w1)
        prop_m_age1519_w2_tob = len(m_age1519_w2_tob) / len(m_age1519_w2)
        prop_f_age1519_w2_tob = len(f_age1519_w2_tob) / len(f_age1519_w2)
        prop_m_age2039_w2_tob = len(m_age2039_w2_tob) / len(m_age2039_w2)
        prop_f_age2039_w2_tob = len(f_age2039_w2_tob) / len(f_age2039_w2)
        prop_m_agege40_w2_tob = len(m_agege40_w2_tob) / len(m_agege40_w2)
        prop_f_agege40_w2_tob = len(f_agege40_w2_tob) / len(f_agege40_w2)
        prop_m_age1519_w3_tob = len(m_age1519_w3_tob) / len(m_age1519_w3)
        prop_f_age1519_w3_tob = len(f_age1519_w3_tob) / len(f_age1519_w3)
        prop_m_age2039_w3_tob = len(m_age2039_w3_tob) / len(m_age2039_w3)
        prop_f_age2039_w3_tob = len(f_age2039_w3_tob) / len(f_age2039_w3)
        prop_m_agege40_w3_tob = len(m_agege40_w3_tob) / len(m_agege40_w3)
        prop_f_agege40_w3_tob = len(f_agege40_w3_tob) / len(f_agege40_w3)
        prop_m_age1519_w4_tob = len(m_age1519_w4_tob) / len(m_age1519_w4)
        prop_f_age1519_w4_tob = len(f_age1519_w4_tob) / len(f_age1519_w4)
        prop_m_age2039_w4_tob = len(m_age2039_w4_tob) / len(m_age2039_w4)
        prop_f_age2039_w4_tob = len(f_age2039_w4_tob) / len(f_age2039_w4)
        prop_m_agege40_w4_tob = len(m_agege40_w4_tob) / len(m_agege40_w4)
        prop_f_agege40_w4_tob = len(f_agege40_w4_tob) / len(f_agege40_w4)
        prop_m_age1519_w5_tob = len(m_age1519_w5_tob) / len(m_age1519_w5)
        prop_f_age1519_w5_tob = len(f_age1519_w5_tob) / len(f_age1519_w5)
        prop_m_age2039_w5_tob = len(m_age2039_w5_tob) / len(m_age2039_w5)
        prop_f_age2039_w5_tob = len(f_age2039_w5_tob) / len(f_age2039_w5)
        prop_m_agege40_w5_tob = len(m_agege40_w5_tob) / len(m_agege40_w5)
        prop_f_agege40_w5_tob = len(f_agege40_w5_tob) / len(f_agege40_w5)

        self.module.o_prop_mar_stat_1['prop_mar_stat_1'].append(prop_mar_stat_1)
        self.module.o_prop_mar_stat_2['prop_mar_stat_2'].append(prop_mar_stat_2)
        self.module.o_prop_mar_stat_3['prop_mar_stat_3'].append(prop_mar_stat_3)
        self.module.o_prop_mar_stat_1_agege60['prop_mar_stat_1_agege60'].append(prop_mar_stat_1_agege60)
        self.module.o_prop_mar_stat_2_agege60['prop_mar_stat_2_agege60'].append(prop_mar_stat_2_agege60)
        self.module.o_prop_mar_stat_3_agege60['prop_mar_stat_3_agege60'].append(prop_mar_stat_3_agege60)

        self.module.o_prop_m_urban_overwt['prop_m_urban_overwt'].append(prop_m_urban_overwt)
        self.module.o_prop_f_urban_overwt['prop_f_urban_overwt'].append(prop_f_urban_overwt)
        self.module.o_prop_m_rural_overwt['prop_m_rural_overwt'].append(prop_m_rural_overwt)
        self.module.o_prop_f_rural_overwt['prop_f_rural_overwt'].append(prop_f_rural_overwt)

        self.module.o_prop_m_urban_low_ex['prop_m_urban_low_ex'].append(prop_m_urban_low_ex)
        self.module.o_prop_f_urban_low_ex['prop_f_urban_low_ex'].append(prop_f_urban_low_ex)
        self.module.o_prop_m_rural_low_ex['prop_m_rural_low_ex'].append(prop_m_rural_low_ex)
        self.module.o_prop_f_rural_low_ex['prop_f_rural_low_ex'].append(prop_f_rural_low_ex)

        self.module.o_prop_urban['prop_urban'].append(prop_urban)
        self.module.o_prop_wealth1['prop_wealth1'].append(prop_wealth1)
        self.module.o_prop_tob['prop_tob'].append(prop_tob)
        self.module.o_prop_m_ex_alc['prop_m_ex_alc'].append(prop_m_ex_alc)
        self.module.o_prop_f_ex_alc['prop_f_ex_alc'].append(prop_f_ex_alc)

        self.module.o_prop_m_age1519_w1_tob['prop_m_age1519_w1_tob'].append(prop_m_age1519_w1_tob)
        self.module.o_prop_m_age2039_w1_tob['prop_m_age2039_w1_tob'].append(prop_m_age2039_w1_tob)
        self.module.o_prop_m_agege40_w1_tob['prop_m_agege40_w1_tob'].append(prop_m_agege40_w1_tob)
        self.module.o_prop_f_age1519_w1_tob['prop_f_age1519_w1_tob'].append(prop_f_age1519_w1_tob)
        self.module.o_prop_f_age2039_w1_tob['prop_f_age2039_w1_tob'].append(prop_f_age2039_w1_tob)
        self.module.o_prop_f_agege40_w1_tob['prop_f_agege40_w1_tob'].append(prop_f_agege40_w1_tob)
        self.module.o_prop_m_age1519_w2_tob['prop_m_age1519_w2_tob'].append(prop_m_age1519_w2_tob)
        self.module.o_prop_m_age2039_w2_tob['prop_m_age2039_w2_tob'].append(prop_m_age2039_w2_tob)
        self.module.o_prop_m_agege40_w2_tob['prop_m_agege40_w2_tob'].append(prop_m_agege40_w2_tob)
        self.module.o_prop_f_age1519_w2_tob['prop_f_age1519_w2_tob'].append(prop_f_age1519_w2_tob)
        self.module.o_prop_f_age2039_w2_tob['prop_f_age2039_w2_tob'].append(prop_f_age2039_w2_tob)
        self.module.o_prop_f_agege40_w2_tob['prop_f_agege40_w2_tob'].append(prop_f_agege40_w2_tob)
        self.module.o_prop_m_age1519_w3_tob['prop_m_age1519_w3_tob'].append(prop_m_age1519_w3_tob)
        self.module.o_prop_m_age2039_w3_tob['prop_m_age2039_w3_tob'].append(prop_m_age2039_w3_tob)
        self.module.o_prop_m_agege40_w3_tob['prop_m_agege40_w3_tob'].append(prop_m_agege40_w3_tob)
        self.module.o_prop_f_age1519_w3_tob['prop_f_age1519_w3_tob'].append(prop_f_age1519_w3_tob)
        self.module.o_prop_f_age2039_w3_tob['prop_f_age2039_w3_tob'].append(prop_f_age2039_w3_tob)
        self.module.o_prop_f_agege40_w3_tob['prop_f_agege40_w3_tob'].append(prop_f_agege40_w3_tob)
        self.module.o_prop_m_age1519_w4_tob['prop_m_age1519_w4_tob'].append(prop_m_age1519_w4_tob)
        self.module.o_prop_m_age2039_w4_tob['prop_m_age2039_w4_tob'].append(prop_m_age2039_w4_tob)
        self.module.o_prop_m_agege40_w4_tob['prop_m_agege40_w4_tob'].append(prop_m_agege40_w4_tob)
        self.module.o_prop_f_age1519_w4_tob['prop_f_age1519_w4_tob'].append(prop_f_age1519_w4_tob)
        self.module.o_prop_f_age2039_w4_tob['prop_f_age2039_w4_tob'].append(prop_f_age2039_w4_tob)
        self.module.o_prop_f_agege40_w4_tob['prop_f_agege40_w4_tob'].append(prop_f_agege40_w4_tob)
        self.module.o_prop_m_age1519_w5_tob['prop_m_age1519_w5_tob'].append(prop_m_age1519_w5_tob)
        self.module.o_prop_m_age2039_w5_tob['prop_m_age2039_w5_tob'].append(prop_m_age2039_w5_tob)
        self.module.o_prop_m_agege40_w5_tob['prop_m_agege40_w5_tob'].append(prop_m_agege40_w5_tob)
        self.module.o_prop_f_age1519_w5_tob['prop_f_age1519_w5_tob'].append(prop_f_age1519_w5_tob)
        self.module.o_prop_f_age2039_w5_tob['prop_f_age2039_w5_tob'].append(prop_f_age2039_w5_tob)
        self.module.o_prop_f_agege40_w5_tob['prop_f_agege40_w5_tob'].append(prop_f_agege40_w5_tob)

        self.module.o_prop_f_1550_on_con['prop_f_1550_on_con'].append(prop_f_1550_on_con)

        wealth_count_alive = df.loc[df.is_alive, 'li_wealth'].value_counts()

        print('%s lifestyle n_m_ge15:%d , prop_wealth1 %f, prop_f_1550_on_con  %f, prop_mar_stat_1 %f,'
              'prop_mar_stat_2 %f, prop_mar_stat_3 %f, prop_m_urban_overwt:%f , newly urban: %d, '
              'wealth: %s' %
              (self.sim.date, n_m_ge15, prop_wealth1, prop_f_1550_on_con, prop_mar_stat_1,  prop_mar_stat_2,
               prop_mar_stat_3, prop_m_urban_overwt,
               newly_urban_in_last_3mths,
               list(wealth_count_alive)),
              flush=True)





