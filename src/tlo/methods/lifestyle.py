"""
Lifestyle module

Documentation: 04 - Methods Repository/Method_Lifestyle.xlsx
"""
import logging

import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Lifestyle(Module):
    """
    Lifestyle module provides properties that are used by all disease modules if they are affected
    by urban/rural, wealth, tobacco usage etc.
    """

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
        'r_tob': Parameter(Types.REAL, 'probability per 3 mths of change from not tob to tob if male '
                                       'age 15-19 wealth level 1'),
        'r_not_tob': Parameter(Types.REAL, 'probability per 3 mths of change from tob to not tob'),
        'rr_tob_age2039': Parameter(Types.REAL, 'risk ratio for tob if age 20-39 compared with 15-19'),
        'rr_tob_agege40': Parameter(Types.REAL, 'risk ratio for tob if age ge 40 compared with 15-19'),
        'rr_tob_f': Parameter(Types.REAL, 'risk ratio for tob if female'),
        'rr_tob_wealth': Parameter(Types.REAL, 'risk ratio for tob per 1 higher wealth level '
                                               '(higher wealth level = lower wealth)'),
        'r_ex_alc': Parameter(Types.REAL, 'probability per 3 mths of change from not ex alc to ex alc'),
        'r_not_ex_alc': Parameter(Types.REAL, 'probability per 3 mths of change from excess alc to not excess alc'),
        'rr_ex_alc_f': Parameter(Types.REAL, 'risk ratio for becoming ex alc if female rather than male'),
        'init_p_urban': Parameter(Types.REAL, 'proportion urban at baseline'),
        'init_p_wealth_urban': Parameter(Types.LIST, 'List of probabilities of category given urban'),
        'init_p_wealth_rural': Parameter(Types.LIST, 'List of probabilities of category given rural'),
        'init_dist_mar_stat_age1320': Parameter(Types.LIST, 'proportions never, current, div_wid age 15-20 baseline'),
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
        'r_contrac_int': Parameter(Types.REAL, 'prob per 3 months of interrupting or stopping contraception '
                                               '(note current method of contrac is a different property'),
        'r_con_from_1': Parameter(Types.LIST, 'probs per 3 months of moving from contraception method 1'),
        'r_con_from_2': Parameter(Types.LIST, 'probs per 3 months of moving from contraception method 2'),
        'r_con_from_3': Parameter(Types.LIST, 'probs per 3 months of moving from contraception method 3'),
        'r_con_from_4': Parameter(Types.LIST, 'probs per 3 months of moving from contraception method 4'),
        'r_con_from_5': Parameter(Types.LIST, 'probs per 3 months of moving from contraception method 5'),
        'r_con_from_6': Parameter(Types.LIST, 'probs per 3 months of moving from contraception method 6'),
        'r_stop_ed': Parameter(Types.REAL, 'prob per 3 months of stopping education if wealth level 5'),
        'rr_stop_ed_lower_wealth': Parameter(Types.REAL, 'relative rate of stopping education per '
                                                         '1 lower wealth quintile'),
        'p_ed_primary': Parameter(Types.REAL, 'probability at age 5 that start primary education if wealth level 5'),
        'rp_ed_primary_higher_wealth': Parameter(Types.REAL, 'relative probability of starting school per 1 '
                                                             'higher wealth level'),
        'p_ed_secondary': Parameter(Types.REAL, 'probability at age 13 that start secondary education at 13 '
                                                'if in primary education and wealth level 5'),
        'rp_ed_secondary_higher_wealth': Parameter(Types.REAL, 'relative probability of starting secondary '
                                                               'school per 1 higher wealth level'),
        'init_age2030_w5_some_ed': Parameter(Types.REAL, 'proportions of low wealth 20-30 year olds with some '
                                                         'education at baseline'),
        'init_rp_some_ed_age0513': Parameter(Types.REAL, 'rel prev of some education at baseline if age 1520'),
        'init_rp_some_ed_age1320': Parameter(Types.REAL, 'rel prev of some education at baseline if age 1520'),
        'init_rp_some_ed_age2030': Parameter(Types.REAL, 'rel prev of some education at baseline if age 2030'),
        'init_rp_some_ed_age3040': Parameter(Types.REAL, 'rel prev of some education at baseline if age 3040'),
        'init_rp_some_ed_age4050': Parameter(Types.REAL, 'rel prev of some education at baseline if age 4050'),
        'init_rp_some_ed_age5060': Parameter(Types.REAL, 'rel prev of some education at baseline if age 5060'),
        'init_rp_some_ed_per_higher_wealth': Parameter(Types.REAL, 'relative prevalence of some education at baseline '
                                                                   'per higher wealth level'),
        'init_prop_age2030_w5_some_ed_sec': Parameter(Types.REAL,
                                                      'proportion of low wealth aged 20-30 with some education who '
                                                      'have secondary education at baseline'),
        'init_rp_some_ed_sec_age1320': Parameter(Types.REAL, 'relative prevalence of sec_ed for age 15-20'),
        'init_rp_some_ed_sec_age3040': Parameter(Types.REAL, 'relative prevalence of sec_ed for age 30-40'),
        'init_rp_some_ed_sec_age4050': Parameter(Types.REAL, 'relative prevalence of sec_ed for age 40-50'),
        'init_rp_some_ed_sec_age5060': Parameter(Types.REAL, 'relative prevalence of sec_ed for age 50-60'),
        'init_rp_some_ed_sec_agege60': Parameter(Types.REAL, 'relative prevalence of sec_ed for age 60+'),
        'init_rp_some_ed_sec_per_higher_wealth': Parameter(Types.REAL,
                                                           'relative prevalence of sec_ed per higher wealth level'),
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
        'li_mar_stat': Property(Types.CATEGORICAL,
                                'marital status - never, current, past (widowed or divorced)',
                                categories=[1, 2, 3]),
        'li_on_con': Property(Types.BOOL, 'on contraceptive'),
        'li_con_t': Property(Types.CATEGORICAL, 'contraceptive type', categories=[1, 2, 3, 4, 5, 6]),
        'li_in_ed': Property(Types.BOOL, 'currently in education'),
        'li_ed_lev': Property(Types.CATEGORICAL, 'education level achieved as of now', categories=[1, 2, 3]),
    }

    def read_parameters(self, data_folder):
        """Setup parameters used by the lifestyle module
        """
        p = self.parameters
        p['r_urban'] = 0.002
        p['r_rural'] = 0.0001
        p['r_overwt'] = 0.0025
        p['r_not_overwt'] = 0.001
        p['rr_overwt_f'] = 0.8
        p['rr_overwt_urban'] = 1.5
        p['r_low_ex'] = 0.001
        p['r_not_low_ex'] = 0.0001
        p['rr_low_ex_f'] = 0.6
        p['rr_low_ex_urban'] = 2.0
        p['r_tob'] = 0.0004
        p['r_not_tob'] = 0.000
        p['rr_tob_f'] = 0.1
        p['rr_tob_age2039'] = 1.2
        p['rr_tob_agege40'] = 1.5
        p['rr_tob_wealth'] = 1.3
        p['r_ex_alc'] = 0.003
        p['r_not_ex_alc'] = 0.000
        p['rr_ex_alc_f'] = 0.07
        p['init_p_urban'] = 0.17
        p['init_p_wealth_urban'] = [0.75, 0.16, 0.05, 0.02, 0.02]
        p['init_p_wealth_rural'] = [0.11, 0.21, 0.22, 0.23, 0.23]
        p['init_p_overwt_agelt15'] = 0.0
        p['init_p_ex_alc_m'] = 0.15
        p['init_p_ex_alc_f'] = 0.01
        p['init_dist_mar_stat_age1520'] = [0.70, 0.30, 0.00]
        p['init_dist_mar_stat_age2030'] = [0.15, 0.80, 0.05]
        p['init_dist_mar_stat_age3040'] = [0.05, 0.70, 0.25]
        p['init_dist_mar_stat_age4050'] = [0.03, 0.50, 0.47]
        p['init_dist_mar_stat_age5060'] = [0.03, 0.30, 0.67]
        p['init_dist_mar_stat_agege60'] = [0.03, 0.20, 0.77]
        p['r_mar'] = 0.03
        p['r_div_wid'] = 0.01
        p['init_p_on_contrac'] = 0.30
        p['init_dist_con_t'] = [0.17, 0.17, 0.17, 0.17, 0.17, 0.15]
        p['r_contrac'] = 0.05
        p['r_contrac_int'] = 0.1
        p['r_con_from_1'] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        p['r_con_from_2'] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        p['r_con_from_3'] = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        p['r_con_from_4'] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        p['r_con_from_5'] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        p['r_con_from_6'] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        p['r_stop_ed'] = 0.001
        p['rr_stop_ed_lower_wealth'] = 1.5
        p['p_ed_primary'] = 0.94
        p['rp_ed_primary_higher_wealth'] = 1.01
        p['p_ed_secondary'] = 0.20
        p['rp_ed_secondary_higher_wealth'] = 1.45
        p['init_age2030_w5_some_ed'] = 0.97
        p['init_rp_some_ed_age0513'] = 1.01
        p['init_rp_some_ed_age1320'] = 1.00
        p['init_rp_some_ed_age3040'] = 1.00
        p['init_rp_some_ed_age4050'] = 0.99
        p['init_rp_some_ed_age5060'] = 0.99
        p['init_rp_some_ed_agege60'] = 0.98
        p['init_rp_some_ed_per_higher_wealth'] = 1.005
        p['init_prop_age2030_w5_some_ed_sec'] = 0.20
        p['init_rp_some_ed_sec_age1320'] = 1.00
        p['init_rp_some_ed_sec_age3040'] = 0.90
        p['init_rp_some_ed_sec_age4050'] = 0.85
        p['init_rp_some_ed_sec_age5060'] = 0.80
        p['init_rp_some_ed_sec_agege60'] = 0.75
        p['init_rp_some_ed_sec_per_higher_wealth'] = 1.48

    def initialise_population(self, population):
        """Set our property values for the initial population.
        :param population: the population of individuals
        """
        df = population.props  # a shortcut to the data-frame storing data for individuals
        p = self.parameters
        rng = self.rng

        df['li_urban'] = False  # default: all individuals rural
        df['li_date_trans_to_urban'] = pd.NaT
        df['li_wealth'].values[:] = 3  # default: all individuals wealth 3
        df['li_overwt'] = False  # default: all not overwt
        df['li_low_ex'] = False  # default all not low ex
        df['li_tob'] = False  # default all not tob
        df['li_ex_alc'] = False  # default all not ex alc
        df['li_mar_stat'].values[:] = 1  # default: all individuals never married
        df['li_on_con'] = False  # default: all not on contraceptives

        # default: call contraceptive type 1, but when li_on_con = False this property becomes most
        # recent contraceptive used
        df['li_con_t'].values[:] = 1

        df['li_in_ed'] = False   # default: not in education
        df['li_ed_lev'].values[:] = 1   # default: education level = 1 - no education

        agelt15_index = df.index[df.is_alive & (df.age_years < 15)]

        # urban
        # randomly selected some individuals as urban
        initial_urban = p['init_p_urban']
        df['li_urban'] = (rng.random_sample(size=len(df)) < initial_urban)

        # get the indices of all individuals who are urban
        urban_index = df.index[df.is_alive & df.li_urban]
        # randomly sample wealth category according to urban wealth probs and assign to urban ind.
        df.loc[urban_index, 'li_wealth'] = rng.choice([1, 2, 3, 4, 5],
                                                      size=len(urban_index),
                                                      p=p['init_p_wealth_urban'])

        # get the indicies of all individual who are rural (i.e. not urban)
        rural_index = df.index[df.is_alive & ~df.li_urban]
        df.loc[rural_index, 'li_wealth'] = rng.choice([1, 2, 3, 4, 5],
                                                      size=len(rural_index),
                                                      p=p['init_p_wealth_rural'])

        # get indices of all individuals over 15 years
        gte_15 = df.index[df.is_alive & (df.age_years >= 15)]

        # overwt;
        overweight_lookup = pd.DataFrame(data=[('M', True, 0.46),
                                               ('M', False, 0.27),
                                               ('F', True, 0.32),
                                               ('F', False, 0.17)],
                                         columns=['sex', 'is_urban', 'p_ow'])

        overweight_probs = df.loc[gte_15, ['sex', 'li_urban']].merge(overweight_lookup,
                                                                     left_on=['sex', 'li_urban'],
                                                                     right_on=['sex', 'is_urban'],
                                                                     how='inner')['p_ow']
        assert len(overweight_probs) == len(gte_15)

        random_draw = rng.random_sample(size=len(gte_15))
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
                                                                 how='inner')['p_low_ex']
        assert len(low_ex_probs) == len(gte_15)

        random_draw = rng.random_sample(size=len(gte_15))
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

        # join the population-with-age dataframe with the tobacco use lookup table (join on sex and age_range)
        tob_probs = df.loc[gte_15].merge(tob_lookup,
                                         left_on=['sex', 'age_range'],
                                         right_on=['sex', 'age_range'],
                                         how='inner')
        assert len(gte_15) == len(tob_probs)

        # each individual has a baseline probability
        # multiply this probability by the wealth level. wealth is a category, so convert to integer
        tob_probs = pd.to_numeric(tob_probs['li_wealth']) * tob_probs['p_tob']

        # we now have the probability of tobacco use for each individual where age >= 15
        # draw a random number between 0 and 1 for all of them
        random_draw = rng.random_sample(size=len(gte_15))

        # decide on tobacco use based on the individual probability is greater than random draw
        # this is a list of True/False. assign to li_tob
        df.loc[gte_15, 'li_tob'] = (random_draw < tob_probs.values)

        # ex alc;
        m_agege15_index = df.index[df.is_alive & (df.age_years >= 15) & (df.sex == 'M')]
        f_agege15_index = df.index[df.is_alive & (df.age_years >= 15) & (df.sex == 'F')]

        df.loc[m_agege15_index, 'li_ex_alc'] = rng.random_sample(size=len(m_agege15_index)) < p['init_p_ex_alc_m']
        df.loc[f_agege15_index, 'li_ex_alc'] = rng.random_sample(size=len(f_agege15_index)) < p['init_p_ex_alc_f']

        # mar stat (marital status)

        age1520_index = df.index[(df.age_years >= 15) & (df.age_years < 20) & df.is_alive]
        age2030_index = df.index[(df.age_years >= 20) & (df.age_years < 30) & df.is_alive]
        age3040_index = df.index[(df.age_years >= 30) & (df.age_years < 40) & df.is_alive]
        age4050_index = df.index[(df.age_years >= 40) & (df.age_years < 50) & df.is_alive]
        age5060_index = df.index[(df.age_years >= 50) & (df.age_years < 60) & df.is_alive]
        agege60_index = df.index[(df.age_years >= 60) & df.is_alive]

        df.loc[age1520_index, 'li_mar_stat'] = rng.choice([1, 2, 3], size=len(age1520_index),
                                                          p=p['init_dist_mar_stat_age1520'])
        df.loc[age2030_index, 'li_mar_stat'] = rng.choice([1, 2, 3], size=len(age2030_index),
                                                          p=p['init_dist_mar_stat_age2030'])
        df.loc[age3040_index, 'li_mar_stat'] = rng.choice([1, 2, 3], size=len(age3040_index),
                                                          p=p['init_dist_mar_stat_age3040'])
        df.loc[age4050_index, 'li_mar_stat'] = rng.choice([1, 2, 3], size=len(age4050_index),
                                                          p=p['init_dist_mar_stat_age4050'])
        df.loc[age5060_index, 'li_mar_stat'] = rng.choice([1, 2, 3], size=len(age5060_index),
                                                          p=p['init_dist_mar_stat_age5060'])
        df.loc[agege60_index, 'li_mar_stat'] = rng.choice([1, 2, 3], size=len(agege60_index),
                                                          p=p['init_dist_mar_stat_agege60'])

        # li_on_con (contraception)

        f_age1550_idx = df.index[df.age_years.between(15, 49) & df.is_alive & (df.sex == 'F')]
        df.loc[f_age1550_idx, 'li_on_con'] = (rng.random_sample(size=len(f_age1550_idx)) < p['init_p_on_contrac'])

        f_age1550_on_con_idx = df.index[(df.age_years >= 15) & (df.age_years < 50) & df.is_alive &
                                        (df.sex == 'F') & df.li_on_con]
        df.loc[f_age1550_on_con_idx, 'li_con_t'] = rng.choice([1, 2, 3, 4, 5, 6],
                                                              size=len(f_age1550_on_con_idx),
                                                              p=p['init_dist_con_t'])

        # education (li_in_ed and li_ed_lev)

        age_ge5_idx = df.index[(df.age_years >= 5) & df.is_alive]

        # calculate the probability of education for all individuals over 5 years old
        eff_prob_some_ed = pd.Series(self.init_age2030_w5_some_ed, index=age_ge5_idx)

        # adjust probability of some education based on age
        eff_prob_some_ed.loc[df.age_years < 13] *= self.init_rp_some_ed_age0513
        eff_prob_some_ed.loc[(df.age_years >= 13) & (df.age_years < 20)] *= self.init_rp_some_ed_age1320
        eff_prob_some_ed.loc[(df.age_years >= 30) & (df.age_years < 40)] *= self.init_rp_some_ed_age3040
        eff_prob_some_ed.loc[(df.age_years >= 40) & (df.age_years < 50)] *= self.init_rp_some_ed_age4050
        eff_prob_some_ed.loc[(df.age_years >= 50) & (df.age_years < 60)] *= self.init_rp_some_ed_age5060
        eff_prob_some_ed.loc[(df.age_years >= 60)] *= self.init_rp_some_ed_agege60

        # adjust probability of some education based on wealth
        eff_prob_some_ed.loc[df.li_wealth == 4] *= self.init_rp_some_ed_per_higher_wealth
        eff_prob_some_ed.loc[df.li_wealth == 3] *= self.init_rp_some_ed_per_higher_wealth**2
        eff_prob_some_ed.loc[df.li_wealth == 2] *= self.init_rp_some_ed_per_higher_wealth**3
        eff_prob_some_ed.loc[df.li_wealth == 1] *= self.init_rp_some_ed_per_higher_wealth**4

        # calculate baseline of education level 3, and adjust for age and wealth
        eff_prob_ed_lev_3 = pd.Series(self.init_prop_age2030_w5_some_ed_sec, index=age_ge5_idx)

        eff_prob_ed_lev_3.loc[(df.age_years < 13)] *= 0
        eff_prob_ed_lev_3.loc[(df.age_years >= 13) & (df.age_years < 20)] *= self.init_rp_some_ed_sec_age1320
        eff_prob_ed_lev_3.loc[(df.age_years >= 30) & (df.age_years < 40)] *= self.init_rp_some_ed_sec_age3040
        eff_prob_ed_lev_3.loc[(df.age_years >= 40) & (df.age_years < 50)] *= self.init_rp_some_ed_sec_age4050
        eff_prob_ed_lev_3.loc[(df.age_years >= 50) & (df.age_years < 60)] *= self.init_rp_some_ed_sec_age5060
        eff_prob_ed_lev_3.loc[(df.age_years >= 60)] *= self.init_rp_some_ed_sec_agege60
        eff_prob_ed_lev_3.loc[df.li_wealth == 4] *= self.init_rp_some_ed_sec_per_higher_wealth
        eff_prob_ed_lev_3.loc[df.li_wealth == 3] *= self.init_rp_some_ed_sec_per_higher_wealth**2
        eff_prob_ed_lev_3.loc[df.li_wealth == 2] *= self.init_rp_some_ed_sec_per_higher_wealth**3
        eff_prob_ed_lev_3.loc[df.li_wealth == 1] *= self.init_rp_some_ed_sec_per_higher_wealth**4

        random_draw_01 = pd.Series(rng.random_sample(size=len(age_ge5_idx)), index=age_ge5_idx)

        dfx = pd.concat([eff_prob_ed_lev_3, eff_prob_some_ed, random_draw_01], axis=1)
        dfx.columns = ['eff_prob_ed_lev_3', 'eff_prob_some_ed', 'random_draw_01']

        dfx['p_ed_lev_1'] = 1 - dfx['eff_prob_some_ed']
        dfx['p_ed_lev_3'] = dfx['eff_prob_ed_lev_3']
        dfx['cut_off_ed_levl_3'] = 1 - dfx['eff_prob_ed_lev_3']

        dfx['li_ed_lev'] = 2
        dfx.loc[dfx['cut_off_ed_levl_3'] < random_draw_01, 'li_ed_lev'] = 3
        dfx.loc[dfx['p_ed_lev_1'] > random_draw_01, 'li_ed_lev'] = 1

        df.loc[age_ge5_idx, 'li_ed_lev'] = dfx['li_ed_lev']

        df.loc[(df.age_years >= 5) & (df.age_years < 13) & (df['li_ed_lev'] == 1) & df.is_alive, 'li_in_ed'] = False
        df.loc[(df.age_years >= 5) & (df.age_years < 13) & (df['li_ed_lev'] == 2) & df.is_alive, 'li_in_ed'] = True
        df.loc[(df.age_years >= 13) & (df.age_years < 20) & (df['li_ed_lev'] == 3) & df.is_alive, 'li_in_ed'] = True

    def initialise_simulation(self, sim):
        """Add lifestyle events to the simulation
        """
        event = LifestyleEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=3))

        event = LifestylesLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=0))

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """

        df = self.sim.population.props

        df.at[child_id, 'li_urban'] = df.at[mother_id, 'li_urban']
        df.at[child_id, 'li_date_trans_to_urban'] = pd.NaT
        df.at[child_id, 'li_wealth'] = df.at[mother_id, 'li_wealth']
        df.at[child_id, 'li_overwt'] = False
        df.at[child_id, 'li_low_ex'] = False
        df.at[child_id, 'li_tob'] = False
        df.at[child_id, 'li_ex_alc'] = False
        df.at[child_id, 'li_mar_stat'] = 1
        df.at[child_id, 'li_on_con'] = False
        df.at[child_id, 'li_con_t'] = 1
        df.at[child_id, 'li_in_ed'] = False
        df.at[child_id, 'li_ed_lev'] = 1


class LifestyleEvent(RegularEvent, PopulationScopeEventMixin):
    """
    Regular event that updates all lifestyle properties for population
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
        df = population.props
        m = self.module

        # -------------------- URBAN-RURAL STATUS --------------------------------------------------

        # get index of current urban/rural status
        currently_rural = df.index[~df.li_urban & df.is_alive]
        currently_urban = df.index[df.li_urban & df.is_alive]

        # handle new transitions
        now_urban: pd.Series = m.rng.random_sample(size=len(currently_rural)) < m.r_urban
        urban_idx = currently_rural[now_urban]
        df.loc[urban_idx, 'li_urban'] = True
        df.loc[urban_idx, 'li_date_trans_to_urban'] = self.sim.date

        # handle new transitions to rural
        now_rural: pd.Series = m.rng.random_sample(size=len(currently_urban)) < m.r_rural
        df.loc[currently_urban[now_rural], 'li_urban'] = False

        # -------------------- OVERWEIGHT ----------------------------------------------------------

        # get all adult who are not overweight
        adults_not_ow = df.index[~df.li_overwt & df.is_alive & (df.age_years >= 15)]

        # calculate the effective prob of becoming overweight; use the index of adults not ow
        eff_p_ow = pd.Series(m.r_overwt, index=adults_not_ow)
        eff_p_ow.loc[df.sex == 'F'] *= m.rr_overwt_f
        eff_p_ow.loc[df.li_urban] *= m.rr_overwt_urban

        # random draw and start of overweight status
        df.loc[adults_not_ow, 'li_overwt'] = (m.rng.random_sample(len(adults_not_ow)) < eff_p_ow)

        # -------------------- LOW EXERCISE --------------------------------------------------------

        adults_not_low_ex = df.index[~df.li_low_ex & df.is_alive & (df.age_years >= 15)]

        eff_p_low_ex = pd.Series(m.r_low_ex, index=adults_not_low_ex)
        eff_p_low_ex.loc[df.sex == 'F'] *= m.rr_low_ex_f
        eff_p_low_ex.loc[df.li_urban] *= m.rr_low_ex_urban

        df.loc[adults_not_low_ex, 'li_low_ex'] = (m.rng.random_sample(len(adults_not_low_ex)) < eff_p_low_ex)

        # -------------------- TOBACCO USE ---------------------------------------------------------

        adults_not_tob = df.index[(df.age_years >= 15) & df.is_alive & ~df.li_tob]
        currently_tob = df.index[df.li_tob & df.is_alive]

        # start tobacco use
        eff_p_tob = pd.Series(m.r_tob, index=adults_not_tob)
        eff_p_tob.loc[(df.age_years >= 20) & (df.age_years < 40)] *= m.rr_tob_age2039
        eff_p_tob.loc[df.age_years >= 40] *= m.rr_tob_agege40
        eff_p_tob.loc[df.sex == 'F'] *= m.rr_tob_f
        eff_p_tob *= m.rr_tob_wealth ** (pd.to_numeric(df.loc[adults_not_tob, 'li_wealth']) - 1)

        df.loc[adults_not_tob, 'li_tob'] = (m.rng.random_sample(len(adults_not_tob)) < eff_p_tob)

        # stop tobacco use
        df.loc[currently_tob, 'li_tob'] = ~(m.rng.random_sample(len(currently_tob)) < m.r_not_tob)

        # -------------------- EXCESSIVE ALCOHOL ---------------------------------------------------

        not_ex_alc_f = df.index[~df.li_ex_alc & df.is_alive & (df.sex == 'F') & (df.age_years >= 15)]
        not_ex_alc_m = df.index[~df.li_ex_alc & df.is_alive & (df.sex == 'M') & (df.age_years >= 15)]
        now_ex_alc = df.index[df.li_ex_alc & df.is_alive]

        df.loc[not_ex_alc_f, 'li_ex_alc'] = m.rng.random_sample(len(not_ex_alc_f)) < m.r_ex_alc * m.rr_ex_alc_f
        df.loc[not_ex_alc_m, 'li_ex_alc'] = m.rng.random_sample(len(not_ex_alc_m)) < m.r_ex_alc
        df.loc[now_ex_alc, 'li_ex_alc'] = ~(m.rng.random_sample(len(now_ex_alc)) < m.r_not_ex_alc)

        # -------------------- MARITAL STATUS ------------------------------------------------------

        curr_never_mar = df.index[df.is_alive & (df.age_years >= 15) & (df.age_years < 30) & (df.li_mar_stat == 1)]
        curr_mar = df.index[df.is_alive & (df.li_mar_stat == 2)]

        # update if now married
        now_mar = m.rng.random_sample(len(curr_never_mar)) < m.r_mar
        df.loc[curr_never_mar[now_mar], 'li_mar_stat'] = 2

        # update if now divorced/widowed
        now_div_wid = m.rng.random_sample(len(curr_mar)) < m.r_div_wid
        df.loc[curr_mar[now_div_wid], 'li_mar_stat'] = 3

        # -------------------- CONTRACEPTIVE STATUS ------------------------------------------------

        curr_not_on_con_idx = df.index[df.is_alive & (df.age_years >= 15) & (df.age_years < 50) &
                                       (df.sex == 'F') & ~df.li_on_con]
        now_on_con = m.rng.choice([True, False], size=len(curr_not_on_con_idx), p=[m.r_contrac, 1 - m.r_contrac])
        if now_on_con.any():
            now_on_con_index = curr_not_on_con_idx[now_on_con]
            df.loc[now_on_con_index, 'li_on_con'] = True

        curr_on_con_idx = df.index[df.is_alive & (df.age_years >= 15) & (df.age_years < 50) &
                                   (df.sex == 'F') & df.li_on_con]
        now_not_on_con = m.rng.choice([True, False],
                                      size=len(curr_on_con_idx),
                                      p=[m.r_contrac_int, 1 - m.r_contrac_int])
        if now_not_on_con.any():
            now_not_on_con_index = curr_on_con_idx[now_not_on_con]
            df.loc[now_not_on_con_index, 'li_on_con'] = False

        f_age50_idx = df.index[df.is_alive & (df.age_years == 50) & (df.sex == 'F') & df.li_on_con]
        df.loc[f_age50_idx, 'li_on_con'] = False

        curr_on_con_t_1_idx = df.index[df.is_alive & (df.age_years >= 15) & (df.age_years < 50) &
                                       (df.sex == 'F') & df.li_on_con & (df.li_con_t == 1)]
        df.loc[curr_on_con_t_1_idx, 'li_con_t'] = m.rng.choice([1, 2, 3, 4, 5, 6],
                                                               size=len(curr_on_con_t_1_idx),
                                                               p=m.r_con_from_1)

        curr_on_con_t_2_idx = df.index[df.is_alive & (df.age_years >= 15) & (df.age_years < 50) &
                                       (df.sex == 'F') & df.li_on_con & (df.li_con_t == 2)]
        df.loc[curr_on_con_t_2_idx, 'li_con_t'] = m.rng.choice([1, 2, 3, 4, 5, 6],
                                                               size=len(curr_on_con_t_2_idx),
                                                               p=m.r_con_from_2)

        curr_on_con_t_3_idx = df.index[df.is_alive & (df.age_years >= 15) & (df.age_years < 50) &
                                       (df.sex == 'F') & df.li_on_con & (df.li_con_t == 3)]
        df.loc[curr_on_con_t_3_idx, 'li_con_t'] = m.rng.choice([1, 2, 3, 4, 5, 6],
                                                               size=len(curr_on_con_t_3_idx),
                                                               p=m.r_con_from_3)

        curr_on_con_t_4_idx = df.index[df.is_alive & (df.age_years >= 15) & (df.age_years < 50) &
                                       (df.sex == 'F') & df.li_on_con & (df.li_con_t == 4)]
        df.loc[curr_on_con_t_4_idx, 'li_con_t'] = m.rng.choice([1, 2, 3, 4, 5, 6],
                                                               size=len(curr_on_con_t_4_idx),
                                                               p=m.r_con_from_4)

        curr_on_con_t_5_idx = df.index[df.is_alive & (df.age_years >= 15) & (df.age_years < 50) &
                                       (df.sex == 'F') & df.li_on_con & (df.li_con_t == 5)]
        df.loc[curr_on_con_t_5_idx, 'li_con_t'] = m.rng.choice([1, 2, 3, 4, 5, 6],
                                                               size=len(curr_on_con_t_5_idx),
                                                               p=m.r_con_from_5)

        curr_on_con_t_6_idx = df.index[df.is_alive & (df.age_years >= 15) & (df.age_years < 50) &
                                       (df.sex == 'F') & df.li_on_con & (df.li_con_t == 6)]
        df.loc[curr_on_con_t_6_idx, 'li_con_t'] = m.rng.choice([1, 2, 3, 4, 5, 6],
                                                               size=len(curr_on_con_t_6_idx),
                                                               p=m.r_con_from_6)

        # -------------------- EDUCATION -----------------------------------------------------------

        # get all individuals currently in education
        in_ed = df.index[df.is_alive & df.li_in_ed]

        # ---- PRIMARY EDUCATION

        # get index of all children who are alive and between 5 and 5.25 years old
        age5 = df.index[(df.age_exact_years >= 5) & (df.age_exact_years < 5.25) & df.is_alive]

        # by default, these children are not in education and have education level 1
        df.loc[age5, 'li_ed_lev'] = 1
        df.loc[age5, 'li_in_ed'] = False

        # create a series to hold the probablity of primary education for children at age 5
        prob_primary = pd.Series(m.p_ed_primary, index=age5)
        prob_primary *= m.rp_ed_primary_higher_wealth**(5 - pd.to_numeric(df.loc[age5, 'li_wealth']))

        # randomly select some to have primary education
        age5_in_primary = m.rng.random_sample(len(age5)) < prob_primary
        df.loc[age5[age5_in_primary], 'li_ed_lev'] = 2
        df.loc[age5[age5_in_primary], 'li_in_ed'] = True

        # ---- SECONDARY EDUCATION

        # get thirteen year olds that are in primary education, any wealth level
        age13_in_primary = df.index[(df.age_years == 13) & df.is_alive & df.li_in_ed & (df.li_ed_lev == 2)]

        # they have a probability of gaining secondary education (level 3), based on wealth
        prob_secondary = pd.Series(m.p_ed_secondary, index=age13_in_primary)
        prob_secondary *= m.rp_ed_secondary_higher_wealth**(5 - pd.to_numeric(df.loc[age13_in_primary, 'li_wealth']))

        # randomly select some to get secondary education
        age13_to_secondary = m.rng.random_sample(len(age13_in_primary)) < prob_secondary
        df.loc[age13_in_primary[age13_to_secondary], 'li_ed_lev'] = 3

        # those who did not go on to secondary education are no longer in education
        df.loc[age13_in_primary[~age13_to_secondary], 'li_in_ed'] = False

        # ---- DROP OUT OF EDUCATION

        # baseline rate of leaving education then adjust for wealth level
        p_leave_ed = pd.Series(m.r_stop_ed, index=in_ed)
        p_leave_ed *= m.rr_stop_ed_lower_wealth**(pd.to_numeric(df.loc[in_ed, 'li_wealth']) - 1)

        # randomly select some individuals to leave education
        now_not_in_ed = m.rng.random_sample(len(in_ed)) < p_leave_ed

        df.loc[in_ed[now_not_in_ed], 'li_in_ed'] = False

        # everyone leaves education at age 20
        df.loc[df.is_alive & df.li_in_ed & (df.age_years == 20), 'li_in_ed'] = False


class LifestylesLoggingEvent(RegularEvent, PopulationScopeEventMixin):
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
        logger.info('%s|li_ed_lev|%s',
                    self.sim.date,
                    df[df.is_alive].groupby(['li_wealth', 'li_ed_lev']).size().to_dict())

        logger.debug('%s|person_one|%s',
                     self.sim.date,
                     df.loc[0].to_dict())
        """

        logger.info('%s|li_urban|%s',
                    self.sim.date,
                    df[df.is_alive].groupby('li_urban').size().to_dict())

        logger.info('%s|li_wealth|%s',
                    self.sim.date,
                    df[df.is_alive].groupby('li_wealth').size().to_dict())

        logger.info('%s|li_overwt|%s',
                    self.sim.date,
                    df[df.is_alive].groupby(['sex', 'li_overwt']).size().to_dict())

        logger.info('%s|li_low_ex|%s',
                    self.sim.date,
                    df[df.is_alive].groupby(['sex', 'li_low_ex']).size().to_dict())

        logger.info('%s|li_tob|%s',
                    self.sim.date,
                    df[df.is_alive].groupby(['sex', 'li_tob']).size().to_dict())

        logger.info('%s|li_ed_lev_by_age|%s',
                    self.sim.date,
                    df[df.is_alive].groupby(['age_range', 'li_in_ed', 'li_ed_lev']).size().to_dict())
