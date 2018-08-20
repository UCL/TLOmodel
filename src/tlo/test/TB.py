import numpy as np
import pandas as pd

# initial pop data #
sim_size = int(100)
TB_prev2018 = 18091575 / sim_size

inds = pd.read_csv('Q:/Thanzi la Onse/HIV/initial_pop_dataframe2018.csv')

incidence = pd.read_excel('Q:/Thanzi la Onse/TB/Method Template TB.xlsx', sheet_name='TB_incidence', header=0)


# Population dynamics parameters #
f = 0.14  # proportion fast progressors, Vynnycky
beta = 7.2  # transmission rate, Juan's model
mu = 0.15  # TB mortality rate

current_time = 2018

# assign infected status using WHO prevalence 2016 by 2 x age-groups
def prevalenceTB_inds(inds, current_time):

    # sample from uninfected population using WHO incidence
    # age 0-14
    tmp = np.random.choice(inds.index[(inds.age < 15) & (inds.sex == 'M')],
                            size=int((incidence['Incident cases'][(incidence.Year == 2016) &
                                                                  (incidence.Sex == 'M') &
                                                             (incidence.Age == '0_14')]) / sim_size),
                           replace=False)


