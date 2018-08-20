import numpy as np
import pandas as pd

# initial pop data #
sim_size = int(100)

inds = pd.read_csv('Q:/Thanzi la Onse/HIV/initial_pop_dataframe2018.csv')

# add column to inds for TB status
inds['tb_status'] = 'U'

incidence = pd.read_excel('Q:/Thanzi la Onse/TB/Method Template TB.xlsx', sheet_name='TB_incidence', header=0)

# Population dynamics parameters #
f = 0.14  # proportion fast progressors, Vynnycky
beta = 7.2  # transmission rate, Juan's model

progression = 0.5  # dummy value, combined rate progression / reinfection / relapse
IRR_CD4 = [0.0198, 0.0681, 0.1339, 0.26297, 0.516]  # relative risk of TB in HIV+ compared with HIV- by CD4
# IRR_CD4 values are scaled to sum to 1, uses Williams' 4x HIV stages, each duration=0.33 years
IRR_ART = 0.39  # relative risk of TB on ART

rel_infectiousness_HIV = 0.52  # relative infectiousness of HIV+ vs HIV-

treatment = 2  # dummy value, combined rate diagnosis / treatment / self-cure

mu = 0.15  # TB mortality rate
RR_mu_HIV = 17.1  # relative risk mortality in HIV+ vs HIV-

current_time = 2018


# baseline population
# assign infected status using WHO prevalence 2016 by 2 x age-groups
def prevalenceTB_inds(inds):
    # create a vector of probabilities depending on HIV status and time since seroconversion
    inds['tmp'][inds.status == 'U'] = IRR_CD4[0]
    inds['tmp'][(inds.status == 'I') & (inds.timeInf <= 3.33)] = IRR_CD4[1]
    inds['tmp'][(inds.status == 'I') & (inds.timeInf > 3.33) & (inds.timeInf <= 6.67)] = IRR_CD4[2]
    inds['tmp'][(inds.status == 'I') & (inds.timeInf > 6.67) & (inds.timeInf <= 10)] = IRR_CD4[3]
    inds['tmp'][(inds.status == 'I') & (inds.timeInf > 10)] = IRR_CD4[4]

    # sample from uninfected population using WHO incidence
    # male age 0-14
    tmp1 = np.random.choice(inds.index[(inds.age < 15) & (inds.sex == 'M')],
                            size=int((incidence['Incident cases'][(incidence.Year == 2016) &
                                                                  (incidence.Sex == 'M') &
                                                                  (incidence.Age == '0_14')]) / sim_size),
                            replace=False, p=inds.tmp)
    inds.loc[tmp1, 'tb_status'] = 'I'  # change status to infected

    # female age 0-14
    tmp2 = np.random.choice(inds.index[(inds.age < 15) & (inds.sex == 'F')],
                            size=int((incidence['Incident cases'][(incidence.Year == 2016) &
                                                                  (incidence.Sex == 'F') &
                                                                  (incidence.Age == '0_14')]) / sim_size),
                            replace=False, p=inds.tmp)
    inds.loc[tmp2, 'tb_status'] = 'I'  # change status to infected

    # male age >=15
    tmp3 = np.random.choice(inds.index[(inds.age >= 15) & (inds.sex == 'M')],
                            size=int((incidence['Incident cases'][(incidence.Year == 2016) &
                                                                  (incidence.Sex == 'M') &
                                                                  (incidence.Age == '15_80')]) / sim_size),
                            replace=False, p=inds.tmp)
    inds.loc[tmp3, 'tb_status'] = 'I'  # change status to infected

    # female age >=15
    tmp4 = np.random.choice(inds.index[(inds.age >= 15) & (inds.sex == 'F')],
                            size=int((incidence['Incident cases'][(incidence.Year == 2016) &
                                                                  (incidence.Sex == 'F') &
                                                                  (incidence.Age == '15_80')]) / sim_size),
                            replace=False, p=inds.tmp)
    inds.loc[tmp4, 'tb_status'] = 'I'  # change status to infected

    del inds['tmp']  # remove temporary column

    return inds


def tb_treatment(inds):
    # apply diagnosis / treatment / self-cure combined rates

    return inds




def force_of_infection_tb(inds):
    infected = len(inds[(inds.tb_status == 'I') & (inds.tb_treat == 0)])  # number infected untreated

    # number co-infected with HIV * relative infectiousness (lower)
    hiv_infected = rel_infectiousness_HIV * len(inds[(inds.tb_status == 'I') & (inds.status == 'I')])

    total_pop = len(inds[(inds.status != 'DH') & (inds.status != 'D')])  # whole population currently alive

    foi = beta * ((infected + hiv_infected) / total_pop)  # force of infection for adults

    return foi

def inf_tb(inds):
    # apply foi to uninfected pop -> latent infection

    return inds

def progression_tb(inds):
    # apply combined progression / relapse / reinfection rates to infected pop

    return inds

def recover_tb(inds):
    # apply combined diagnosis / treatment / self-cure rates to TB cases

    return inds

