import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent


# initial pop data #
sim_size = int(100)

inds = pd.read_csv('Q:/Thanzi la Onse/HIV/initial_pop_dataframe2018.csv')

# add column to inds for TB status
inds['tb_status'] = 'U'

TBincidence = pd.read_excel('Q:/Thanzi la Onse/TB/Method Template TB.xlsx', sheet_name='TB_incidence', header=0)

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

# this class contains all the methods required to set up the baseline population
class TB(Module):
    """Models baseline TB prevalence.

    Methods required:
    * `read_parameters(data_folder)`
    * `initialise_population(population)`
    * `initialise_simulation(sim)`
    * `on_birth(mother, child)`
    """

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    PARAMETERS = {
        'prop_fast_progressor': Parameter(
            Types.REAL,
            'Proportion of infections that progress directly to active stage, Vynnycky'),
        'transmission_rate': Parameter(
            Types.REAL,
            'TB transmission rate, estimated by Juan'),
        'progression_to_active_rate': Parameter(
            Types.REAL,
            'Combined rate of progression/reinfection/relapse from Juan'),
        'rr_TB_with_HIV_stages': Parameter(
            Types.REAL,
            'relative risk of TB hin HIV+ compared with HIV- by CD4 stage'),
        'rr_ART': Parameter(
            Types.REAL,
            'relative risk of TB in HIV+ on ART'),
        'rr_infectiousness_HIV': Parameter(
            Types.REAL,
            'relative infectiousness of TB in HIV+ compared with HIV-'),
        'recovery': Parameter(
            Types.REAL,
            'combined rate of diagnosis, treatment and self-cure, from Juan'),
        'TB_mortality_rate': Parameter(
            Types.REAL,
            'mortality rate with active TB'),
        'rr_TB_mortality_HIV': Parameter(
            Types.REAL,
            'relative risk of mortality from TB in HIV+ compared with HIV-'),
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'has_TB': Property(Types.BOOL, 'TB status'),
        'date_TB_infection': Property(Types.DATE, 'Date acquired TB infection'),
        'date_TB_death': Property(Types.DATE, 'Projected time of TB death if untreated'),
        'on_treatment': Property(Types.BOOL, 'Currently on treatment for TB'),
        'date_ART_treatment start': Property(Types.DATE, 'Date treatment started'),
        'date_death': Property(Types.DATE, 'Date of death'),
    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.
        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """

        params = self.parameters
        params['prop_fast_progressor'] = 0.14
        params['transmission_rate'] = 7.2
        params['progression_to_active_rate'] = 0.5
        params['rr_TB_with_HIV_stages'] = [0.0198, 0.0681, 0.1339, 0.26297, 0.516]
        params['rr_ART'] = 0.39
        params['rr_infectiousness_HIV'] = 0.52
        params['recovery'] = 2
        params['TB_mortality_rate'] = 0.15
        params['rr_TB_mortality_HIV'] = 17.1


    # baseline population
    # assign infected status using WHO prevalence 2016 by 2 x age-groups
    def prevalenceTB(self, df, current_time):

        self.current_time = current_time
        self.HIVstage1 = 3.33 # Williams equal duration HIV stages (x4)
        self.HIVstage2 = 6.67
        self.HIVstage3 = 10

        # create a vector of probabilities depending on HIV status and time since seroconversion
        df['tmp'][df.has_HIV == 1] = IRR_CD4[0]
        df['tmp'][(df.has_HIV == 1) & (self.current_time - df.date_HIV_infection <= self.HIVstage1)] = IRR_CD4[1]
        df['tmp'][(df.has_HIV == 1) & (self.current_time - df.date_HIV_infection > self.HIVstage1) & (self.current_time - df.date_HIV_infection <= self.HIVstage2)] = IRR_CD4[2]
        df['tmp'][(df.has_HIV == 1) & (self.current_time - df.date_HIV_infection > self.HIVstage2) & (self.current_time - df.date_HIV_infection <= self.HIVstage3)] = IRR_CD4[3]
        df['tmp'][(df.has_HIV == 1) & (self.current_time - df.date_HIV_infection > self.HIVstage3)] = IRR_CD4[4]

        # sample from uninfected population using WHO incidence
        # male age 0-14
        tmp1 = np.random.choice(df.index[(df.age < 15) & (df.sex == 'M')],
                                size=int((TBincidence['Incident cases'][(TBincidence.Year == 2016) &
                                                                      (TBincidence.Sex == 'M') &
                                                                      (TBincidence.Age == '0_14')])),
                                replace=False, p=df.tmp)
        df.loc[tmp1, 'tb_status'] = 'A'  # change status to active infection

        # female age 0-14
        tmp2 = np.random.choice(df.index[(df.age < 15) & (df.sex == 'F')],
                                size=int((TBincidence['Incident cases'][(TBincidence.Year == 2016) &
                                                                      (TBincidence.Sex == 'F') &
                                                                      (TBincidence.Age == '0_14')])),
                                replace=False, p=df.tmp)
        df.loc[tmp2, 'tb_status'] = 'A'  # change status to infected

        # male age >=15
        tmp3 = np.random.choice(df.index[(df.age >= 15) & (df.sex == 'M')],
                                size=int((TBincidence['Incident cases'][(TBincidence.Year == 2016) &
                                                                      (TBincidence.Sex == 'M') &
                                                                      (TBincidence.Age == '15_80')])),
                                replace=False, p=df.tmp)
        df.loc[tmp3, 'tb_status'] = 'A'  # change status to infected

        # female age >=15
        tmp4 = np.random.choice(df.index[(df.age >= 15) & (df.sex == 'F')],
                                size=int((TBincidence['Incident cases'][(TBincidence.Year == 2016) &
                                                                      (TBincidence.Sex == 'F') &
                                                                      (TBincidence.Age == '15_80')])),
                                replace=False, p=df.tmp)
        df.loc[tmp4, 'tb_status'] = 'A'  # change status to infected

        del df['tmp']  # remove temporary column

        return df



















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

