"""
TB infections
"""

import pandas as pd
import numpy as np

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent, Event, IndividualScopeEventMixin

# file_path = '/Users/Tara/Desktop/TLO/TB/Method_TB.xlsx'  # laptop
from tlo.methods import demography

file_path = 'Q:/Thanzi la Onse/TB/Method_TB.xlsx'  # desktop

tb_data = pd.read_excel(file_path, sheet_name=None, header=0)
Active_tb_prob, Latent_tb_prop = tb_data['Active_TB_prob'], tb_data['Latent_TB_prob']


class tb_baseline(Module):
    """ Set up the baseline population with TB prevalence
    """

    def __init__(self, name=None):
        super().__init__(name)
        self.store = {'Time': [], 'Total_active_tb': [], 'Total_co-infected': [], 'TB_deaths': [],
                      'Time_death_TB': []}

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
        'rr_tb_hiv_stages': Parameter(
            Types.REAL,
            'relative risk of tb in hiv+ compared with hiv- by cd4 stage'),
        'rr_tb_art': Parameter(
            Types.REAL,
            'relative risk of tb in hiv+ on art'),
        'rr_tb_malnourished': Parameter(Types.REAL, 'relative risk of tb with malnourishment'),
        'rr_tb_diabetes1': Parameter(Types.REAL, 'relative risk of tb with diabetes type 1'),
        'rr_tb_alcohol': Parameter(Types.REAL, 'relative risk of tb with heavy alcohol use'),
        'rr_tb_smoking': Parameter(Types.REAL, 'relative risk of tb with smoking'),
        'rr_tb_pollution': Parameter(Types.REAL, 'relative risk of tb with indoor air pollution'),
        'rel_infectiousness_hiv': Parameter(
            Types.REAL,
            'relative infectiousness of tb in hiv+ compared with hiv-'),
        'prob_self_cure': Parameter(
            Types.REAL,
            'probability of self-cure'),
        'self_cure': Parameter(
            Types.REAL,
            'annual rate of self-cure'),
        'tb_mortality_rate': Parameter(
            Types.REAL,
            'mortality rate with active tb'),
        'tb_mortality_HIV': Parameter(
            Types.REAL,
            'mortality from tb with concurrent HIV'),
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'has_tb': Property(Types.CATEGORICAL, categories=['Uninfected', 'Latent', 'Active'], description='tb status'),
        'date_active_tb': Property(Types.DATE, 'Date active tb started'),
        'date_latent_tb': Property(Types.DATE, 'Date acquired tb infection (latent stage)'),
        'date_tb_death': Property(Types.DATE, 'Projected time of tb death if untreated'),
    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.
        Here we do nothing.
        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        params = self.parameters
        params['prop_fast_progressor'] = 0.14
        params['transmission_rate'] = 0.00001  # 4.9 (Juan) but no treatment yet so too high for test runs
        params['progression_to_active_rate'] = 0.5

        params['rr_tb_with_hiv_stages'] = [3.44, 6.76, 13.28, 26.06]
        params['rr_tb_art'] = 0.39
        params['rr_tb_malnourished'] = 2.1
        params['rr_tb_diabetes1'] = 3
        params['rr_tb_alcohol'] = 2.9
        params['rr_tb_smoking'] = 2.6
        params['rr_tb_pollution'] = 1.5

        params['rel_infectiousness_hiv'] = 0.68  # Juan
        params['prob_self_cure'] = 0.15
        params['self_cure'] = 0.33  # tiemersma plos one 2011, self-cure/death in 3 yrs
        params['tb_mortality_rate'] = 0.15  # Juan
        params['tb_mortality_HIV'] = 0.84  # Juan

    def initialise_population(self, population):
        """Set our property values for the initial population.
        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.
        :param population: the population of individuals
        initial pop is 2014
        then 2015-2018+ needs to run with beta in the FOI
        """
        df = population.props
        age = population.age
        now = self.sim.date

        # set-up baseline population
        df['has_tb'].values[:] = 'Uninfected'
        df['date_active_tb'] = pd.NaT
        df['date_latent_tb'] = pd.NaT
        df['date_tb_death'] = pd.NaT

        # TB infections - active / latent
        # baseline infections not weighted by RR, randomly assigned
        # can include RR values in the sample command (weights)

        active_tb_prob_year = Active_tb_prob.loc[
            Active_tb_prob.Year == now.year, ['ages', 'Sex', 'incidence_per_capita']]

        for i in range(0, 81):
            # male
            idx = (age.years == i) & (df.sex == 'M') & (df.has_tb == 'Uninfected')

            if idx.any():
                # sample from uninfected population using WHO prevalence
                fraction_latent_tb = Latent_tb_prop.loc[
                    (Latent_tb_prop.sex == 'M') & (Latent_tb_prop.age == i), 'prob_latent_tb']
                male_latent_tb = df[idx].sample(frac=fraction_latent_tb).index
                df.loc[male_latent_tb, 'has_tb'] = 'Latent'
                df.loc[male_latent_tb, 'date_latent_tb'] = now

            idx_uninfected = (age.years == i) & (df.sex == 'M') & (df.has_tb == 'Uninfected')

            if idx_uninfected.any():
                fraction_active_tb = active_tb_prob_year.loc[
                    (active_tb_prob_year.Sex == 'M') & (active_tb_prob_year.ages == i), 'incidence_per_capita']
                male_active_tb = df[idx_uninfected].sample(frac=fraction_active_tb).index
                df.loc[male_active_tb, 'has_tb'] = 'Active'
                df.loc[male_active_tb, 'date_active_tb'] = now

            # female
            idx = (age.years == i) & (df.sex == 'F') & (df.has_tb == 'Uninfected')

            if idx.any():
                # sample from uninfected population using WHO prevalence
                fraction_latent_tb = Latent_tb_prop.loc[
                    (Latent_tb_prop.sex == 'F') & (Latent_tb_prop.age == i), 'prob_latent_tb']
                female_latent_tb = df[idx].sample(frac=fraction_latent_tb).index
                df.loc[female_latent_tb, 'has_tb'] = 'Latent'
                df.loc[female_latent_tb, 'date_latent_tb'] = now

            idx_uninfected = (age.years == i) & (df.sex == 'F') & (df.has_tb == 'Uninfected')

            if idx.any():
                fraction_active_tb = active_tb_prob_year.loc[
                    (active_tb_prob_year.Sex == 'F') & (active_tb_prob_year.ages == i), 'incidence_per_capita']
                female_active_tb = df[idx_uninfected].sample(frac=fraction_active_tb).index
                df.loc[female_active_tb, 'has_tb'] = 'Active'
                df.loc[female_active_tb, 'date_active_tb'] = now

        # print(df.head(20))

        # date of death of the infected individuals (in the future)
        infected_count = len(df[(df.has_tb == 'Active') & df.is_alive])  # get all the infected alive individuals
        # print('infected_count: ', infected_count)

        death_years_ahead = np.random.exponential(scale=5, size=infected_count)
        death_td_ahead = pd.to_timedelta(death_years_ahead * 365.25, unit='d')
        # death_td_ahead = pd.Series(death_td_ahead).dt.floor("S")  # remove microseconds
        # print('death_td_ahead: ', death_td_ahead)

        # set the properties of infected individuals
        df.loc[(df.has_tb == 'Active') & df.is_alive, 'date_tb_death'] = now + death_td_ahead

    def initialise_simulation(self, sim):
        """Get ready for simulation start.
        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """
        # add the basic event (we will implement below)
        sim.schedule_event(tb_event(self), sim.date + DateOffset(months=12))

        # add an event to log to screen
        sim.schedule_event(tb_LoggingEvent(self), sim.date + DateOffset(months=12))

        # add the death event of infected individuals
        # df = sim.population.props
        # infected_individuals = df[(df.has_tb == 'Active') & df.is_alive].index
        # for index in infected_individuals:
        #     individual = self.sim.population[index]
        #     death_event = tbDeathEvent(self, individual)
        #     self.sim.schedule_event(death_event, individual.date_tb_death)

        sim.schedule_event(tbDeathEvent(self), sim.date + DateOffset(
            months=12))


    def on_birth(self, mother, child):
        """Initialise our properties for a newborn individual.
        This is called by the simulation whenever a new person is born.
        :param mother: the mother for this child
        :param child: the new child
        """

        child.has_tb.values[:] = 'Uninfected'
        child.date_active_tb = pd.NaT
        child.date_latent_tb = pd.NaT
        child.date_tb_death = pd.NaT


class tb_event(RegularEvent, PopulationScopeEventMixin):
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
        super().__init__(module, frequency=DateOffset(months=12))  # every 12 months
        # make sure any rates are annual if frequency of event is annual

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """

        params = self.module.parameters
        now = self.sim.date
        rng = self.module.rng

        df = population.props

        # apply a force of infection to produce new latent cases
        # no age distribution for FOI but the relative risks would affect distribution of active infection
        # remember event is occurring annually so scale rates accordingly
        active_hiv_neg = len(df[(df.has_tb == 'Active') & ~df.has_hiv & df.is_alive])
        active_hiv_pos = len(df[(df.has_tb == 'Active') & df.has_hiv & df.is_alive])
        uninfected_total = len(df[(df.has_tb == 'Uninfected') & df.is_alive])
        total_population = len(df[df.is_alive])

        force_of_infection = (params['transmission_rate'] * active_hiv_neg * (active_hiv_pos * params[
            'rel_infectiousness_hiv']) * uninfected_total) / total_population
        # print('force_of_infection: ', force_of_infection)

        prob_tb_new = pd.Series(force_of_infection, index=df[(df.has_tb == 'Uninfected') & df.is_alive].index)
        # print('prob_tb_new: ', prob_tb_new)
        is_newly_infected = prob_tb_new > rng.rand(len(prob_tb_new))
        new_case = is_newly_infected[is_newly_infected].index
        df.loc[new_case, 'has_tb'] = 'Latent'
        df.loc[new_case, 'date_latent_tb'] = now

        # if any newly infected latent cases, 14% become active directly
        new_latent = df[(df.has_tb == 'Latent') & (df.date_latent_tb == now) & df.is_alive].sum()
        # print(new_latent)

        if new_latent.any():
            fast_progression = df[(df.has_tb == 'Latent') & (df.date_latent_tb == now) & df.is_alive].sample(
                frac=params['prop_fast_progressor']).index
            df.loc[fast_progression, 'has_tb'] = 'Active'
            df.loc[fast_progression, 'date_active_tb'] = now

        # slow progressors with latent TB become active
        # random sample with weights for RR of active disease
        eff_prob_active_tb = pd.Series(0, index=df.index)
        # print('eff_prob_active_tb - all zeros: ', eff_prob_active_tb)

        eff_prob_active_tb.loc[df.has_tb == 'Latent'] = params['progression_to_active_rate']
        # print('eff_prob_active_tb: ', eff_prob_active_tb)

        hiv_stage1 = df.index[df.has_hiv & (df.has_tb == 'Latent') &
                              (((now - df.date_hiv_infection).dt.days / 365.25) < 3.33)]
        # print('hiv_stage1', hiv_stage1)

        hiv_stage2 = df.index[df.has_hiv & (df.has_tb == 'Latent') &
                              (((now - df.date_hiv_infection).dt.days / 365.25) >= 3.33) &
                              (((now - df.date_hiv_infection).dt.days / 365.25) < 6.67)]
        # print('hiv_stage2', hiv_stage2)

        hiv_stage3 = df.index[df.has_hiv & (df.has_tb == 'Latent') &
                              (((now - df.date_hiv_infection).dt.days / 365.25) >= 6.67) &
                              (((now - df.date_hiv_infection).dt.days / 365.25) < 10)]
        # print('hiv_stage3', hiv_stage3)

        hiv_stage4 = df.index[df.has_hiv & (df.has_tb == 'Latent') &
                              (((now - df.date_hiv_infection).dt.days / 365.25) >= 10)]
        # print('hiv_stage4', hiv_stage4)

        eff_prob_active_tb.loc[hiv_stage1] *= params['rr_tb_with_hiv_stages'][0]
        eff_prob_active_tb.loc[hiv_stage2] *= params['rr_tb_with_hiv_stages'][1]
        eff_prob_active_tb.loc[hiv_stage3] *= params['rr_tb_with_hiv_stages'][2]
        eff_prob_active_tb.loc[hiv_stage4] *= params['rr_tb_with_hiv_stages'][3]
        # print('eff_prob_active_tb with hiv stages: ', eff_prob_active_tb)
        # eff_prob_active_tb.to_csv('Q:/Thanzi la Onse/TB/eff_prob.csv', sep=',')

        # eff_prob_active_tb.loc[df.on_art] *= params['rr_tb_art']
        # eff_prob_active_tb.loc[df.is_malnourished] *= params['rr_tb_malnourished']
        # eff_prob_active_tb.loc[df.has_diabetes1] *= params['rr_tb_diabetes1']
        # eff_prob_active_tb.loc[df.high_alcohol] *= params['rr_tb_alcohol']
        # eff_prob_active_tb.loc[df.is_smoker] *= params['rr_tb_smoking']
        # eff_prob_active_tb.loc[df.high_pollution] *= params['rr_tb_pollution']

        prog_to_active = eff_prob_active_tb > rng.rand(len(eff_prob_active_tb))
        # print('prog_to_active: ', prog_to_active )
        new_active_case = prog_to_active[prog_to_active].index
        # print('new_active_case: ', new_active_case)
        df.loc[new_active_case, 'has_tb'] = 'Active'
        df.loc[new_active_case, 'date_active_tb'] = now

        # self-cure - move back from active to latent, make sure it's not the ones that just became active
        self_cure_tb = df[(df.has_tb == 'Active') & (df.date_active_tb < now)].sample(
            frac=(params['prob_self_cure'] * params['self_cure'])).index
        df.loc[self_cure_tb, 'has_tb'] = 'Latent'


class tbDeathEvent(RegularEvent, PopulationScopeEventMixin):
    """The regular event that actually kills people.

    Regular events automatically reschedule themselves at a fixed frequency,
    and thus implement discrete timestep type behaviour. The frequency is
    specified when calling the base class constructor in our __init__ method.
    """

    def __init__(self, module):
        """Create a new random death event.

        We need to pass the frequency at which we want to occur to the base class
        constructor using super(). We also pass the module that created this event,
        so that random number generators can be scoped per-module.

        :param module: the module that created this event
        :param death_probability: the per-person probability of death each month
        """
        super().__init__(module, frequency=DateOffset(months=12))

    def apply(self, population):
        """Apply this event to the population.

        For efficiency, we use pandas operations to scan the entire population
        and kill individuals at random.

        :param population: the current population
        """
        params = self.module.parameters
        df = population.props
        now = self.sim.date
        rng = self.module.rng

        # Generate a series of random numbers, one per individual
        mortality_rate = pd.Series(0, index=df.index)
        mortality_rate.loc[(df.has_tb == 'Active') & ~df.has_hiv] = params['tb_mortality_rate']
        mortality_rate.loc[(df.has_tb == 'Active') & df.has_hiv] = params['tb_mortality_HIV']
        # print('mort_rate: ', mortality_rate)

        probs = rng.rand(len(df))
        deaths = df.is_alive & (probs < mortality_rate)
        # print('deaths: ', deaths)
        will_die = (df[deaths]).index
        # print('will_die: ', will_die)

        for i in will_die:
            person = population[i]
            death = demography.InstantaneousDeath(self.module, person, cause='tb')  # make that death event
            self.sim.schedule_event(death, now)  # schedule the death for "now"

        total_deaths = len(will_die)
        self.module.store['TB_deaths'].append(total_deaths)
        self.module.store['Time_death_TB'].append(self.sim.date)




class tb_LoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """ produce some outputs to check
        """
        # run this event every 12 months (every year)
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props

        active_tb_total = len(df[df.has_tb == 'Active'])
        coinfected_total = len(df[(df.has_tb == 'Active') & df.has_hiv])

        self.module.store['Time'].append(self.sim.date)
        self.module.store['Total_active_tb'].append(active_tb_total)
        self.module.store['Total_co-infected'].append(coinfected_total)

        # print('tb outputs: ', self.sim.date, active_tb_total, coinfected_total)
