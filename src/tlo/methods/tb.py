"""
TB infections
"""

import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent
from tlo.methods import demography


class tb_baseline(Module):
    """ Set up the baseline population with TB prevalence
    """

    def __init__(self, name=None, workbook_path=None):
        super().__init__(name)
        self.workbook_path = workbook_path
        self.store = {'Time': [], 'Total_active_tb': [], 'Total_active_tb_mdr': [], 'Total_co-infected': [],
                      'TB_deaths': [],
                      'Time_death_TB': []}

    PARAMETERS = {
        'prop_fast_progressor': Parameter(Types.REAL,
                                          'Proportion of infections that progress directly to active stage'),
        'transmission_rate': Parameter(Types.REAL, 'TB transmission rate, estimated by Juan'),
        'progression_to_active_rate': Parameter(Types.REAL,
                                                'Combined rate of progression/reinfection/relapse from Juan'),
        'rr_tb_hiv_stages': Parameter(Types.REAL, 'relative risk of tb in hiv+ compared with hiv- by cd4 stage'),
        'rr_tb_art': Parameter(Types.REAL, 'relative risk of tb in hiv+ on art'),
        'rr_tb_ipt': Parameter(Types.REAL, 'relative risk of tb on ipt'),
        'rr_tb_malnourished': Parameter(Types.REAL, 'relative risk of tb with malnourishment'),
        'rr_tb_diabetes1': Parameter(Types.REAL, 'relative risk of tb with diabetes type 1'),
        'rr_tb_alcohol': Parameter(Types.REAL, 'relative risk of tb with heavy alcohol use'),
        'rr_tb_smoking': Parameter(Types.REAL, 'relative risk of tb with smoking'),
        'rr_tb_pollution': Parameter(Types.REAL, 'relative risk of tb with indoor air pollution'),
        'rel_infectiousness_hiv': Parameter(Types.REAL, 'relative infectiousness of tb in hiv+ compared with hiv-'),
        'prob_self_cure': Parameter(Types.REAL, 'probability of self-cure'),
        'rate_self_cure': Parameter(Types.REAL, 'annual rate of self-cure'),
        'tb_mortality_rate': Parameter(Types.REAL, 'mortality rate with active tb'),
        'tb_mortality_HIV': Parameter(Types.REAL, 'mortality from tb with concurrent HIV'),
        'prop_mdr2010': Parameter(Types.REAL, 'prevalence of mdr in TB cases 2010'),
        'prop_mdr_new': Parameter(Types.REAL, 'prevalence of mdr in new tb cases'),
        'prop_mdr_retreated': Parameter(Types.REAL, 'prevalence of mdr in previously treated cases'),
    }

    PROPERTIES = {
        'tb_inf': Property(Types.CATEGORICAL,
                           categories=['uninfected', 'latent_susc', 'active_susc', 'latent_mdr', 'active_mdr'],
                           description='tb status'),
        'tb_date_active': Property(Types.DATE, 'Date active tb started'),
        'tb_date_latent': Property(Types.DATE, 'Date acquired tb infection (latent stage)'),
        'tb_date_death': Property(Types.DATE, 'Projected time of tb death if untreated'),
    }

    def read_parameters(self, data_folder):

        params = self.parameters
        params['param_list'] = pd.read_excel(self.workbook_path,
                                             sheet_name='parameters')
        self.param_list.set_index("parameter", inplace=True)

        params['prop_fast_progressor'] = self.param_list.loc['prop_fast_progressor', 'value1']
        params['transmission_rate'] = self.param_list.loc['transmission_rate', 'value1']
        params['progression_to_active_rate'] = self.param_list.loc['progression_to_active_rate', 'value1']

        params['rr_tb_with_hiv_stages'] = self.param_list.loc['transmission_rate'].values
        params['rr_tb_art'] = self.param_list.loc['rr_tb_art', 'value1']
        params['rr_tb_ipt'] = self.param_list.loc['rr_tb_ipt', 'value1']
        params['rr_tb_malnourished'] = self.param_list.loc['rr_tb_malnourished', 'value1']
        params['rr_tb_diabetes1'] = self.param_list.loc['rr_tb_diabetes1', 'value1']
        params['rr_tb_alcohol'] = self.param_list.loc['rr_tb_alcohol', 'value1']
        params['rr_tb_smoking'] = self.param_list.loc['rr_tb_smoking', 'value1']
        params['rr_tb_pollution'] = self.param_list.loc['rr_tb_pollution', 'value1']
        params['rel_infectiousness_hiv'] = self.param_list.loc['rel_infectiousness_hiv', 'value1']
        params['prob_self_cure'] = self.param_list.loc['prob_self_cure', 'value1']
        params['rate_self_cure'] = self.param_list.loc['rate_self_cure', 'value1']
        params['tb_mortality_rate'] = self.param_list.loc['tb_mortality_rate', 'value1']
        params['tb_mortality_hiv'] = self.param_list.loc['tb_mortality_hiv', 'value1']
        params['prop_mdr2010'] = self.param_list.loc['prop_mdr2010', 'value1']
        params['prop_mdr_new'] = self.param_list.loc['prop_mdr_new', 'value1']
        params['prop_mdr_retreated'] = self.param_list.loc['prop_mdr_retreated', 'value1']

        params['tb_data'] = pd.read_excel(self.workbook_path,
                                          sheet_name=None)

        params['Active_tb_prob'], params['Latent_tb_prob'] = self.tb_data['Active_TB_prob'], \
                                                             self.tb_data['Latent_TB_prob']

    def initialise_population(self, population):
        """Set our property values for the initial population.
        """
        df = population.props
        now = self.sim.date

        # set-up baseline population
        df['tb_inf'].values[:] = 'uninfected'
        df['tb_date_active'] = pd.NaT
        df['tb_date_latent'] = pd.NaT
        df['tb_date_death'] = pd.NaT

        # TB infections - active / latent
        # baseline infections not weighted by RR, randomly assigned
        active_tb_data = self.parameters['Active_tb_prob']
        latent_tb_data = self.parameters['Latent_tb_prob']

        active_tb_prob_year = active_tb_data.loc[
            active_tb_data.Year == now.year, ['ages', 'Sex', 'incidence_per_capita']]

        # TODO: condense this with a merge function and remove if statements
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~ MALE ~~~~~~~~~~~~~~~~~~~~~~~~~~

        for i in range(0, 81):
            idx = (df.age_years == i) & (df.sex == 'M') & (df.tb_inf == 'uninfected') & df.is_alive

            # LATENT
            if idx.any():
                # sample from uninfected population using WHO prevalence
                fraction_latent_tb = latent_tb_data.loc[
                    (latent_tb_data.sex == 'M') & (latent_tb_data.age == i), 'prob_latent_tb']
                male_latent_tb = df[idx].sample(frac=fraction_latent_tb).index
                df.loc[male_latent_tb, 'tb_inf'] = 'latent_susc'
                df.loc[male_latent_tb, 'tb_date_latent'] = now

                # allocate some latent infections as mdr-tb
                if len(df[df.is_alive & (df.sex == 'M') & (df.tb_inf == 'latent_susc')]) > 10:
                    idx_c = df[df.is_alive & (df.sex == 'M') & (df.tb_inf == 'latent_susc')].sample(
                        frac=self.parameters['prop_mdr2010']).index

                    df.loc[idx_c, 'tb_inf'] = 'latent_mdr'  # change to mdr-tb

            idx_uninfected = (df.age_years == i) & (df.sex == 'M') & (df.tb_inf == 'uninfected') & df.is_alive

            # ACTIVE
            if idx_uninfected.any():
                fraction_active_tb = active_tb_prob_year.loc[
                    (active_tb_prob_year.Sex == 'M') & (active_tb_prob_year.ages == i), 'incidence_per_capita']
                male_active_tb = df[idx_uninfected].sample(frac=fraction_active_tb).index
                df.loc[male_active_tb, 'tb_inf'] = 'active_susc'
                df.loc[male_active_tb, 'tb_date_active'] = now

                # allocate some active infections as mdr-tb
                if len(df[df.is_alive & (df.sex == 'M') & (df.tb_inf == 'active_susc')]) > 10:
                    idx_c = df[df.is_alive & (df.sex == 'M') & (df.tb_inf == 'active_susc')].sample(
                        frac=self.parameters['prop_mdr2010']).index

                    df.loc[idx_c, 'tb_inf'] = 'active_mdr'  # change to mdr-tb

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~ FEMALE ~~~~~~~~~~~~~~~~~~~~~~~~~~

            idx = (df.age_years == i) & (df.sex == 'F') & (df.tb_inf == 'uninfected') & df.is_alive

            # LATENT
            if idx.any():
                # sample from uninfected population using WHO latent prevalence estimates
                fraction_latent_tb = latent_tb_data.loc[
                    (latent_tb_data.sex == 'F') & (latent_tb_data.age == i), 'prob_latent_tb']
                female_latent_tb = df[idx].sample(frac=fraction_latent_tb).index
                df.loc[female_latent_tb, 'tb_inf'] = 'latent_susc'
                df.loc[female_latent_tb, 'tb_date_latent'] = now

            # allocate some latent infections as mdr-tb
            if len(df[df.is_alive & (df.sex == 'F') & (df.tb_inf == 'latent_susc')]) > 10:
                idx_c = df[df.is_alive & (df.sex == 'F') & (df.tb_inf == 'latent_susc')].sample(
                    frac=self.parameters['prop_mdr2010']).index

                df.loc[idx_c, 'tb_inf'] = 'latent_mdr'  # change to mdr-tb

            idx_uninfected = (df.age_years == i) & (df.sex == 'F') & (df.tb_inf == 'uninfected') & df.is_alive

            # ACTIVE
            if idx.any():
                fraction_active_tb = active_tb_prob_year.loc[
                    (active_tb_prob_year.Sex == 'F') & (active_tb_prob_year.ages == i), 'incidence_per_capita']
                female_active_tb = df[idx_uninfected].sample(frac=fraction_active_tb).index
                df.loc[female_active_tb, 'tb_inf'] = 'active_susc'
                df.loc[female_active_tb, 'tb_date_active'] = now

            # allocate some active infections as mdr-tb
            if len(df[df.is_alive & (df.sex == 'F') & (df.tb_inf == 'active_susc')]) > 10:
                idx_c = df[df.is_alive & (df.sex == 'F') & (df.tb_inf == 'active_susc')].sample(
                    frac=self.parameters['prop_mdr2010']).index

                df.loc[idx_c, 'tb_inf'] = 'active_mdr'  # change to mdr-tb

    def initialise_simulation(self, sim):
        sim.schedule_event(tb_event(self), sim.date + DateOffset(months=12))
        sim.schedule_event(tb_mdr_event(self), sim.date + DateOffset(months=12))

        sim.schedule_event(tbDeathEvent(self), sim.date + DateOffset(
            months=12))

        # add an event to log to screen
        sim.schedule_event(tb_LoggingEvent(self), sim.date + DateOffset(months=12))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.
        """
        df = self.sim.population.props

        df.at[child_id, 'tb_inf'] = 'uninfected'
        df.at[child_id, 'tb_date_active'] = pd.NaT
        df.at[child_id, 'tb_date_latent'] = pd.NaT
        df.at[child_id, 'tb_date_death'] = pd.NaT


class tb_event(RegularEvent, PopulationScopeEventMixin):
    """ tb infection events
    """

    def __init__(self, module):
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

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~ FORCE OF INFECTION ~~~~~~~~~~~~~~~~~~~~~~~~~~

        # apply a force of infection to produce new latent cases
        # no age distribution for FOI but the relative risks would affect distribution of active infection
        # remember event is occurring annually so scale rates accordingly
        active_hiv_neg = len(df[(df.tb_inf == 'active_susc') & ~df.hiv_inf & df.is_alive])
        active_hiv_pos = len(df[(df.tb_inf == 'active_susc') & df.hiv_inf & df.is_alive])
        uninfected_total = len(df[(df.tb_inf == 'uninfected') & df.is_alive])
        total_population = len(df[df.is_alive])

        force_of_infection = (params['transmission_rate'] * active_hiv_neg * (active_hiv_pos * params[
            'rel_infectiousness_hiv']) * uninfected_total) / total_population
        # print('force_of_infection: ', force_of_infection)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~ NEW INFECTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~

        #  everyone at same risk of latent infection
        prob_tb_new = pd.Series(force_of_infection, index=df[(df.tb_inf == 'uninfected') & df.is_alive].index)
        # print('prob_tb_new: ', prob_tb_new)
        is_newly_infected = prob_tb_new > rng.rand(len(prob_tb_new))
        new_case = is_newly_infected[is_newly_infected].index
        df.loc[new_case, 'tb_inf'] = 'latent_susc'
        df.loc[new_case, 'tb_date_latent'] = now

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~ FAST PROGRESSORS TO ACTIVE DISEASE ~~~~~~~~~~~~~~~~~~~~~~~~~~

        # if any newly infected latent cases, 14% become active directly
        new_latent = df[(df.tb_inf == 'latent_susc') & (df.tb_date_latent == now) & df.is_alive].sum()
        # print(new_latent)

        if new_latent.any():
            fast_progression = df[(df.tb_inf == 'latent_susc') & (df.tb_date_latent == now) & df.is_alive].sample(
                frac=params['prop_fast_progressor']).index
            df.loc[fast_progression, 'tb_inf'] = 'active_susc'
            df.loc[fast_progression, 'tb_date_active'] = now

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~ SLOW PROGRESSORS TO ACTIVE DISEASE ~~~~~~~~~~~~~~~~~~~~~~~~~~
        # this could also be a relapse event

        # slow progressors with latent TB become active
        # random sample with weights for RR of active disease
        eff_prob_active_tb = pd.Series(0, index=df.index)
        eff_prob_active_tb.loc[(df.tb_inf == 'latent_susc')] = params['progression_to_active_rate']
        # print('eff_prob_active_tb: ', eff_prob_active_tb)

        hiv_stage1 = df.index[df.hiv_inf & (df.tb_inf == 'latent_susc') &
                              (((now - df.hiv_date_inf).dt.days / 365.25) < 3.33)]
        # print('hiv_stage1', hiv_stage1)

        hiv_stage2 = df.index[df.hiv_inf & (df.tb_inf == 'latent_susc') &
                              (((now - df.hiv_date_inf).dt.days / 365.25) >= 3.33) &
                              (((now - df.hiv_date_inf).dt.days / 365.25) < 6.67)]
        # print('hiv_stage2', hiv_stage2)

        hiv_stage3 = df.index[df.hiv_inf & (df.tb_inf == 'latent_susc') &
                              (((now - df.hiv_date_inf).dt.days / 365.25) >= 6.67) &
                              (((now - df.hiv_date_inf).dt.days / 365.25) < 10)]
        # print('hiv_stage3', hiv_stage3)

        hiv_stage4 = df.index[df.hiv_inf & (df.tb_inf == 'latent_susc') &
                              (((now - df.hiv_date_inf).dt.days / 365.25) >= 10)]
        # print('hiv_stage4', hiv_stage4)

        eff_prob_active_tb.loc[hiv_stage1] *= params['rr_tb_with_hiv_stages'][0]
        eff_prob_active_tb.loc[hiv_stage2] *= params['rr_tb_with_hiv_stages'][1]
        eff_prob_active_tb.loc[hiv_stage3] *= params['rr_tb_with_hiv_stages'][2]
        eff_prob_active_tb.loc[hiv_stage4] *= params['rr_tb_with_hiv_stages'][3]
        eff_prob_active_tb.loc[df.hiv_on_art == '2'] *= params['rr_tb_art']
        # eff_prob_active_tb.loc[df.is_malnourished] *= params['rr_tb_malnourished']
        # eff_prob_active_tb.loc[df.has_diabetes1] *= params['rr_tb_diabetes1']
        # eff_prob_active_tb.loc[df.high_alcohol] *= params['rr_tb_alcohol']
        # eff_prob_active_tb.loc[df.is_smoker] *= params['rr_tb_smoking']
        # eff_prob_active_tb.loc[df.high_pollution] *= params['rr_tb_pollution']

        prog_to_active = eff_prob_active_tb > rng.rand(len(eff_prob_active_tb))
        # print('prog_to_active: ', prog_to_active )
        new_active_case = prog_to_active[prog_to_active].index
        # print('new_active_case: ', new_active_case)
        df.loc[new_active_case, 'tb_inf'] = 'active_susc'
        df.loc[new_active_case, 'tb_date_active'] = now

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~ SELF CURE ~~~~~~~~~~~~~~~~~~~~~~~~~~

        # self-cure - move back from active to latent, make sure it's not the ones that just became active
        self_cure_tb = df[(df.tb_inf == 'active_susc') & df.is_alive & (df.tb_date_active < now)].sample(
            frac=(params['prob_self_cure'] * params['rate_self_cure'])).index
        df.loc[self_cure_tb, 'tb_inf'] = 'latent_susc'


# TODO: tb_mdr should also be a risk for people with latent_susc status
# when they move back to latent after treatment / self-cure, they stay as latent_mdr


class tb_mdr_event(RegularEvent, PopulationScopeEventMixin):
    """ tb infection events
    """

    def __init__(self, module):
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

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~ FORCE OF INFECTION ~~~~~~~~~~~~~~~~~~~~~~~~~~

        active_hiv_neg = len(df[(df.tb_inf == 'active_mdr') & ~df.hiv_inf & df.is_alive])
        active_hiv_pos = len(df[(df.tb_inf == 'active_mdr') & df.hiv_inf & df.is_alive])
        # TODO: include latent_susc as a susceptible pop here also?
        uninfected_total = len(df[(df.tb_inf == 'uninfected') & df.is_alive])
        total_population = len(df[df.is_alive])

        force_of_infection = (params['transmission_rate'] * active_hiv_neg * (active_hiv_pos * params[
            'rel_infectiousness_hiv']) * uninfected_total) / total_population
        # print('force_of_infection: ', force_of_infection)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~ NEW INFECTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~

        #  everyone at same risk of latent infection
        prob_tb_new = pd.Series(force_of_infection, index=df[(df.tb_inf == 'uninfected') & df.is_alive].index)
        # print('prob_tb_new: ', prob_tb_new)
        is_newly_infected = prob_tb_new > rng.rand(len(prob_tb_new))
        new_case = is_newly_infected[is_newly_infected].index
        df.loc[new_case, 'tb_inf'] = 'latent_mdr'
        df.loc[new_case, 'tb_date_latent'] = now

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~ FAST PROGRESSORS TO ACTIVE DISEASE ~~~~~~~~~~~~~~~~~~~~~~~~~~

        # if any newly infected latent cases, 14% become active directly
        new_latent = df[(df.tb_inf == 'latent_mdr') & (df.tb_date_latent == now) & df.is_alive].sum()
        # print(new_latent)

        if new_latent.any():
            fast_progression = df[(df.tb_inf == 'latent_mdr') & (df.tb_date_latent == now) & df.is_alive].sample(
                frac=params['prop_fast_progressor']).index
            df.loc[fast_progression, 'tb_inf'] = 'active_mdr'
            df.loc[fast_progression, 'tb_date_active'] = now

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~ SLOW PROGRESSORS TO ACTIVE DISEASE ~~~~~~~~~~~~~~~~~~~~~~~~~~
        # this could also be a relapse event

        # slow progressors with latent TB become active
        # random sample with weights for RR of active disease
        eff_prob_active_tb = pd.Series(0, index=df.index)
        eff_prob_active_tb.loc[(df.tb_inf == 'latent_mdr')] = params['progression_to_active_rate']
        # print('eff_prob_active_tb: ', eff_prob_active_tb)

        hiv_stage1 = df.index[df.hiv_inf & (df.tb_inf == 'latent_mdr') &
                              (((now - df.hiv_date_inf).dt.days / 365.25) < 3.33)]
        # print('hiv_stage1', hiv_stage1)

        hiv_stage2 = df.index[df.hiv_inf & (df.tb_inf == 'latent_mdr') &
                              (((now - df.hiv_date_inf).dt.days / 365.25) >= 3.33) &
                              (((now - df.hiv_date_inf).dt.days / 365.25) < 6.67)]
        # print('hiv_stage2', hiv_stage2)

        hiv_stage3 = df.index[df.hiv_inf & (df.tb_inf == 'latent_mdr') &
                              (((now - df.hiv_date_inf).dt.days / 365.25) >= 6.67) &
                              (((now - df.hiv_date_inf).dt.days / 365.25) < 10)]
        # print('hiv_stage3', hiv_stage3)

        hiv_stage4 = df.index[df.hiv_inf & (df.tb_inf == 'latent_mdr') &
                              (((now - df.hiv_date_inf).dt.days / 365.25) >= 10)]
        # print('hiv_stage4', hiv_stage4)

        eff_prob_active_tb.loc[hiv_stage1] *= params['rr_tb_with_hiv_stages'][0]
        eff_prob_active_tb.loc[hiv_stage2] *= params['rr_tb_with_hiv_stages'][1]
        eff_prob_active_tb.loc[hiv_stage3] *= params['rr_tb_with_hiv_stages'][2]
        eff_prob_active_tb.loc[hiv_stage4] *= params['rr_tb_with_hiv_stages'][3]
        eff_prob_active_tb.loc[df.hiv_on_art == '2'] *= params['rr_tb_art']
        # eff_prob_active_tb.loc[df.is_malnourished] *= params['rr_tb_malnourished']
        # eff_prob_active_tb.loc[df.has_diabetes1] *= params['rr_tb_diabetes1']
        # eff_prob_active_tb.loc[df.high_alcohol] *= params['rr_tb_alcohol']
        # eff_prob_active_tb.loc[df.is_smoker] *= params['rr_tb_smoking']
        # eff_prob_active_tb.loc[df.high_pollution] *= params['rr_tb_pollution']

        prog_to_active = eff_prob_active_tb > rng.rand(len(eff_prob_active_tb))
        # print('prog_to_active: ', prog_to_active )
        new_active_case = prog_to_active[prog_to_active].index
        # print('new_active_case: ', new_active_case)
        df.loc[new_active_case, 'tb_inf'] = 'active_mdr'
        df.loc[new_active_case, 'tb_date_active'] = now

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~ SELF CURE ~~~~~~~~~~~~~~~~~~~~~~~~~~

        # self-cure - move back from active to latent, make sure it's not the ones that just became active
        if len(df[(df.tb_inf == 'active_mdr') & df.is_alive & (df.tb_date_active < now)]) > 10:
            self_cure_tb = df[(df.tb_inf == 'active_mdr') & df.is_alive & (df.tb_date_active < now)].sample(
                frac=(params['prob_self_cure'] * params['rate_self_cure'])).index
            df.loc[self_cure_tb, 'tb_inf'] = 'latent_mdr'


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

        mortality_rate = pd.Series(0, index=df.index)
        mortality_rate.loc[((df.tb_inf == 'active_susc') | (df.tb_inf == 'active_mdr')) & ~df.hiv_inf] = params[
            'tb_mortality_rate']
        mortality_rate.loc[((df.tb_inf == 'active_susc') | (df.tb_inf == 'active_mdr')) & df.hiv_inf] = params[
            'tb_mortality_hiv']
        # print('mort_rate: ', mortality_rate)

        # Generate a series of random numbers, one per individual
        probs = rng.rand(len(df))
        deaths = df.is_alive & (probs < mortality_rate)
        # print('deaths: ', deaths)
        will_die = (df[deaths]).index
        # print('will_die: ', will_die)

        # TODO: add in treatment status as conditions for death

        for person in will_die:
            if df.at[person, 'is_alive']:
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id=person, cause='tb'),
                                        now)
                df.at[person, 'tb_date_death'] = now


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

        active_tb_susc = len(df[(df.tb_inf == 'active_susc') & df.is_alive])
        active_tb_mdr = len(df[(df.tb_inf == 'active_mdr') & df.is_alive])

        coinfected_total = len(
            df[((df.tb_inf == 'active_susc') | (df.tb_inf == 'active_mdr')) & df.hiv_inf & df.is_alive])

        self.module.store['Time'].append(self.sim.date)
        self.module.store['Total_active_tb'].append(active_tb_susc)
        self.module.store['Total_active_tb_mdr'].append(active_tb_mdr)
        self.module.store['Total_co-infected'].append(coinfected_total)

        # print('tb outputs: ', self.sim.date, active_tb_total, coinfected_total)
