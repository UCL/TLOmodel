"""
First draft of Labour module (Natural history)
Documentation:
"""

import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent


class Labour (Module):
    """
    This module models labour and delivery and generates the properties for "complications" of delivery """

    PARAMETERS = {
        'r_initial_prob_labour_term': Parameter(
            Types.REAL, 'probabilities of a woman entering one of the 4 states of labour'),
       # ??? REMOVE ???
        'r_SL': Parameter(
            Types.REAL, 'probability of a woman entering spontaneous labour'),
        'r_induction': Parameter(
            Types.REAL, 'probability of a womans labour being induced'),
        'r_planned_CS': Parameter(
            Types.REAL, 'probability of a woman undergoing a planned caesarean section'),
        'r_abortion': Parameter(
            Types.REAL, 'probability of a woman experiencing an abortion before 28 weeks gestation'),
        'r_PL/OL': Parameter(
            Types.REAL, 'probability of a woman entering prolonged/obstructed labour'),
        'rr_PL/OL_nuliparity': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if they are nuliparous'),
        'rr_PL/OL_parity+3':Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if her parity is >3'),
        'rr_PL/OL_age<20': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if her age is less than '
                            '20 years'),
        'rr_PL/OL_baby>3.5g': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if her baby weighs more than 3.5'
                        'kilograms'),
        'rr_PL/OL_baby<1.5g': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if her baby weighs less than 1.5'
                        'kilograms'),
        'rr_PL/OL_bmi<18': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if her BMI is less than 18'),
        'rr_PL/OL_bmi>25': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if her BMI is greater than 25'),
        'r_ptb': Parameter (
            Types.REAL, 'probability of a woman entering labour at <37 weeks gestation'),
        'rr_ptb_malaria': Parameter (
            Types.REAL, 'relative risk of a woman with malaria experiencing pre term labour'), # determine malaria definition
         'r_live_birth_SL' : Parameter(
             Types.REAL, 'probability of live birth following spontaneous labour'),
        'r_live_birth_IL': Parameter(
            Types.REAL, 'probability of live birth following induced labour'),
        'r_live_birth_planned_CS': Parameter(
            Types.REAL, 'probability of live birth following a planned caesarean section'),
        'r_live_birth_PL': Parameter(
            Types.REAL, 'probability of live birth following prolonged labour'),
        'r_live_birth_OL': Parameter(
            Types.REAL, 'probability of live birth following obstructed labour'),
        'r_still_birth_SL': Parameter (
            Types.REAL, 'probability of still birth following spontaneous labour'),
        'r_still_birth_IL': Parameter(
            Types.REAL, 'probability of still birth following induced labour'),
        'r_still_birth_PL': Parameter(
            Types.REAL, 'probability of still birth following prolonged labour'),
        'r_still_birth_OL': Parameter(
            Types.REAL, 'probability of still birth following obstructed labour'),
        'r_still_birth_abortion': Parameter(
            Types.REAL, 'probability of still birth an abortion')
    }

    PROPERTIES = {
        'la_labour': Property(Types.CATEGORICAL,'not in labour, spontaneous unobstructed labour, prolonged or '
                                                'obstructed labour, Preterm Labour',
                                categories=['not_in_labour', 'spontaneous_unobstructed_labour',
                                            'prolonged_or_obstructed_labour','pretterm_labour']),

        # Have made 4 categories, combined PL and OL into one "state" and added preterm labour
        # for initialisation = r_labour_term = [0.05, 0.70, 0.10, 0.05]

        'la_planned_cs': Property(Types.BOOL, 'pregnant woman undergoes an elective caesarean section'),
        'la_induced_labour': Property(Types.BOOL, 'pregnant woman has labour induced'),
        'la_abortion': Property(Types.DATE, 'the date on which a pregnant has had an abortion'),
        # date variable - what do we set this as for intialisation
        'la_live_birth': Property(Types.BOOL, 'labour ends in a live birth'),
        'la_still_birth': Property(Types.BOOL, 'labour ends in a still birth'),
        'la_parity': Property(Types.INT, 'total number of deliveries'),
        'la_previous_cs': Property(Types.INT, 'total number of previous deliveries by caesarean section'),
        'la_immediate_postpartum': Property(Types.BOOL, ' postpartum period from delivery to 48 hours post')

    }


    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we do nothing.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        params = self.parameters
        params['r_initial_prob_labour_term'] = [0.05, 0.70, 0.10, 0.05] # ??? REMOVE ???
        params['r_SL'] = 0.80
        params['r_induction'] = 0.10
        params['r_planned_CS'] = 0.03
        params['r_abortion'] = 0.15
        params['r_PL/OL'] = 0.06
        params['rr_PL/OL_nuliparity'] = 1.8
        params['r_PL/OL_parity3+'] = 0.8
        params['r_PL/OL_age<20'] = 1.3
        params['r_PL/OL_baby>3.5kg'] = 1.2
        params['r_PL/OL_baby<1.5kg'] = 0.7
        params['r_PL/OL_bmi<18'] = 0.8
        params['r_PL/OL_bmi>25'] = 1.4
        params['r_PLT'] = 0.10
        params['rr_PLT_malaria'] = 1.4
        params['r_live_birth_SL'] = 0.85
        params['r_live_birth_IL'] = 0.8
        params['r_live_birth_PL'] = 0.75
        params['r_live_birth_OL'] = 0.7
        params['r_still_birth_SL'] = 0.15
        params['r_still_birth_IL'] = 0.2
        params['r_still_birth_PL'] = 0.25
        params['r_still_birth_OL'] = 0.3

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        df = population.props  # a shortcut to the data-frame storing data for individuals
        m = self
        rng = m.rng

    # ----------------------------------------- DEFAULTS----------------------------------------------------------------

        df['la_labour'] = 'not_in_labour' # im not sure this is right
        df['la_planned_CS'] = False
        df['la_induced_labour'] = False
        df['la_abortion'] = pd.NaT
        df['la_live_birth'] = False
        df['la_still_birth'] = False
        df['la_partiy'] = 0
        df['la_previous_cs'] = 0
        df['la_immediate_postpartum']= False

    # -----------------------------------ASSIGN LIKELIHOOD OF LABOUR AT BASELINE--------------------------------------

        # create data frame of the probabilities of a pregnant (term) woman entering labour
        # this could by just setting a likelihood of term pregnant women entering labour on baseline
        # would need to generate previous ANC and other obstetric history factors

        pregnant_nolab = df.index[(df.age_years >= 15) & df.is_alive & df.is_pregnant]

        # how do i offest by the date of pregnancy to ensure only women at term are selected

        p_preg_lab = pd.DataFrame(data=[m.r_labour_term],
                                  columns=['not in labour', 'spontaneous labour', 'prolonged labour',
                                           'obstructed labour'], index=pregnant_nolab)

        # ISSUE: no women are generated as pregnant by demography?
        # TO DO consider women who enter labour through induction and planned CS on initialisation


    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        event = LabourEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=1))

        event = LabourLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=0))


    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the mother for this child
        :param child_id: the new child
        """

        df = self.sim.population.props

        df.at[child_id, 'la_labour'] = 'not_in_labour'
        df.at[child_id, 'la_planned_CS'] = False
        df.at[child_id, 'la_induced_labor'] = False
        df.at[child_id, 'la_abortion'] = pd.NaT
        df.at[child_id, 'la_live_birth'] = False
        df.at[child_id, 'la_still_birth'] = False
        df.at[child_id, 'la_parity'] = 0
        df.at[child_id, 'la_previous_cs'] = 0
        df.at[child_id, 'la_immediate_postpartum'] = False


class LabourEvent(RegularEvent, PopulationScopeEventMixin):
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
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        """Apply this event to the population.

        :param population: the current population
        """

        df = population.props
        m = self.module
        rng = m.rng

        f_pregnant_not_in_labour_idx = df.index[df.is_alive & (df.sex == 'F') & df.age_years >= 15
                                                & ~df.la_spontaneous_labour]

        eff_prob_spontaneous_labour = m.r_SL
        random_draw = pd.Series(rng.random_sample(size=len(f_pregnant_not_in_labour_idx)),
                                index=df.index[df.is_alive & (df.sex == 'F') & df.age_years >= 16
                                               & ~df.bi_spontaneous_labour],
        dfx = pd.concat([eff_prob_spontaneous_labour, random_draw], axis=1),)


class LabourLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """Handles lifestyle logging"""
    def __init__(self, module):
        """schedule logging to repeat every 3 months
        """
        self.repeat = 1
        super().__init__(module, frequency=DateOffset(days=self.repeat))

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        # get some summary statistics
 #      df = population.props

        # calculate incidence of oesophageal cancer diagnosis in people aged > 60+
        # (this includes people diagnosed with dysplasia, but diagnosis rate at this stage is very low)

#       incident_oes_cancer_diagnosis_agege60_idx = df.index[df.ca_incident_oes_cancer_diagnosis_this_3_month_period
#       & (df.age_years >= 60)]
#       agege60_without_diagnosed_oes_cancer_idx = df.index[(df.age_years >= 60) & ~df.ca_oesophagus_diagnosed]

#       incidence_per_year_oes_cancer_diagnosis = (4 * 100000 * len(incident_oes_cancer_diagnosis_agege60_idx))/\
#                                                 len(agege60_without_diagnosed_oes_cancer_idx)

#       incidence_per_year_oes_cancer_diagnosis = round(incidence_per_year_oes_cancer_diagnosis, 3)

#       logger.debug('%s|person_one|%s',
#                      self.sim.date,
#                      df.loc[0].to_dict())

#       logger.info('%s|ca_oesophagus|%s',
#                   self.sim.date,
#                   df[df.is_alive].groupby(['ca_oesophagus']).size().to_dict())

        # note below remove is_alive
#       logger.info('%s|ca_oesophagus_death|%s',
#                   self.sim.date,
#                   df[df.age_years >= 20].groupby(['ca_oesophageal_cancer_death']).size().to_dict())


#       logger.info('%s|ca_incident_oes_cancer_diagnosis_this_3_month_period|%s',
#                   self.sim.date,
#                   incidence_per_year_oes_cancer_diagnosis)


#       logger.info('%s|ca_oesophagus_diagnosed|%s',
#                   self.sim.date,
#                   df[df.age_years >= 20].groupby(['ca_oesophagus', 'ca_oesophagus_diagnosed']).size().to_dict())

#       logger.info('%s|ca_oesophagus|%s',
#                   self.sim.date,
#                   df[df.is_alive].groupby(['age_range', 'ca_oesophagus']).size().to_dict())
