"""
This is a simplified births Module. it aims at implementing some simple events to generate births thereby avoid
registering heavy modules(contraception, labour and pregnant supervisor) to do the same

"""
import numpy as np
import pandas as pd
from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import PopulationScopeEventMixin, RegularEvent, IndividualScopeEventMixin, Event

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Simplifiedbirths(Module):
    """
    a simplified births module responsible for generating births in a simplified way and assign mother ids to newborns
    """
    # Declare Metadata
    METADATA = {}

    # Here we declare module parameters. we specify the name, type and a longer description
    PARAMETERS = {
        'pregnancy_prob': Parameter(
            Types.REAL, 'probability of pregnancies in a month'),
        'min_age': Parameter(
            Types.REAL, 'minimum age of individuals'),
        'max_age': Parameter(
            Types.REAL, 'maximum age of individuals'),
        'days_until_delivery': Parameter(
            Types.DATE, 'number of days a pregnant woman has to go through until delivery'
                        ' (assuming term delivery (37-41 weeks)), '),
        'prob_breastfeeding_type': Parameter(
            Types.LIST, 'probabilities for breastfeeding status: none, non_exclusive, exclusive')

    }

    # Next we declare module properties. again we specify the name, type and a longer description
    PROPERTIES = {
        'is_pregnant': Property(Types.BOOL, 'Whether this individual is currently pregnant'),
        'date_of_last_pregnancy': Property(Types.DATE,
                                           'Date of the last pregnancy of this individual'),
        'si_date_of_delivery': Property(Types.DATE,
                                        'expected date of delivery for this individual'),
        'nb_breastfeeding_status': Property(Types.CATEGORICAL, 'Breastfeeding status of a newborn',
                                            categories=['none', 'non_exclusive', 'exclusive']),
    }

    def __init__(self, name=None, resourcefilepath=None):
        # NB. Parameters passed to the module can be inserted in the __init__ definition.

        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    def read_parameters(self, data_folder):
        # initialising our parameters

        param = self.parameters
        param['pregnancy_prob'] = 0.01
        param['min_age'] = 15
        param['max_age'] = 49
        param['days_until_delivery'] = pd.DateOffset(days=7 * 37 + self.rng.randint(0, 7 * 4))
        self.parameters['prob_breastfeeding_type'] = [0.101, 0.289, 0.61]

    def initialise_population(self, population):
        # Set our property values for the initial population.

        df = population.props  # a shortcut to the dataframe storing data for individuals

        # Assign the characteristics
        df.loc[df.is_alive, 'is_pregnant'] = False
        df.loc[df.is_alive, 'date_of_last_pregnancy'] = pd.NaT
        df.loc[df.is_alive, 'si_date_of_delivery'] = pd.NaT
        df.loc[df.is_alive, 'nb_breastfeeding_status'] = 'none'

    def initialise_simulation(self, sim):
        """Get ready for simulation start.
        """
        # check all population to determine who will get pregnant (repeats every month)
        sim.schedule_event(SimplifiedPregnancyEvent(self), sim.date + DateOffset(months=1))

        # Launch the repeating event that will store statistics about the population structure
        # sim.schedule_event(SimplifiedBirthsLoggingEvent(self), sim.date + DateOffset(years=1))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.
        """
        df = self.sim.population.props
        params = self.parameters

        # Assign the date of last delivery to a newborn
        df.at[child_id, 'is_pregnant'] = False
        df.at[child_id, 'date_of_last_pregnancy'] = pd.NaT
        df.at[child_id, 'si_date_of_delivery'] = pd.NaT

        # Assign breastfeeding status to newborns
        random_draw = self.rng.choice(('none', 'non_exclusive', 'exclusive'), p=params['prob_breastfeeding_type'])
        df.at[child_id, 'nb_breastfeeding_status'] = random_draw

        # logger.info('%s|post_birth_info|%s',
        #             self.sim.date,
        #             {
        #                 'woman_index': mother_id,
        #                 'child_index': child_id
        #
        #             })


class SimplifiedPregnancyEvent(RegularEvent, PopulationScopeEventMixin):
    """ A class responsible for making women pregnant and scheduling birth event at their delivery date
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        self.age_low = module.parameters['min_age']  # min preferred age
        self.age_high = module.parameters['max_age']  # max preferred age
        self.pregnancy_prob = module.parameters['pregnancy_prob']   # probability of women to get pregnant in
        # a particular month
        self.days_until_delivery = module.parameters['days_until_delivery']  # number of days until delivery
        # (term delivery)

    def apply(self, population):
        df = self.sim.population.props  # get the population dataframe

        # select females from dataframe who are not pregnant and of age between 15 - 49
        selected_women = df.loc[(df.sex == 'F') & df.is_alive & ~df.is_pregnant & df.age_years.between(self.age_low,
                                                                                                       self.age_high)]

        # determining which woman should get pregnant
        new_pregnancies = (self.module.rng.random_sample(size=len(selected_women.index)) < self.pregnancy_prob)

        pregnant_women_ids = selected_women.index[new_pregnancies]

        # updating properties for selected women
        df.loc[pregnant_women_ids, 'is_pregnant'] = True
        df.loc[pregnant_women_ids, 'date_of_last_pregnancy'] = self.sim.date
        df.loc[pregnant_women_ids, 'si_date_of_delivery'] = self.sim.date + self.days_until_delivery

        # print(pregnant_women_ids)

        for pregnant_woman_id in pregnant_women_ids:
            self.sim.schedule_event(SimplifiedBirthsEvent(self, pregnant_woman_id), df.loc[pregnant_woman_id,
                                                                                           'si_date_of_delivery'])


class SimplifiedBirthsEvent(Event, IndividualScopeEventMixin):
    """This is the BirthEvent. It is scheduled by SimplifiedBirthsEvent to allow women who have been pregnant
    for 9 months give birth"""
    def __init__(self, module, mother_id):
        super().__init__(module, person_id=mother_id)

    def apply(self, mother_id):

        logger.debug(f'{self.sim.date} | @@@@ A Birth is now occuring, to mother {mother_id} '
                     f' on | {self.sim.date}')
        self.sim.do_birth(mother_id)


# class SimplifiedBirthsLoggingEvent(RegularEvent, PopulationScopeEventMixin):
#     def __init__(self, module):
#         """Logs outputs for simplified_births module
#         """
#         # run this event every 12 months (every year)
#         self.repeat = 12
#         super().__init__(module, frequency=DateOffset(months=self.repeat))
#
#     def apply(self, population):
#         df = population.props
#
#         one_year_prior = self.sim.date - np.timedelta64(1, 'Y')
#
#         total_births_last_year = len(
#             df.index[(df.date_of_birth > one_year_prior) & (df.date_of_birth < self.sim.date)])  #
#         # logger.info(f'{self.sim.date} |total_births| {total_births_last_year}')
#         logger.info('%s|total_births|%s',
#                     self.sim.date - np.timedelta64(1, 'Y') , {'total': total_births_last_year})
