"""
This is a simplified births Module. it aims at implementing some simple events to generate births thereby avoid
registering of some heavy modules to do the same

"""
import numpy as np
import pandas as pd
from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import PopulationScopeEventMixin, RegularEvent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Simplifiedbirths(Module):
    """
    a simplified births module responsible for generating births in a simplified way and assign mother ids to newborns
    """
    # Declare Metadata (this is for a typical 'Disease Module')
    METADATA = {}

    PARAMETERS = {
        'birth_prob': Parameter(
            Types.REAL, 'probability of births in a month')
    }

    PROPERTIES = {
        'si_date_of_last_delivery': Property(Types.DATE, 'woman\'s date of last delivery')
    }

    # Declare the non-generic symptoms that this module will use.
    SYMPTOMS = {}

    def __init__(self, name=None, resourcefilepath=None):
        # NB. Parameters passed to the module can be inserted in the __init__ definition.

        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.
        :param data_folder: path of a folder supplied to the Simulation containing data files.
        """
        self.parameters['birth_prob'] = 0.01

    def initialise_population(self, population):
        """Set our property values for the initial population.
        :param population: the population of individuals
        """
        df = population.props  # a shortcut to the dataframe storing data for individuals

        # Assign the characteristics
        df.loc[df.is_alive, 'si_date_of_last_delivery'] = pd.NaT

    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.

        """
        # check all population to determine if birth should be triggered (repeats every month)
        sim.schedule_event(SimplifiedBirthsEvent(self), sim.date + DateOffset(months=1))

        # Launch the repeating event that will store statistics about the population structure
        sim.schedule_event(SimplifiedBirthsLoggingEvent(self), sim.date + DateOffset(days=0))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        df = self.sim.population.props

        # Assign the date of last delivery to a newborn
        df.at[child_id, 'si_date_of_last_delivery'] = pd.NaT

        logger.info('%s|post_birth_info|%s',
                    self.sim.date,
                    {
                        'woman_index': mother_id,
                        'child_index': child_id

                    })


class SimplifiedBirthsEvent(RegularEvent, PopulationScopeEventMixin):
    """A Simplified births class for an event

    Regular events automatically reschedule themselves at a fixed frequency,
    and thus implement discrete timestep type behaviour. The frequency is
    specified when calling the base class constructor in our __init__ method.
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        self.age_low = 15
        self.age_high = 49
        self.birth_prob = module.parameters['birth_prob']

    def apply(self, population):
        df = self.sim.population.props  # get the population dataframe

        selected_women = (df.sex == 'F') & df.is_alive & df.age_years.between(self.age_low, self.age_high)

        # print(selected_women)

        # determining which woman should give birth
        new_births = (self.module.rng.random_sample(size=len(selected_women)) < self.birth_prob)

        births_ids = selected_women.index[new_births]

        # assigning the date of delivery to the selected women
        df.loc[births_ids, 'si_date_of_last_delivery'] = self.sim.date

        for female_id in births_ids:
            # generating a  child
            logger.info(f'@@@@ A Birth is now occurring, to mother {female_id} on date {self.sim.date}')
            self.sim.do_birth(female_id)


class SimplifiedBirthsLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """Logs outputs for simplified_births module
        """
        # run this event every 12 months (every year)
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        df = population.props

        one_year_prior = self.sim.date - np.timedelta64(1, 'Y')

        total_births_last_year = len(df.index[(df.date_of_birth > one_year_prior) & (df.date_of_birth < self.sim.date)])

        # if total_births_last_year == 0:
        #     total_births_last_year = 1

        # logger.info(f'{self.sim.date} |total_births| {total_births_last_year}')
        logger.info('%s|total_births|%s',
                    self.sim.date,
                    {
                        'total': total_births_last_year
                    })
