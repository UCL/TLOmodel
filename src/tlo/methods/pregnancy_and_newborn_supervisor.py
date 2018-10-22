""""
Introducing the ... pregnancy_and_newborn_supervisor

The module look after pregnant women and their children until the time that the pregnancy is ended or the newborn reaches 4 weeks, whichever is later

"""

import numpy as np
import pandas as pd

from tlo import Module, Parameter, Property, Types, DateOffset
from tlo.events import Event, PopulationScopeEventMixin, RegularEvent, IndividualScopeEventMixin

class Pregnancy_And_Newborn_Supervisor(Module):

    # Module parameters
    PARAMETERS = {
    }

    # Properties of individuals 'owned' by this module
    PROPERTIES = {
    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we just assign parameter values explicitly.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        params = self.parameters  # To save typing!
        pass


    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.
        """

        pass


    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """


    def on_birth(self, mother, child):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother: the mother for this child
        :param child: the new child
        """



#
# class Pregnancy_And_Newborn_Supervisor_Event(RegularEvent,IndividualScopeEventMixin):
#     """
#     This is called every week during a woman's pregnancy and until their child is 4 weeks old
#
#     """
#
#     def __init__(self, module, individual):
#         # set this event to repeat weekly
#         super().__init__(module, person=individual, frequency=DateOffset(weeks=1))
#
#
#     def apply(self, person):
#         # print("We are checking up on the pregnancy of ", person, "....")
#
#
