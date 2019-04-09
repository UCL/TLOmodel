"""
Contraception module covering baseline contraception methods use, failure (pregnancy),
Switching contraceptive methods, and discontiuation rates by age
please see Dropbox/Thanzi la Onse/05 - Resources/Programming Notes/Contraception-Pregnancy.pdf
for conceptual diagram
"""
import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent
from tlo.methods.demography import Demography

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Contraception(Module):
    """
    Contraception module covering baseline contraception methods use, failure (pregnancy),
    Switching contraceptive methods, and discontiuation rates by age
    please see Dropbox/Thanzi la Onse/05 - Resources/Programming Notes/Contraception-Pregnancy.pdf
    for conceptual diagram (lucid chart)
    """

    def __init__(self, name=None, workbook_path=None):
        super().__init__(name)
        self.workbook_path = workbook_path

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    PARAMETERS = {
        'fertility_schedule': Parameter(Types.DATA_FRAME, 'Age_spec fertility'),
        # TODO: fertlity schedule (baseline fertlity) is currently in demography should it only be here in contraception?
        'contraception_initiation1': Parameter(Types.DATA_FRAME, 'irate1'),
        'contraception_initiation2': Parameter(Types.DATA_FRAME, 'irate2'),
        'contraception_discontinuation_failure_switching': Parameter(Types.DATA_FRAME,
                                                                     'contraception_failure_discontin')
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'contraception': Property(Types.CATEGORICAL, 'Current contraceptive method',
                                  categories=['not_using', 'pill', 'IUD', 'injections', 'implant', 'male_condom',
                                              'female_sterilization', 'other_modern', 'periodic_abstinence',
                                              'withdrawal', 'other_traditional'])
        # These are the 11 categories of contraception ('not using' + 10 methods) from the DHS analysis of initiation,
        # discontinuation, failure and switching rates
        # 'other modern' includes Male sterilization, Female Condom, Emergency contraception
        # 'other traditional' includes lactational amenohroea (LAM), standard days method (SDM), 'other traditional method'
        # Have replaced Age-spec fertility sheet in demography.xlsx (in this branch) with the one in contraception.xlsx
        # (has 11 categories and one row for each age with baseline contraceptopn prevalences for each of the 11 categories)
    }

    def read_parameters(self, data_folder):
        """
        Please see Contraception-Pregnancy.pdf for lucid chart explaining the relationships between
        baseline fertility rate, intitiation rates, discontinuation, failure and switching rates, and being on
        contraception or not, and being pregnant
        """
        workbook = pd.read_excel(self.workbook_path, sheet_name=None)
        self.parameters['fertility_schedule'] = workbook['Age_spec fertility']
        self.parameters['contraception_initiation1'] = workbook['irate1']
        # this Excel sheet is irate1_all.csv output from 'initiation rates_age_stcox.do' Stata analysis of DHS contraception calendar data
        self.parameters['contraception_initiation2'] = workbook['irate2']
        # this Excel sheet is irate2_all.csv output from 'initiation rates_age_stcox.do' Stata analysis of DHS contraception calendar data
        self.parameters['contraception_discontinuation_failure_switching'] = workbook['contraception_failure_discontin']
        # this Excel sheet is contraception_failure_discontinuation_switching_by_age.csv output from 'discontinuation & switching rates_age.do' Stata analysis of DHS contraception calendar data

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.
        """
        df = population.props
        fertility = self.parameters['fertility_schedule']

        # Assign contraception method
        # 1. select females aged 15-49 from population, for current year
        possibly_using = df.is_alive & (df.sex == 'F') & df.age_years.between(15, 49)
        females1549 = df.index[possibly_using]
        # 2. Get probabilities of using each method of contraception by age
        # contraception_probs = df.loc[fertility, ['not_using', 'pill', 'IUD', 'injections', 'implant', 'male_condom',
        #                                         'female_sterilization', 'other_modern', 'periodic_abstinence',
        #                                         'withdrawal', 'other_traditional']].age
        # 2. merge the probabilities into each row in sim population
        df_new = df.merge(fertility, left_on=['age_years'], right_on=['age'], how='left')

        probabilities = df_new.loc[females1549, ['not_using', 'pill', 'IUD', 'injections', 'implant', 'male_condom',
                                              'female_sterilization', 'other_modern', 'periodic_abstinence',
                                              'withdrawal', 'other_traditional']]

        # 3. apply probabilities of each contraception type to sim population
        for woman in probabilities.index:
            her_p=np.asarray(probabilities.loc[woman,:])
            her_op=np.asarray(probabilities.columns)

            her_method=self.rng.choice(her_op,p=her_p/her_p.sum())  # /her_p.sum() added becasue probs sometimes add to not quite 1 due to rounding

            df.loc[woman,'contraception']=her_method



        # 3. apply probabilities of each contraception type to sim population
        categories = ['not using', 'pill', 'IUD', 'injection', 'implant', 'male condom', 'female sterilization',
                      'other modern', 'periodic abstinence', 'withdrawal', 'other traditional']
        random_choice = self.rng.choice(categories, size=len(probabilities), p=probabilities)
        df.loc[females1549, 'contraception'].values[:] = random_choice



    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """
        raise NotImplementedError

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        raise NotImplementedError


class ContraceptionEvent(RegularEvent, PopulationScopeEventMixin):
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
        raise NotImplementedError
