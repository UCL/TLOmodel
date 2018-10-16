"""
The core demography module and its associated events.

Expects input in format of the 'Demography.xlsx'  of TimH, sent 3/10. Uses the 'Interpolated
population structure' worksheet within to initialise the age & sex distribution of population.
"""
import numpy as np
import pandas as pd

from tlo import Module, Parameter, Property, Types, DateOffset
from tlo.events import Event, PopulationScopeEventMixin, RegularEvent, IndividualScopeEventMixin


class Demography(Module):
    """
    The core demography modules handling age and sex of individuals. Also is responsible for their
    'is_alive' status
    """

    def __init__(self, name=None, workbook_path=None):
        super().__init__(name)
        self.workbook_path = workbook_path

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    PARAMETERS = {
        'interpolated_pop': Parameter(Types.DATA_FRAME, 'Interpolated population structure'),
        'fertility_schedule': Parameter(Types.DATA_FRAME, 'Age-spec fertility rates'),
        'mortality_schedule': Parameter(Types.DATA_FRAME, 'Age-spec fertility rates')
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'date_of_birth': Property(Types.DATE, 'Date of birth of this individual'),
        'sex': Property(Types.CATEGORICAL, 'Male or female', categories=['M', 'F']),
        'mother_id': Property(Types.INT, 'Unique identifier of mother of this individual'),
        'is_alive': Property(Types.BOOL, 'Whether this individual is alive'),
        'is_pregnant': Property(Types.BOOL,'Whether this individual is currently pregnant'),
        'is_married': Property(Types.BOOL,'Whether this individual is currently married'),
        'contraception': Property(Types.CATEGORICAL, 'Contraception method',
                 categories=['Not using','Pill','iud','Injections','Condom','Female Sterilization',
                             'Male Sterilization','Periodic Abstinence','Withdrawal','Other','Norplant'])
    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Loads the 'Interpolated Pop Structure' worksheet from the Demography Excel workbook.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        #TODO: Read in excel file only once
        self.parameters['interpolated_pop'] = pd.read_excel(self.workbook_path,
                                                            sheet_name='Interpolated Pop Structure')

        self.parameters['fertility_schedule'] = pd.read_excel(self.workbook_path,
                                                        sheet_name='Age_spec fertility')

        self.parameters['mortality_schedule'] = pd.read_excel(self.workbook_path,
                                                        sheet_name='Mortality Rate')


    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        worksheet = self.parameters['interpolated_pop']

        # get a subset of the rows from the worksheet
        intpop = worksheet.loc[worksheet.year == self.sim.date.year].copy().reset_index()
        intpop['probability'] = intpop.value / intpop.value.sum()
        intpop['month_range'] = 12
        is_age_range = (intpop['age_to'] != intpop['age_from'])
        intpop.loc[is_age_range, 'month_range'] = (intpop['age_to'] - intpop['age_from']) * 12

        pop_sample = intpop.iloc[np.random.choice(intpop.index.values,
                                                  size=len(population),
                                                  p=intpop.probability.values)]
        pop_sample = pop_sample.reset_index()
        months = pd.Series(pd.to_timedelta(np.random.randint(low=0,
                                                             high=12,
                                                             size=len(population)),
                                           unit='M',
                                           box=False))

        df = population.props
        df.date_of_birth = self.sim.date - (pd.to_timedelta(pop_sample['age_from'], unit='Y') + months)
        df.sex = pd.Categorical(pop_sample['gender'].map({'female': 'F', 'male': 'M'}))
        df.mother_id = -1  # we can't use np.nan because that casts the series into a float
        df.is_alive = True


        # assign that half the adult population is married (will be done in lifestyle module)
        df.is_married=False

        adults=(population.age.years>=18)
        df.loc[adults,'is_married']=np.random.choice([True,False],size=adults.sum(),p=[0.5,0.5])

        df.contraception.values[:]='Not using'


        # assign that none of the adult (woman) population is pregnant
        df.is_pregnant = False



    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        event=PregnancyPoll(self)

        sim.schedule_event(event,sim.date+DateOffset(months=1)) # check all population to determine if pregnancy should be triggered



        pass

    def on_birth(self, mother, child):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother: the mother for this child
        :param child: the new child
        """
        child.date_of_birth = self.sim.date
        child.sex = np.random.choice(['M', 'F'])
        child.mother_id = mother.index
        child.is_alive = True

        print("A new child has been born:",child.index)



class PregnancyPoll(RegularEvent,PopulationScopeEventMixin):
    """
    This event looks across each woman in the population to determine who will become pregnant
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):

        print('Checking to see if anyone should become pregnant....')


        df=population.props # get the population dataframe

        female=df.loc[df.sex=='F',['contraception','is_married','is_pregnant']]  # get the subset of women from the population dataframe and relevant characteristics
        female=pd.merge(female, population.age,left_index=True,right_index=True)  # merge in the ages

        fert_schedule=self.module.parameters['fertility_schedule']  # load the fertility schedule (imported datasheet from excel workbook)
        fert_schedule=fert_schedule.loc[fert_schedule.year==self.sim.date.year] # get the subset of fertility rates for this year.


        # add age-groups to each dataframe (this to be done by the population object later)
        # TODO: add age-group cateogory to population method so that this isn't done here
        female['agegrp']=0
        female['agegrp']=np.floor(female['years']/5)

        fert_schedule['agegrp']=0
        fert_schedule['agegrp']=np.floor(fert_schedule['age_from']/5)

        female=pd.merge(female, s, left_on=['agegrp','contraception','is_married'], right_on=['agegrp','cmeth','married'], how='inner', left_index=False, right_index=False)

        # TODO: TH took the following line out as it didn't work and the above stuff is working ok for now.
        # female=female.reset_index().merge(s, left_on=['agegrp','contraception','is_married'], right_on=['agegrp','cmeth','married'], how='left').set_index('person')
        # female=female[female.age.notna()]

        # zero-out risk of pregnancy if already pregnant
        female.loc[female['is_pregnant']==True,'value']=0

        outcome=(np.random.random(size=len(female))<female.value)  # flipping the coin to determine if this woman will become pregnant

        df.loc[female.index, 'is_pregnant'] = outcome


        #
        # print(df[df['is_pregnant'] == True])


        # loop through each newly pregnant women in order to schedule them a 'delayed birth event'

        # get indicies for the pregnant women
        pregnantwomen=(df[df['is_pregnant']==True]).index

        print("The time is now",self.sim.date)
        print("The following women are now pregnnat....")
        for i in pregnantwomen:

            print('Woman number: ',i)
            print('Her age is:', df.loc[i,'years'])

            # schedule the birth event for this woman (9 months plus/minus 2 wks)
            birth_date_of_child = self.sim.date + DateOffset(months=9) + DateOffset(weeks=-2+4*np.random.random())

            mother = population[i]
            birth = DelayedBirthEvent(self.module, mother)
            self.sim.schedule_event(birth, birth_date_of_child)

            print("The birth is now booked for: ", birth_date_of_child)







class DelayedBirthEvent(Event, IndividualScopeEventMixin):
    """A one-off event in which a pregnant mother gives birth.
    """

    def __init__(self, module, mother):
        """Create a new birth event.

        We need to pass the person this event happens to to the base class constructor
        using super(). We also pass the module that created this event, so that random
        number generators can be scoped per-module.

        :param module: the module that created this event
        :param mother: the person giving birth
        """
        super().__init__(module, person=mother)

    def apply(self, mother):
        """Apply this event to the given person.
        Assuming the person is still alive, we ask the simulation to create a new offspring.
        :param person: the person the event happens to, i.e. the mother giving birth
        """

        print("@@@@ A Birth is now occuring, to mother", mother)
        print("The time is", self.sim.date)
        if mother.is_alive:
            self.sim.do_birth(mother)

        # Reset the mother's is_pregnant status showing that she is no longer pregnant
        mother.is_pregnant=False


