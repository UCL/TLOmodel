"""
The core demography module and its associated events.

Expects input in format of the 'Demography.xlsx'  of TimH, sent 3/10. Uses the 'Interpolated
population structure' worksheet within to initialise the age & sex distribution of population.
"""
import numpy as np
import pandas as pd

from tlo import Module, Parameter, Property, Types, DateOffset
from tlo.events import Event, PopulationScopeEventMixin, RegularEvent, IndividualScopeEventMixin
# from tlo.methods.pregnancy_and_newborn_supervisor import Pregnancy_And_Newborn_Supervisor_Event


class Demography(Module):
    """
    The core demography modules handling age and sex of individuals. Also is responsible for their
    'is_alive' status
    """

    def __init__(self, name=None, workbook_path=None):
        super().__init__(name)
        self.workbook_path = workbook_path
        self.store={'Time':[],
                    'Population_Total':[],
                    'Population_Men':[],
                    'Population_Men_0-4':[],
                    'Population_Men_5-9':[],
                    'Population_Men_10-14': [],
                    'Population_Men_15-19': [],
                    'Population_Men_20-24': [],
                    'Population_Men_25-29': [],
                    'Population_Men_30-34': [],
                    'Population_Men_35-39': [],
                    'Population_Men_40-44': [],
                    'Population_Men_45-49': [],
                    'Population_Men_50-54': [],
                    'Population_Men_55-59': [],
                    'Population_Men_60-64': [],
                    'Population_Men_65-69': [],
                    'Population_Men_70-74': [],
                    'Population_Men_75-79': [],
                    'Population_Men_80-84': [],
                    'Population_Men_85-': [],
                    'Population_Women': [],
                    'Population_Women_0-4': [],
                    'Population_Women_5-9': [],
                    'Population_Women_10-14': [],
                    'Population_Women_15-19': [],
                    'Population_Women_20-24': [],
                    'Population_Women_25-29': [],
                    'Population_Women_30-34': [],
                    'Population_Women_35-39': [],
                    'Population_Women_40-44': [],
                    'Population_Women_45-49': [],
                    'Population_Women_50-54': [],
                    'Population_Women_55-59': [],
                    'Population_Women_60-64': [],
                    'Population_Women_65-69': [],
                    'Population_Women_70-74': [],
                    'Population_Women_75-79': [],
                    'Population_Women_80-84': [],
                    'Population_Women_85-': [],
                    'DeathEvent_Time': [],
                    'DeathEvent_Age': [],
                    'DeathEvent_Cause': [],
                    'BirthEvent_Time': [],
                    'BirthEvent_AgeOfMother': [],
                    'BirthEvent_Outcome': []}



    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    PARAMETERS = {
        'interpolated_pop': Parameter(Types.DATA_FRAME, 'Interpolated population structure'),
        'fertility_schedule': Parameter(Types.DATA_FRAME, 'Age-spec fertility rates'),
        'mortality_schedule': Parameter(Types.DATA_FRAME, 'Age-spec fertility rates'),
        'fraction_of_births_male': Parameter(Types.REAL, 'Birth Sex Ratio')
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
        'date_of_last_pregnancy': Property(Types.DATE,'Date of the last pregnancy of this individual'),
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

        self.parameters['fraction_of_births_male'] = 0.5


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

        pop_sample = intpop.iloc[self.sim.rng.choice(intpop.index.values,
                                                  size=len(population),
                                                  p=intpop.probability.values)]
        pop_sample = pop_sample.reset_index()
        months = pd.Series(pd.to_timedelta(self.sim.rng.randint(low=0,
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
        df.loc[adults,'is_married']=self.sim.rng.choice([True,False],size=adults.sum(),p=[0.5,0.5])

        df.contraception.values[:]='Not using'

        # assign that none of the adult (woman) population is pregnant
        df.is_pregnant = False
        df.date_of_last_pregnancy = pd.DateOffset(months=0)


    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        # Launch the repeating events for demoraphy...
        sim.schedule_event(PregnancyPoll(self),sim.date+DateOffset(months=1))       # check all population to determine if pregnancy should be triggered (repeats every month)
        sim.schedule_event(OtherDeathPoll(self),sim.date+DateOffset(months=1))      # check all population to determine if person should die (from causes other than those explicitly modelled) (repeats every month)

        # Launch the repeating event that will store statistics about the population structure
        sim.schedule_event(DemographyLoggingEvent(self),sim.date+DateOffset(months=12))



    def on_birth(self, mother, child):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother: the mother for this child
        :param child: the new child
        """
        child.date_of_birth = self.sim.date
        child.sex = sim.rng.choice(['M', 'F'],p=[self.parameters['fraction_of_births_male'],(1-self.parameters['fraction_of_births_male'])])
        child.mother_id = mother.index
        child.is_alive = True

        if self.sim.verboseoutput:
            print("A new child has been born:",child.index)



class PregnancyPoll(RegularEvent,PopulationScopeEventMixin):
    """
    This event looks across each woman in the population to determine who will become pregnant
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):

        if self.sim.verboseoutput:
            print('Checking to see if anyone should become pregnant....')

        df=population.props # get the population dataframe

        female=df.loc[df.sex=='F',['contraception','is_married','is_pregnant']]  # get the subset of women from the population dataframe and relevant characteristics
        female=pd.merge(female, population.age,left_index=True,right_index=True)  # merge in the ages

        fert_schedule=self.module.parameters['fertility_schedule']  # load the fertility schedule (imported datasheet from excel workbook)
        fert_schedule=fert_schedule.loc[fert_schedule.year==self.sim.date.year] # get the subset of fertility rates for this year.

        # --------
        # add age-groups to each dataframe (this to be done by the population object later)
        # (NB. Why does fert_schedule['agegrp']=0 create a warning when the same line on 'females' doesn't)
        # TODO: add age-group cateogory to population method so that this isn't done here
        female['agegrp']=0
        female['agegrp']=np.floor(female['years']/5)
        fert_schedule['agegrp']=0
        fert_schedule['agegrp']=np.floor(fert_schedule['age_from']/5)
        # --------

        # get the probability of pregnancy for each woman in the model, through merging with the fert_schedule data
        female=female.reset_index().merge(fert_schedule, left_on=['agegrp','contraception','is_married'], right_on=['agegrp','cmeth','married'], how='left').set_index('person')
        female=female[female.value.notna()]  # clean up items that didn't merge (those with ages not referenced in the fert_schedule sheet)

        female.loc[female['is_pregnant']==True,'value']=0 # zero-out risk of pregnancy if already pregnant

        newlypregnant=(self.sim.rng.random_sample(size=len(female))<female.value/12)          # flipping the coin to determine if this woman will become pregnant
                                                                                    # the imported number is a yearly proportion. So adjust the rate according
                                                                                    # to the frequency with which the event is recurring
                                                                                    # TODO: this should be linked to the self.frequency value

        df.loc[female.index[newlypregnant], 'is_pregnant'] = True             # updating the pregancy status for that women
        df.loc[female.index[newlypregnant], 'date_of_last_pregnancy']= self.sim.date


        # loop through each newly pregnant women in order to schedule them a 'delayed birth event'
        personnumbers_of_newlypregnant=female.index[newlypregnant]
        for personnumber in personnumbers_of_newlypregnant:

            if self.sim.verboseoutput:
                print('Woman number: ',population[personnumber])
                print('Her age is:', female.loc[personnumber,'years'])

            # schedule the birth event for this woman (9 months plus/minus 2 wks)
            birth_date_of_child = self.sim.date + DateOffset(months=9) + DateOffset(weeks=-2+4*self.sim.rng.random_sample())

            mother = population[personnumber] # get the index of the mother (**Q: what data type is this....)

            # Schedule the Birth
            birth = DelayedBirthEvent(self.module, mother)  # prepare for that birth event
            self.sim.schedule_event(birth, birth_date_of_child)

            if self.sim.verboseoutput:
                print("The birth is now booked for: ", birth_date_of_child)

            # Launch the Pregnancy_And_Newborn_Supervisor
            # p_and_n_supervisor=Pregnancy_And_Newborn_Supervisor_Event(self.module,mother)
            # self.sim.schedule_event(p_and_n_supervisor, DateOffset(weeks=1))


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

        if self.sim.verboseoutput:
            print("@@@@ A Birth is now occuring, to mother", mother)

        if mother.is_alive:
            self.sim.do_birth

        # Reset the mother's is_pregnant status showing that she is no longer pregnant
        mother.is_pregnant=False


        # Log the birth:
        self.module.store['BirthEvent_Time'].append(self.sim.date)
        self.module.store['BirthEvent_AgeOfMother'].append(self.sim.population.age.years[mother.index])
        self.module.store['BirthEvent_Outcome'].append(0)



class OtherDeathPoll(RegularEvent,PopulationScopeEventMixin):
    """
    This event looks across each person in the population to see if they should die.
    During development this will be applying WPP all-cause death rates...
    ... But once we have diseases causing their own deaths, it will become the mortality schedule
    ... from all cases other than those explicitly modelled
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        if self.sim.verboseoutput:
            print("Checking to see if anyone should die....")

        #get the mortality schedule for now...
        mort_schedule=self.module.parameters['mortality_schedule']  # load the mortality schedule (imported datasheet from excel workbook)
        mort_schedule=mort_schedule.loc[mort_schedule.year==self.sim.date.year] # get the subset of mortality rates for this year.
        # create new variable that will align with population.sex
        mort_schedule['sex']=np.where(mort_schedule['gender']=='male','M','F')

        # get the population
        df=population.props
        df=pd.merge(df, population.age,left_index=True,right_index=True)  # merge in the ages

        # --------
        # add age-groups to each dataframe (this to be done by the population object later)
        # NB... the age-groups here are -1:<0 year-olds; 0:1-4 yearolds; 1:5-9 yearolds; etc..)
        df['agegrp']=-99
        df['agegrp']=1+np.floor(df.years/5)
        df.loc[df.years==0,'agegrp']=-1 # overwriting with -1 for the <0 year-olds

        mort_schedule['agegrp']=-99
        mort_schedule['agegrp']=np.floor(mort_schedule.age_from/5)
        mort_schedule.loc[mort_schedule.age_from==0,'agegrp'] =-1
        #-------

        # merge the popualtion dataframe with the parameter dataframe to pick-up the risk of mortality for each person in the model
        df=df.reset_index().merge(mort_schedule, left_on=['agegrp','sex'], right_on=['agegrp','sex'], how='left').set_index('person')

        outcome = (self.sim.rng.random_sample(size=len(df)) < df.value / 12)  # flipping the coin to determine if this woman will die
                                                            # the imported number is a yearly proportion. So adjust the rate according
                                                            # to the frequency with which the event is recurring
                                                            # TODO: this should be linked to the self.frequency value

        # loop through to see who is going to die:
        willdie = (df[outcome == True]).index
        cause='other'
        for i in willdie:
            person=population[i]
            death = InstantaneousDeath(self.module, person,cause='Other')  # make that death event
            self.sim.schedule_event(death, self.sim.date) # schedule the death for "now"





class InstantaneousDeath(Event, IndividualScopeEventMixin):

    """
    Performs the Death operation on an individual and logs it.

    """

    def __init__(self, module, individual,cause):
        super().__init__(module, person=individual)
        self.cause=cause

    def apply(self, individual):

        if self.sim.verboseoutput:
            print("@@@@ A Death is now occuring, to person", individual)

        if individual.is_alive:
            # here comes the death..
            individual.is_alive = False
            # the person is now dead

        if self.sim.verboseoutput:
            print("*******************************************The person", individual, "is now officially dead and has died of :", self.cause)

        # Log the death
        self.module.store['DeathEvent_Time'].append(self.sim.date)
        self.module.store['DeathEvent_Age'].append(self.sim.population.age.years[individual.index])
        self.module.store['DeathEvent_Cause'].append(self.cause)


class DemographyLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """comments...
        """
        # run this event every 12 months (every year)
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props

        Num_Women=((df['sex']=='F') & (df['is_alive']==True)).sum()
        Num_Men=((df['sex']=='M') & (df['is_alive']==True)).sum()
        Num_Total=(df['is_alive']==True).sum()

        counts = {'Num_Women': Num_Women, 'Num_Men': Num_Men, 'Num_Total': Num_Total}


        if self.sim.verboseoutput:
            print('>>>> %s - Demography: {Num_Women: %d; Num_Men: %d; Num_Total: %d}' %
              (self.sim.date,Num_Women,Num_Men,Num_Total), flush=True)



        self.module.store['Time'].append(self.sim.date)
        self.module.store['Population_Total'].append(Num_Total)



