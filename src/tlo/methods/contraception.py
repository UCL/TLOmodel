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

from tlo import Date, DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, IndividualScopeEventMixin, RegularEvent
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
        'contraception_initiation1': Parameter(Types.DATA_FRAME, 'irate1_'),
        'contraception_initiation2': Parameter(Types.DATA_FRAME, 'irate2_'),
        'contraception_discontinuation': Parameter(Types.DATA_FRAME, 'Discontinuation'),
        'contraception_failure': Parameter(Types.DATA_FRAME, 'Failure'),
        'r_fail_age': Parameter(Types.REAL, 'change in drate (failure) per year of age increase'),
        'r_fail_age_sq': Parameter(Types.REAL, 'change in drate (failure) per year of age squared increase'),
        'r_fail_cons': Parameter(Types.REAL, 'drate (failure) at age zero - constant term from regression')
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'contraception': Property(Types.CATEGORICAL, 'Current contraceptive method',
                                  categories=['not_using', 'pill', 'IUD', 'injections', 'implant', 'male_condom',
                                              'female_sterilization', 'other_modern', 'periodic_abstinence',
                                              'withdrawal', 'other_traditional']),
        # These are the 11 categories of contraception ('not using' + 10 methods) from the DHS analysis of initiation,
        # discontinuation, failure and switching rates
        # 'other modern' includes Male sterilization, Female Condom, Emergency contraception
        # 'other traditional' includes lactational amenohroea (LAM), standard days method (SDM), 'other traditional method'
        # Have replaced Age-spec fertility sheet in demography.xlsx (in this branch) with the one in contraception.xlsx
        # (has 11 categories and one row for each age with baseline contraceptopn prevalences for each of the 11 categories)
        'date_of_childbirth': Property(Types.DATE, 'Due date of child for those who become pregnant'),
    }

    def read_parameters(self, data_folder):
        """
        Please see Contraception-Pregnancy.pdf for lucid chart explaining the relationships between
        baseline fertility rate, intitiation rates, discontinuation, failure and switching rates, and being on
        contraception or not, and being pregnant
        """
        workbook = pd.read_excel(self.workbook_path, sheet_name=None)
        self.parameters['fertility_schedule'] = workbook['Age_spec fertility']
        self.parameters['contraception_initiation1'] = workbook['irate1_']  # 'irate_1_' sheet created manually as a work around to address to do point on line 39
        # this Excel sheet is irate1_all.csv output from 'initiation rates_age_stcox.do' Stata analysis of DHS contraception calendar data
        self.parameters['contraception_initiation2'] = workbook['irate2_']
        # this Excel sheet is irate2_all.csv output from 'initiation rates_age_stcox.do' Stata analysis of DHS contraception calendar data
        self.parameters['contraception_discontinuation'] = workbook['Discontinuation']
        # this Excel sheet is from contraception_failure_discontinuation_switching.csv output from 'discontinuation & switching rates_age.do' Stata analysis of DHS contraception calendar data
        self.parameters['contraception_failure'] = workbook['Failure']
        # this Excel sheet is from contraception_failure_discontinuation_switching.csv output from 'discontinuation & switching rates_age.do' Stata analysis of DHS contraception calendar data
        p = self.parameters
        p['r_fail_age'] = np.random.normal(loc=0.0049588, scale=0.0006043, size=1)   # from Stata analysis Step 3.5 of discontinuation & switching rates_age_30apr2019.do
        p['r_fail_age_sq'] = np.random.normal(loc=-0.000073, scale=0.00000986, size=1)    # to generate parameter with uncertainty; loc is mean (coefficient from regression) and scale is SD (Standard Error), size 1 means just a single draw for use in Fail event
        p['r_fail_cons'] = np.random.normal(loc=-0.018882, scale=0.0089585, size=1)  # should constant term have uncertainty too or be fixed?


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
        #       and merge the probabilities into each row in sim population
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


        # TODO: need to do it without above for loop:
        # probabilities['p_list'] = probabilities.apply(lambda row: row[:].tolist(), axis=1)  # doesn't work as p_list is dtype['O'] (object) rather than float64
        #categories = ['not_using', 'pill', 'IUD', 'injections', 'implant', 'male_condom', 'female_sterilization',
        #              'other_modern', 'periodic_abstinence', 'withdrawal', 'other_traditional']
        #random_choice = self.rng.choice(categories, size=len(df), p=probabilities['p_list'])
        #df.loc[females1549, 'contraception'].values[:] = random_choice


    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """
        # check all females not using contraception to determine if contraception starts i.e. category should change from 'not_using' (repeats every month)
        sim.schedule_event(Init1(self), sim.date + DateOffset(months=1))

        # check all females using contraception to determine if contraception discontinues i.e. category should change to 'not_using' (repeats every month)
        sim.schedule_event(Discontinue(self), sim.date + DateOffset(months=1))

        # check all females using contraception to determine if contraception fails i.e. woman becomes pregnant whilst using contraception (repeats every month)
        sim.schedule_event(Fail(self), sim.date + DateOffset(months=1))

        # check all women after birth to determine subsequent contraception method (including not_using)
        # This should only be called after birth, though should be repeated every month i.e. following new births every month
        sim.schedule_event(Init2(self), sim.date + DateOffset(months=1))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        df = self.sim.population.props

         # Log the birth:
        logger.info('%s|on_birth|%s',
                    self.sim.date,
                    {
                        'mother': mother_id,
                        'child': child_id,
                        'mother_age': df.at[mother_id, 'age_years'],
                        'xxx': 0
                    })


class Init1(RegularEvent, PopulationScopeEventMixin):
    """
    This event looks across all women who are 'not_using' contraception to determine if they start using each
    method according to initiation_rate1 (irate_1)
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        self.age_low = 15
        self.age_high = 49

    def apply(self, population):
        logger.debug('Checking to see if anyone should start using contraception')

        df = population.props  # get the population dataframe
        m = self.module
        rng = self.module.rng

        # get the indices of the women from the population with the relevant characterisitcs
        not_using_idx = df.index[(df.sex == 'F') &
                                 df.is_alive &
                                 df.age_years.between(self.age_low, self.age_high) &
                                 ~df.is_pregnant &
                                 (df.contraception == 'not_using')]

        # prepare the probabilities
        c_worksheet = m.parameters['contraception_initiation1']
        c_probs = c_worksheet.loc[0].values.tolist()
        c_names = c_worksheet.columns.tolist()

        # sample contraceptive method for everyone not using
        # and put in a series which has index of all currently not using
        sampled_method = pd.Series(rng.choice(c_names, p=c_probs, size=len(not_using_idx), replace=True),
                                   index=not_using_idx)

        # only update those starting on contraception
        now_using_idx = not_using_idx[sampled_method != 'not_using']
        df.loc[now_using_idx, 'contraception'] = sampled_method[now_using_idx]

        # output some logging if any start contraception
        if len(now_using_idx):
            for woman_id in now_using_idx:
                logger.info('%s|start_contraception|%s',
                            self.sim.date,
                            {
                                'woman_index': woman_id,
                                'contraception': df.at[woman_id, 'contraception']
                            })


class Discontinue(RegularEvent, PopulationScopeEventMixin):
    """
    This event looks across all women who are using a contraception method to determine if they stop using it
    according to discontinuation_rate
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        self.age_low = 15
        self.age_high = 49

    def apply(self, population):
        logger.debug('Checking to see if anyone should stop using contraception')

        df = population.props  # get the population dataframe
        m = self.module
        rng = self.module.rng

        # get the indices of the women from the population with the relevant characterisitcs
        using_idx = df.index[(df.contraception != 'not_using')]

        # prepare the probabilities
        c_worksheet = m.parameters['contraception_discontinuation']

        # add the probabilities and copy to each row of the sim population (population dataframe)
        df_new = pd.concat([df, c_worksheet], axis=1)
        df_new.loc[:, ['pill', 'IUD', 'injections', 'implant', 'male_condom', 'female_sterilization', 'other_modern',
                   'periodic_abstinence', 'withdrawal', 'other_traditional']] = df_new.loc[0, ['pill', 'IUD', 'injections',
                                                                                           'implant', 'male_condom',
                                                                                           'female_sterilization',
                                                                                           'other_modern',
                                                                                           'periodic_abstinence',
                                                                                           'withdrawal',
                                                                                           'other_traditional']].tolist()
        probabilities = df_new.loc[using_idx, ['contraception','pill', 'IUD', 'injections', 'implant', 'male_condom',
                                                 'female_sterilization', 'other_modern', 'periodic_abstinence',
                                                 'withdrawal', 'other_traditional']]
        probabilities['prob'] = probabilities.lookup(probabilities.index, probabilities['contraception'])
        probabilities['1-prob'] = 1-probabilities['prob']
        probabilities['not_using'] = 'not_using'

        # apply the probabilities of discontinuation for each contraception method
        # to series which has index of all currently using
        # need to use a for loop to loop through each method
        for woman in using_idx:
            her_p = np.asarray(probabilities.loc[woman,['prob','1-prob']], dtype='float64')
            her_op = np.asarray(probabilities.loc[woman,['not_using', 'contraception']])

            her_method = rng.choice(her_op,p=her_p)

            df.loc[woman,'contraception']=her_method
            # output some logging if any stop contraception
            if her_method == 'not_using':
                logger.info('%s|stop_contraception|%s',
                                self.sim.date,
                                {
                                    'woman_index': woman,
                                    'contraception': df.at[woman, 'contraception']
                                })


class Fail(RegularEvent, PopulationScopeEventMixin):
    """
    This event looks across all women who are using a contraception method to determine if they become pregnant
    i.e. the method fails according to failure_rate
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        self.age_low = 15
        self.age_high = 49

    def apply(self, population):
        logger.debug('Checking to see if anyone becomes pregnant whilst on contraception')

        df = population.props  # get the population dataframe
        m = self.module
        rng = self.module.rng

        # get the indices of the women from the population with the relevant characterisitcs
        using_idx = df.index[(df.contraception != 'not_using')]

        # prepare the probabilities
        c_worksheet = m.parameters['contraception_failure']

        # add the probabilities and copy to each row of the sim population (population dataframe)
        df_new = pd.concat([df, c_worksheet], axis=1)
        df_new.loc[:, ['pill', 'IUD', 'injections', 'implant', 'male_condom', 'female_sterilization', 'other_modern',
                   'periodic_abstinence', 'withdrawal', 'other_traditional']] = df_new.loc[0, ['pill', 'IUD', 'injections',
                                                                                           'implant', 'male_condom',
                                                                                           'female_sterilization',
                                                                                           'other_modern',
                                                                                           'periodic_abstinence',
                                                                                           'withdrawal',
                                                                                           'other_traditional']].tolist()
        probabilities = df_new.loc[using_idx, ['is_pregnant','contraception','pill', 'IUD', 'injections', 'implant', 'male_condom',
                                                 'female_sterilization', 'other_modern', 'periodic_abstinence',
                                                 'withdrawal', 'other_traditional']]
        probabilities['prob'] = probabilities.lookup(probabilities.index, probabilities['contraception'])
        probabilities['1-prob'] = 1-probabilities['prob']
        probabilities['preg'] = 'preg'

        # TODO: modify failure probabilities by age according to newly added regression parameters
        # apply the probabilities of failure for each contraception method
        # to series which has index of all currently using
        # need to use a for loop to loop through each method
        for woman in using_idx:
            her_p = np.asarray(probabilities.loc[woman,['prob','1-prob']], dtype='float64')
            her_op = np.asarray(probabilities.loc[woman,['preg', 'contraception']])

            her_method = rng.choice(her_op,p=her_p)
            if her_method == 'preg':
                df.loc[woman,'is_pregnant']=True
                df.loc[woman, 'date_of_last_pregnancy'] = self.sim.date
                df.loc[woman, 'preg'] = 'newly_preg'
                # schedule the birth event for each newly pregnant woman (9 months plus/minus 2 wks)
                df.loc[woman, 'date_of_childbirth'] = self.sim.date + DateOffset(months=9,
                                                           weeks=-2 + 4 * rng.random_sample())
                # Schedule the Birth
                # TODO: not working: despite importing Demography I get the error: AttributeError: type object 'Demography' has no attribute 'DelayedBirthEvent'
                # self.sim.schedule_event(Demography.DelayedBirthEvent(self.sim.modules['Demography'], woman),
                #                    date_of_birth)

                # output some logging if any pregnancy (contraception failure)
                logger.info('%s|fail_contraception|%s',
                                self.sim.date,
                                {
                                    'woman_index': woman,
                                    'Preg': df.at[woman, 'is_pregnant'],
                                    'birth booked for': df.at[woman, 'date_of_childbirth']
                                })


class Init2(RegularEvent, PopulationScopeEventMixin):
    """
    This event looks across all women who have given birth and decides what contraceptive method they then have
    (including not_using, according to initiation_rate2 (irate_2)
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        self.age_low = 15
        self.age_high = 49

    def apply(self, population):
        logger.debug('Checking to see what contraception method women who have just given birth should have')

        df = population.props  # get the population dataframe
        m = self.module
        rng = self.module.rng

        # prepare the probabilities
        c_worksheet = m.parameters['contraception_initiation2']
        c_probs = c_worksheet.loc[0].values.tolist()
        c_names = c_worksheet.columns.tolist()
        # Note the irate2s are low as they are just for the month after pregnancy and
        # 	then for the 99.48% who 'initiate' to 'not_using' (i.e. 1 - sum(irate2s))
        # 	they are then subject to the usual irate1s per month
        # 	- see Contraception-Pregnancy.pdf schematic


        # get the indices of the women from the population with the relevant characterisitcs
        # those who have just given birth within the last month
        # the if statement below is to keep it running if there are no births within the last month
        if df[((df.date_of_childbirth < self.sim.date) & (df.date_of_childbirth > self.sim.date - DateOffset(months=1)))].empty == True:
            pass
        else:
            birth_idx = df.index[((df.date_of_childbirth < self.sim.date) & (df.date_of_childbirth > self.sim.date - DateOffset(months=1)))]
            # sample contraceptive method for everyone just given birth
            sampled_method = pd.Series(rng.choice(c_names, p=c_probs, size=len(birth_idx), replace=True),
                                   index=birth_idx)
            # update contraception method for all women who have just given birth
            df.loc[birth_idx, 'contraception'] = sampled_method[birth_idx]
            # output some logging if any post-birth contraception
            if len(birth_idx):
                for woman_id in birth_idx:
                    logger.info('%s|post-birth_contraception|%s',
                                self.sim.date,
                                {
                                    'woman_index': woman_id,
                                    'contraception': df.at[woman_id, 'contraception']
                                })

