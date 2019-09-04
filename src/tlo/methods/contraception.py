"""
Contraception module covering baseline fertility, contraception methods use, failure (pregnancy),
Switching contraceptive methods, and discontiuation rates by age
please see Dropbox/Thanzi la Onse/05 - Resources/Model design/Contraception-Pregnancy.pdf
for conceptual diagram
"""
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import math

from pathlib import Path
from tlo import Date, DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, PopulationScopeEventMixin, IndividualScopeEventMixin, RegularEvent
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

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    PARAMETERS = {
        'fertility_schedule': Parameter(Types.DATA_FRAME, 'Age_spec fertility'),
        'contraception_initiation1': Parameter(Types.DATA_FRAME, 'irate1_'),  # 2011-2016 rate
        'contraception_initiation2': Parameter(Types.DATA_FRAME, 'irate2_'),  # 2011-2016 rate
        'contraception_switching': Parameter(Types.DATA_FRAME, 'Switching'),
        'contraception_switching_matrix': Parameter(Types.DATA_FRAME, 'switching_matrix'),
        'contraception_discontinuation': Parameter(Types.DATA_FRAME, 'Discontinuation'),
        'contraception_failure': Parameter(Types.DATA_FRAME, 'Failure'),
        'r_init1_age': Parameter(Types.REAL, 'proportioniate change in irate1 for each age in years'),    # from Fracpoly regression
        'r_discont_age': Parameter(Types.REAL, 'proportioniate change in drate for each age in years'),     # from Fracpoly regression
        'rr_fail_under25': Parameter(Types.REAL, 'Increase in Failure rate for under-25s')
        # TODO: add relative fertility rates for HIV+ compared to HIV- by age group from Marston et al 2017
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'co_contraception': Property(Types.CATEGORICAL, 'Current contraceptive method',
                                     categories=['not_using', 'pill', 'IUD', 'injections', 'implant', 'male_condom',
                                                 'female_sterilization', 'other_modern', 'periodic_abstinence',
                                                 'withdrawal', 'other_traditional']),
        # These are the 11 categories of contraception ('not using' + 10 methods) from the DHS analysis of initiation,
        # discontinuation, failure and switching rates
        # 'other modern' includes Male sterilization, Female Condom, Emergency contraception
        # 'other traditional' includes lactational amenohroea (LAM), standard days method (SDM), 'other traditional method'
        # Have replaced Age-spec fertility sheet in ResourceFile_DemographicData.xlsx (in this branch) with the one in ResourceFile_Contraception.xlsx
        # (has 11 categories and one row for each age with baseline contraceptopn prevalences for each of the 11 categories)
        'co_date_of_childbirth': Property(Types.DATE, 'Due date of child for those who become pregnant'),
        'is_pregnant': Property(Types.BOOL, 'Whether this individual is currently pregnant'),
        'date_of_last_pregnancy': Property(Types.DATE,
                                           'Date of the last pregnancy of this individual'),
    }

    def read_parameters(self, data_folder):
        """
        Please see Contraception-Pregnancy.pdf for lucid chart explaining the relationships between
        baseline fertility rate, intitiation rates, discontinuation, failure and switching rates, and being on
        contraception or not, and being pregnant
        """
        workbook = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_Contraception.xlsx', sheet_name=None)

        self.parameters['fertility_schedule'] = workbook['Age_spec fertility']

        self.parameters['contraception_initiation1'] = workbook[
            'irate1_']  # 'irate_1_' sheet created manually as a work around to address to do point on line 39
        # this Excel sheet is irate1_all.csv output from 'initiation rates_age_stcox.do' Stata analysis of DHS contraception calendar data

        self.parameters['contraception_initiation2'] = workbook['irate2_']
        # this Excel sheet is irate2_all.csv output from 'initiation rates_age_stcox.do' Stata analysis of DHS contraception calendar data

        self.parameters['contraception_switching'] = workbook['Switching']
        # this Excel sheet is from contraception_failure_discontinuation_switching.csv output from 'failure discontinuation switching rates.do' Stata analysis of DHS contraception calendar data

        self.parameters['contraception_switching_matrix'] = workbook['switching_matrix']
        # this Excel sheet is from contraception switching matrix output from line 144 of 'failure discontinuation switching rates.do' Stata analysis of DHS contraception calendar data

        self.parameters['contraception_discontinuation'] = workbook['Discontinuation']
        # this Excel sheet is from contraception_failure_discontinuation_switching.csv output from 'failure discontinuation switching rates.do' Stata analysis of DHS contraception calendar data

        self.parameters['contraception_failure'] = workbook['Failure']
        # this Excel sheet is from contraception_failure_discontinuation_switching.csv output from 'failure discontinuation switching rates.do' Stata analysis of DHS contraception calendar data

        self.parameters['rr_fail_under25'] = 2.2
        # From Guttmacher analysis - do not apply to female steriliztion or male sterilization - note that as these are already 0 (see 'Failure' Excel sheet) the rate will remain 0

        self.parameters['r_init1_age'] = workbook['r_init1_age']
        # from Stata analysis line 250 of initiation rates_age_stcox_2005_2016_5yrPeriods.do: fracpoly: regress _d age_ // fracpoly exact age (better fitting model, higher F statistic) - see 'Initiation1 by age' worksheet, results are in 'r_init1_age' sheet

        self.parameters['r_discont_age'] = workbook['r_discont_age']
        # from Stata analysis Step 3.5 of discontinuation & switching rates_age.do: fracpoly: regress drate_allmeth age: - see 'Discontinuation by age' worksheet, results are in 'r_discont_age' sheet

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.
        """
        df = population.props

        df.loc[df.is_alive, 'co_contraception'] = 'not_using'

        # Assign contraception method
        # 1. select females aged 15-49 from population, for current year
        females1549 = df.is_alive & (df.sex == 'F') & df.age_years.between(15, 49)

        # 2. Prepare probabilities lookup table
        co_types_prob_lookup = self.parameters['fertility_schedule'].set_index('age')
        co_types_prob_lookup.drop(columns=['year', 'basefert_dhs'], inplace=True)  # drop unused columns
        co_types = list(co_types_prob_lookup.columns)

        # normalise the values so they sum to 1 and collapse into single array
        co_types_prob_lookup = co_types_prob_lookup.apply(lambda x: x.values / sum(x.values), axis=1)

        # 3. apply probabilities of each contraception type to sim population
        def pick_contraceptive(age):
            """a function that randomly chooses a contraceptive based on age"""
            return self.rng.choice(co_types, p=co_types_prob_lookup.loc[age])

        df.loc[females1549, 'co_contraception'] = df.loc[females1549, 'age_years'].apply(pick_contraceptive)

        # TODO: need to do it without above for loop:
        # probabilities['p_list'] = probabilities.apply(lambda row: row[:].tolist(), axis=1)  # doesn't work as p_list is dtype['O'] (object) rather than float64
        # categories = ['not_using', 'pill', 'IUD', 'injections', 'implant', 'male_condom', 'female_sterilization',
        #              'other_modern', 'periodic_abstinence', 'withdrawal', 'other_traditional']
        # random_choice = self.rng.choice(categories, size=len(df), p=probabilities['p_list'])
        # df.loc[females1549, 'co_contraception'].values[:] = random_choice

    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """
        # check all females not using contraception to determine if contraception starts i.e. category should change from 'not_using' (starts at month 0)
        sim.schedule_event(Init1(self), sim.date + DateOffset(months=0))

        # check all females using contraception to determine if contraception discontinues i.e. category should change to 'not_using' (starts at month 0)
        sim.schedule_event(Discontinue(self), sim.date + DateOffset(months=0))

        # check all females using contraception to determine if contraception Switches i.e. category should change from any method to a new method (not including 'not_using') (starts at month 0)
        sim.schedule_event(Switch(self), sim.date + DateOffset(months=0))

        # check all females using contraception to determine if contraception fails i.e. woman becomes pregnant whilst using contraception (starts at month 0)
        sim.schedule_event(Fail(self), sim.date + DateOffset(months=0))

        # check all women after birth to determine subsequent contraception method (including not_using) (starts at month 0)
        # This should only be called after birth, though should be repeated every month i.e. following new births every month
        sim.schedule_event(Init2(self), sim.date + DateOffset(months=0))

        # check all population to determine if pregnancy should be triggered (repeats every month)
        sim.schedule_event(PregnancyPoll(self), sim.date + DateOffset(months=1))

        # Launch the repeating event that will store statistics about the population structure
        sim.schedule_event(ContraceptionLoggingEvent(self), sim.date + DateOffset(days=0))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        df = self.sim.population.props

        df.at[child_id, 'mother_id'] = mother_id
        df.at[child_id, 'date_of_last_pregnancy'] = pd.NaT
        df.at[child_id, 'is_pregnant'] = False
        df.at[child_id, 'co_contraception'] = 'not_using'

        # Reset the mother's is_pregnant status showing that she is no longer pregnant
        df.at[mother_id, 'is_pregnant'] = False

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
        super().__init__(module, frequency=DateOffset(months=1))  # runs every month
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
                                 (df.co_contraception == 'not_using')]

        # prepare the probabilities
        c_worksheet = m.parameters['contraception_initiation1']
        c_worksheet2 = m.parameters['r_init1_age']

        # add the probabilities and copy to each row of the sim population (population dataframe)
        df_new = pd.concat([df, c_worksheet], axis=1)
        df_new.loc[:, ['not_using', 'pill', 'IUD', 'injections', 'implant', 'male_condom', 'female_sterilization', 'other_modern',
                       'periodic_abstinence', 'withdrawal', 'other_traditional']] = df_new.loc[
            0, ['not_using', 'pill', 'IUD', 'injections',
                'implant', 'male_condom',
                'female_sterilization',
                'other_modern',
                'periodic_abstinence',
                'withdrawal',
                'other_traditional']].tolist()
        probabilities = df_new.loc[
            not_using_idx, ['age_years', 'co_contraception', 'not_using', 'pill', 'IUD', 'injections', 'implant', 'male_condom',
                        'female_sterilization', 'other_modern', 'periodic_abstinence',
                        'withdrawal', 'other_traditional']]

        # get the proportioniate change in irate1 for each age in years, through merging with the r_init1_age data
        len_before_merge = len(probabilities)
        probabilities = probabilities.reset_index().merge(c_worksheet2,
                                              left_on=['age_years'],
                                              right_on=['age'],
                                              how='left').set_index('person')
        assert len(probabilities) == len_before_merge

        ##c_probs = c_worksheet.loc[0].values.tolist()
        c_names = c_worksheet.columns.tolist()

        # adjust the probabilities of initiation1 of each method according to the proportionate change by age in years (which is based on the fracpoly regression parameters):
        c_probs = probabilities[['pill', 'IUD', 'injections',
                'implant', 'male_condom', 'female_sterilization', 'other_modern',
                'periodic_abstinence', 'withdrawal', 'other_traditional']]
        c_probs_age_extra = probabilities[['pill', 'IUD', 'injections',
                'implant', 'male_condom', 'female_sterilization', 'other_modern',
                'periodic_abstinence', 'withdrawal', 'other_traditional']].mul(probabilities['r_init1_age'], axis='index')
        c_probs = c_probs + c_probs_age_extra
        c_probs['not_using'] = 1 - c_probs.sum(axis=1)

        # apply probabilities of initiation of each method to everyone (depends on their age)
        # (note this includes 'not_using' i.e no initiation)
        for woman in not_using_idx:
            her_p = np.asarray(c_probs.loc[woman, :], dtype='float64')
            her_op = np.asarray(c_probs.columns)

            her_method = rng.choice(her_op, p=her_p)

            df.loc[woman, 'co_contraception'] = her_method

            # output some logging if any start of contraception
            if her_method != 'not_using':
                logger.info('%s|start_contraception1|%s',
                            self.sim.date,
                            {
                                'woman_index': woman,
                                'age': df.at[woman, 'age_years'],
                                'co_contraception': df.at[woman, 'co_contraception']
                            })


class Switch(RegularEvent, PopulationScopeEventMixin):
    """
    This event looks across all women who are using a contraception method to determine if they switch to another
     method according to switching_rate, and then for those that switch directs towards a new method according to
     switching_matrix
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))  # runs every month
        self.age_low = 15
        self.age_high = 49

    def apply(self, population):
        logger.debug('Checking to see if anyone should switch contraception methods')

        df = population.props  # get the population dataframe
        m = self.module
        rng = self.module.rng

        # get the indices of the women from the population with the relevant characterisitcs
        using_idx = df.index[(df.co_contraception != 'not_using') & (df.co_contraception != 'female_sterilization')]

        # prepare the probabilities for Switch by method (i.e. any switch - different probability for each method)
        c_worksheet = m.parameters['contraception_switching']

        # add the probabilities of Switching and copy to each row of the sim population (population dataframe)
        df_new = pd.concat([df, c_worksheet], axis=1)
        df_new.loc[:, ['pill', 'IUD', 'injections', 'implant', 'male_condom', 'female_sterilization', 'other_modern',
                       'periodic_abstinence', 'withdrawal', 'other_traditional']] = df_new.loc[
            0, ['pill', 'IUD', 'injections',
                'implant', 'male_condom',
                'female_sterilization',
                'other_modern',
                'periodic_abstinence',
                'withdrawal',
                'other_traditional']].tolist()
        probabilities = df_new.loc[
            using_idx, ['co_contraception', 'pill', 'IUD', 'injections', 'implant', 'male_condom',
                        'female_sterilization', 'other_modern', 'periodic_abstinence',
                        'withdrawal', 'other_traditional']]
        probabilities['prob'] = probabilities.lookup(probabilities.index, probabilities['co_contraception'])
        probabilities['1-prob'] = 1 - probabilities['prob']
        probabilities['switch'] = 'switch'  # for flagging those who switch from a method below

        # prepare the switching matrix to determine which method the woman switches on to:
        switch = m.parameters['contraception_switching_matrix']

        # Get probabilities of switching to each method of contraception by co_contraception method
        #       and merge the probabilities into each row in sim population - note set_index is needed to keep index
        df_switch = probabilities.merge(switch, left_on=['co_contraception'], right_on=['switchfrom'],
                                        how='left').set_index(probabilities.index)

        # apply the probabilities of switching for each contraception method to series which has index of all
        # currently using (not including female sterilization as can't switch from this)
        # need to use a for loop to loop through each method
        for woman in using_idx:
            her_p = np.asarray(probabilities.loc[woman, ['prob', '1-prob']], dtype='float64')
            her_op = np.asarray(probabilities.loc[woman, ['switch', 'co_contraception']])

            her_method = rng.choice(her_op, p=her_p)

            # Switch to new method according to switching matrix probs for current method if chosen to switch:
            if her_method == 'switch':
                # output some logging for contraception switch to new method
                logger.info('%s|switchfrom_contraception|%s',
                            self.sim.date,
                            {
                                'woman_index': woman,
                                'contraception switched from': df.at[woman, 'co_contraception']
                            })

                # apply probabilities of switching to each contraception type to sim population
                # for woman2 in using_idx:
                her_p2 = np.asarray(df_switch.loc[woman, ['pill_y', 'IUD_y', 'injections_y', 'implant_y',
                                                          'male_condom_y', 'female_sterilization_y',
                                                          'other_modern_y', 'periodic_abstinence_y', 'withdrawal_y',
                                                          'other_traditional_y']], dtype='float64')

                her_op2 = np.asarray(['pill', 'IUD', 'injections', 'implant', 'male_condom',
                                      'female_sterilization', 'other_modern', 'periodic_abstinence',
                                      'withdrawal', 'other_traditional'])

                her_method2 = rng.choice(her_op2, p=her_p2)

                df.loc[woman, 'co_contraception'] = her_method2

                # output some logging for contraception switch to new method
                logger.info('%s|switchto_contraception|%s',
                            self.sim.date,
                            {
                                'woman_index': woman,
                                'contraception switched to': df.at[woman, 'co_contraception']
                            })


class Discontinue(RegularEvent, PopulationScopeEventMixin):
    """
    This event looks across all women who are using a contraception method to determine if they stop using it
    according to discontinuation_rate
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))  # runs every month
        self.age_low = 15
        self.age_high = 49

    def apply(self, population):
        logger.debug('Checking to see if anyone should stop using contraception')

        df = population.props  # get the population dataframe
        m = self.module
        rng = self.module.rng

        # get the indices of the women from the population with the relevant characterisitcs
        using_idx = df.index[(df.co_contraception != 'not_using')]

        # prepare the probabilities
        c_worksheet = m.parameters['contraception_discontinuation']
        c_worksheet2 = m.parameters['r_discont_age']

        # add the probabilities and copy to each row of the sim population (population dataframe)
        df_new = pd.concat([df, c_worksheet], axis=1)
        df_new.loc[:, ['pill', 'IUD', 'injections', 'implant', 'male_condom', 'female_sterilization', 'other_modern',
                       'periodic_abstinence', 'withdrawal', 'other_traditional']] = df_new.loc[
            0, ['pill', 'IUD', 'injections',
                'implant', 'male_condom',
                'female_sterilization',
                'other_modern',
                'periodic_abstinence',
                'withdrawal',
                'other_traditional']].tolist()
        probabilities = df_new.loc[
            using_idx, ['age_years', 'co_contraception', 'pill', 'IUD', 'injections', 'implant', 'male_condom',
                        'female_sterilization', 'other_modern', 'periodic_abstinence',
                        'withdrawal', 'other_traditional']]

        # get the proportioniate change in drate for each age in years, through merging with the r_discont_age data
        len_before_merge = len(probabilities)
        probabilities = probabilities.reset_index().merge(c_worksheet2,
                                              left_on=['age_years'],
                                              right_on=['age'],
                                              how='left').set_index('person')
        assert len(probabilities) == len_before_merge

        probabilities['not_using'] = 'not_using'
        probabilities['prob'] = probabilities.lookup(probabilities.index, probabilities['co_contraception'])
        # adjust the probabilities of discontinuation according to the proportionate change by age in years (which is based on the fracpoly regression parameters):
        probabilities['prob'] = probabilities['prob'] + (probabilities['prob']* probabilities['r_discont_age'])
        probabilities['1-prob'] = 1 - probabilities['prob']

        # apply the probabilities of discontinuation for each contraception method
        # to series which has index of all currently using
        # need to use a for loop to loop through each method
        for woman in using_idx:
            her_p = np.asarray(probabilities.loc[woman, ['prob', '1-prob']], dtype='float64')
            her_op = np.asarray(probabilities.loc[woman, ['not_using', 'co_contraception']])

            her_method = rng.choice(her_op, p=her_p)

            df.loc[woman, 'co_contraception'] = her_method
            # output some logging if any stop contraception
            if her_method == 'not_using':
                logger.info('%s|stop_contraception|%s',
                            self.sim.date,
                            {
                                'woman_index': woman,
                                'co_contraception': df.at[woman, 'co_contraception']
                            })


class Fail(RegularEvent, PopulationScopeEventMixin):
    """
    This event looks across all women who are using a contraception method to determine if they become pregnant
    i.e. the method fails according to failure_rate
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))  # runs every month
        self.age_low = 15
        self.age_high = 49

    def apply(self, population):
        logger.debug('Checking to see if anyone becomes pregnant whilst on contraception')

        df = population.props  # get the population dataframe
        m = self.module
        rng = self.module.rng

        # get the indices of the women from the population with the relevant characterisitcs
        using_idx = df.index[(df.co_contraception != 'not_using')]

        # prepare the probabilities
        c_worksheet = m.parameters['contraception_failure']

        # add the probabilities and copy to each row of the sim population (population dataframe)
        df_new = pd.concat([df, c_worksheet], axis=1)
        df_new.loc[:, ['pill', 'IUD', 'injections', 'implant', 'male_condom', 'female_sterilization', 'other_modern',
                       'periodic_abstinence', 'withdrawal', 'other_traditional']] = df_new.loc[
            0, ['pill', 'IUD', 'injections',
                'implant', 'male_condom',
                'female_sterilization',
                'other_modern',
                'periodic_abstinence',
                'withdrawal',
                'other_traditional']].tolist()
        probabilities = df_new.loc[
            using_idx, ['age_years', 'is_pregnant', 'co_contraception', 'pill', 'IUD', 'injections', 'implant',
                        'male_condom',
                        'female_sterilization', 'other_modern', 'periodic_abstinence',
                        'withdrawal', 'other_traditional']]
        probabilities['prob'] = probabilities.lookup(probabilities.index, probabilities['co_contraception'])

        # apply increased risk of failure to under 25s
        under25 = probabilities.index[probabilities.age_years.between(15, 25)]
        probabilities.loc[under25, 'prob'] = probabilities['prob'] * m.parameters['rr_fail_under25']

        probabilities['1-prob'] = 1 - probabilities['prob']
        probabilities['preg'] = 'preg'

        # apply the probabilities of failure for each contraception method
        # to series which has index of all currently using
        # need to use a for loop to loop through each method
        for woman in using_idx:
            her_p = np.asarray(probabilities.loc[woman, ['prob', '1-prob']], dtype='float64')
            her_op = np.asarray(probabilities.loc[woman, ['preg', 'co_contraception']])

            her_method = rng.choice(her_op, p=her_p)
            if her_method == 'preg':
                df.loc[woman, 'is_pregnant'] = True
                df.loc[woman, 'date_of_last_pregnancy'] = self.sim.date
                df.loc[
                    woman, 'preg'] = 'new_unintented_preg'  # as these are contraceptive failures these pregnancies are unintended
                # schedule the birth event for each newly pregnant woman (9 months plus/minus 2 wks)
                df.loc[woman, 'co_date_of_childbirth'] = self.sim.date + DateOffset(months=9,
                                                                                    weeks=-2 + 4 * rng.random_sample())
                # Schedule the Birth
                scheduled_birth_date = df.at[woman, 'co_date_of_childbirth']
                birth = DelayedBirthEvent(self, mother_id=woman)
                self.sim.schedule_event(birth, scheduled_birth_date)

                # output some logging if any pregnancy (contraception failure)
                logger.info('%s|fail_contraception|%s',
                            self.sim.date,
                            {
                                'woman_index': woman,
                                'Preg': df.at[woman, 'is_pregnant'],
                                'birth booked for': str(df.at[woman, 'co_date_of_childbirth'])
                            })


class Init2(RegularEvent, PopulationScopeEventMixin):
    """
    This event looks across all women who have given birth and decides what contraceptive method they then have
    (including not_using, according to initiation_rate2 (irate_2)
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))  # runs every month
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
        if df[((df.co_date_of_childbirth < self.sim.date) & (
            df.co_date_of_childbirth > self.sim.date - DateOffset(months=1)))].empty == True:
            pass
        else:
            birth_idx = df.index[((df.co_date_of_childbirth < self.sim.date) & (
                    df.co_date_of_childbirth > self.sim.date - DateOffset(months=1)))]
            # sample contraceptive method for everyone just given birth
            sampled_method = pd.Series(rng.choice(c_names, p=c_probs, size=len(birth_idx), replace=True),
                                       index=birth_idx)
            # update contraception method for all women who have just given birth
            df.loc[birth_idx, 'co_contraception'] = sampled_method[birth_idx]
            # output some logging if any post-birth contraception
            if len(birth_idx):
                for woman_id in birth_idx:
                    logger.info('%s|post_birth_contraception|%s',
                                self.sim.date,
                                {
                                    'woman_index': woman_id,
                                    'co_contraception': df.at[woman_id, 'co_contraception']
                                })


class PregnancyPoll(RegularEvent, PopulationScopeEventMixin):
    """
    This event looks across each woman in the population to determine who will become pregnant
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        self.age_low = 15
        self.age_high = 49

    def apply(self, population):

        logger.debug('Checking to see if anyone should become pregnant....')

        if self.sim.date > Date(2035, 1, 1):
            logger.debug('Now after 2035')

        df = population.props  # get the population dataframe

        # get the subset of women from the population dataframe and relevant characteristics
        subset = (df.sex == 'F') & df.is_alive & df.age_years.between(self.age_low, self.age_high) & ~df.is_pregnant & (
                df.co_contraception == 'not_using')
        females = df.loc[subset, ['co_contraception', 'age_years']]

        # load the fertility schedule (imported datasheet from excel workbook)
        fertility_schedule = self.module.parameters['fertility_schedule']

        # --------

        # get the probability of pregnancy for each woman in the model, through merging with the fert_schedule data
        len_before_merge = len(females)
        females = females.reset_index().merge(fertility_schedule,
                                              left_on=['age_years'],
                                              # TimC: got rid of 'contraception' here as just one basefert_dhs per age (see new 'Age_spec fertility' sheet)
                                              right_on=['age'],
                                              # TimC: got rid of 'cmeth' here as just one basefert_dhs per age (see new 'Age_spec fertility' sheet)
                                              how='inner').set_index('person')
        assert len(females) == len_before_merge

        # flipping the coin to determine if this woman will become pregnant (basefert_dhs is in the Excel sheet)
        newly_pregnant = (self.module.rng.random_sample(size=len(females)) < females.basefert_dhs / 12)

        # the imported number is a yearly proportion. So adjust the rate accordingly
        # to the frequency with which the event is recurring
        # TODO: this should be linked to the self.frequency value

        newly_pregnant_ids = females.index[newly_pregnant]

        # updating the pregancy status for that women
        df.loc[newly_pregnant_ids, 'is_pregnant'] = True
        df.loc[newly_pregnant_ids, 'date_of_last_pregnancy'] = self.sim.date

        # loop through each newly pregnant women in order to schedule them a 'delayed birth event'
        for female_id in newly_pregnant_ids:
            logger.info('%s|female %d pregnant at age: %d', self.sim.date, female_id,
                        females.at[female_id, 'age_years'])

            # schedule the birth event for this woman (9 months plus/minus 2 wks)
            date_of_birth = self.sim.date + DateOffset(months=9,
                                                       weeks=-2 + 4 * self.module.rng.random_sample())

            # Schedule the Birth
            self.sim.schedule_event(DelayedBirthEvent(self.module, female_id),
                                    date_of_birth)

            logger.info('birth booked for: %s', date_of_birth)


class DelayedBirthEvent(Event, IndividualScopeEventMixin):
    """A one-off event in which a pregnant mother gives birth.
    """

    def __init__(self, module, mother_id):
        """Create a new birth event.

        We need to pass the person this event happens to to the base class constructor
        using super(). We also pass the module that created this event, so that random
        number generators can be scoped per-module.

        :param module: the module that created this event
        :param mother_id: the person giving birth
        """
        super().__init__(module, person_id=mother_id)

    def apply(self, mother_id):
        """Apply this event to the given person.
        Assuming the person is still alive, we ask the simulation to create a new offspring.
        :param mother_id: the person the event happens to, i.e. the mother giving birth
        """

        logger.info('%s|@@@@ A Birth is now occuring, to mother %s', self.sim.date, mother_id)

        df = self.sim.population.props

        # If the mother is alive and still pregnant
        if df.at[mother_id, 'is_alive'] and df.at[mother_id, 'is_pregnant']:
            self.sim.do_birth(mother_id)


class ContraceptionLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """Logs output for analysis_contraception
        """
        # run this event every 12 months (every year)
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))
        self.age_low = 15
        self.age_high = 49

    def apply(self, population):
        df = population.props

        contraception_count = df[df.is_alive & df.age_years.between(self.age_low, self.age_high)].groupby(
            'co_contraception').size()

        logger.info('%s|contraception|%s',
                    self.sim.date,
                    {
                        'total': sum(contraception_count),
                        'not_using': contraception_count['not_using'],
                        'using': sum(contraception_count) - contraception_count['not_using'],
                        'pill': contraception_count['pill'],
                        'IUD': contraception_count['IUD'],
                        'injections': contraception_count['injections'],
                        'implant': contraception_count['implant'],
                        'male_condom': contraception_count['male_condom'],
                        'female_sterilization': contraception_count['female_sterilization'],
                        'other_modern': contraception_count['other_modern'],
                        'periodic_abstinence': contraception_count['periodic_abstinence'],
                        'withdrawal': contraception_count['withdrawal'],
                        'other_traditional': contraception_count['other_traditional']
                    })

        preg_counts = df[df.is_alive & df.age_years.between(self.age_low, self.age_high)].is_pregnant.value_counts()
        is_preg_count = (df.is_alive & df.age_years.between(self.age_low, self.age_high) & df.is_pregnant).sum()
        is_notpreg_count = (df.is_alive & df.age_years.between(self.age_low, self.age_high) & ~df.is_pregnant).sum()

        logger.info('%s|pregnancy|%s', self.sim.date,
                    {
                        'total': sum(preg_counts),
                        'pregnant': is_preg_count,
                        'not_pregnant': is_notpreg_count
                    })
