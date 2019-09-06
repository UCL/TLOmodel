import logging

import pandas as pd

from pathlib import Path

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent

from tlo.methods import demography, labour


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AbortionAndMiscarriage(Module):
    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    PARAMETERS = {
        'cumu_risk_miscarriage': Parameter(
            Types.REAL, 'cumulative risk of miscarriage in the first 24 weeks of pregnancy'),
        'rr_miscarriage_prevmiscarriage': Parameter(
            Types.REAL, 'relative risk of miscarriage for women who have previously miscarried'),
        'rr_miscarriage_35': Parameter(
            Types.REAL, 'relative risk of miscarriage for women who is over 35 years old'),
        'rr_miscarriage_3134': Parameter(
            Types.REAL, 'relative risk of miscarriage for women who is between 31 and 34 years old'),
        'rr_miscarriage_grav4': Parameter(
            Types.REAL, 'relative risk of miscarriage for women who has a gravidity of greater than 4'),
        'incidence_induced_abortion': Parameter(
            Types.REAL, 'incidence of induced abortion before 28 weeks gestation'),
        'prob_pph_misc': Parameter(
            Types.REAL, 'probability of a postpartum haemorrhage following a miscarriage  before 28 weeks'
                        ' gestation'),
        'prob_sepsis_misc': Parameter(
            Types.REAL, 'probability of a sepsis  following a miscarriage before 28 weeks gestation'),
        'cfr_sepsis_misc': Parameter(
            Types.REAL, 'case fatality rate for maternal sepsis following a miscarriage'),
        'cfr_haem_misc': Parameter(
            Types.REAL, 'case fatality rate for haemorrhage following a miscarriage'),
    }

    PROPERTIES = {
        'am_total_miscarriages': Property(Types.INT, 'the number of miscarriages a woman has experienced'),
        'am_date_most_recent_miscarriage': Property(Types.DATE, 'the date this woman has last experienced spontaneous'
                                                                ' miscarriage'),
        'am_total_induced_abortion': Property(Types.INT, 'the number of induced abortions a woman has experienced'),
        'am_date_most_recent_abortion': Property(Types.DATE, 'the date this woman has last undergone an induced'
                                                             ' abortion'),
        'am_safety_of_abortion': Property(Types.CATEGORICAL, 'Null, Safe, Less Safe, Least Safe',
                                          categories=['null', 'safe',  'less_safe', 'least_safe']),
    }

    # Todo: Consider the relevance for dating both abortion/miscarriage

    def read_parameters(self, data_folder):
        params = self.parameters

        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_AbortionAndMiscarriage.xlsx',
                            sheet_name='parameter_values')
        dfd.set_index('parameter_name', inplace=True)

        params['cumu_risk_miscarriage'] = dfd.loc['cumu_risk_miscarriage', 'value']
        params['rr_miscarriage_prevmiscarriage'] = dfd.loc['rr_miscarriage_prevmiscarriage', 'value']
        params['rr_miscarriage_35'] = dfd.loc['rr_miscarriage_35', 'value']
        params['rr_miscarriage_3134'] = dfd.loc['rr_miscarriage_3134', 'value']
        params['rr_miscarriage_grav4'] = dfd.loc['rr_miscarriage_grav4', 'value']
        params['incidence_induced_abortion'] = dfd.loc['incidence_induced_abortion', 'value']
        params['prob_pph_misc'] = dfd.loc['prob_pph_misc', 'value']
        params['prob_sepsis_misc'] = dfd.loc['prob_sepsis_misc', 'value']
        params['cfr_sepsis_misc'] = dfd.loc['cfr_sepsis_misc', 'value']
        params['cfr_haem_misc'] = dfd.loc['cfr_haem_misc', 'value']

        if 'HealthBurden' in self.sim.modules.keys():
            params['daly_wt_abortive_outcome'] = self.sim.modules['HealthBurden'].get_daly_weight(352)

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        df = population.props
        df.loc[df.sex == 'F', 'am_total_miscarriages'] = 0
        df.loc[df.sex == 'F', 'am_date_most_recent_miscarriage'] = pd.NaT
        df.loc[df.sex == 'F', 'am_total_induced_abortion'] = 0
        df.loc[df.sex == 'F', 'am_date_most_recent_abortion'] = pd.NaT
        df.loc[df.sex == 'F', 'am_safety_of_abortion'] = 'null'

        self.sim.modules['HealthSystem'].register_disease_module(self)

    def initialise_simulation(self, sim):

        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        # add the basic event
        event = InducedAbortionEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=6))

        # add an event to log to screen
        sim.schedule_event(AbortionAndMiscarriageLoggingEvent(self), sim.date + DateOffset(months=6))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the ID for the mother for this child
        :param child_id: the ID for the new child
        """

        df = self.sim.population.props

        if df.at[child_id, 'sex'] == 'F':
            df.at[child_id, 'am_total_miscarriages'] = 0
            df.at[child_id, 'am_date_most_recent_miscarriage'] = pd.NaT
            df.at[child_id, 'am_total_induced_abortion'] = 0
            df.at[child_id, 'am_date_most_recent_abortion'] = pd.NaT
            df.at[child_id, 'am_safety_of_abortion'] = 'null'

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug('This is AbortionAndMiscarriage, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)

    def report_daly_values(self):

        logger.debug('This is Abortion and Miscarriage reporting my health values')
        df = self.sim.population.props
        params = self.parameters

        one_month_prior = self.sim.date - pd.to_timedelta(1, unit='m')

        recent_abortion = df.index[df.is_alive & (df.am_total_induced_abortion == 1) &
                                   (df.am_date_most_recent_abortion > one_month_prior)]

        recent_miscarriage = df.index[df.is_alive & (df.am_total_miscarriages == 1) &
                                      (df.am_date_most_recent_miscarriage > one_month_prior)]

        health_values_1 = df.loc[recent_abortion, 'am_total_induced_abortion'].map(
            {
                0: 0,
                1: params['daly_wt_abortive_outcome']
            })
        health_values_2 = df.loc[recent_miscarriage, 'am_total_miscarriages'].map(
            {
                0: 0,
                1: params['daly_wt_abortive_outcome']
            })

        health_values_df = pd.concat([health_values_1.loc[df.is_alive], health_values_2.loc[df.is_alive]], axis=1)

        return health_values_df

        # Todo: Have yet to confirm this runs as HealthBurden is not registered presently


class CheckIfNewlyPregnantWomanWillMiscarry(Event, IndividualScopeEventMixin):
    """This event checks if a woman who is newly pregnant will experience a miscarriage. """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        # First we identify if a newly pregnant woman has any risk factors that predispose her to a miscarriage
        if (df.at[individual_id, 'am_total_miscarriages'] >= 1) & (df.at[individual_id, 'age_years'] <= 30) & \
           (df.at[individual_id, 'la_parity'] < 1 > 3):
            rf1 = params['rr_miscarriage_prevmiscarriage']
        else:
            rf1 = 1

        if (df.at[individual_id, 'am_total_miscarriages'] == 0) & (df.at[individual_id, 'age_years'] >= 35) & \
           (df.at[individual_id, 'la_parity'] < 1 > 3):
            rf2 = params['rr_miscarriage_35']
        else:
            rf2 = 1

        if (df.at[individual_id, 'am_total_miscarriages'] == 0) & (df.at[individual_id, 'age_years'] >= 31) & \
           (df.at[individual_id, 'age_years'] <= 34) & (df.at[individual_id,'la_parity'] < 1 > 3):
            rf3 = params['rr_miscarriage_3134']
        else:
            rf3 = 1

        if (df.at[individual_id, 'am_total_miscarriages'] == 0) & (df.at[individual_id, 'age_years'] <= 30) & \
           (df.at[individual_id, 'la_parity'] >= 1 <= 3):
            rf4 = params['rr_miscarriage_grav4']
        else:
            rf4 = 1

        # Next we multiply the baseline rate of miscarriage in the reference population who are absent of riskfactors
        # by the product of the relative rates for any risk factors this mother may have
        riskfactors = rf1 * rf2 * rf3 * rf4

        if riskfactors == 1:
            eff_cumu_risk_miscarriage = params['cumu_risk_miscarriage']
        else:
            eff_cumu_risk_miscarriage = riskfactors * params['cumu_risk_miscarriage']

        # Finally a random draw is used to determine if this woman will experience a miscarriage for this pregnancy
        random = self.sim.rng.random_sample(size=1)
        if random < eff_cumu_risk_miscarriage:

            # If a newly pregnant woman will miscarry her pregnancy, a random date within 24 weeks of conception is
            # generated, the date of miscarriage is scheduled for this day

            # (N.b malawi date of viability is 28 weeks, we using 24 as cut off to allow for very PTB.)

            random_draw = self.sim.rng.exponential(scale=0.5, size=1)  # Todo: confirm the correct value for the scale
            random_days = pd.to_timedelta(random_draw[0] * 168, unit='d')
            miscarriage_date = self.sim.date + random_days

            self.sim.schedule_event(MiscarriageAndPostMiscarriageComplicationsEvent(self.module, individual_id,
                                                                                    cause='post miscarriage'),
                                    miscarriage_date)

            # And for women who do not have a miscarriage we move them to labour scheduler to determine at what
            # gestation they will go into labour

        else:
            self.sim.schedule_event(labour.LabourScheduler(self.sim.modules['Labour'], individual_id, cause='pregnancy')
                                    , self.sim.date)


class MiscarriageAndPostMiscarriageComplicationsEvent(Event, IndividualScopeEventMixin):

    """reset a woman's pregnancy properties as she has miscarried. Also calculates risk of complications following
     miscarriage """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        # Here we reset the pregnancy properties of a woman who has miscarried
        if df.at[individual_id, 'is_alive'] & df.at[individual_id, 'is_pregnant']:
            df.at[individual_id, 'is_pregnant'] = False
            df.at[individual_id, 'am_total_miscarriages'] = +1
            df.at[individual_id, 'am_date_most_recent_miscarriage'] = self.sim.date

            df.at[individual_id, 'ac_gestational_age'] = 0

            logger.info('This is MiscarriageAndPostMiscarriageComplicationsEvent, person %d has experienced a '
                        'miscarriage on date %s and is no longer pregnant', individual_id, self.sim.date)

        # As with the other complication events here we determine if this woman will experience any complications
        # following her miscarriage
            riskfactors = 1  # rf1
            eff_prob_pph = riskfactors * params['prob_pph_misc']
            random = self.sim.rng.random_sample(size=1)
            if random < eff_prob_pph:
                df.at[individual_id, 'la_pph'] = True

            riskfactors = 1  # rf1
            eff_prob_pp_sepsis = riskfactors * params['prob_sepsis_misc']
            random = self.sim.rng.random_sample(size=1)
            if random < eff_prob_pp_sepsis:
                df.at[individual_id, 'la_sepsis'] = True

        # Currently if she has experienced any of these complications she is scheduled to pass through the DeathEvent
        # to determine if they are fatal
            if df.at[individual_id, 'la_sepsis'] or df.at[individual_id, 'la_pph']:
                logger.info('This is MiscarriageAndPostMiscarriageComplicationsEvent scheduling a possible death for'
                            ' person %d after suffering complications of miscarriage', individual_id)
                self.sim.schedule_event(MiscarriageAndAbortionDeathEvent(self.module, individual_id, cause='miscarriage'),
                                        self.sim.date)


class InducedAbortionEvent(RegularEvent, PopulationScopeEventMixin):
    """
    This event is occurring regularly at 6 montly intervals to determine if women will undergo an induced abortion
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=6))

    def apply(self, population):
        df = population.props
        params = self.module.parameters

        # First we create an index of all the women who have become pregnant in the previous 6 months
        pregnant_past_six_mths = df.index[df.is_pregnant & df.is_alive & (df.ac_gestational_age <= 27) &
                                          (df.date_of_last_pregnancy > self.sim.start_date)]

        # We then apply the risk of induced abortion to these women
        incidence_abortion = pd.Series(params['incidence_induced_abortion'], index=pregnant_past_six_mths)

        random_draw = pd.Series(self.module.rng.random_sample(size=len(pregnant_past_six_mths)),
                                index=df.index[df.is_pregnant & df.is_alive &
                                               (df.ac_gestational_age <= 27) &
                                               (df.date_of_last_pregnancy > self.sim.start_date)])

        dfx = pd.concat([incidence_abortion, random_draw], axis=1)
        dfx.columns = ['incidence_abortion', 'random_draw']
        idx_induced_abortion = dfx.index[dfx.incidence_abortion > dfx.random_draw]

        # For women who will undergo induced abortion we reset their pregnancy properties and update the data frame
        df.loc[idx_induced_abortion, 'is_pregnant'] = False
        df.loc[idx_induced_abortion, 'am_total_induced_abortion'] = +1
        df.loc[idx_induced_abortion, 'am_date_most_recent_abortion'] = self.sim.date  # (is this needed)
        df.loc[idx_induced_abortion, 'ac_gestational_age'] = 0

        # Todo: Please see SBA master sheet for epi queires
        # Todo: Apply safety level and complications
        # Todo : schedule care seeking for HSI and schedule death event


class MiscarriageAndAbortionDeathEvent (Event, IndividualScopeEventMixin):

    """handles death following miscarriage"""

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        if df.at[individual_id, 'la_pph']:
            random = self.sim.rng.random_sample(size=1)
            if random < params['cfr_haem_misc']:
                df.at[individual_id, 'la_maternal_death'] = True
                df.at[individual_id, 'la_maternal_death_date'] = self.sim.date
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                      cause=' complication of miscarriage'),
                                        self.sim.date)
            else:
                df.at[individual_id, 'la_pph'] = False

        if df.at[individual_id, 'la_sepsis']:
            random = self.sim.rng.random_sample(size=1)
            if random < params['cfr_sepsis_misc']:
                df.at[individual_id, 'la_maternal_death'] = True
                df.at[individual_id, 'la_maternal_death_date'] = self.sim.date
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                      cause='complication of miscarriage'),
                                        self.sim.date)
                logger.info('This is MiscarriageDeathEvent scheduling a death for person %d on date %s who died '
                            'following complications of miscarriage', individual_id,
                             self.sim.date)

                logger.info('%s|maternal_death|%s', self.sim.date,
                        {
                            'age': df.at[individual_id, 'age_years'],
                            'person_id': individual_id
                        })
            else:
                df.at[individual_id, 'la_pph'] = False




class HSI_AbortionAndMiscarriage_PresentsForPostAbortionCare(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    This interaction manages the initial appointment for women presenting to hospital for post aboriton care for
    incomplete loss of pregnancy
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['MinorSurg'] = 1  # todo: medical management would require less

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        abortion_pkg_code = pd.unique(consumables.loc[consumables[
                                                       'Intervention_Pkg'] == 'Post-abortion case management',
                                                      'Intervention_Pkg_Code'])[0]
        the_cons_footprint = {
            'Intervention_Package_Code': [abortion_pkg_code],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'AbortionAndMiscarriage_PresentsForPostAbortionCare'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = [0]
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):

        logger.debug('This is HSI_AbortionAndMiscarriage_PresentsForPostAbortionCare, a first appointment for person %d',
                     person_id)

        df = self.sim.population.props

        #  Treatment for: Incomplete abortion (medical/surgical), sepsis?, trauma, bleeding


class AbortionAndMiscarriageLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """Produce a summmary of the numbers of people with respect to their 'mockitis status'
        """
        # run this event every month
        self.repeat = 6
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props

