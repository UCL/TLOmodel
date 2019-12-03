import logging
import os
from pathlib import Path

import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import demography
from tlo.methods.healthsystem import HSI_Event

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Malaria(Module):

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    PARAMETERS = {
        'mal_inc': Parameter(Types.REAL, 'monthly incidence of malaria in all ages'),
        'interv': Parameter(Types.REAL, 'data frame of intervention coverage by year'),
        'clin_inc': Parameter(Types.REAL, 'data frame of clinical incidence by age, district, intervention coverage'),
        'inf_inc': Parameter(Types.REAL, 'data frame of infection incidence by age, district, intervention coverage'),
        'sev_inc': Parameter(Types.REAL, 'data frame of severe case incidence by age, district, intervention coverage'),

        'p_infection': Parameter(
            Types.REAL, 'Probability that an uninfected individual becomes infected'),
        'sensitivity_rdt': Parameter(
            Types.REAL, 'Sensitivity of rdt'),
        'cfr': Parameter(
            Types.REAL, 'case-fatality rate for severe malaria'),
        'dur_asym': Parameter(
            Types.REAL, 'duration (days) of asymptomatic malaria'),
        'dur_clin': Parameter(
            Types.REAL, 'duration (days) of clinical symptoms of malaria'),
        'dur_clin_para': Parameter(
            Types.REAL, 'duration (days) of parasitaemia for clinical malaria cases'),
        'rr_hiv': Parameter(
            Types.REAL, 'relative risk of clinical malaria if hiv-positive'),
    }

    PROPERTIES = {
        'ma_is_infected': Property(
            Types.BOOL, 'Current status of malaria'),
        'ma_date_infected': Property(
            Types.DATE, 'Date of latest infection'),
        'ma_date_death': Property(
            Types.DATE, 'Date of scheduled death due to malaria'),
        'ma_tx': Property(Types.BOOL, 'Currently on anti-malarial treatment'),
        'ma_date_tx': Property(
            Types.DATE, 'Date treatment started for most recent malaria episode'),
        'ma_specific_symptoms': Property(
            Types.CATEGORICAL,
            'specific symptoms with malaria infection', categories=['none', 'clinical', 'severe']),
        'ma_unified_symptom_code': Property(Types.CATEGORICAL, 'level of symptoms on the standardised scale, 0-4',
                                            categories=[0, 1, 2, 3, 4]),
        'ma_district_edited': Property(
            Types.STRING, 'edited districts to match with malaria data'),
        'ma_age_edited': Property(
            Types.REAL, 'age values redefined to match with malaria data')

    }

    def read_parameters(self, data_folder):

        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_malaria.xlsx', sheet_name='parameters')
        self.load_parameters_from_dataframe(dfd)

        p = self.parameters

        workbook = pd.read_excel(os.path.join(self.resourcefilepath,
                                              'ResourceFile_malaria.xlsx'), sheet_name=None)

        # baseline characteristics
        p['mal_inc'] = workbook['incidence']
        p['interv'] = workbook['interventions']
        p['clin_inc'] = workbook['clin_inc_age']
        p['inf_inc'] = workbook['inf_age']
        p['sev_inc'] = workbook['sev_age']

        # get the DALY weight that this module will use from the weight database (these codes are just random!)
        if 'HealthBurden' in self.sim.modules.keys():
            p['daly_wt_none'] = self.sim.modules['HealthBurden'].get_daly_weight(50)
            p['daly_wt_clinical'] = self.sim.modules['HealthBurden'].get_daly_weight(50)
            p['daly_wt_severe'] = self.sim.modules['HealthBurden'].get_daly_weight(589)

    def initialise_population(self, population):
        df = population.props
        p = self.parameters
        now = self.sim.date

        # Set default for properties
        df['ma_is_infected'] = False
        df['ma_date_infected'] = pd.NaT
        df['ma_date_death'] = pd.NaT
        df['ma_tx'] = False
        df['ma_date_tx'] = pd.NaT
        df['ma_specific_symptoms'].values[:] = 'none'
        df['ma_unified_symptom_code'].values[:] = 0
        df['ma_district_edited'] = df['district_of_residence']
        df['ma_age_edited'] = 0

        # ----------------------------------- RENAME DISTRICTS -----------------------------------
        # rename districts to match malaria data
        df.loc[(df.district_of_residence == 'Lilongwe City'), 'ma_district_edited'] = 'Lilongwe'
        df.loc[(df.district_of_residence == 'Blantyre City'), 'ma_district_edited'] = 'Blantyre'
        df.loc[(df.district_of_residence == 'Zomba City'), 'ma_district_edited'] = 'Zomba'
        df.loc[(df.district_of_residence == 'Mzuzu City'), 'ma_district_edited'] = 'Mzimba'
        df.loc[(df.district_of_residence == 'Nkhata Bay'), 'ma_district_edited'] = 'Mzimba'

        assert (not pd.isnull(df['ma_district_edited']).any())

        # ----------------------------------- INTERVENTION COVERAGE -----------------------------------
        interv = p['interv']

        # find annual intervention coverage levels rate for 2010
        # TODO: replace with district-level intervention coverage estimates when available
        itn_2010 = interv.loc[interv.Year == now.year, 'ITN_coverage'].values[0]
        # print(itn_2010)
        irs_2010 = interv.loc[interv.Year == now.year, 'IRS_coverage'].values[0]

        # need to select values[0] maybe????
        itn_2010 = round(itn_2010, 1)  # round to nearest 0.1 to match Pete's data
        irs_2010 = 0.8 if irs_2010 > 0.5 else 0  # round to 0 or 0.8

        inf_inc = p['inf_inc']
        inf_inc_Jan2010 = inf_inc.loc[(inf_inc.month == 1) & (inf_inc.llin == itn_2010) & (inf_inc.irs == irs_2010)]
        clin_inc = p['clin_inc']
        clin_inc_Jan2010 = clin_inc.loc[
            (clin_inc.month == 1) & (clin_inc.llin == itn_2010) & (clin_inc.irs == irs_2010)]
        sev_inc = p['sev_inc']
        sev_inc_Jan2010 = sev_inc.loc[(sev_inc.month == 1) & (sev_inc.llin == itn_2010) & (sev_inc.irs == irs_2010)]

        # create new district list to match Pete's list and keep/use for all malaria events

        # ----------------------------------- INCIDENCE ESTIMATES -----------------------------------

        # for each district and age, look up incidence estimate using itn_2010 and irs_2010
        # create new age column with 0, 0.5, 1, 2, ...
        df.loc[df.age_exact_years.between(0, 0.5), 'ma_age_edited'] = 0
        df.loc[df.age_exact_years.between(0.5, 1), 'ma_age_edited'] = 0.5
        df.loc[(df.age_exact_years >= 1), 'ma_age_edited'] = df.age_years[df.age_years >= 1]
        assert (not pd.isnull(df['ma_age_edited']).any())
        df['ma_age_edited'] = df['ma_age_edited'].astype('float')  # for merge with malaria data

        df_inf = df.reset_index().merge(inf_inc_Jan2010, left_on=['ma_district_edited', 'ma_age_edited'],
                                        right_on=['admin', 'age'],
                                        how='left', indicator=True).set_index('person')
        df_inf['monthly_prob_inf'] = df_inf['monthly_prob_inf'].fillna(0)  # 0 if over 80 yrs
        assert (not pd.isnull(df_inf['monthly_prob_inf']).any())

        df_clin = df.reset_index().merge(clin_inc_Jan2010, left_on=['ma_district_edited', 'ma_age_edited'],
                                         right_on=['admin', 'age'],
                                         how='left').set_index('person')
        df_clin['monthly_prob_clin'] = df_clin['monthly_prob_clin'].fillna(0)  # 0 if over 80 yrs
        assert (not pd.isnull(df_clin['monthly_prob_clin']).any())

        df_sev = df.reset_index().merge(sev_inc_Jan2010, left_on=['ma_district_edited', 'ma_age_edited'],
                                        right_on=['admin', 'age'],
                                        how='left').set_index('person')
        df_sev['monthly_prob_sev'] = df_sev['monthly_prob_sev'].fillna(0)  # 0 if over 80 yrs
        assert (not pd.isnull(df_sev['monthly_prob_sev']).any())

        # ----------------------------------- BASELINE INFECTION STATUS -----------------------------------
        ## infected
        risk_ml = pd.Series(0, index=df.index)
        risk_ml.loc[df.is_alive] = 1  # applied to everyone
        # risk_ml.loc[df.hv_inf ] *= p['rr_hiv']  # then have to scale within every subgroup

        random_draw = self.rng.random_sample(size=len(df_inf))
        ml_idx = df_inf[df_inf.is_alive & (random_draw < df_inf.monthly_prob_inf)].index
        df.loc[ml_idx, 'ma_is_infected'] = True
        df.loc[
            ml_idx, 'ma_date_infected'] = now  # TODO: scatter dates across month, then have to scatter clinical/severe

        ## clinical - subset of infected
        random_draw = self.rng.random_sample(size=len(df_clin))
        clin_idx = df_clin[df_clin.is_alive & df_clin.ma_is_infected & (random_draw < df_clin.monthly_prob_clin)].index
        df.loc[clin_idx, 'ma_specific_symptoms'] = 'clinical'
        df.loc[clin_idx, 'ma_unified_symptom_code'] = 1

        ## severe - subset of clinical
        random_draw = self.rng.random_sample(size=len(df_sev))
        sev_idx = df_sev[df_sev.is_alive & (df_sev.ma_specific_symptoms == 'clinical') & (
                random_draw < df_sev.monthly_prob_sev)].index
        df.loc[sev_idx, 'ma_specific_symptoms'] = 'severe'
        df.loc[sev_idx, 'ma_unified_symptom_code'] = 2

        ## tidy up
        del df_inf, df_clin, df_sev

        # ----------------------------------- SCHEDULED DEATHS -----------------------------------
        # schedule deaths within the next week
        # Assign time of infections across the month
        random_draw = self.rng.random_sample(size=len(df))
        death = df.index[
            (df.ma_specific_symptoms == 'severe') & (df.ma_date_infected == now) & (random_draw < p['cfr'])]

        for person in death:
            random_date = self.rng.randint(low=0, high=7)
            random_days = pd.to_timedelta(random_date, unit='d')

            death_event = MalariaDeathEvent(self, individual_id=person,
                                            cause='malaria')  # make that death event
            self.sim.schedule_event(death_event, self.sim.date + random_days)  # schedule the death

        # ----------------------------------- HEALTHCARE-SEEKING -----------------------------------

        # find annual intervention coverage levels rate for 2010
        act = interv.loc[interv.Year == now.year, 'ACT_coverage'].values[0]

        seeks_care = pd.Series(data=False, index=df.index[
            (df.ma_specific_symptoms == 'clinical') | (df.ma_specific_symptoms == 'severe')])

        # assume only clinical cases will seek care - includes severe
        for i in df.loc[clin_idx].index:
            # prob = self.sim.modules['HealthSystem'].get_prob_seek_care(i, symptom_code=4)
            seeks_care[i] = self.rng.rand() < act  # placeholder for coverage / testing rates

        if seeks_care.sum() > 0:

            for person_index in seeks_care.index[seeks_care]:
                # print(person_index)

                logger.debug(
                    'This is Malaria, scheduling HSI_Malaria_rdt for person %d',
                    person_index)

                event = HSI_Malaria_rdt(self, person_id=person_index)
                self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                    priority=2,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(weeks=2)
                                                                    )
        else:
            logger.debug(
                'This is Malaria, There is no new healthcare seeking')

        # ----------------------------------- PARASITE CLEARANCE - NO TREATMENT -----------------------------------
        # schedule self-cure if no treatment, no self-cure from severe malaria

        # asymptomatic
        asym = df.index[df.ma_is_infected & (df.ma_specific_symptoms == 'none') & (df.ma_date_infected == now)]

        for person in df.loc[asym].index:
            logger.debug(
                'This is Malaria, scheduling parasite clearance for asymptomatic person %d', person)

            random_date = self.rng.randint(low=0, high=p['dur_asym'])
            random_days = pd.to_timedelta(random_date, unit='d')

            cure = MalariaParasiteClearanceEvent(self, person)
            self.sim.schedule_event(cure, (self.sim.date + random_days))

        # clinical
        clin = df.index[(df.ma_specific_symptoms == 'clinical') & (df.ma_date_infected == now)]

        for person in df.loc[clin].index:
            logger.debug(
                'This is Malaria, scheduling parasite clearance and symptom end for symptomatic person %d', person)

            date_para = self.rng.randint(low=0, high=p['dur_clin_para'])
            date_para_days = pd.to_timedelta(date_para, unit='d')

            date_clin = self.rng.randint(low=0, high=p['dur_clin'])
            date_clin_days = pd.to_timedelta(date_clin, unit='d')

            cure = MalariaParasiteClearanceEvent(self, person)
            self.sim.schedule_event(cure, (self.sim.date + date_para_days))

            # schedule symptom end (5 days)
            symp_end = MalariaSympEndEvent(self, person)
            self.sim.schedule_event(symp_end, self.sim.date + date_clin_days)

        # ----------------------------------- REGISTER WITH HEALTH SYSTEM -----------------------------------
        self.sim.modules['HealthSystem'].register_disease_module(self)

    def initialise_simulation(self, sim):
        # add the basic event
        event = MalariaEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=1))

        # add an event to log to screen - output on last day of each year
        sim.schedule_event(MalariaLoggingEvent(self), sim.date + DateOffset(months=0))

    def on_birth(self, mother_id, child_id):

        df = self.sim.population.props

        df.at[child_id, 'ma_is_infected'] = False
        df.at[child_id, 'ma_date_infected'] = pd.NaT
        df.at[child_id, 'ma_date_death'] = pd.NaT
        df.at[child_id, 'ma_tx'] = False
        df.at[child_id, 'ma_date_tx'] = pd.NaT
        df.at[child_id, 'ma_specific_symptoms'].values[:] = 'none'
        df.at[child_id, 'ma_unified_symptom_code'].values[:] = 0
        df.at[child_id, 'ma_district_edited'] = df['district_of_residence']
        df.at[child_id, 'ma_age_edited'] = 0

        # ----------------------------------- RENAME DISTRICTS -----------------------------------
        # rename districts to match malaria data
        df.at[child_id, (df.district_of_residence == 'Lilongwe City'), 'ma_district_edited'] = 'Lilongwe'
        df.at[child_id, (df.district_of_residence == 'Blantyre City'), 'ma_district_edited'] = 'Blantyre'
        df.at[child_id, (df.district_of_residence == 'Zomba City'), 'ma_district_edited'] = 'Zomba'
        df.at[child_id, (df.district_of_residence == 'Mzuzu City'), 'ma_district_edited'] = 'Mzimba'
        df.at[child_id, (df.district_of_residence == 'Nkhata Bay'), 'ma_district_edited'] = 'Mzimba'

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug('This is Malaria, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        logger.debug('This is malaria reporting my health values')

        df = self.sim.population.props  # shortcut to population properties dataframe

        p = self.parameters

        health_values = df.loc[df.is_alive, 'ma_specific_symptoms'].map({
            'none': 0,
            'clinical': p['daly_wt_clinical'],
            'severe': p['daly_wt_severe']
        })
        health_values.name = 'Malaria Symptoms'  # label the cause of this disability

        return health_values.loc[df.is_alive]  # returns the series


class MalariaEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):

        logger.debug('This is MalariaEvent, tracking the disease progression of the population.')

        df = population.props
        p = self.module.parameters
        rng = self.module.rng
        now = self.sim.date

        # ----------------------------------- INTERVENTION COVERAGE -----------------------------------
        interv = p['interv']

        # find annual intervention coverage levels rate for 2010
        # TODO: replace with district-level intervention coverage estimates when available
        itn_2010 = interv.loc[interv.Year == now.year, 'ITN_coverage'].values[0]
        irs_2010 = interv.loc[interv.Year == now.year, 'IRS_coverage'].values[0]
        itn_2010 = round(itn_2010, 1)  # round to nearest 0.1 to match Pete's data
        irs_2010 = 0.8 if irs_2010 > 0.5 else 0  # round to 0 or 0.8

        # select incidence based on intervention coverage
        month = now.month

        inf_inc = p['inf_inc']
        inf_inc_month = inf_inc.loc[(inf_inc.month == month) & (inf_inc.llin == itn_2010) & (inf_inc.irs == irs_2010)]
        clin_inc = p['clin_inc']
        clin_inc_month = clin_inc.loc[
            (clin_inc.month == month) & (clin_inc.llin == itn_2010) & (clin_inc.irs == irs_2010)]
        sev_inc = p['sev_inc']
        sev_inc_month = sev_inc.loc[(sev_inc.month == month) & (sev_inc.llin == itn_2010) & (sev_inc.irs == irs_2010)]

        # for each district and age, look up incidence estimate using itn_2010 and irs_2010
        # create new age column with 0, 0.5, 1, 2, ...
        df.loc[df.age_exact_years.between(0, 0.5), 'ma_age_edited'] = 0
        df.loc[df.age_exact_years.between(0.5, 1), 'ma_age_edited'] = 0.5
        df.loc[(df.age_exact_years >= 1), 'ma_age_edited'] = df.age_years[df.age_years >= 1]
        assert (not pd.isnull(df['ma_age_edited']).any())
        df['ma_age_edited'] = df['ma_age_edited'].astype('float')  # for merge with malaria data

        df_inf = df.reset_index().merge(inf_inc_month, left_on=['ma_district_edited', 'ma_age_edited'],
                                        right_on=['admin', 'age'],
                                        how='left', indicator=True).set_index('person')
        df_inf['monthly_prob_inf'] = df_inf['monthly_prob_inf'].fillna(0)  # 0 if over 80 yrs
        assert (not pd.isnull(df_inf['monthly_prob_inf']).any())

        df_clin = df.reset_index().merge(clin_inc_month, left_on=['ma_district_edited', 'ma_age_edited'],
                                         right_on=['admin', 'age'],
                                         how='left').set_index('person')
        df_clin['monthly_prob_clin'] = df_clin['monthly_prob_clin'].fillna(0)  # 0 if over 80 yrs
        assert (not pd.isnull(df_clin['monthly_prob_clin']).any())

        df_sev = df.reset_index().merge(sev_inc_month, left_on=['ma_district_edited', 'ma_age_edited'],
                                        right_on=['admin', 'age'],
                                        how='left').set_index('person')
        df_sev['monthly_prob_sev'] = df_sev['monthly_prob_sev'].fillna(0)  # 0 if over 80 yrs
        assert (not pd.isnull(df_sev['monthly_prob_sev']).any())

        # ----------------------------------- NEW INFECTIONS -----------------------------------
        ## infected
        risk_ml = pd.Series(0, index=df.index)
        risk_ml.loc[df.is_alive] = 1  # applied to everyone
        # risk_ml.loc[df.hv_inf ] *= p['rr_hiv']  # then have to scale within every subgroup

        random_draw = rng.random_sample(size=len(df_inf))
        ml_idx = df_inf[df_inf.is_alive & (random_draw < df_inf.monthly_prob_inf)].index
        df.loc[ml_idx, 'ma_is_infected'] = True
        df.loc[ml_idx, 'ma_date_infected'] = now  # TODO: scatter dates across month

        ## clinical - subset of infected
        random_draw = rng.random_sample(size=len(df_clin))
        clin_idx = df_clin[df_clin.is_alive & df_clin.ma_is_infected & (random_draw < df_clin.monthly_prob_clin)].index
        df.loc[clin_idx, 'ma_specific_symptoms'] = 'clinical'
        df.loc[clin_idx, 'ma_unified_symptom_code'] = 1

        ## severe - subset of clinical
        random_draw = rng.random_sample(size=len(df_sev))
        sev_idx = df_sev[df_sev.is_alive & (df_sev.ma_specific_symptoms == 'clinical') & (
                random_draw < df_sev.monthly_prob_sev)].index
        df.loc[sev_idx, 'ma_specific_symptoms'] = 'severe'
        df.loc[sev_idx, 'ma_unified_symptom_code'] = 2

        ## tidy up
        del df_inf, df_clin, df_sev

        # if any are infected
        if len(ml_idx):
            logger.debug('This is MalariaEvent, assigning new malaria infections')

            # ----------------------------------- PARASITE CLEARANCE - NO TREATMENT -----------------------------------
            # schedule self-cure if no treatment, no self-cure from severe malaria

            # asymptomatic
            asym = df.index[(df.ma_specific_symptoms == 'none') & (df.ma_date_infected == now)]

            for person in df.loc[asym].index:
                logger.debug(
                    'This is Malaria Event, scheduling parasite clearance for asymptomatic person %d', person)

                random_date = rng.randint(low=0, high=p['dur_asym'])
                random_days = pd.to_timedelta(random_date, unit='d')

                cure = MalariaParasiteClearanceEvent(self.module, person)
                self.sim.schedule_event(cure, (self.sim.date + random_days))

            # clinical
            clin = df.index[(df.ma_specific_symptoms == 'clinical') & (df.ma_date_infected == now)]

            for person in df.loc[clin].index:
                logger.debug(
                    'This is Malaria Event, scheduling parasite clearance and symptom end for symptomatic person %d',
                    person)

                date_para = rng.randint(low=0, high=p['dur_clin_para'])
                date_para_days = pd.to_timedelta(date_para, unit='d')
                # print('date_para_days', date_para_days)

                date_clin = rng.randint(low=0, high=p['dur_clin'])
                date_clin_days = pd.to_timedelta(date_clin, unit='d')
                # print('date_clin_days', date_clin_days)

                cure = MalariaParasiteClearanceEvent(self.module, person)
                self.sim.schedule_event(cure, (self.sim.date + date_para_days))

                # schedule symptom end (5 days)
                symp_end = MalariaSympEndEvent(self.module, person)
                self.sim.schedule_event(symp_end, self.sim.date + date_clin_days)

            # ----------------------------------- SCHEDULED DEATHS -----------------------------------
            # schedule deaths within the next week
            # Assign time of infections across the month
            random_draw = rng.random_sample(size=len(df))

            # the cfr applies to all clinical malaria - specific to severe may be even higher
            death = df.index[
                (df.ma_specific_symptoms == 'severe') & (df.ma_date_infected == now) & (random_draw < p['cfr'])]

            for person in death:
                logger.debug(
                    'This is MalariaEvent, scheduling malaria death for person %d',
                    person)

                random_date = rng.randint(low=0, high=7)
                random_days = pd.to_timedelta(random_date, unit='d')

                death_event = MalariaDeathEvent(self.module, individual_id=person,
                                                cause='malaria')  # make that death event
                self.sim.schedule_event(death_event, self.sim.date + random_days)  # schedule the death

            # ----------------------------------- HEALTHCARE-SEEKING -----------------------------------

            interv = p['interv']

            # find annual intervention coverage levels rate for 2010
            act = interv.loc[interv.Year == now.year, 'ACT_coverage'].values[0]
            # act = 1

            # all symptomatic cases can seek care
            symp = df.index[(df.ma_specific_symptoms == 'clinical') | (df.ma_specific_symptoms == 'severe')]
            seeks_care = pd.Series(data=False, index=df.loc[symp].index)

            for i in df.loc[symp].index:
                # prob = self.sim.modules['HealthSystem'].get_prob_seek_care(i, symptom_code=4)
                seeks_care[i] = rng.rand() < act  # placeholder for coverage / testing rates

            if seeks_care.sum() > 0:

                for person_index in seeks_care.index[seeks_care]:
                    # print(person_index)

                    logger.debug(
                        'This is MalariaEvent, scheduling HSI_Malaria_rdt for person %d',
                        person_index)

                    event = HSI_Malaria_rdt(self.module, person_id=person_index)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                        priority=2,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset(weeks=2)
                                                                        )
            else:
                logger.debug(
                    'This is MalariaEvent, There is no new healthcare seeking')
        else:
            logger.debug('This is MalariaEvent, no one is newly infected.')


class MalariaDeathEvent(Event, IndividualScopeEventMixin):
    """
    Performs the Death operation on an individual and logs it.
    """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)
        self.cause = cause

    def apply(self, individual_id):
        df = self.sim.population.props

        if df.at[individual_id, 'is_alive']:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id, cause='malaria'),
                                    self.sim.date)

            df.at[individual_id, 'ma_date_death'] = self.sim.date


# ---------------------------------------------------------------------------------
# Health System Interaction Events
# ---------------------------------------------------------------------------------


class HSI_Malaria_rdt(HSI_Event, IndividualScopeEventMixin):
    """
    this is a point-of-care malaria rapid diagnostic test, with results within 2 minutes
    """

    def __init__(self, module, person_id):
        super().__init__(module=Malaria, person_id=person_id)
        assert isinstance(module, Malaria)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['LabParasit'] = 1

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Malaria_RDT'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVELS = [1]
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):

        df = self.sim.population.props
        params = self.module.parameters

        logger.debug('HSI_Malaria_rdt: rdt test for person %d', person_id)

        # the OneHealth consumables have Intervention_Pkg_Code= -99 which causes errors
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        # this package contains treatment too
        pkg_code1 = pd.unique(
            consumables.loc[
                consumables['Items'] == 'Malaria test kit (RDT)',
                'Intervention_Pkg_Code'])[0]

        consumables_needed = {
            'Intervention_Package_Code': [{pkg_code1: 1}],
            'Item_Code': [],
        }

        # request the RDT
        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed, log=False
        )

        if outcome_of_request_for_consumables:

            # check if diagnosed
            if df.at[person_id, 'ma_is_infected']:

                # ----------------------------------- SEVERE MALARIA -----------------------------------

                # if severe malaria, treat for complicated malaria
                if (df.at[person_id, 'ma_specific_symptoms'] == 'severe'):

                    if (df.at[person_id, 'age_years'] < 15):

                        logger.debug(
                            "This is HSI_Malaria_rdt scheduling HSI_Malaria_tx_compl_child for person %d on date %s",
                            person_id, (self.sim.date + DateOffset(days=1)))

                        treat = HSI_Malaria_tx_compl_child(self.module, person_id=person_id)
                        self.sim.modules['HealthSystem'].schedule_hsi_event(treat,
                                                                            priority=1,
                                                                            topen=self.sim.date + DateOffset(days=1),
                                                                            tclose=None)

                    else:
                        logger.debug(
                            "This is HSI_Malaria_rdt scheduling HSI_Malaria_tx_compl_adult for person %d on date %s",
                            person_id, (self.sim.date + DateOffset(days=1)))

                        treat = HSI_Malaria_tx_compl_adult(self.module, person_id=person_id)
                        self.sim.modules['HealthSystem'].schedule_hsi_event(treat,
                                                                            priority=1,
                                                                            topen=self.sim.date + DateOffset(days=1),
                                                                            tclose=None)

                # ----------------------------------- TREATMENT CLINICAL DISEASE -----------------------------------

                else:
                    # diagnosis of clinical disease dependent on RDT sensitivity
                    diagnosed = self.sim.rng.choice([True, False], size=1, p=[params['sensitivity_rdt'],
                                                                              (1 - params['sensitivity_rdt'])])

                    # diagnosis / treatment for children <5
                    if diagnosed & (df.at[person_id, 'age_years'] < 5):
                        logger.debug("This is HSI_Malaria_rdt scheduling HSI_Malaria_tx_0_5 for person %d on date %s",
                                     person_id, (self.sim.date + DateOffset(days=1)))

                        treat = HSI_Malaria_tx_0_5(self.module, person_id=person_id)
                        self.sim.modules['HealthSystem'].schedule_hsi_event(treat,
                                                                            priority=1,
                                                                            topen=self.sim.date + DateOffset(days=1),
                                                                            tclose=None)

                    # diagnosis / treatment for children 5-15
                    if diagnosed & (df.at[person_id, 'age_years'] >= 5) & (df.at[person_id, 'age_years'] < 15):
                        logger.debug("This is HSI_Malaria_rdt scheduling HSI_Malaria_tx_5_15 for person %d on date %s",
                                     person_id, (self.sim.date + DateOffset(days=1)))

                        treat = HSI_Malaria_tx_5_15(self.module, person_id=person_id)
                        self.sim.modules['HealthSystem'].schedule_hsi_event(treat,
                                                                            priority=1,
                                                                            topen=self.sim.date + DateOffset(days=1),
                                                                            tclose=None)

                    # diagnosis / treatment for adults
                    if diagnosed & (df.at[person_id, 'age_years'] >= 15):
                        logger.debug("This is HSI_Malaria_rdt scheduling HSI_Malaria_tx_adult for person %d on date %s",
                                     person_id, (self.sim.date + DateOffset(days=1)))

                        treat = HSI_Malaria_tx_adult(self.module, person_id=person_id)
                        self.sim.modules['HealthSystem'].schedule_hsi_event(treat,
                                                                            priority=1,
                                                                            topen=self.sim.date + DateOffset(days=1),
                                                                            tclose=None)

            # log the consumables used
            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self, cons_req_as_footprint=consumables_needed, log=True
            )

    def did_not_run(self):
        logger.debug('HSI_Malaria_rdt: did not run')
        pass


class HSI_Malaria_tx_0_5(HSI_Event, IndividualScopeEventMixin):
    """
    this is anti-malarial treatment for children <15 kg. Includes treatment plus one rdt
    """

    def __init__(self, module, person_id):
        super().__init__(module=Malaria, person_id=person_id)
        assert isinstance(module, Malaria)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Under5OPD'] = 1  # This requires one out patient


        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Malaria_treatment_child0_5'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVELS = [1]
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        logger.debug('HSI_Malaria_tx_0_5: malaria treatment for child %d',
                     person_id)

        df = self.sim.population.props

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(
            consumables.loc[
                consumables['Intervention_Pkg'] == 'Uncomplicated (children, <15 kg)',
                'Intervention_Pkg_Code'])[0]  # this pkg_code includes another rdt

        the_cons_footprint = {
            'Intervention_Package_Code': [{pkg_code1: 1}],
            'Item_Code': []
        }

        # request the treatment
        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=the_cons_footprint, log=False
        )

        if outcome_of_request_for_consumables:
            df.at[person_id, 'ma_tx'] = True
            df.at[person_id, 'ma_date_tx'] = self.sim.date

            self.sim.schedule_event(MalariaCureEvent(self.module, person_id), self.sim.date + DateOffset(weeks=1))

            # log the consumables
            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self, cons_req_as_footprint=the_cons_footprint, log=True
            )

    def did_not_run(self):
        logger.debug('HSI_Malaria_tx_0_5: did not run')
        pass


class HSI_Malaria_tx_5_15(HSI_Event, IndividualScopeEventMixin):
    """
    this is anti-malarial treatment for children >15 kg. Includes treatment plus one rdt
    """

    def __init__(self, module, person_id):
        super().__init__(module=Malaria, person_id=person_id)
        assert isinstance(module, Malaria)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Under5OPD'] = 1  # This requires one out patient

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Malaria_treatment_child5_15'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVELS = [1]
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        logger.debug('HSI_Malaria_tx_5_15: malaria treatment for child %d',
                     person_id)

        df = self.sim.population.props

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(
            consumables.loc[
                consumables['Intervention_Pkg'] == 'Uncomplicated (children, >15 kg)',
                'Intervention_Pkg_Code'])[0]  # this pkg_code includes another rdt

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1],
            'Item_Code': []
        }

        # request the treatment
        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=the_cons_footprint, log=False
        )

        if outcome_of_request_for_consumables:
            df.at[person_id, 'ma_tx'] = True
            df.at[person_id, 'ma_date_tx'] = self.sim.date

            self.sim.schedule_event(MalariaCureEvent(self.module, person_id), self.sim.date + DateOffset(weeks=1))

            # log the consumables
            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self, cons_req_as_footprint=the_cons_footprint, log=True
            )

    def did_not_run(self):
        logger.debug('HSI_Malaria_tx_5_15: did not run')
        pass


class HSI_Malaria_tx_adult(HSI_Event, IndividualScopeEventMixin):
    """
    this is anti-malarial treatment for adults. Includes treatment plus one rdt
    """

    def __init__(self, module, person_id):
        super().__init__(module=Malaria, person_id=person_id)
        assert isinstance(module, Malaria)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Malaria_treatment_adult'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVELS = [1]
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        logger.debug('HSI_Malaria_tx_adult: malaria treatment for person %d',
                     person_id)

        df = self.sim.population.props

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(
            consumables.loc[
                consumables['Intervention_Pkg'] == 'Uncomplicated (adult, >36 kg)',
                'Intervention_Pkg_Code'])[0]  # this pkg_code includes another rdt

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1],
            'Item_Code': []
        }

        # request the treatment
        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=the_cons_footprint, log=False
        )

        if outcome_of_request_for_consumables:
            df.at[person_id, 'ma_tx'] = True
            df.at[person_id, 'ma_date_tx'] = self.sim.date

            self.sim.schedule_event(MalariaCureEvent(self.module, person_id), self.sim.date + DateOffset(weeks=1))

            # log the consumables
            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self, cons_req_as_footprint=the_cons_footprint, log=True
            )

    def did_not_run(self):
        logger.debug('HSI_Malaria_tx_adult: did not run')
        pass


class HSI_Malaria_tx_compl_child(HSI_Event, IndividualScopeEventMixin):
    """
    this is anti-malarial treatment for complicated malaria in children
    """

    def __init__(self, module, person_id):
        super().__init__(module=Malaria, person_id=person_id)
        assert isinstance(module, Malaria)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Under5OPD'] = 1  # This requires one out patient

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Malaria_treatment_complicated_child'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVELS = [3]
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        logger.debug('HSI_Malaria_tx_compl_child: complicated malaria treatment for child %d',
                     person_id)

        df = self.sim.population.props

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(
            consumables.loc[
                consumables['Intervention_Pkg'] == 'Complicated (children, injectable artesunate)',
                'Intervention_Pkg_Code'])[0]

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1],
            'Item_Code': []
        }

        # request the treatment
        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=the_cons_footprint, log=False
        )

        if outcome_of_request_for_consumables:
            df.at[person_id, 'ma_tx'] = True
            df.at[person_id, 'ma_date_tx'] = self.sim.date

            self.sim.schedule_event(MalariaCureEvent(self.module, person_id), self.sim.date + DateOffset(weeks=1))

            # log the consumables
            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self, cons_req_as_footprint=the_cons_footprint, log=True
            )

    def did_not_run(self):
        logger.debug('HSI_Malaria_tx_compl_child: did not run')
        pass


class HSI_Malaria_tx_compl_adult(HSI_Event, IndividualScopeEventMixin):
    """
    this is anti-malarial treatment for complicated malaria in adults
    """

    def __init__(self, module, person_id):
        super().__init__(module=Malaria, person_id=person_id)
        assert isinstance(module, Malaria)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Malaria_treatment_complicated_adult'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVELS = [3]
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        logger.debug('HSI_Malaria_tx_compl_adult: complicated malaria treatment for person %d',
                     person_id)

        df = self.sim.population.props

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(
            consumables.loc[
                consumables['Intervention_Pkg'] == 'Complicated (adults, injectable artesunate)',
                'Intervention_Pkg_Code'])[0]

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1],
            'Item_Code': []
        }

        # request the treatment
        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=the_cons_footprint, log=False
        )

        if outcome_of_request_for_consumables:
            df.at[person_id, 'ma_tx'] = True
            df.at[person_id, 'ma_date_tx'] = self.sim.date

            self.sim.schedule_event(MalariaCureEvent(self.module, person_id), self.sim.date + DateOffset(weeks=1))

            # log the consumables
            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self, cons_req_as_footprint=the_cons_footprint, log=True
            )

    def did_not_run(self):
        logger.debug('HSI_Malaria_tx_compl_adult: did not run')
        pass


# ---------------------------------------------------------------------------------
# Recovery Events
# ---------------------------------------------------------------------------------
class MalariaCureEvent(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        logger.debug("Stopping malaria treatment and curing person %d", person_id)

        df = self.sim.population.props

        # stop treatment
        if df.at[person_id, 'is_alive']:
            df.at[person_id, 'ma_tx'] = False

            df.at[person_id, 'ma_is_infected'] = False


class MalariaParasiteClearanceEvent(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        logger.debug("This is MalariaParasiteClearanceEvent for person %d", person_id)

        df = self.sim.population.props

        if df.at[person_id, 'is_alive']:
            df.at[person_id, 'ma_is_infected'] = False


class MalariaSympEndEvent(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        logger.debug("This is MalariaSympEndEvent ending symptoms of clinical malaria for person %d", person_id)

        df = self.sim.population.props

        if df.at[person_id, 'is_alive']:
            df.at[person_id, 'mi_specific_symptoms'] = 'none'


# ---------------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------------

class MalariaLoggingEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        self.repeat = 1
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props
        now = self.sim.date

        # ------------------------------------ INCIDENCE ------------------------------------

        # infected in the last year, clinical and severe cases only
        # incidence rate per 1000 person-years
        tmp = len(
            df.loc[df.is_alive & (df.ma_date_infected > (now - DateOffset(months=self.repeat)))])
        pop = len(df[df.is_alive])

        inc_1000py = (tmp / pop) * 1000

        # incidence rate in 2-10 yr olds
        tmp2 = len(
            df.loc[df.is_alive & (df.age_years.between(2, 10)) & (
                df.ma_date_infected > (now - DateOffset(months=self.repeat)))])
        pop2_10 = len(df[df.is_alive & (df.age_years.between(2, 10))])
        inc_1000py_2_10 = (tmp2 / pop2_10) * 1000

        inc_1000py_hiv = 0  # if running without hiv/tb

        logger.info('%s|incidence|%s', now,
                    {
                        'number_new_cases': tmp,
                        'population': pop,
                        'inc_1000py': inc_1000py,
                        'inc_1000py_hiv': inc_1000py_hiv,
                        'new_cases_2_10': tmp2,
                        'population2_10': pop2_10,
                        'inc_1000py_2_10': inc_1000py_2_10
                    })

        # ------------------------------------ RUNNING COUNTS ------------------------------------

        counts = {'None': 0, 'clinical': 0, 'severe': 0}
        counts.update(df.loc[df.is_alive, 'ma_specific_symptoms'].value_counts().to_dict())

        logger.info('%s|status_counts|%s', now, counts)

        # ------------------------------------ PREVALENCE BY AGE ------------------------------------

        # clinical cases including severe
        # assume asymptomatic parasitaemia is too low to detect in surveys
        child2_10_clin = len(df[df.is_alive & (df.ma_specific_symptoms == 'clinical') & (df.age_years.between(2, 10))])

        # population size - children
        child2_10_pop = len(df[df.is_alive & (df.age_years.between(2, 10))])

        # prevalence in children aged 2-10
        child_prev = child2_10_clin / child2_10_pop if child2_10_pop else 0

        # prevalence of clinical including severe in all ages
        total_clin = len(df[df.is_alive & (df.ma_specific_symptoms == 'clinical')])
        pop2 = len(df[df.is_alive])
        prev_clin = total_clin / pop2

        logger.info('%s|prevalence|%s', now,
                    {
                        'child2_10_prev': child_prev,
                        'clinical_prev': prev_clin,
                    })

        # ------------------------------------ TREATMENT COVERAGE ------------------------------------
        # prop on treatment, all ages

        # people who had treatment start date within last year
        tx = len(df[df.is_alive & df.ma_tx])

        # people who had malaria infection date within last year
        # includes asymp
        denom = len(df[df.is_alive & ((df.ma_specific_symptoms == 'clinical') | (df.ma_specific_symptoms == 'severe'))])
        tx_coverage = tx / denom if denom else 0

        logger.info('%s|tx_coverage|%s', now,
                    {
                        'treatment_coverage': tx_coverage,
                    })

        # ------------------------------------ MORTALITY ------------------------------------
        # deaths reported in the last 12 months / pop size
        deaths = len(df[(df.ma_date_death > (now - DateOffset(months=self.repeat)))])

        mort_rate = deaths / len(df[df.is_alive])

        logger.info('%s|ma_mortality|%s', now,
                    {
                        'mort_rate': mort_rate,
                    })
