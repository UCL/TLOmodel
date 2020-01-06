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

    def __init__(self, name=None, resourcefilepath=None, level=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.level = level  # set to national or district-level malaria infection events

        logger.info(f'Malaria infection event running at level {self.level}')

    PARAMETERS = {
        'mal_inc': Parameter(Types.REAL, 'monthly incidence of malaria in all ages'),
        'interv': Parameter(Types.REAL, 'data frame of intervention coverage by year'),
        'clin_inc': Parameter(Types.REAL, 'data frame of clinical incidence by age, district, intervention coverage'),
        'inf_inc': Parameter(Types.REAL, 'data frame of infection incidence by age, district, intervention coverage'),
        'sev_inc': Parameter(Types.REAL, 'data frame of severe case incidence by age, district, intervention coverage'),
        'itn_district': Parameter(Types.REAL, 'data frame of ITN usage rates by district'),

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
        'treatment_adjustment': Parameter(
            Types.REAL, 'probability of death from severe malaria if on treatment'
        )
    }

    PROPERTIES = {
        'ma_is_infected': Property(
            Types.BOOL, 'Current status of malaria'),
        'ma_date_infected': Property(
            Types.DATE, 'Date of latest infection'),
        'ma_date_clinical': Property(
            Types.DATE, 'Date of latest clinical infection'),
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
            Types.REAL, 'age values redefined to match with malaria data'),
        'ma_clinical_counter': Property(
            Types.INT, 'annual counter for malaria clinical episodes'),
        'ma_tx_counter': Property(
            Types.INT, 'annual counter for malaria treatment episodes')
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
        p['itn_district'] = workbook['MAP_ITNrates']

        # get the DALY weight that this module will use from the weight database (these codes are just random!)
        if 'HealthBurden' in self.sim.modules.keys():
            p['daly_wt_none'] = self.sim.modules['HealthBurden'].get_daly_weight(50)
            p['daly_wt_clinical'] = self.sim.modules['HealthBurden'].get_daly_weight(50)
            p['daly_wt_severe'] = self.sim.modules['HealthBurden'].get_daly_weight(589)

    def initialise_population(self, population):
        df = population.props
        p = self.parameters
        now = self.sim.date
        rng = self.rng

        # ----------------------------------- REGISTER WITH HEALTH SYSTEM -----------------------------------
        # need to register before any health system, stuff / symptom manager happens
        self.sim.modules['HealthSystem'].register_disease_module(self)

        # ----------------------------------- INITIALISE THE POPULATION-----------------------------------
        # Set default for properties
        df['ma_is_infected'] = False
        df['ma_date_infected'] = pd.NaT
        df['ma_date_clinical'] = pd.NaT
        df['ma_date_death'] = pd.NaT
        df['ma_tx'] = False
        df['ma_date_tx'] = pd.NaT
        df['ma_specific_symptoms'].values[:] = 'none'
        df['ma_unified_symptom_code'].values[:] = 0
        df['ma_district_edited'] = df['district_of_residence']
        df['ma_age_edited'] = 0

        df['ma_clinical_counter'] = 0
        df['ma_tx_counter'] = 0

        if self.level == 0:
            # ----------------------------------- INCIDENCE - NATIONAL -----------------------------------
            # select incidence based on year
            curr_year = now.year

            # these values are just clinical incidence
            inf_inc = p['mal_inc']
            inf_inc_month = inf_inc.loc[(inf_inc.year == curr_year), 'monthly_inc_rate']

            # ----------------------------------- NEW CLINICAL INFECTIONS -----------------------------------
            # new clinical infections
            uninf = df.index[(df.ma_specific_symptoms == 'none') & df.is_alive]
            now_infected = rng.choice([True, False],
                                      size=len(uninf),
                                      p=[inf_inc_month.values[0], 1 - inf_inc_month.values[0]])

            # if any are infected
            if now_infected.sum():

                infected_idx = uninf[now_infected]
                len_new = len(infected_idx)
                logger.debug(f'Malaria Module: assigning {len_new} clinical malaria infections')

                df.loc[infected_idx, 'ma_is_infected'] = True
                df.loc[infected_idx, 'ma_date_infected'] = now  # TODO: scatter dates across month
                df.loc[infected_idx, 'ma_specific_symptoms'] = 'clinical'
                df.loc[infected_idx, 'ma_unified_symptom_code'] = 1
                df.loc[infected_idx, 'ma_date_clinical'] = now
                df.loc[infected_idx, 'ma_clinical_counter'] += 1  # counter only for new clinical cases (inc severe)

                ## severe - subset of newly clinical
                prob_sev = 0.1  # tmp value for prob of clinical case becoming severe

                new_inf = df.index[
                    df.is_alive & df.ma_is_infected & (df.ma_date_infected == now) & (
                        df.ma_specific_symptoms == 'clinical')]
                now_severe = rng.choice([True, False],
                                        size=len(new_inf),
                                        p=[prob_sev, 1 - prob_sev])

                if now_severe.sum():
                    severe_idx = new_inf[now_severe]
                    len_sev = len(severe_idx)
                    logger.debug(f'Malaria Module: assigning {len_sev} severe malaria infections')

                    df.loc[severe_idx, 'ma_specific_symptoms'] = 'severe'
                    df.loc[severe_idx, 'ma_unified_symptom_code'] = 2

        else:
            # ----------------------------------- INCIDENCE - DISTRICT -----------------------------------

            # ----------------------------------- RENAME DISTRICTS -----------------------------------
            # rename districts to match malaria data
            df.loc[(df.district_of_residence == 'Lilongwe City'), 'ma_district_edited'] = 'Lilongwe'
            df.loc[(df.district_of_residence == 'Blantyre City'), 'ma_district_edited'] = 'Blantyre'
            df.loc[(df.district_of_residence == 'Zomba City'), 'ma_district_edited'] = 'Zomba'
            df.loc[(df.district_of_residence == 'Mzuzu City'), 'ma_district_edited'] = 'Mzimba'
            df.loc[(df.district_of_residence == 'Nkhata Bay'), 'ma_district_edited'] = 'Mzimba'

            assert (not pd.isnull(df['ma_district_edited']).any())

            # ----------------------------------- INTERVENTION COVERAGE -----------------------------------

            # get ITN usage rates for current year by district
            itn = p['itn_district']

            interv = p['interv']

            # find annual intervention coverage levels rate for 2010
            irs_2010 = interv.loc[interv.Year == now.year, 'IRS_coverage'].values[0]
            irs_2010 = 0.8 if irs_2010 > 0.5 else 0  # round to 0 or 0.8

            # ITN select current year for each district
            itn_curr = itn.loc[itn.Year == now.year]
            itn_curr.itn_rates = round(itn_curr.itn_rates, 1)  # round to nearest 0.1 to match Pete's data
            itn_curr.distr_rate = itn_curr["District"].map(str) + itn_curr["itn_rates"]
            print(itn_curr.distr_rate)

            # ----------------------------------- INCIDENCE ESTIMATES -----------------------------------
            # inf_inc select current month and irs
            month = now.month

            inf_inc = p['inf_inc']  # datasheet with infection incidence
            inf_inc.distr_llin = inf_inc["admin"].map(str) + inf_inc["llin"]
            print(inf_inc.distr_llin)

            inf_inc_month = inf_inc.loc[(inf_inc.month == month) & (inf_inc.irs == irs_2010)]

            # need to select based on itn.itn_rates also
            inf_inc_month = inf_inc_month.loc[inf_inc_month.distr_llin.isin(itn_curr.distr_rate)]

            # there should now be only one llin level per district, and lots of ages





            # ----------------------------------- INCIDENCE ESTIMATES -----------------------------------
            month = now.month

            inf_inc = p['inf_inc']
            inf_inc_month = inf_inc.loc[
                (inf_inc.month == month) & (inf_inc.llin == itn_2010) & (inf_inc.irs == irs_2010)]
            clin_inc = p['clin_inc']
            clin_inc_month = clin_inc.loc[
                (clin_inc.month == month) & (clin_inc.llin == itn_2010) & (clin_inc.irs == irs_2010)]
            sev_inc = p['sev_inc']
            sev_inc_month = sev_inc.loc[
                (sev_inc.month == month) & (sev_inc.llin == itn_2010) & (sev_inc.irs == irs_2010)]
            ######################################################################################################
            # for each district and age, look up incidence estimate using itn_2010 and irs_2010
            # create new age column with 0, 0.5, 1, 2, ...
            df.loc[df.age_exact_years.between(0, 0.5), 'ma_age_edited'] = 0
            df.loc[df.age_exact_years.between(0.5, 1), 'ma_age_edited'] = 0.5
            df.loc[(df.age_exact_years >= 1), 'ma_age_edited'] = df.age_years[df.age_years >= 1]
            assert (not pd.isnull(df['ma_age_edited']).any())
            df['ma_age_edited'] = df['ma_age_edited'].astype('float')  # for merge with malaria data

            # merge the incidence into the main df and replace each event call
            df_ml = df.reset_index().merge(inf_inc_month, left_on=['ma_district_edited', 'ma_age_edited'],
                                           right_on=['admin', 'age'],
                                           how='left', indicator=True).set_index('person')
            df_ml['monthly_prob_inf'] = df_ml['monthly_prob_inf'].fillna(0)  # 0 if over 80 yrs
            assert (not pd.isnull(df_ml['monthly_prob_inf']).any())

            df_ml = df_ml.reset_index().merge(clin_inc_month, left_on=['ma_district_edited', 'ma_age_edited'],
                                              right_on=['admin', 'age'],
                                              how='left').set_index('person')
            df_ml['monthly_prob_clin'] = df_ml['monthly_prob_clin'].fillna(0)  # 0 if over 80 yrs
            assert (not pd.isnull(df_ml['monthly_prob_clin']).any())

            df_ml = df_ml.reset_index().merge(sev_inc_month, left_on=['ma_district_edited', 'ma_age_edited'],
                                              right_on=['admin', 'age'],
                                              how='left').set_index('person')
            df_ml['monthly_prob_sev'] = df_ml['monthly_prob_sev'].fillna(0)  # 0 if over 80 yrs
            assert (not pd.isnull(df_ml['monthly_prob_sev']).any())

            # ----------------------------------- NEW INFECTIONS -----------------------------------
            ## infected
            # risk_ml = pd.Series(0, index=df.index)
            # risk_ml.loc[df.is_alive] = 1  # applied to everyone
            # risk_ml.loc[df.hv_inf ] *= p['rr_hiv']  # then have to scale within every subgroup

            # # new infections
            # # uninf = df_ml.index[(df_ml.ma_specific_symptoms == 'none') & df_ml.is_alive]
            # # now_infected = rng.choice([True, False],
            # #                           size=len(uninf),
            # #                           p=[df_ml.monthly_prob_inf[uninf], 1 - df_ml.monthly_prob_inf[uninf]])
            #
            # # randomly sample from completely susceptible population
            # newly_infected_index = df_ml.index[(rng.random_sample(size=len(df_ml)) < df_ml.monthly_prob_inf)]
            #
            # # if any are infected
            # if newly_infected_index.any():
            #     # infected_idx = uninf[now_infected]
            #     len_new = len(newly_infected_index)
            #     logger.debug(f'MalariaEventDistrict: assigning {len_new} malaria infections')
            #
            #     df_ml.loc[newly_infected_index, 'ma_is_infected'] = True
            #     df_ml.loc[newly_infected_index, 'ma_date_infected'] = now  # TODO: scatter dates across month
            #
            #     new_clinical = df_ml.index[
            #         df_ml.is_alive & df_ml.ma_is_infected & (df_ml.ma_date_infected == now) &
            #         (rng.random_sample(size=len(df_ml)) < df_ml.monthly_prob_clin)]
            #
            #     # now_clinical = rng.choice([True, False],
            #     #                           size=len(new_inf),
            #     #                           p=[df_ml.monthly_prob_inf[new_inf],
            #     #                              1 - df_ml.monthly_prob_inf[new_inf]])
            #
            #     # newly_clinical = rng.random_sample(size=len(new_inf) < df_ml.monthly_prob_clin[new_inf])
            #
            #     # assign new clinical infections
            #     if new_clinical.any():
            #         # clinical_idx = new_inf[now_clinical]
            #         len_new = len(new_clinical)
            #         logger.debug(f'MalariaEventDistrict: assigning {len_new} clinical malaria infections')
            #
            #         df_ml.loc[new_clinical, 'ma_specific_symptoms'] = 'clinical'
            #
            #         # update main dataframe
            #         df.loc[new_clinical, 'ma_specific_symptoms'] = 'clinical'
            #         df.loc[new_clinical, 'ma_unified_symptom_code'] = 1
            #         df.loc[new_clinical, 'ma_date_clinical'] = now
            #         df.loc[new_clinical, 'ma_clinical_counter'] += 1  # counter only for new clinical cases (inc severe)
            #
            #         # assign new severe infections
            #         new_severe = df_ml.index[
            #             df_ml.is_alive & df_ml.ma_is_infected & (df_ml.ma_date_infected == now) &
            #             (df_ml.ma_specific_symptoms == 'clinical') &
            #             (rng.random_sample(size=len(df_ml)) < df_ml.monthly_prob_sev)]
            #
            #         # now_severe = rng.choice([True, False],
            #         #                         size=len(new_clin),
            #         #                         p=[df_ml.monthly_prob_sev[new_clin],
            #         #                            1 - df_ml.monthly_prob_sev[new_clin]])
            #
            #         if new_severe.any():
            #             # severe_idx = new_clin[now_severe]
            #             len_new = len(new_severe)
            #             logger.debug(f'MalariaEventDistrict: assigning {len_new} severe malaria infections')
            #
            #             # update main dataframe
            #             df.loc[new_severe, 'ma_specific_symptoms'] = 'severe'
            #             df.loc[new_severe, 'ma_unified_symptom_code'] = 2
            ## infected
            risk_ml = pd.Series(0, index=df.index)
            risk_ml.loc[df.is_alive] = 1  # applied to everyone
            # risk_ml.loc[df.hv_inf ] *= p['rr_hiv']  # then have to scale within every subgroup

            # update new df_ml dataframe using appended monthly probabilities, then update main df
            # new infections
            random_draw = rng.random_sample(size=len(df_ml))
            ml_idx = df_ml[df_ml.is_alive & (df_ml.ma_specific_symptoms == 'none') &
                           (random_draw < df_ml.monthly_prob_inf)].index
            df_ml.loc[ml_idx, 'ma_is_infected'] = True
            df_ml.loc[ml_idx, 'ma_date_infected'] = now  # TODO: scatter dates across month
            # print('ml_idx', ml_idx)

            ## clinical - subset of newly infected
            random_draw = rng.random_sample(size=len(df))
            clin_idx = df_ml[
                df_ml.is_alive & df_ml.ma_is_infected & (df_ml.ma_date_infected == now) & (
                    random_draw < df_ml.monthly_prob_clin)].index
            df_ml.loc[clin_idx, 'ma_specific_symptoms'] = 'clinical'
            # print('clin_idx', clin_idx)

            ## severe - subset of newly clinical
            random_draw = rng.random_sample(size=len(df))
            sev_idx = df_ml[
                df_ml.is_alive & df_ml.ma_is_infected & (df_ml.ma_date_infected == now) & (
                    df_ml.ma_specific_symptoms == 'clinical') & (
                    random_draw < df_ml.monthly_prob_sev)].index
            # print('sev_idx', sev_idx)

            # update the main dataframe
            df.loc[ml_idx, 'ma_date_infected'] = now
            df.loc[ml_idx, 'ma_is_infected'] = True
            df.loc[ml_idx, 'ma_specific_symptoms'] = 'none'

            df.loc[clin_idx, 'ma_specific_symptoms'] = 'clinical'
            df.loc[clin_idx, 'ma_unified_symptom_code'] = 1
            df.loc[clin_idx, 'ma_date_clinical'] = now
            df.loc[clin_idx, 'ma_clinical_counter'] += 1  # counter only for new clinical cases (inc severe)

            df.loc[sev_idx, 'ma_specific_symptoms'] = 'severe'
            df.loc[sev_idx, 'ma_unified_symptom_code'] = 2

            ## tidy up
            del df_ml

        # ----------------------------------- PARASITE CLEARANCE - NO TREATMENT -----------------------------------
        # schedule self-cure if no treatment, no self-cure from severe malaria

        # asymptomatic
        asym = df.index[(df.ma_specific_symptoms == 'none') & (df.ma_date_infected == now)]

        for person in df.loc[asym].index:
            # logger.debug(
            #     'Malaria Event: scheduling parasite clearance for asymptomatic person %d', person)

            random_date = rng.randint(low=0, high=p['dur_asym'])
            random_days = pd.to_timedelta(random_date, unit='d')

            cure = MalariaParasiteClearanceEvent(self, person)
            self.sim.schedule_event(cure, (self.sim.date + random_days))

        # clinical
        clin = df.index[(df.ma_specific_symptoms == 'clinical') & (df.ma_date_infected == now)]

        for person in df.loc[clin].index:
            # logger.debug(
            #     'Malaria Event: scheduling parasite clearance and symptom end for symptomatic person %d',
            #     person)

            date_para = rng.randint(low=0, high=p['dur_clin_para'])
            date_para_days = pd.to_timedelta(date_para, unit='d')
            # print('date_para_days', date_para_days)

            # # TODO: if clinical symptoms too short maybe no healthcare seeking scheduled
            # projected date moving from clinical back to asymptomatic
            date_clin = rng.randint(low=p['dur_clin'] - 2, high=p['dur_clin'] + 2)
            date_clin_days = pd.to_timedelta(date_clin, unit='d')
            print('date_clin_days', date_clin_days)

            cure = MalariaParasiteClearanceEvent(self, person)
            self.sim.schedule_event(cure, (self.sim.date + date_para_days))

            # schedule symptom end (5 days +/- 2 days)
            symp_end = MalariaClinEndEvent(self, person)
            self.sim.schedule_event(symp_end, self.sim.date + date_clin_days)

        # ----------------------------------- HEALTHCARE-SEEKING -----------------------------------
        # clinical cases will seek care with some probability
        # severe cases will definitely seek care
        interv = p['interv']

        # find annual intervention coverage levels rate for 2010
        act = interv.loc[interv.Year == now.year, 'ACT_coverage'].values[0]
        # act = 1

        # CLINICAL CASES
        if len(clin) > 0:
            self.sim.modules['SymptomManager'].chg_symptom(
                person_id=list(clin),
                symptom_string='fever',
                add_or_remove='+',
                disease_module=self,
                duration_in_days=p['dur_clin'])

            self.sim.modules['SymptomManager'].chg_symptom(
                person_id=list(clin),
                symptom_string='headache',
                add_or_remove='+',
                disease_module=self,
                duration_in_days=p['dur_clin'])

            self.sim.modules['SymptomManager'].chg_symptom(
                person_id=list(clin),
                symptom_string='vomiting',
                add_or_remove='+',
                disease_module=self,
                duration_in_days=p['dur_clin'])

            self.sim.modules['SymptomManager'].chg_symptom(
                person_id=list(clin),
                symptom_string='stomachache',
                add_or_remove='+',
                disease_module=self,
                duration_in_days=p['dur_clin'])

        # seeks_care = pd.Series(data=False, index=df.loc[clin].index)
        #
        # for i in df.loc[clin].index:
        #     # prob = self.sim.modules['HealthSystem'].get_prob_seek_care(i, symptom_code=4)
        #     seeks_care[i] = rng.rand() < act  # placeholder for coverage / testing rates
        #
        # if seeks_care.sum() > 0:
        #
        #     for person_index in seeks_care.index[seeks_care]:
        #         # print(person_index)
        #
        #         logger.debug(
        #             'MalariaEvent: scheduling HSI_Malaria_rdt for clinical malaria case %d',
        #             person_index)
        #
        #         delay = rng.choice(7)
        #         # print(delay)
        #
        #         event = HSI_Malaria_rdt(self, person_id=person_index)
        #         self.sim.modules['HealthSystem'].schedule_hsi_event(event,
        #                                                             priority=2,
        #                                                             topen=self.sim.date + DateOffset(
        #                                                                 days=delay),
        #                                                             tclose=self.sim.date + DateOffset(
        #                                                                 days=(delay + 14))
        #                                                             )
        # else:
        #     logger.debug(
        #         'MalariaEvent: There is no new healthcare seeking for clinical cases')

        # SEVERE CASES
        # todo check only severe selected here
        severe = df.index[(df.ma_specific_symptoms == 'severe') & (df.ma_date_infected == now)]

        for person_index in severe:
            # print(person_index)

            # adult severe malaria case
            if df.at[person_index, 'age_exact_years'] >= 5:
                logger.debug(
                    'MalariaEventNational: scheduling HSI_Malaria_tx_compl_adult for severe malaria case %d',
                    person_index)

                delay = rng.choice(2)  # random delay 0-2 days
                # print(delay)

                # severe malaria intervention happen with priority=0 (highest)
                event = HSI_Malaria_tx_compl_adult(self, person_id=person_index)
                self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                    priority=0,
                                                                    topen=self.sim.date + DateOffset(
                                                                        days=delay),
                                                                    tclose=self.sim.date + DateOffset(
                                                                        days=(delay + 14))
                                                                    )
            if df.at[person_index, 'age_exact_years'] < 5:
                logger.debug(
                    'MalariaEventNational: scheduling HSI_Malaria_tx_compl_child for severe malaria case %d',
                    person_index)

                delay = rng.choice(2)  # random delay 0-2 days
                # print(delay)

                event = HSI_Malaria_tx_compl_child(self, person_id=person_index)
                self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                    priority=0,
                                                                    topen=self.sim.date + DateOffset(
                                                                        days=delay),
                                                                    tclose=self.sim.date + DateOffset(
                                                                        days=(delay + 14))
                                                                    )

        # ----------------------------------- SCHEDULED DEATHS -----------------------------------
        # schedule deaths within the next week
        # Assign time of infections across the month
        random_draw = rng.random_sample(size=len(df))

        # the cfr applies to all severe malaria
        death = df.index[
            (df.ma_specific_symptoms == 'severe') & (df.ma_date_infected == now) & (random_draw < p['cfr'])]

        for person in death:
            logger.debug(
                'MalariaEvent: scheduling malaria death for person %d',
                person)

            random_date = rng.randint(low=0, high=7)
            random_days = pd.to_timedelta(random_date, unit='d')

            death_event = MalariaDeathEvent(self, individual_id=person,
                                            cause='malaria')  # make that death event
            self.sim.schedule_event(death_event, self.sim.date + random_days)  # schedule the death

    def initialise_simulation(self, sim):

        if self.level == 0:
            sim.schedule_event(MalariaEventNational(self), sim.date + DateOffset(months=1))
        else:
            sim.schedule_event(MalariaEventDistrict(self), sim.date + DateOffset(months=1))

        sim.schedule_event(MalariaResetCounterEvent(self), sim.date + DateOffset(days=365))  # 01 jan each year

        # add an event to log to screen - 31st Dec each year
        sim.schedule_event(MalariaLoggingEvent(self), sim.date + DateOffset(days=364))
        sim.schedule_event(MalariaTxLoggingEvent(self), sim.date + DateOffset(days=364))
        sim.schedule_event(MalariaPrevDistrictLoggingEvent(self), sim.date + DateOffset(months=1))

    def on_birth(self, mother_id, child_id):

        df = self.sim.population.props

        df.at[child_id, 'ma_is_infected'] = False
        df.at[child_id, 'ma_date_infected'] = pd.NaT
        df.at[child_id, 'ma_date_clinical'] = pd.NaT
        df.at[child_id, 'ma_date_death'] = pd.NaT
        df.at[child_id, 'ma_tx'] = False
        df.at[child_id, 'ma_date_tx'] = pd.NaT
        df.at[child_id, 'ma_specific_symptoms'] = 'none'
        df.at[child_id, 'ma_unified_symptom_code'] = 0
        df.at[child_id, 'ma_district_edited'] = df.at[child_id, 'district_of_residence']
        df.at[child_id, 'ma_age_edited'] = 0
        df.at[child_id, 'ma_clinical_counter'] = 0
        df.at[child_id, 'ma_tx_counter'] = 0

        # ----------------------------------- RENAME DISTRICTS -----------------------------------
        # rename districts to match malaria data
        if df.at[child_id, 'ma_district_edited'] == 'Lilongwe City':
            df.at[child_id, 'ma_district_edited'] = 'Lilongwe'

        elif df.at[child_id, 'ma_district_edited'] == 'Blantyre City':
            df.at[child_id, 'ma_district_edited'] = 'Blantyre'

        elif df.at[child_id, 'ma_district_edited'] == 'Zomba City':
            df.at[child_id, 'ma_district_edited'] = 'Zomba'

        elif df.at[child_id, 'ma_district_edited'] == 'Mzuzu City':
            df.at[child_id, 'ma_district_edited'] = 'Mzuzu'

        elif df.at[child_id, 'ma_district_edited'] == 'Nkhata Bay':
            df.at[child_id, 'ma_district_edited'] = 'Mzimba'

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


class MalariaEventNational(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):

        logger.debug('MalariaEventNational: tracking the disease progression of the population.')

        df = population.props
        p = self.module.parameters
        rng = self.module.rng
        now = self.sim.date

        # ----------------------------------- INCIDENCE - NATIONAL -----------------------------------
        # select incidence based on year
        curr_year = now.year

        # these values are just clinical incidence
        inf_inc = p['mal_inc']
        inf_inc_month = inf_inc.loc[(inf_inc.year == curr_year), 'monthly_inc_rate']
        # print('inf_inc_month', inf_inc_month.values[0])
        # print(self.sim.date)

        # ----------------------------------- NEW CLINICAL INFECTIONS -----------------------------------
        # new clinical infections: susceptible = uninfected and asym
        uninf = df.index[(df.ma_specific_symptoms == 'none') & df.is_alive]
        now_infected = rng.choice([True, False],
                                  size=len(uninf),
                                  p=[inf_inc_month.values[0], 1 - inf_inc_month.values[0]])
        # print('now_infected', now_infected.sum())

        # if any are infected
        if now_infected.sum():

            infected_idx = uninf[now_infected]
            len_new = len(infected_idx)
            logger.debug(f'MalariaEventNational: assigning {len_new} clinical malaria infections')

            df.loc[infected_idx, 'ma_is_infected'] = True
            df.loc[infected_idx, 'ma_date_infected'] = now  # TODO: scatter dates across month
            df.loc[infected_idx, 'ma_specific_symptoms'] = 'clinical'
            df.loc[infected_idx, 'ma_unified_symptom_code'] = 1
            df.loc[infected_idx, 'ma_date_clinical'] = now
            df.loc[infected_idx, 'ma_clinical_counter'] += 1  # counter only for new clinical cases (inc severe)

            ## severe - subset of newly clinical
            prob_sev = 0.05  # tmp value for prob of clinical case becoming severe

            new_inf = df.index[
                df.is_alive & df.ma_is_infected & (df.ma_date_infected == now) & (
                    df.ma_specific_symptoms == 'clinical')]
            now_severe = rng.choice([True, False],
                                    size=len(new_inf),
                                    p=[prob_sev, 1 - prob_sev])

            if now_severe.sum():
                severe_idx = new_inf[now_severe]
                len_sev = len(severe_idx)
                logger.debug(f'MalariaEventNational: assigning {len_sev} severe malaria infections')

                df.loc[severe_idx, 'ma_specific_symptoms'] = 'severe'
                df.loc[severe_idx, 'ma_unified_symptom_code'] = 2

            # ----------------------------------- PARASITE CLEARANCE - NO TREATMENT -----------------------------------
            # schedule self-cure if no treatment, no self-cure from severe malaria

            # asymptomatic
            asym = df.index[(df.ma_specific_symptoms == 'none') & (df.ma_date_infected == now)]

            for person in df.loc[asym].index:
                # logger.debug(
                #     'Malaria Event: scheduling parasite clearance for asymptomatic person %d', person)

                random_date = rng.randint(low=0, high=p['dur_asym'])
                random_days = pd.to_timedelta(random_date, unit='d')

                cure = MalariaParasiteClearanceEvent(self.module, person)
                self.sim.schedule_event(cure, (self.sim.date + random_days))

            # clinical
            clin = df.index[(df.ma_specific_symptoms == 'clinical') & (df.ma_date_infected == now)]

            for person in df.loc[clin].index:
                # logger.debug(
                #     'Malaria Event: scheduling parasite clearance and symptom end for symptomatic person %d',
                #     person)

                date_para = rng.randint(low=0, high=p['dur_clin_para'])
                date_para_days = pd.to_timedelta(date_para, unit='d')
                # print('date_para_days', date_para_days)

                date_clin = rng.randint(low=p['dur_clin'] - 2, high=p['dur_clin'] + 2)
                date_clin_days = pd.to_timedelta(date_clin, unit='d')
                # print('date_clin_days', date_clin_days)

                cure = MalariaParasiteClearanceEvent(self.module, person)
                self.sim.schedule_event(cure, (self.sim.date + date_para_days))

                # schedule move back to asymptomatic (5 days +/- 2 days)
                symp_end = MalariaClinEndEvent(self.module, person)
                self.sim.schedule_event(symp_end, self.sim.date + date_clin_days)

            # ----------------------------------- HEALTHCARE-SEEKING -----------------------------------
            # clinical cases will seek care with some probability
            # severe cases will definitely seek care

            # Give everyone with clinical malaria generic symptom fever (some will go to care for this)
            # Report this to the unified symptom manager:
            # the symptom manager just gives the person the symptom
            if len(clin) > 0:
                # TODO: if clinical symptoms too short maybe no healthcare seeking scheduled in time
                # TODO: symptoms may occur with different probabilities, lasting for diff durations

                # this also schedules symptom resolution in 5 days
                self.sim.modules['SymptomManager'].chg_symptom(
                    person_id=list(clin),
                    symptom_string='fever',
                    add_or_remove='+',
                    disease_module=self.module,
                    duration_in_days=p['dur_clin'])

                self.sim.modules['SymptomManager'].chg_symptom(
                    person_id=list(clin),
                    symptom_string='headache',
                    add_or_remove='+',
                    disease_module=self.module,
                    duration_in_days=p['dur_clin'])

                self.sim.modules['SymptomManager'].chg_symptom(
                    person_id=list(clin),
                    symptom_string='vomiting',
                    add_or_remove='+',
                    disease_module=self.module,
                    duration_in_days=p['dur_clin'])

                self.sim.modules['SymptomManager'].chg_symptom(
                    person_id=list(clin),
                    symptom_string='stomachache',
                    add_or_remove='+',
                    disease_module=self.module,
                    duration_in_days=p['dur_clin'])


                # TODO:'duration_in_days' schedules symptom resolution and treatment also schedules resolution!
                # TODO: check symptoms still present if treated before scheduling symptom resolution

            # the HealthSeekingBehaviourPoll will check for generic symptoms daily and schedule a generic appt
            # in generic appt, code 'if fever: schedule HSI_malaria_rdt
            # if severe malaria, schedule my own HSI_severe_tx combined with rdt

            # Q. how to calibrate tx coverage levels, where is the gating step?

            # interv = p['interv']
            #
            # # find annual intervention coverage levels rate for 2010
            # act = interv.loc[interv.Year == now.year, 'ACT_coverage'].values[0]
            # # act = 1
            # act = act * 3  # correction for HSI capacity restrictions -> actual act coverage
            #
            # # todo: MIS 2017 54% children <5 with fever sought care
            # # todo: 30.5% sought care same/next day
            # # CLINICAL CASES
            # seeks_care = pd.Series(data=False, index=df.loc[clin].index)
            #
            # # todo check no-one with symptoms=none is in clin index
            # for i in df.loc[clin].index:
            #     # prob = self.sim.modules['HealthSystem'].get_prob_seek_care(i, symptom_code=4)
            #     seeks_care[i] = rng.rand() < act  # placeholder for coverage / testing rates
            #
            # # print('seeks_care', seeks_care.sum())
            # if seeks_care.sum() > 0:
            #
            #     for person_index in seeks_care.index[seeks_care]:
            #         # print(person_index)
            #
            #         logger.debug(
            #             'MalariaEvent: scheduling HSI_Malaria_rdt for clinical malaria case %d',
            #             person_index)
            #
            #         delay = rng.choice(7)
            #         # print(delay)
            #
            #         event = HSI_Malaria_rdt(self.module, person_id=person_index)
            #         self.sim.modules['HealthSystem'].schedule_hsi_event(event,
            #                                                             priority=2,
            #                                                             topen=self.sim.date + DateOffset(days=delay),
            #                                                             tclose=self.sim.date + DateOffset(
            #                                                                 days=(delay + 14))
            #                                                             )
            # else:
            #     logger.debug(
            #         'MalariaEventNational: There is no new healthcare seeking for clinical cases')
            #
            # SEVERE CASES
            # todo check only severe cases are passing through here
            # all severe cases WILL SEEK CARE - but some may not receive any / be too late
            severe = df.index[(df.ma_specific_symptoms == 'severe') & (df.ma_date_infected == now)]

            for person_index in severe:
                # print(person_index)

                # adult severe malaria case
                if (df.at[person_index, 'age_exact_years'] >= 5):
                    print('test')

                    logger.debug(
                        'MalariaEventNational: scheduling HSI_Malaria_tx_compl_adult for severe malaria case %d',
                        person_index)

                    delay = rng.choice(2)  # random delay 0-2 days
                    # print(delay)

                    # severe malaria intervention happen with priority=0 (highest)
                    event = HSI_Malaria_tx_compl_adult(self.module, person_id=person_index)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                        priority=0,
                                                                        topen=self.sim.date + DateOffset(days=delay),
                                                                        tclose=self.sim.date + DateOffset(
                                                                            days=(delay + 14))
                                                                        )
                if (df.at[person_index, 'age_exact_years'] < 5):
                    logger.debug(
                        'MalariaEventNational: scheduling HSI_Malaria_tx_compl_child for severe malaria case %d',
                        person_index)

                    delay = rng.choice(2)  # random delay 0-2 days
                    # print(delay)

                    event = HSI_Malaria_tx_compl_child(self.module, person_id=person_index)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                        priority=0,
                                                                        topen=self.sim.date + DateOffset(days=delay),
                                                                        tclose=self.sim.date + DateOffset(
                                                                            days=(delay + 14))
                                                                        )

            # ----------------------------------- SCHEDULED DEATHS -----------------------------------
            # schedule deaths within the next week
            # Assign time of infections across the month
            random_draw = rng.random_sample(size=len(df))

            # the cfr applies to all severe cases
            death = df.index[
                (df.ma_specific_symptoms == 'severe') & (df.ma_date_infected == now) & (random_draw < p['cfr'])]

            for person in death:
                logger.debug(
                    'MalariaEvent: scheduling malaria death for person %d',
                    person)

                random_date = rng.randint(low=0, high=7)
                random_days = pd.to_timedelta(random_date, unit='d')

                death_event = MalariaDeathEvent(self.module, individual_id=person,
                                                cause='malaria')  # make that death event
                self.sim.schedule_event(death_event, self.sim.date + random_days)  # schedule the death

        else:
            logger.debug('MalariaEventNational: no one is newly infected.')


class MalariaEventDistrict(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):

        logger.debug('MalariaEvent: tracking the disease progression of the population.')

        df = population.props
        p = self.module.parameters
        rng = self.module.rng
        now = self.sim.date

        # ----------------------------------- INTERVENTION COVERAGE -----------------------------------
        interv = p['interv']

        # find annual intervention coverage levels
        # TODO: replace with district-level intervention coverage estimates when available
        # TODO: using 'slept under bednet' because ITN coverage >80% in 2013 and no more cases
        # itn = interv.loc[interv.Year == now.year, 'MIS_sleptUnderITN'].values[0]
        irs = interv.loc[interv.Year == now.year, 'IRS_coverage'].values[0]
        irs = 0.8 if irs > 0.5 else 0  # round to 0 or 0.8

        # get ITN usage rates for current year by district
        itn = p['itn_district']

        itn_curr = itn.loc[itn.Year == now.year]
        # check this returns all values for the current year

        itn.rate_distr = itn["Year"].map(str) + itn["itn_rates"]


        # select incidence based on intervention coverage
        month = now.month

        # need to pull out the district values with district matching from itn dataset
        inf_inc = p['inf_inc']  # datasheet with infection incidence

        # the districts from itn need to match with the districts from inf_inc_month
        # get the dataset using month and irs
        # match the inf_inc districts with the itn_districts to get district-level inf_inc
        # then produce new dataset with distr, age and inf_inc
        inf_inc.llin_distr = inf_inc["admin"].map(str) + inf_inc["llin"]

        # maybe do these steps separately
        inf_inc_month = inf_inc.loc[(inf_inc.month == month) & (inf_inc.irs == irs)]
        itn.monthly_prob_inf = round(itn.monthly_prob_inf, 1)  # round to nearest 0.1 to match Pete's data

        # should be a dataset with one entry per district/age combo but only one llin level per district

        # select the district entries which match the district-level itn usage rates
        inf_inc_month = inf_inc.loc[inf_inc.llin_distr.isin(itn.rate_distr)]





        clin_inc = p['clin_inc']
        clin_inc_month = clin_inc.loc[
            (clin_inc.month == month) & (clin_inc.llin == itn) & (clin_inc.irs == irs)]
        sev_inc = p['sev_inc']
        sev_inc_month = sev_inc.loc[(sev_inc.month == month) & (sev_inc.llin == itn) & (sev_inc.irs == irs)]

        # for each district and age, look up incidence estimate using itn_2010 and irs_2010
        # create new age column with 0, 0.5, 1, 2, ...
        df.loc[df.age_exact_years.between(0, 0.5), 'ma_age_edited'] = 0
        df.loc[df.age_exact_years.between(0.5, 1), 'ma_age_edited'] = 0.5
        df.loc[(df.age_exact_years >= 1), 'ma_age_edited'] = df.age_years[df.age_years >= 1]
        assert (not pd.isnull(df['ma_age_edited']).any())
        df['ma_age_edited'] = df['ma_age_edited'].astype('float')  # for merge with malaria data

        # merge the incidence into the main df and replace each event call
        df_ml = df.reset_index().merge(inf_inc_month, left_on=['ma_district_edited', 'ma_age_edited'],
                                       right_on=['admin', 'age'],
                                       how='left', indicator=True).set_index('person')
        df_ml['monthly_prob_inf'] = df_ml['monthly_prob_inf'].fillna(0)  # 0 if over 80 yrs
        assert (not pd.isnull(df_ml['monthly_prob_inf']).any())

        df_ml = df_ml.reset_index().merge(clin_inc_month, left_on=['ma_district_edited', 'ma_age_edited'],
                                          right_on=['admin', 'age'],
                                          how='left').set_index('person')
        df_ml['monthly_prob_clin'] = df_ml['monthly_prob_clin'].fillna(0)  # 0 if over 80 yrs
        assert (not pd.isnull(df_ml['monthly_prob_clin']).any())

        df_ml = df_ml.reset_index().merge(sev_inc_month, left_on=['ma_district_edited', 'ma_age_edited'],
                                          right_on=['admin', 'age'],
                                          how='left').set_index('person')
        df_ml['monthly_prob_sev'] = df_ml['monthly_prob_sev'].fillna(0)  # 0 if over 80 yrs
        assert (not pd.isnull(df_ml['monthly_prob_sev']).any())

        # ----------------------------------- NEW INFECTIONS -----------------------------------
        ## infected
        risk_ml = pd.Series(0, index=df.index)
        risk_ml.loc[df.is_alive] = 1  # applied to everyone
        # risk_ml.loc[df.hv_inf ] *= p['rr_hiv']  # then have to scale within every subgroup

        # new infections - sample from uninfected/asymptomatic
        random_draw = rng.random_sample(size=len(df_ml))
        ml_idx = df_ml[
            df_ml.is_alive & (df.ma_specific_symptoms == 'none') & (random_draw < df_ml.monthly_prob_inf)].index
        df_ml.loc[ml_idx, 'ma_is_infected'] = True
        df_ml.loc[ml_idx, 'ma_date_infected'] = now  # TODO: scatter dates across month
        # print('len ml_idx', len(ml_idx))

        ## clinical - subset of newly infected
        random_draw = rng.random_sample(size=len(df))
        clin_idx = df_ml[
            df_ml.is_alive & df_ml.ma_is_infected & (df_ml.ma_date_infected == now) & (
                random_draw < df_ml.monthly_prob_clin)].index
        df_ml.loc[clin_idx, 'ma_specific_symptoms'] = 'clinical'
        # print('len clin_idx', len(clin_idx))

        ## severe - subset of newly clinical
        random_draw = rng.random_sample(size=len(df))
        sev_idx = df_ml[
            df_ml.is_alive & df_ml.ma_is_infected & (df_ml.ma_date_infected == now) & (
                df_ml.ma_specific_symptoms == 'clinical') & (
                random_draw < df_ml.monthly_prob_sev)].index
        # print('len sev_idx', len(sev_idx))

        # update the main dataframe
        df.loc[ml_idx, 'ma_date_infected'] = now
        df.loc[ml_idx, 'ma_is_infected'] = True
        df.loc[ml_idx, 'ma_specific_symptoms'] = 'none'

        df.loc[clin_idx, 'ma_specific_symptoms'] = 'clinical'
        df.loc[clin_idx, 'ma_unified_symptom_code'] = 1
        df.loc[clin_idx, 'ma_date_clinical'] = now
        df.loc[clin_idx, 'ma_clinical_counter'] += 1  # counter only for new clinical cases (inc severe)
        # print('clin counter', df['ma_clinical_counter'].sum())

        df.loc[sev_idx, 'ma_specific_symptoms'] = 'severe'
        df.loc[sev_idx, 'ma_unified_symptom_code'] = 2

        ## tidy up
        del df_ml

        # if any are infected
        if len(ml_idx):
            logger.debug('This is MalariaEvent, assigning new malaria infections')

            # ----------------------------------- PARASITE CLEARANCE - NO TREATMENT -----------------------------------
            # schedule self-cure if no treatment, no self-cure from severe malaria

            # asymptomatic
            asym = df.index[(df.ma_specific_symptoms == 'none') & (df.ma_date_infected == now)]

            for person in df.loc[asym].index:
                # logger.debug(
                #     'Malaria Event: scheduling parasite clearance for asymptomatic person %d', person)

                random_date = rng.randint(low=0, high=p['dur_asym'])
                random_days = pd.to_timedelta(random_date, unit='d')

                cure = MalariaParasiteClearanceEvent(self.module, person)
                self.sim.schedule_event(cure, (self.sim.date + random_days))

            # clinical
            clin = df.index[(df.ma_specific_symptoms == 'clinical') & (df.ma_date_infected == now)]

            for person in df.loc[clin].index:
                # logger.debug(
                #     'Malaria Event: scheduling parasite clearance person %d',
                #     person)

                date_para = rng.randint(low=0, high=p['dur_clin_para'])
                date_para_days = pd.to_timedelta(date_para, unit='d')
                # print('date_para_days', date_para_days)

                date_clin = rng.randint(low=0, high=p['dur_clin'])
                date_clin_days = pd.to_timedelta(date_clin, unit='d')
                # print('date_clin_days', date_clin_days)

                cure = MalariaParasiteClearanceEvent(self.module, person)
                self.sim.schedule_event(cure, (self.sim.date + date_para_days))

                # schedule move back to asymptomatic(5 days)
                symp_end = MalariaClinEndEvent(self.module, person)
                self.sim.schedule_event(symp_end, self.sim.date + date_clin_days)

            # # ----------------------------------- HEALTHCARE-SEEKING -----------------------------------
            # # clinical cases will seek care with some probability
            # # severe cases will definitely seek care
            # interv = p['interv']
            #
            # # find annual intervention coverage levels rate for 2010
            # act = interv.loc[interv.Year == now.year, 'ACT_coverage'].values[0]
            # # act = 1
            # act = act * 3  # adjustment to match tx coverage estimates
            #
            # # CLINICAL CASES
            # seeks_care = pd.Series(data=False, index=df.loc[clin].index)
            #
            # for i in df.loc[clin].index:
            #     # prob = self.sim.modules['HealthSystem'].get_prob_seek_care(i, symptom_code=4)
            #     seeks_care[i] = rng.rand() < act  # placeholder for coverage / testing rates
            #
            # if seeks_care.sum() > 0:
            #
            #     for person_index in seeks_care.index[seeks_care]:
            #         # print(person_index)
            #
            #         logger.debug(
            #             'MalariaEvent: scheduling HSI_Malaria_rdt for clinical malaria case %d',
            #             person_index)
            #
            #         delay = rng.choice(7)
            #         # print(delay)
            #
            #         event = HSI_Malaria_rdt(self.module, person_id=person_index)
            #         self.sim.modules['HealthSystem'].schedule_hsi_event(event,
            #                                                             priority=2,
            #                                                             topen=self.sim.date + DateOffset(days=delay),
            #                                                             tclose=self.sim.date + DateOffset(
            #                                                                 days=(delay + 14))
            #                                                             )
            # else:
            #     logger.debug(
            #         'MalariaEvent: There is no new healthcare seeking for clinical cases')
            # clinical cases will seek care with some probability
            # severe cases will definitely seek care

            # Give everyone with clinical malaria generic symptom fever (some will go to care for this)
            # Report this to the unified symptom manager:
            # the symptom manager just gives the person the symptom
            if len(clin) > 0:
                # TODO: if clinical symptoms too short maybe no healthcare seeking scheduled in time

                # this also schedules symptom resolution in 5 days
                self.sim.modules['SymptomManager'].chg_symptom(
                    person_id=list(clin),
                    symptom_string='fever',
                    add_or_remove='+',
                    disease_module=self.module,
                    duration_in_days=p['dur_clin'])

                self.sim.modules['SymptomManager'].chg_symptom(
                    person_id=list(clin),
                    symptom_string='headache',
                    add_or_remove='+',
                    disease_module=self.module,
                    duration_in_days=p['dur_clin'])

                self.sim.modules['SymptomManager'].chg_symptom(
                    person_id=list(clin),
                    symptom_string='vomiting',
                    add_or_remove='+',
                    disease_module=self.module,
                    duration_in_days=p['dur_clin'])

                self.sim.modules['SymptomManager'].chg_symptom(
                    person_id=list(clin),
                    symptom_string='stomachache',
                    add_or_remove='+',
                    disease_module=self.module,
                    duration_in_days=p['dur_clin'])

                # TODO:'duration_in_days' schedules symptom resolution and treatment also schedules resolution!
                # TODO: check symptoms still present if treated before scheduling symptom resolution

            # the HealthSeekingBehaviourPoll will check for generic symptoms daily and schedule a generic appt
            # in generic appt, code 'if fever: schedule HSI_malaria_rdt
            # if severe malaria, schedule my own HSI_severe_tx combined with rdt

            # Q. how to calibrate tx coverage levels, where is the gating step?
            # SEVERE CASES
            # todo check only severe cases are passing through here
            severe = df.index[(df.ma_specific_symptoms == 'severe') & (df.ma_date_infected == now)]

            for person_index in severe:
                # print(person_index)

                logger.debug(
                    'MalariaEvent: scheduling HSI_Malaria_rdt for severe malaria case %d',
                    person_index)

                delay = rng.choice(2)
                # print(delay)

                event = HSI_Malaria_rdt(self.module, person_id=person_index)
                self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                    priority=2,
                                                                    topen=self.sim.date + DateOffset(days=delay),
                                                                    tclose=self.sim.date + DateOffset(days=(delay + 14))
                                                                    )

            # ----------------------------------- SCHEDULED DEATHS -----------------------------------
            # schedule deaths within the next week
            # Assign time of infections across the month
            random_draw = rng.random_sample(size=len(df))

            # the cfr applies to all severe malaria cases
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

        else:
            logger.debug('MalariaEvent: no one is newly infected.')


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

            # if on treatment, will reduce probability of death
            # use random number generator - currently param treatment_adjustment set to 0.5
            if df.at[individual_id, 'ma_tx']:
                prob = self.module.rng.random_sample(size=1)

                if prob < self.module.parameters['treatment_adjustment']:
                    self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id, cause='malaria'),
                                            self.sim.date)

                    df.at[individual_id, 'ma_date_death'] = self.sim.date

            else:
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

        super().__init__(module, person_id=person_id)
        assert isinstance(module, Malaria)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['LabPOC'] = 1
        # print(the_appt_footprint)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Malaria_RDT'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props
        params = self.module.parameters

        district = df.at[person_id, 'district_of_residence']
        logger.debug(f'HSI_Malaria_rdt: rdt test for person {person_id} in {district}')

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
            hsi_event=self, cons_req_as_footprint=consumables_needed, to_log=False
        )

        if outcome_of_request_for_consumables:

            # check if still alive
            if df.at[person_id, 'is_alive']:

                # ----------------------------------- SEVERE MALARIA -----------------------------------

                # if severe malaria, treat for complicated malaria
                if df.at[person_id, 'ma_is_infected'] & (df.at[person_id, 'ma_specific_symptoms'] == 'severe'):

                    if (df.at[person_id, 'age_years'] < 15):

                        logger.debug(
                            "HSI_Malaria_rdt: scheduling HSI_Malaria_tx_compl_child for person %d on date %s",
                            person_id, (self.sim.date + DateOffset(days=1)))

                        treat = HSI_Malaria_tx_compl_child(self.module, person_id=person_id)
                        self.sim.modules['HealthSystem'].schedule_hsi_event(treat,
                                                                            priority=1,
                                                                            topen=self.sim.date,
                                                                            tclose=None)

                    else:
                        logger.debug(
                            "HSI_Malaria_rdt: scheduling HSI_Malaria_tx_compl_adult for person %d on date %s",
                            person_id, (self.sim.date + DateOffset(days=1)))

                        treat = HSI_Malaria_tx_compl_adult(self.module, person_id=person_id)
                        self.sim.modules['HealthSystem'].schedule_hsi_event(treat,
                                                                            priority=1,
                                                                            topen=self.sim.date,
                                                                            tclose=None)

                # ----------------------------------- TREATMENT CLINICAL DISEASE -----------------------------------

                elif df.at[person_id, 'ma_is_infected'] & (df.at[person_id, 'ma_specific_symptoms'] == 'clinical'):

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
                                                                            topen=self.sim.date,
                                                                            tclose=None)

                    # diagnosis / treatment for children 5-15
                    if diagnosed & (df.at[person_id, 'age_years'] >= 5) & (df.at[person_id, 'age_years'] < 15):
                        logger.debug("HSI_Malaria_rdt: scheduling HSI_Malaria_tx_5_15 for person %d on date %s",
                                     person_id, (self.sim.date + DateOffset(days=1)))

                        treat = HSI_Malaria_tx_5_15(self.module, person_id=person_id)
                        self.sim.modules['HealthSystem'].schedule_hsi_event(treat,
                                                                            priority=1,
                                                                            topen=self.sim.date,
                                                                            tclose=None)

                    # diagnosis / treatment for adults
                    if diagnosed & (df.at[person_id, 'age_years'] >= 15):
                        logger.debug("HSI_Malaria_rdt: scheduling HSI_Malaria_tx_adult for person %d on date %s",
                                     person_id, (self.sim.date + DateOffset(days=1)))

                        treat = HSI_Malaria_tx_adult(self.module, person_id=person_id)
                        self.sim.modules['HealthSystem'].schedule_hsi_event(treat,
                                                                            priority=1,
                                                                            topen=self.sim.date,
                                                                            tclose=None)

            # log the consumables used
            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self, cons_req_as_footprint=consumables_needed, to_log=True
            )

    def did_not_run(self):
        logger.debug('HSI_Malaria_rdt: did not run')
        pass


class HSI_Malaria_tx_0_5(HSI_Event, IndividualScopeEventMixin):
    """
    this is anti-malarial treatment for children <15 kg. Includes treatment plus one rdt
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Malaria)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Under5OPD'] = 1  # This requires one out patient

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Malaria_treatment_child0_5'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props
        if not df.at[person_id, 'ma_tx']:

            logger.debug('HSI_Malaria_tx_0_5: requesting malaria treatment for child %d',
                         person_id)

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
                hsi_event=self, cons_req_as_footprint=the_cons_footprint, to_log=False
            )

            if outcome_of_request_for_consumables:
                logger.debug('HSI_Malaria_tx_0_5: giving malaria treatment for child %d',
                             person_id)

                if df.at[person_id, 'is_alive']:
                    df.at[person_id, 'ma_tx'] = True
                    df.at[person_id, 'ma_date_tx'] = self.sim.date
                    df.at[person_id, 'ma_tx_counter'] += 1

                    self.sim.schedule_event(MalariaCureEvent(self.module, person_id),
                                            self.sim.date + DateOffset(weeks=1))

                    # log the consumables
                    outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                        hsi_event=self, cons_req_as_footprint=the_cons_footprint, to_log=True
                    )

    def did_not_run(self):
        logger.debug('HSI_Malaria_tx_0_5: did not run')
        pass


class HSI_Malaria_tx_5_15(HSI_Event, IndividualScopeEventMixin):
    """
    this is anti-malarial treatment for children >15 kg. Includes treatment plus one rdt
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Malaria)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Under5OPD'] = 1  # This requires one out patient

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Malaria_treatment_child5_15'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props
        if not df.at[person_id, 'ma_tx']:

            logger.debug('HSI_Malaria_tx_5_15: requesting malaria treatment for child %d',
                     person_id)

            consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
            pkg_code1 = pd.unique(
                consumables.loc[
                    consumables['Intervention_Pkg'] == 'Uncomplicated (children, >15 kg)',
                    'Intervention_Pkg_Code'])[0]  # this pkg_code includes another rdt

            the_cons_footprint = {
                'Intervention_Package_Code': [{pkg_code1: 1}],
                'Item_Code': []
            }

            # request the treatment
            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self, cons_req_as_footprint=the_cons_footprint, to_log=False
            )

            if outcome_of_request_for_consumables:
                logger.debug('HSI_Malaria_tx_5_15: giving malaria treatment for child %d',
                             person_id)

                if df.at[person_id, 'is_alive']:
                    df.at[person_id, 'ma_tx'] = True
                    df.at[person_id, 'ma_date_tx'] = self.sim.date
                    df.at[person_id, 'ma_tx_counter'] += 1

                    self.sim.schedule_event(MalariaCureEvent(self.module, person_id),
                                            self.sim.date + DateOffset(weeks=1))

                    # log the consumables
                    outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                        hsi_event=self, cons_req_as_footprint=the_cons_footprint, to_log=True
                    )

    def did_not_run(self):
        logger.debug('HSI_Malaria_tx_5_15: did not run')
        pass


class HSI_Malaria_tx_adult(HSI_Event, IndividualScopeEventMixin):
    """
    this is anti-malarial treatment for adults. Includes treatment plus one rdt
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Malaria)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Malaria_treatment_adult'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props
        if not df.at[person_id, 'ma_tx']:

            logger.debug('HSI_Malaria_tx_adult: requesting malaria treatment for person %d',
                         person_id)

            consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
            pkg_code1 = pd.unique(
                consumables.loc[
                    consumables['Intervention_Pkg'] == 'Uncomplicated (adult, >36 kg)',
                    'Intervention_Pkg_Code'])[0]  # this pkg_code includes another rdt

            the_cons_footprint = {
                'Intervention_Package_Code': [{pkg_code1: 1}],
                'Item_Code': []
            }

            # request the treatment
            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self, cons_req_as_footprint=the_cons_footprint, to_log=False
            )

            if outcome_of_request_for_consumables:
                logger.debug('HSI_Malaria_tx_adult: giving malaria treatment for person %d',
                             person_id)

                if df.at[person_id, 'is_alive']:
                    df.at[person_id, 'ma_tx'] = True
                    df.at[person_id, 'ma_date_tx'] = self.sim.date
                    df.at[person_id, 'ma_tx_counter'] += 1

                    self.sim.schedule_event(MalariaCureEvent(self.module, person_id),
                                            self.sim.date + DateOffset(weeks=1))

                    # log the consumables
                    outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                        hsi_event=self, cons_req_as_footprint=the_cons_footprint, to_log=True
                    )

    def did_not_run(self):
        logger.debug('HSI_Malaria_tx_adult: did not run')
        pass


class HSI_Malaria_tx_compl_child(HSI_Event, IndividualScopeEventMixin):
    """
    this is anti-malarial treatment for complicated malaria in children
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Malaria)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 5

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Malaria_treatment_complicated_child'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props
        if not df.at[person_id, 'ma_tx']:

            logger.debug('HSI_Malaria_tx_compl_child: requesting complicated malaria treatment for child %d',
                     person_id)

            consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
            pkg_code1 = pd.unique(
                consumables.loc[
                    consumables['Intervention_Pkg'] == 'Complicated (children, injectable artesunate)',
                    'Intervention_Pkg_Code'])[0]

            the_cons_footprint = {
                'Intervention_Package_Code': [{pkg_code1: 1}],
                'Item_Code': []
            }

            # request the treatment
            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self, cons_req_as_footprint=the_cons_footprint, to_log=False
            )

            if outcome_of_request_for_consumables:
                logger.debug('HSI_Malaria_tx_compl_child: giving complicated malaria treatment for child %d',
                             person_id)

                if df.at[person_id, 'is_alive']:
                    df.at[person_id, 'ma_tx'] = True
                    df.at[person_id, 'ma_date_tx'] = self.sim.date
                    df.at[person_id, 'ma_tx_counter'] += 1

                    self.sim.schedule_event(MalariaCureEvent(self.module, person_id),
                                            self.sim.date + DateOffset(weeks=1))

                    # log the consumables
                    outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                        hsi_event=self, cons_req_as_footprint=the_cons_footprint, to_log=True
                    )

    def did_not_run(self):
        logger.debug('HSI_Malaria_tx_compl_child: did not run')
        pass


class HSI_Malaria_tx_compl_adult(HSI_Event, IndividualScopeEventMixin):
    """
    this is anti-malarial treatment for complicated malaria in adults
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Malaria)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 5

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Malaria_treatment_complicated_adult'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props
        if not df.at[person_id, 'ma_tx']:

            logger.debug('HSI_Malaria_tx_compl_adult: requesting complicated malaria treatment for person %d',
                     person_id)

            consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
            pkg_code1 = pd.unique(
                consumables.loc[
                    consumables['Intervention_Pkg'] == 'Complicated (adults, injectable artesunate)',
                    'Intervention_Pkg_Code'])[0]

            the_cons_footprint = {
                'Intervention_Package_Code': [{pkg_code1: 1}],
                'Item_Code': []
            }

            # request the treatment
            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self, cons_req_as_footprint=the_cons_footprint, to_log=False
            )

            if outcome_of_request_for_consumables:
                logger.debug('HSI_Malaria_tx_compl_adult: giving complicated malaria treatment for person %d',
                             person_id)

                if df.at[person_id, 'is_alive']:
                    df.at[person_id, 'ma_tx'] = True
                    df.at[person_id, 'ma_date_tx'] = self.sim.date
                    df.at[person_id, 'ma_tx_counter'] += 1

                    self.sim.schedule_event(MalariaCureEvent(self.module, person_id),
                                            self.sim.date + DateOffset(weeks=1))

                    # log the consumables
                    outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                        hsi_event=self, cons_req_as_footprint=the_cons_footprint, to_log=True
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
        logger.debug("MalariaCureEvent: Stopping malaria treatment and curing person %d", person_id)

        df = self.sim.population.props

        # stop treatment
        if df.at[person_id, 'is_alive']:

            # check that a fever is present and was caused by malaria before resolving it
            if (df.at[person_id, 'sy_fever'] > 0) & (df.at[person_id, 'ma_specific_symptoms'] == 'clinical'):
                self.sim.modules['SymptomManager'].chg_symptom(
                    person_id=person_id,
                    symptom_string='fever',
                    add_or_remove='-',
                    disease_module=self.module,
                    duration_in_days=None)

                self.sim.modules['SymptomManager'].chg_symptom(
                    person_id=person_id,
                    symptom_string='headache',
                    add_or_remove='-',
                    disease_module=self.module,
                    duration_in_days=None)

                self.sim.modules['SymptomManager'].chg_symptom(
                    person_id=person_id,
                    symptom_string='vomiting',
                    add_or_remove='-',
                    disease_module=self.module,
                    duration_in_days=None)

                self.sim.modules['SymptomManager'].chg_symptom(
                    person_id=person_id,
                    symptom_string='stomachache',
                    add_or_remove='-',
                    disease_module=self.module,
                    duration_in_days=None)

            # change treatment and infection status
            df.at[person_id, 'ma_tx'] = False

            df.at[person_id, 'ma_is_infected'] = False
            df.at[person_id, 'ma_specific_symptoms'] = 'none'
            df.at[person_id, 'ma_unified_symptom_code'] = 0


class MalariaParasiteClearanceEvent(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        # logger.debug("This is MalariaParasiteClearanceEvent for person %d", person_id)

        df = self.sim.population.props

        if df.at[person_id, 'is_alive']:
            df.at[person_id, 'ma_is_infected'] = False
            df.at[person_id, 'ma_specific_symptoms'] = 'none'
            df.at[person_id, 'ma_unified_symptom_code'] = 0


class MalariaClinEndEvent(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        logger.debug("This is MalariaClinEndEvent changing person %d from clinical to asymptomatic", person_id)
        # symptom manager will resolve the symptoms
        # need to move clinical people back from ma_specific_symptoms = clinical to asymptomatic ('none')

        df = self.sim.population.props

        if df.at[person_id, 'is_alive']:
            df.at[person_id, 'ma_specific_symptoms'] = 'none'
            df.at[person_id, 'ma_unified_symptom_code'] = 0


# ---------------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------------

class MalariaLoggingEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props
        now = self.sim.date

        # ------------------------------------ INCIDENCE ------------------------------------

        # infected in the last time-step, clinical and severe cases only
        # incidence rate per 1000 person-years
        # include those cases that have died in the case load
        tmp = len(
            df.loc[(df.ma_date_clinical > (now - DateOffset(months=self.repeat)))])
        pop = len(df[df.is_alive])

        inc_1000py = (tmp / pop) * 1000

        # incidence rate in 2-10 yr olds
        tmp2 = len(
            df.loc[(df.age_years.between(2, 10)) & (
                df.ma_date_clinical > (now - DateOffset(months=self.repeat)))])

        pop2_10 = len(df[df.is_alive & (df.age_years.between(2, 10))])
        inc_1000py_2_10 = (tmp2 / pop2_10) * 1000

        inc_1000py_hiv = 0  # if running without hiv/tb

        # using clinical counter
        # sum all the counters for previous year
        clin_episodes = df['ma_clinical_counter'].sum()  # clinical episodes (inc severe)
        incCounter_1000py = (clin_episodes / pop) * 1000

        logger.info('%s|incidence|%s', now,
                    {
                        'number_new_cases': tmp,
                        'population': pop,
                        'inc_1000py': inc_1000py,
                        'inc_1000py_hiv': inc_1000py_hiv,
                        'new_cases_2_10': tmp2,
                        'population2_10': pop2_10,
                        'inc_1000py_2_10': inc_1000py_2_10,
                        'inc_clin_counter': incCounter_1000py
                    })

        # ------------------------------------ RUNNING COUNTS ------------------------------------

        counts = {'None': 0, 'clinical': 0, 'severe': 0}
        counts.update(df.loc[df.is_alive, 'ma_specific_symptoms'].value_counts().to_dict())

        logger.info('%s|status_counts|%s', now, counts)

        # ------------------------------------ PARASITE PREVALENCE BY AGE ------------------------------------

        # includes all parasite positive cases: some may have low parasitaemia (undetectable)
        child2_10_clin = len(df[df.is_alive & df.ma_is_infected & (df.age_years.between(2, 10))])

        # population size - children
        child2_10_pop = len(df[df.is_alive & (df.age_years.between(2, 10))])

        # prevalence in children aged 2-10
        child_prev = child2_10_clin / child2_10_pop if child2_10_pop else 0

        # prevalence of clinical including severe in all ages
        total_clin = len(
            df[df.is_alive & ((df.ma_specific_symptoms == 'clinical') | (df.ma_specific_symptoms == 'severe'))])
        pop2 = len(df[df.is_alive])
        prev_clin = total_clin / pop2

        logger.info('%s|prevalence|%s', now,
                    {
                        'child2_10_prev': child_prev,
                        'clinical_prev': prev_clin,
                    })

        # ------------------------------------ MORTALITY ------------------------------------
        # deaths reported in the last 12 months / pop size
        deaths = len(df[(df.ma_date_death > (now - DateOffset(months=self.repeat)))])

        mort_rate = deaths / len(df[df.is_alive])

        logger.info('%s|ma_mortality|%s', now,
                    {
                        'mort_rate': mort_rate,
                    })


class MalariaTxLoggingEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props
        now = self.sim.date

        # ------------------------------------ TREATMENT COVERAGE ------------------------------------
        # prop clinical episodes which had treatment, all ages

        # sum all the counters for previous year
        tx = df['ma_tx_counter'].sum()  # treatment (inc severe)
        clin = df['ma_clinical_counter'].sum()  # clinical episodes (inc severe)

        tx_coverage = tx / clin if clin else 0

        logger.info('%s|tx_coverage|%s', now,
                    {
                        'number treated': tx,
                        'number clinical episodes': clin,
                        'treatment_coverage': tx_coverage,
                    })


class MalariaPrevDistrictLoggingEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        self.repeat = 1
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props

        # todo this could be PfPR in 2-10 yr olds and clinical incidence too
        # ------------------------------------ PREVALENCE OF INFECTION ------------------------------------
        infected = df[df.is_alive & df.ma_is_infected].groupby('ma_district_edited').size()
        pop = df[df.is_alive].groupby('ma_district_edited').size()
        prev = infected / pop
        prev_ed = prev.fillna(0)
        assert prev_ed.all() >= 0  # checks
        assert prev_ed.all() <= 1

        logger.info('%s|prev_district|%s', self.sim.date,
                    prev_ed.to_dict())
        logger.info('%s|pop_district|%s', self.sim.date,
                    pop.to_dict())


# ---------------------------------------------------------------------------------
# Reset counters
# ---------------------------------------------------------------------------------
class MalariaResetCounterEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # reset all the counters to zero each year
        df = population.props
        now = self.sim.date

        logger.info(f'Resetting the malaria counter {now}')

        df['ma_clinical_counter'] = 0
        df['ma_tx_counter'] = 0
