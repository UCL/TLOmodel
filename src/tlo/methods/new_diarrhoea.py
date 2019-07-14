"""
Childhood diarrhoea module
Documentation: 04 - Methods Repository/Method_Child_EntericInfection.xlsx
"""
import logging

import numpy as np
import pandas as pd
from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent, Event, IndividualScopeEventMixin
from tlo.methods.iCCM import HSI_Sick_Child_Seeks_Care_From_HSA

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class NewDiarrhoea(Module):
    PARAMETERS = {
        'base_incidence_diarrhoea_by_rotavirus':
            Parameter(Types.LIST, 'incidence of diarrhoea caused by rotavirus in age groups 0-11, 12-23, 24-59 months '
                      ),
        'base_incidence_diarrhoea_by_shigella':
            Parameter(Types.LIST,
                      'incidence of diarrhoea caused by shigella spp in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_incidence_diarrhoea_by_adenovirus':
            Parameter(Types.LIST,
                      'incidence of diarrhoea caused by adenovirus 40/41 in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_incidence_diarrhoea_by_crypto':
            Parameter(Types.LIST,
                      'incidence of diarrhoea caused by cryptosporidium in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_incidence_diarrhoea_by_campylo':
            Parameter(Types.LIST,
                      'incidence of diarrhoea caused by campylobacter spp in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_incidence_diarrhoea_by_ETEC':
            Parameter(Types.LIST,
                      'incidence of diarrhoea caused by ST-ETEC in age groups 0-11, 12-23, 24-59 months'
                      ),
        'rr_gi_diarrhoea_HHhandwashing':
            Parameter(Types.REAL, 'relative rate of diarrhoea with household handwashing with soap'
                      ),
        'rr_gi_diarrhoea_improved_sanitation':
            Parameter(Types.REAL, 'relative rate of diarrhoea for improved sanitation'
                      ),
        'rr_gi_diarrhoea_clean_water':
            Parameter(Types.REAL, 'relative rate of diarrhoea for access to clean drinking water'
                      ),
        'rr_gi_diarrhoea_HIV':
            Parameter(Types.REAL, 'relative rate of diarrhoea for HIV positive status'
                      ),
        'rr_gi_diarrhoea_SAM':
            Parameter(Types.REAL, 'relative rate of diarrhoea for severe malnutrition'
                      ),
        'rr_gi_diarrhoea_cont_breast':
            Parameter(Types.REAL, 'relative rate of diarrhoea for exclusive breastfeeding upto 6 months'
                      ),
        'rr_gi_diarrhoea_excl_breast':
            Parameter(Types.REAL, 'relative rate of diarrhoea for continued breastfeeding 6 months to 2 years'
                      ),

    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'gi_diarrhoea_status': Property(Types.BOOL, 'symptomatic infection - diarrhoea disease'),
        'gi_diarrhoea_pathogen': Property(Types.CATEGORICAL, 'attributable pathogen for diarrhoea',
                                          categories=['rotavirus', 'shigella', 'adenovirus', 'cryptosporidium',
                                                      'campylobacter', 'ST-ETEC']),
        'gi_diarrhoea_acute_type': Property(Types.CATEGORICAL, 'clinical acute diarrhoea type',
                                            categories=['dysentery', 'acute watery diarrhoea']),
        'gi_dehydration_status': Property(Types.CATEGORICAL, 'dehydration status',
                                          categories=['no dehydration', 'some dehydration', 'severe dehydration']),
        'gi_persistent_diarrhoea': Property(Types.BOOL, 'diarrhoea episode longer than 14 days - persistent type'),

        'gi_diarrhoea_death': Property(Types.BOOL, 'death caused by diarrhoea'),
        'date_of_onset_diarrhoea': Property(Types.DATE, 'date of onset of diarrhoea'),
        'gi_recovered_date': Property(Types.DATE, 'date of recovery from enteric infection'),
        'gi_diarrhoea_death_date': Property(Types.DATE, 'date of death from enteric infection'),
        'gi_diarrhoea_count': Property(Types.REAL, 'number of diarrhoea episodes per individual'),
        'has_hiv': Property(Types.BOOL, 'temporary property - has hiv'),
        'malnutrition': Property(Types.BOOL, 'temporary property - malnutrition status'),
        'exclusive_breastfeeding': Property(Types.BOOL, 'temporary property - exclusive breastfeeding upto 6 mo'),
        'continued_breastfeeding': Property(Types.BOOL, 'temporary property - continued breastfeeding 6mo-2years'),
        # symptoms of diarrhoea for care seeking
        'di_diarrhoea_loose_watery_stools': Property(Types.BOOL, 'diarrhoea symptoms - loose or watery stools'),
        'di_blood_in_stools': Property(Types.BOOL, 'dysentery symptoms - blood in the stools'),
        'di_diarrhoea_over14days': Property(Types.BOOL, 'persistent diarrhoea - diarrhoea for 14 days or more'),
    }

    def read_parameters(self, data_folder):
        """ Setup parameters values used by the module
        """
        p = self.parameters

        p['base_incidence_diarrhoea_by_rotavirus'] = [0.061, 0.02225, 0.00125]
        p['base_incidence_diarrhoea_by_shigella'] = [0.061, 0.02225, 0.00125]
        p['base_incidence_diarrhoea_by_adenovirus'] = [0.061, 0.02225, 0.00125]
        p['base_incidence_diarrhoea_by_crypto'] = [0.061, 0.02225, 0.00125]
        p['base_incidence_diarrhoea_by_campylo'] = [0.061, 0.02225, 0.00125]
        p['base_incidence_diarrhoea_by_ETEC'] = [0.061, 0.02225, 0.00125]
        p['rr_gi_diarrhoea_HHhandwashing'] = 0.5
        p['rr_gi_diarrhoea_improved_sanitation'] = 0.5
        p['rr_gi_diarrhoea_clean_water'] = 0.5
        p['rr_gi_diarrhoea_HIV'] = 1.4
        p['rr_gi_diarrhoea_SAM'] = 1.5
        p['rr_gi_diarrhoea_excl_breast'] = 0.5
        p['rr_gi_diarrhoea_cont_breast'] = 0.9

    def initialise_population(self, population):
        """Set our property values for the initial population.
        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.
        :param population: the population of individuals
        """
        df = population.props  # a shortcut to the data-frame storing data for individuals
        m = self
        rng = m.rng
        now = self.sim.date

        # DEFAULTS
        df['gi_diarrhoea_status'] = False
        df['gi_dehydration_status'] = 'no dehydration'
        df['date_of_onset_diarrhoea'] = pd.NaT
        df['gi_recovered_date'] = pd.NaT
        df['gi_diarrhoea_death_date'] = pd.NaT
        df['gi_diarrhoea_count'] = 0
        df['gi_diarrhoea_death'] = False
        df['malnutrition'] = False
        df['has_hiv'] = False
        df['exclusive_breastfeeding'] = False
        df['continued_breastfeeding'] = False

    def initialise_simulation(self, sim):
        """
        Get ready for simulation start.
        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        # add the basic event for dysentery ---------------------------------------------------
        event_diarrhoea = DiarrhoeaEvent(self)
        sim.schedule_event(event_diarrhoea, sim.date + DateOffset(months=3))

        # add an event to log to screen
        # sim.schedule_event(DysenteryLoggingEvent(self), sim.date + DateOffset(months=6))

        '''# death event
        death_all_diarrhoea = DeathDiarrhoeaEvent(self)
        sim.schedule_event(death_all_diarrhoea, sim.date)

        # add an event to log to screen
        sim.schedule_event(AcuteDiarrhoeaLoggingEvent(self), sim.date + DateOffset(months=6))

        # add the basic event for persistent diarrhoea ------------------------------------------
        event_persistent_diar = PersistentDiarrhoeaEvent(self)
        sim.schedule_event(event_persistent_diar, sim.date + DateOffset(months=3))

        # add an event to log to screen
        sim.schedule_event(PersistentDiarrhoeaLoggingEvent(self), sim.date + DateOffset(months=6))
        '''

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        This is called by the simulation whenever a new person is born.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """

        pass

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug('This is Diarrhoea, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)


class DiarrhoeaEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=2))

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props
        m = self.module
        rng = m.rng

        no_diarrhoea = df.is_alive & (df.gi_diarrhoea_status == False) & (df.age_exact_years < 1)
        no_diarrhoea0 = df.is_alive & (df.gi_diarrhoea_status == False) & (df.age_exact_years < 1)
        no_diarrhoea1 = df.is_alive & (df.gi_diarrhoea_status == False) & (df.age_years > 1) & (df.age_years > 2)
        no_diarrhoea2 = df.is_alive & (df.gi_diarrhoea_status == False) & (df.age_years > 2) & (df.age_years < 5)
        no_diarrhoea_under5 = df.is_alive & (df.gi_diarrhoea_status == False) & (df.age_years < 5)
        current_no_diarrhoea = df.index[no_diarrhoea_under5]

        diarrhoea_rotavirus0 = pd.Series(m.base_incidence_diarrhoea_by_rotavirus[0],
                                         index=df.index[no_diarrhoea_under5])
        diarrhoea_rotavirus1 = pd.Series(m.base_incidence_diarrhoea_by_rotavirus[1],
                                         index=df.index[no_diarrhoea_under5])
        diarrhoea_rotavirus2 = pd.Series(m.base_incidence_diarrhoea_by_rotavirus[2],
                                         index=df.index[no_diarrhoea_under5])

        diarrhoea_shigella0 = pd.Series(m.base_incidence_diarrhoea_by_shigella[0],
                                        index=df.index[no_diarrhoea_under5])
        diarrhoea_shigella1 = pd.Series(m.base_incidence_diarrhoea_by_shigella[1],
                                        index=df.index[no_diarrhoea_under5])
        diarrhoea_shigella2 = pd.Series(m.base_incidence_diarrhoea_by_shigella[2],
                                        index=df.index[no_diarrhoea_under5])

        diarrhoea_adenovirus0 = pd.Series(m.base_incidence_diarrhoea_by_adenovirus[0],
                                          index=df.index[no_diarrhoea_under5])
        diarrhoea_adenovirus1 = pd.Series(m.base_incidence_diarrhoea_by_adenovirus[1],
                                          index=df.index[no_diarrhoea_under5])
        diarrhoea_adenovirus2 = pd.Series(m.base_incidence_diarrhoea_by_adenovirus[2],
                                          index=df.index[no_diarrhoea_under5])

        diarrhoea_crypto0 = pd.Series(m.base_incidence_diarrhoea_by_crypto[0],
                                      index=df.index[no_diarrhoea_under5])
        diarrhoea_crypto1 = pd.Series(m.base_incidence_diarrhoea_by_crypto[1],
                                      index=df.index[no_diarrhoea_under5])
        diarrhoea_crypto2 = pd.Series(m.base_incidence_diarrhoea_by_crypto[2],
                                      index=df.index[no_diarrhoea_under5])

        diarrhoea_campylo0 = pd.Series(m.base_incidence_diarrhoea_by_campylo[0],
                                       index=df.index[no_diarrhoea_under5])
        diarrhoea_campylo1 = pd.Series(m.base_incidence_diarrhoea_by_campylo[1],
                                       index=df.index[no_diarrhoea_under5])
        diarrhoea_campylo2 = pd.Series(m.base_incidence_diarrhoea_by_campylo[2],
                                       index=df.index[no_diarrhoea_under5])

        diarrhoea_ETEC0 = pd.Series(m.base_incidence_diarrhoea_by_ETEC[0],
                                    index=df.index[no_diarrhoea_under5])
        diarrhoea_ETEC1 = pd.Series(m.base_incidence_diarrhoea_by_ETEC[1],
                                    index=df.index[no_diarrhoea_under5])
        diarrhoea_ETEC2 = pd.Series(m.base_incidence_diarrhoea_by_ETEC[2],
                                    index=df.index[no_diarrhoea_under5])

        eff_prob_gi_diarrhoea0 = pd.concat([diarrhoea_rotavirus0, diarrhoea_shigella0, diarrhoea_adenovirus0,
                                            diarrhoea_crypto0, diarrhoea_campylo0, diarrhoea_ETEC0], axis=1)
        eff_prob_gi_diarrhoea1 = pd.concat([diarrhoea_rotavirus1, diarrhoea_shigella1, diarrhoea_adenovirus1,
                                            diarrhoea_crypto1, diarrhoea_campylo1, diarrhoea_ETEC1], axis=1)
        eff_prob_gi_diarrhoea2 = pd.concat([diarrhoea_rotavirus2, diarrhoea_shigella2, diarrhoea_adenovirus2,
                                            diarrhoea_crypto2, diarrhoea_campylo2, diarrhoea_ETEC2], axis=1)

        # for age 0-11 months
        eff_prob_gi_diarrhoea0.loc[no_diarrhoea0 & df.li_no_access_handwashing == False] \
            *= m.rr_gi_diarrhoea_HHhandwashing
        eff_prob_gi_diarrhoea0.loc[no_diarrhoea0 & df.li_no_clean_drinking_water == False] \
            *= m.rr_gi_diarrhoea_clean_water
        eff_prob_gi_diarrhoea0.loc[no_diarrhoea0 & df.li_unimproved_sanitation == False] \
            *= m.rr_gi_diarrhoea_improved_sanitation
        eff_prob_gi_diarrhoea0.loc[no_diarrhoea0 & (df.has_hiv == True)] \
            *= m.rr_gi_diarrhoea_HIV
        eff_prob_gi_diarrhoea0.loc[no_diarrhoea0 & df.malnutrition == True] \
            *= m.rr_gi_diarrhoea_SAM
        eff_prob_gi_diarrhoea0.loc[no_diarrhoea0 & df.exclusive_breastfeeding == True & (df.age_exact_years <= 0.5)] \
            *= m.rr_gi_diarrhoea_excl_breast
        eff_prob_gi_diarrhoea0.loc[no_diarrhoea0 & df.continued_breastfeeding == True & (df.age_exact_years > 0.5) &
                                   (df.age_exact_years < 2)] *= m.rr_gi_diarrhoea_cont_breast

        # for age 12-23 months
        eff_prob_gi_diarrhoea1.loc[no_diarrhoea1 & df.li_no_access_handwashing == False] \
            *= m.rr_gi_diarrhoea_HHhandwashing
        eff_prob_gi_diarrhoea1.loc[no_diarrhoea1 & df.li_no_clean_drinking_water == False] \
            *= m.rr_gi_diarrhoea_clean_water
        eff_prob_gi_diarrhoea1.loc[no_diarrhoea1 & df.li_unimproved_sanitation == False] \
            *= m.rr_gi_diarrhoea_improved_sanitation
        eff_prob_gi_diarrhoea1.loc[no_diarrhoea1 & (df.has_hiv == True)] \
            *= m.rr_gi_diarrhoea_HIV
        eff_prob_gi_diarrhoea1.loc[no_diarrhoea1 & df.malnutrition == True] \
            *= m.rr_gi_diarrhoea_SAM
        eff_prob_gi_diarrhoea1.loc[no_diarrhoea1 & df.continued_breastfeeding == True & (df.age_exact_years > 0.5) &
                                   (df.age_exact_years < 2)] *= m.rr_gi_diarrhoea_cont_breast

        # for age 24-59 months
        eff_prob_gi_diarrhoea2.loc[no_diarrhoea2 & df.li_no_access_handwashing == False] \
            *= m.rr_gi_diarrhoea_HHhandwashing
        eff_prob_gi_diarrhoea2.loc[no_diarrhoea2 & df.li_no_clean_drinking_water == False] \
            *= m.rr_gi_diarrhoea_clean_water
        eff_prob_gi_diarrhoea2.loc[no_diarrhoea2 & df.li_unimproved_sanitation == False] \
            *= m.rr_gi_diarrhoea_improved_sanitation
        eff_prob_gi_diarrhoea2.loc[no_diarrhoea2 & (df.has_hiv == True)] \
            *= m.rr_gi_diarrhoea_HIV
        eff_prob_gi_diarrhoea2.loc[no_diarrhoea2 & df.malnutrition == True] \
            *= m.rr_gi_diarrhoea_SAM

        random_draw = pd.Series(rng.random_sample(size=len(no_diarrhoea0)), index=eff_prob_gi_diarrhoea)
        eff_prob_gi_diarrhoea['random_draw'] = random_draw

        incident_diarrh_rotavirus_idx = eff_prob_gi_diarrhoea.index['rotavirus 0-11' > random_draw]


        idx_get_acute_diarrhoea = eff_prob_gi_diarrhoea.index[incident_acute_diarrhoea]
        df.loc[idx_get_acute_diarrhoea, 'gi_diarrhoea_status'] = True

        # WHEN THEY GET ACUTE WATERY DIARRHOEA - DATE
        random_draw_days = np.random.randint(0, 60, size=len(incident_acute_diarrhoea))
        adding_days = pd.to_timedelta(random_draw_days, unit='d')
        date_of_aquisition = self.sim.date + adding_days
        df.loc[idx_get_acute_diarrhoea, 'date_of_onset_diarrhoea'] = date_of_aquisition

        # # # # # # ASSIGN DEHYDRATION LEVELS FOR ACUTE WATERY DIARRHOEA # # # # # #

        under5_acute_diarrhoea_idx = df.index[
            (df.age_years < 5) & df.is_alive & (df.gi_diarrhoea_status == 'acute watery diarrhoea')]

        eff_prob_some_dehydration_acute_diarrhoea = pd.Series(0.5, index=under5_acute_diarrhoea_idx)
        eff_prob_severe_dehydration_acute_diarrhoea = pd.Series(0.3, index=under5_acute_diarrhoea_idx)
        random_draw_b = pd.Series(self.sim.rng.random_sample(size=len(under5_acute_diarrhoea_idx)),
                                  index=under5_acute_diarrhoea_idx)

        no_dehydration_acute_diarrhoea = \
            1 - (eff_prob_some_dehydration_acute_diarrhoea + eff_prob_severe_dehydration_acute_diarrhoea)
        some_dehydration_acute_diarrhoea = \
            (random_draw_b > no_dehydration_acute_diarrhoea) & \
            (random_draw_b < (no_dehydration_acute_diarrhoea + eff_prob_some_dehydration_acute_diarrhoea))
        severe_dehydration_acute_diarrhoea = \
            ((no_dehydration_acute_diarrhoea + eff_prob_some_dehydration_acute_diarrhoea) < random_draw_b) & \
            ((no_dehydration_acute_diarrhoea + eff_prob_some_dehydration_acute_diarrhoea +
              eff_prob_severe_dehydration_acute_diarrhoea) > random_draw_b)

        idx_some_dehydration_acute_diarrhoea = eff_prob_some_dehydration_acute_diarrhoea.index[
            some_dehydration_acute_diarrhoea]
        idx_severe_dehydration_acute_diarrhoea = eff_prob_severe_dehydration_acute_diarrhoea.index[
            severe_dehydration_acute_diarrhoea]

        df.loc[idx_some_dehydration_acute_diarrhoea, 'gi_dehydration_status'] = 'some dehydration'
        df.loc[idx_severe_dehydration_acute_diarrhoea, 'gi_dehydration_status'] = 'severe dehydration'

        # # # # # # # # SYMPTOMS FROM ACUTE WATERY DIARRHOEA # # # # # # # # # # # # # # # # # # # # # # #
        df.loc[idx_get_acute_diarrhoea, 'di_diarrhoea_loose_watery_stools'] = True
        df.loc[idx_get_acute_diarrhoea, 'di_blood_in_stools'] = False
        df.loc[idx_get_acute_diarrhoea, 'di_diarrhoea_over14days'] = False

        # --------------------------------------------------------------------------------------------------------
        # SEEKING CARE FOR ACUTE WATERY DIARRHOEA
        # --------------------------------------------------------------------------------------------------------

        acute_diarrhoea_symptoms = \
            df.index[df.is_alive & (df.age_years < 5) & (df.di_diarrhoea_loose_watery_stools == True) &
                     (df.di_blood_in_stools == False) & df.di_diarrhoea_over14days == False]

        seeks_care = pd.Series(data=False, index=acute_diarrhoea_symptoms)
        for individual in acute_diarrhoea_symptoms:
            prob = self.sim.modules['HealthSystem'].get_prob_seek_care(individual, symptom_code=1)
            seeks_care[individual] = self.module.rng.rand() < prob
            event = HSI_Sick_Child_Seeks_Care_From_HSA(self.module, person_id=individual)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=2,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(weeks=2)
                                                                )


class DeathDiarrhoeaEvent(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, population):

        df = population.props
        m = self.module
        rng = m.rng

        # ------------------------------------------------------------------------------------------------------
        # DEATH DUE TO ACUTE BLOODY DIARRHOEA - DYSENTERY
        # ------------------------------------------------------------------------------------------------------
        eff_prob_death_dysentery = \
            pd.Series(m.r_death_dysentery,
                      index=df.index[df.is_alive & (df.gi_diarrhoea_status == 'dysentery') & (df.age_years < 5)])

        eff_prob_death_dysentery.loc[df.is_alive & (df.gi_diarrhoea_status == 'dysentery') &
                                     (df.age_exact_years >= 1) & (df.age_exact_years < 2)] *= \
            m.rr_death_dysentery_age12to23mo
        eff_prob_death_dysentery.loc[df.is_alive & (df.gi_diarrhoea_status == 'dysentery') &
                                     (df.age_exact_years >= 2) & (df.age_exact_years < 5)] *= \
            m.rr_death_dysentery_age24to59mo
        eff_prob_death_dysentery.loc[df.is_alive & (df.gi_diarrhoea_status == 'dysentery') &
                                     (df.has_hiv == True) & (df.age_exact_years < 5)] *= m.rr_death_dysentery_HIV
        eff_prob_death_dysentery.loc[df.is_alive & (df.gi_diarrhoea_status == 'dysentery') &
                                     df.malnutrition == True & (df.age_exact_years < 5)] *= m.rr_death_dysentery_SAM

        under5_dysentery_idx = df.index[(df.age_years < 5) & df.is_alive & (df.gi_diarrhoea_status == 'dysentery')]

        random_draw = pd.Series(rng.random_sample(size=len(under5_dysentery_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive &
                                               (df.gi_diarrhoea_status == 'dysentery')])
        dfx = pd.concat([eff_prob_death_dysentery, random_draw], axis=1)
        dfx.columns = ['eff_prob_death_dysentery', 'random_draw']

        for person_id in under5_dysentery_idx:
            if dfx.index[dfx.eff_prob_death_dysentery > dfx.random_draw]:
                df.at[person_id, 'gi_diarrhoea_death'] = True
            else:
                df.at[person_id, 'gi_diarrhoea_status'] = 'none'

        # ------------------------------------------------------------------------------------------------------
        # DEATH DUE TO ACUTE WATERY DIARRHOEA
        # ------------------------------------------------------------------------------------------------------

        eff_prob_death_acute_diarrhoea = \
            pd.Series(m.r_death_acute_diarrhoea,
                      index=df.index[
                          df.is_alive & (df.gi_diarrhoea_status == 'acute watery diarrhoea') & (df.age_years < 5)])
        eff_prob_death_acute_diarrhoea.loc[df.is_alive & (df.gi_diarrhoea_status == 'acute watery diarrhoea') &
                                           (df.age_exact_years >= 1) & (df.age_exact_years < 2)] *= \
            m.rr_death_acute_diar_age12to23mo
        eff_prob_death_acute_diarrhoea.loc[df.is_alive & (df.gi_diarrhoea_status == 'acute watery diarrhoea') &
                                           (df.age_exact_years >= 2) & (df.age_exact_years < 5)] *= \
            m.rr_death_acute_diar_age24to59mo
        eff_prob_death_acute_diarrhoea.loc[df.is_alive & (df.gi_diarrhoea_status == 'acute watery diarrhoea') &
                                           (df.has_hiv == True) & (df.age_exact_years < 5)] *= \
            m.rr_death_acute_diar_HIV
        eff_prob_death_acute_diarrhoea.loc[df.is_alive & (df.gi_diarrhoea_status == 'acute watery diarrhoea') &
                                           df.malnutrition == True & (df.age_exact_years < 5)] *= \
            m.rr_death_acute_diar_SAM

        under5_acute_diarrhoea_idx = df.index[(df.age_years < 5) & df.is_alive &
                                              (df.gi_diarrhoea_status == 'acute watery diarrhoea')]

        random_draw = pd.Series(rng.random_sample(size=len(under5_acute_diarrhoea_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive &
                                               (df.gi_diarrhoea_status == 'acute watery diarrhoea')])
        dfx = pd.concat([eff_prob_death_acute_diarrhoea, random_draw], axis=1)
        dfx.columns = ['eff_prob_death_acute_diarrhoea', 'random_draw']

        for person_id in under5_acute_diarrhoea_idx:
            if dfx.index[dfx.eff_prob_death_acute_diarrhoea > dfx.random_draw]:
                df.at[person_id, 'gi_diarrhoea_death'] = True
            else:
                df.at[person_id, 'gi_diarrhoea_status'] = 'none'

        # ------------------------------------------------------------------------------------------------------
        # DEATH DUE TO PERSISTENT DIARRHOEA
        # ------------------------------------------------------------------------------------------------------

        eff_prob_death_persistent_diarrhoea = \
            pd.Series(m.r_death_persistent_diarrhoea,
                      index=df.index[
                          df.is_alive & (df.gi_diarrhoea_status == 'persistent diarrhoea') & (df.age_years < 5)])
        eff_prob_death_persistent_diarrhoea.loc[df.is_alive & (df.gi_diarrhoea_status == 'persistent diarrhoea') &
                                                (df.age_exact_years >= 1) & (df.age_exact_years < 2)] *= \
            m.rr_death_persistent_diar_age12to23mo
        eff_prob_death_persistent_diarrhoea.loc[df.is_alive & (df.gi_diarrhoea_status == 'persistent diarrhoea') &
                                                (df.age_exact_years >= 2) & (df.age_exact_years < 5)] *= \
            m.rr_death_persistent_diar_age24to59mo
        eff_prob_death_persistent_diarrhoea.loc[df.is_alive & (df.gi_diarrhoea_status == 'persistent diarrhoea') &
                                                (df.has_hiv == True) & (df.age_exact_years < 5)] *= \
            m.rr_death_persistent_diar_HIV
        eff_prob_death_persistent_diarrhoea.loc[df.is_alive & (df.gi_diarrhoea_status == 'persistent diarrhoea') &
                                                df.malnutrition == True & (df.age_exact_years < 5)] *= \
            m.rr_death_persistent_diar_SAM

        under5_persistent_diarrhoea_idx = df.index[(df.age_years < 5) & df.is_alive &
                                                   (df.gi_diarrhoea_status == 'persistent diarrhoea')]

        random_draw = pd.Series(rng.random_sample(size=len(under5_persistent_diarrhoea_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive &
                                               (df.gi_diarrhoea_status == 'persistent diarrhoea')])
        dfx = pd.concat([eff_prob_death_persistent_diarrhoea, random_draw], axis=1)
        dfx.columns = ['eff_prob_death_persistent_diarrhoea', 'random_draw']

        for person_id in under5_persistent_diarrhoea_idx:
            if dfx.index[dfx.eff_prob_death_persistent_diarrhoea > dfx.random_draw]:
                df.at[person_id, 'gi_diarrhoea_death'] = True
            else:
                df.at[person_id, 'ei_diarrhoea_status'] = 'none'

        death_this_period = df.index[(df.gi_diarrhoea_death == True)]
        for individual_id in death_this_period:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id, 'ChildhoodDiarrhoea'),
                                    self.sim.date)
