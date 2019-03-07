"""
Childhood Respiratory Infection module
Documentation: 04 - Methods Repository/Method_Child_RespiratoryInfection.xlsx
"""
import logging

import pandas as pd
from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ChildhoodPneumonia (Module):

    PARAMETERS = {
        'initial_prev_cold_base': Parameter(Types.REAL, 'initial prevalence of common cold, among children '
                                                        'aged 2-11 months, HIV negative, '
                                                        'no SAM, no indoor air pollution '
                                                            ),
        'rp_cold_agelt2mo': Parameter(Types.REAL, 'relative prevalence of common cold for age < 2 months'
                                              ),
        'rp_cold_age_12to23mo': Parameter(Types.REAL, 'relative prevalence of common cold for age 12 to 23 months'
                                             ),
        'rp_cold_age_24to59mo': Parameter(Types.REAL, 'relative prevalence of common cold for age 24 to 59 months'
                                             ),
        'rp_cold_HIV': Parameter(Types.REAL, 'relative prevalence of common cold for HIV positive'
                                             ),
        'rp_cold_malnutrition': Parameter(Types.REAL, 'relative prevalence of common cold for severe acute malnutrition'
                                                    ),
        'rp_cold_indoor_air_pollution': Parameter(Types.REAL, 'relative prevalence of common cold '
                                                              'for indoor air pollution' ),
        'base_incidence_ri_cold': Parameter(Types.REAL, 'baseline incidence of common cold, among children '
                                                        'aged 2-11 months, HIV negative, '
                                                        'no SAM, no indoor air pollution '
                                                            ),
        'rr_cold_agelt2mo': Parameter(Types.REAL, 'relative rate of common cold for age < 2 months'
                                      ),
        'rr_cold_age_12to23mo': Parameter(Types.REAL, 'relative rate of common cold for age 12 to 23 months'
                                          ),
        'rr_cold_age_24to59mo': Parameter(Types.REAL, 'relative rate of common cold for age 24 to 59 months'
                                          ),
        'rr_cold_HIV': Parameter(Types.REAL, 'relative rate of common cold for HIV positive'
                                 ),
        'rr_cold_malnutrition': Parameter(Types.REAL, 'relative rate of common cold for severe acute malnutrition'
                                          ),
        'rr_cold_indoor_air_pollution': Parameter(Types.REAL,
                                                  'relative rate of common cold for indoor air pollution'
                                                  ),
        'initial_prev_pneumonia_base': Parameter(Types.REAL, 'initial prevalence of pneumonia, among children '
                                                             'aged 2-11 months, HIV negative, '
                                                             'no SAM, no indoor air pollution '
                                                    ),
        'rp_pneumonia_agelt2mo': Parameter(Types.REAL, 'relative prevalence of pneumonia for age < 2 months'
                                          ),
        'rp_pneumonia_age_12to23mo': Parameter(Types.REAL, 'relative prevalence of pneumonia for age 12 to 23 months'
                                             ),
        'rp_pneumonia_age_24to59mo': Parameter(Types.REAL, 'relative prevalence of pneumonia for age 24 to 59 months'
                                             ),
        'rp_pneumonia_HIV': Parameter(Types.REAL, 'relative prevalence of pneumonia for HIV positive'
                                 ),
        'rp_pneumonia_malnutrition': Parameter(Types.REAL, 'relative prevalence of pneumonia '
                                                           'for severe acute malnutrition'
                                          ),
        'rp_pneumonia_indoor_air_pollution': Parameter(Types.REAL, 'relative prevalence of pneumonia '
                                                                   'for indoor air pollution'
                                                  ),
        'base_incidence_ri_pneumonia': Parameter(Types.REAL, 'baseline incidence of common cold, among children '
                                                             'aged 2-11 months, HIV negative,'
                                                             'no SAM, no indoor air pollution '
                                            ),
        'rr_pneumonia_agelt2mo': Parameter(Types.REAL, 'relative rate of pneumonia for age < 2 months'
                                           ),
        'rr_pneumonia_age_12to23mo': Parameter(Types.REAL, 'relative rate of pneumonia for age 12 to 23 months'
                                               ),
        'rr_pneumonia_age_24to59mo': Parameter(Types.REAL, 'relative rate of pneumonia for age 24 to 59 months'
                                               ),
        'rr_pneumonia_HIV': Parameter(Types.REAL, 'relative rate of pneumonia for HIV positive'
                                      ),
        'rr_pneumonia_malnutrition': Parameter(Types.REAL,
                                               'relative rate of pneumonia for severe acute malnutrition'
                                               ),
        'rr_pneumonia_indoor_air_pollution': Parameter(Types.REAL,
                                                       'relative rate of pneumonia for indoor air pollution'
                                                       ),
        'initial_prev_severe_pneumonia_base': Parameter(Types.REAL, 'inicial prevalence of severe pneumonia,'
                                                                    'among children aged 3-11 months, HIV negative, '
                                                                    'normal weight, no SAM, no indoor air pollution '
                                                   ),
        'rp_severe_pneumonia_agelt2mo': Parameter(Types.REAL, 'relative prevalence of severe pneumonia'
                                                              ' for age <2 months'
                                                          ),
        'rp_severe_pneumonia_age_12to23mo': Parameter(Types.REAL, 'relative prevalence of severe pneumonia '
                                                                  'for age 12 to 23 months'
                                                         ),
        'rp_severe_pneumonia_age_24to59mo': Parameter(Types.REAL, 'relative prevalence of severe pneumonia'
                                                                  ' for age 24 to 59 months'
                                                         ),
        'rp_severe_pneumonia_HIV': Parameter(Types.REAL, 'relative prevalence of severe pneumonia '
                                                         'for HIV positive status'
                                      ),
        'rp_severe_pneumonia_malnutrition': Parameter(Types.REAL, 'relative prevalence of severe pneumonia '
                                                                  'for severe acute malnutrition'
                                               ),
        'rp_severe_pneumonia_indoor_air_pollution': Parameter(Types.REAL, 'relative prevalence of severe pneumonia '
                                                                          'for indoor air pollution'
                                                       ),
        'rr_severe_pneumonia_agelt2mo': Parameter(Types.REAL, 'relative rate of severe pneumonia for age <2 months'
                                                  ),
        'rr_severe_pneumonia_age_12to23mo': Parameter(Types.REAL,
                                                      'relative rate of severe pneumonia for age 12 to 23 months'),

        'rr_severe_pneumonia_age_24to59mo': Parameter(Types.REAL, 'relative rate of severe pneumonia'
                                                                  ' for age 24 to 59 months'
                                                      ),
        'rr_severe_pneumonia_HIV': Parameter(Types.REAL, 'relative rate of severe pneumonia for HIV positive status'
                                             ),
        'rr_severe_pneumonia_malnutrition': Parameter(Types.REAL, 'relative prevalence of severe pneumonia '
                                                                  'for severe acute malnutrition'
                                                      ),
        'rr_severe_pneumonia_indoor_air_pollution': Parameter(Types.REAL, 'relative prevalence of severe pneumonia '
                                                                          'for indoor air pollution'
                                                              ),
        'r_progress_to_severe_pneumonia': Parameter(Types.REAL, 'probability of progressing from pneumonia to severe pneumonia'
                                                    'among children aged 2-11 months, HIV negative, normal weight,'
                                                    'no SAM, no indoor air pollution'
                                                    ),
        'rr_progress_severe_pneumonia_agelt2mo': Parameter(Types.REAL, 'relative rate of progression '
                                                                       'to severe pneumonia for age <2 months'
                                                      ),
        'rr_progress_severe_pneumonia_age_12to23mo': Parameter(Types.REAL, 'relative rate of progression '
                                                                           'to severe pneumonia for age 12 to 23 months'
                                                         ),
        'rr_progress_severe_pneumonia_age_24to59mo': Parameter(Types.REAL, 'relative rate of progression '
                                                                           'to severe pneumonia for age 24 to 59 months'
                                                         ),
        'rr_progress_severe_pneumonia_HIV': Parameter(Types.REAL, 'relative risk of progression to '
                                                                  'severe pneumonia for HIV positive status'
                                             ),
        'rr_progress_severe_pneumonia_malnutrition': Parameter(Types.REAL, 'relative rate of progression to '
                                                                           'severe pneumonia for severe acute malnutrition'
                                                      ),
        'rr_progress_severe_pneumonia_indoor_air_pollution': Parameter(Types.REAL, 'relative risk of progression to '
                                                                                   'severe pneumonia for indoor air pollution'
                                                              ),
        'r_death_pneumonia': Parameter(Types.REAL, 'death rate from pneumonia among children aged 2-11 months,'
                                                   'HIV negative, no SAM, no indoor air pollution'),
        'rr_death_pneumonia_age_<2months': Parameter(Types.REAL, 'relative risk of common cold for age < 2 months'
                                          ),
        'rr_death_pneumonia_age_12-23months': Parameter(Types.REAL, 'relative risk of death from pneumonia '
                                                                    'for age 12 to 23 months'
                                             ),
        'rr_death_pneumonia_age_24-59months': Parameter(Types.REAL, 'relative risk of death from pneumonia '
                                                                    'for age 24 to 59 months'
                                             ),
        'rr_death_pneumonia_HIV': Parameter(Types.REAL, 'relative risk of death from pneumonia for HIV positive'
                                 ),
        'rr_death_pneumonia_malnutrition': Parameter(Types.REAL, 'relative risk of death from pneumonia '
                                                                 'for severe acute malnutrition'
                                          ),
        'rr_death_pneumonia_indoor_air_pollution': Parameter(Types.REAL, 'relative risk of death from pneumonia '
                                                                         'for indoor air pollution'
                                                  ),
        'rr_death_pneumonia_treatment_adherence': Parameter(Types.REAL, 'relative risk of death from pneumonia '
                                                                        'for completed treatment'),
        'r_recovery_cold': Parameter(Types.REAL, 'recovery rate from common cold among children aged 2-11 months,'
                                                 ' HIV negative, no SAM, no indoor air pollution'
                                     ),
        'rr_recovery_cold_agelt2mo': Parameter(Types.REAL, 'relative rate of recovery from common cold '
                                                           'for age < 2 months'
                                          ),
        'rr_recovery_cold_age_12to23mo': Parameter(Types.REAL, 'relative rate of recovery from common cold '
                                                               'for age 12 to 23 months'
                                             ),
        'rr_recovery_cold_age_24to59months': Parameter(Types.REAL, 'relative rate of recovery from common cold '
                                                                   'for age 24 to 59 months'
                                             ),
        'rr_recovery_cold_HIV': Parameter(Types.REAL, 'relative rate of recovery from common cold for HIV positive'
                                 ),
        'rr_recovery_cold_malnutrition': Parameter(Types.REAL, 'relative rate of recovery from common cold '
                                                               'for severe acute malnutrition'
                                          ),
        'rr_recovery_cold_indoor_air_pollution': Parameter(Types.REAL, 'relative rate of recovery from common cold '
                                                                       'for indoor air pollution'
                                                  ),
        'r_recovery_pneumonia': Parameter(Types.REAL, 'recovery rate from pneumonia among children aged 2-11 months,'
                                                      ' HIV negative, no SAM, no indoor air pollution '),
        'rr_recovery_pneumonia_agelt2mo': Parameter(Types.REAL,
                                                        'relative rate of recovery from pneumonia for age < 2 months'
                                                        ),
        'rr_recovery_pneumonia_age_12to23mo': Parameter(Types.REAL, 'relative rate of recovery from pneumonia '
                                                           'for age between 12 to 23 months'
                                                           ),
        'rr_recovery_pneumonia_age_24to59mo': Parameter(Types.REAL, 'relative rate of recovery from pneumonia '
                                                                    'for age between 24 to 59 months'
                                                           ),
        'rr_recovery_pneumonia_HIV': Parameter(Types.REAL,
                                               'relative rate of recovery from pneumonia for HIV positive status'
                                               ),
        'rr_recovery_pneumonia_malnutrition': Parameter(Types.REAL, 'relative rate of recovery from pneumonia '
                                                                    'for acute malnutrition'
                                                        ),
        'rr_recovery_pneumonia_treatment_adherence': Parameter(Types.REAL,
                                                               'relative rate of recovery from pneumonia if incompleted treatment'
                                                               ),
        'r_recovery_severe_pneumonia': Parameter(Types.REAL, 'baseline recovery rate from severe pneumonia '
                                                             'among children ages 2 to 11 months,'
                                                             'HIV negative, no SAM, no indoor air pollution'),
        'rr_recovery_severe_pneumonia_agelt2mo': Parameter(Types.REAL, 'relative rate of recovery from severe pneumonia'
                                                                       ' for age <2 months'
                                                          ),
        'rr_recovery_severe_pneumonia_age_12to23mo': Parameter(Types.REAL, 'relative rate of recovery from'
                                                                  ' severe pneumonia for age between 12 to 23 months'
                                                         ),
        'rr_recovery_severe_pneumonia_age_24to59mo': Parameter(Types.REAL, 'relative rate of recovery from '
                                                                           'severe pneumonia for age between 24 to 59 months'
                                                 ),
        'rr_recovery_severe_pneumonia_HIV': Parameter(Types.REAL, 'relative rate of recovery from severe pneumonia'
                                                                  ' for HIV positive status'
                                          ),
        'rr_recovery_severe_pneumonia_malnutrition': Parameter(Types.REAL, 'relative rate of recovery from '
                                                                           'severe pneumonia for acute malnutrition'
                                                   ),
        'rr_recovery_severe_pneumonia_treatment_adherence': Parameter(Types.REAL, 'relative rate of recovery from '
                                                                                  'severe pneumonia if incompleted treatment'
                                                               ),
        'init_prop_resp_infection_status': Parameter(Types.REAL, 'initial proportions in ri_respiratory_infection_status categories '
                                                                 'for children aged 2-11 months, HIV negative,'
                                                                 'no SAM, no indoor air pollution')
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'ri_respiratory_infection_status': Property(Types.CATEGORICAL, 'respiratory infection status',
                                  categories=['none', 'common cold', 'pneumonia', 'severe pneumonia']),
        'ri_cough': Property(Types.BOOL, 'respiratory infection symptoms cough'),
        'ri_fever': Property(Types.BOOL, 'respiratory infection symptoms fever'),
        'ri_fast_breathing': Property(Types.BOOL, 'respiratory infection symptoms fast breathing'),
        'ri_chest_indraw': Property(Types.BOOL, 'respiratory infection symptoms chest indrawing'),
        'ri_stridor': Property(Types.BOOL, 'respiratory infection symptoms stridor'),
        'ri_not_able_drink_breastfeed': Property(Types.BOOL, 'resp infection symptoms not able to drink or breastfeed'),
        'ri_convulsions': Property(Types.BOOL, 'respiratory infection symptoms convulsions'),
        'ri_lethargic_unconscious': Property(Types.BOOL, 'respiratory infection symptoms lethargic or unconscious'),
        'ri_diagnosis': Property(Types.BOOL, 'respiratory infection diagnosis'),
        'has_hiv': Property(Types.BOOL, 'temporary property - has hiv'),
        'ri_weight': Property(Types.CATEGORICAL, 'weight',
                                                    categories=['normal', 'underweight', 'severely underweight'
                                                            ]),
        'malnutrition': Property(Types.BOOL, 'temporary property - malnutrition status'),
        'indoor_air_pollution': Property(Types.BOOL, 'temporary property - indoor air pollution'),
    }

    def read_parameters(self, data_folder):
        """ Setup parameters values used by the module
        """
        p = self.parameters
        p['initial_prev_cold_base'] = 0.2
        p['rp_cold_agelt2mo'] = 1.2
        p['rp_cold_age12to23mo'] = 0.8
        p['rp_cold_age24to59mo'] = 0.5
        p['rp_cold_HIV'] = 1.3
        p['rp_cold_weight_moderate_underw'] = 1.1
        p['rp_cold_weight_severe_underw'] = 1.2
        p['rp_cold_malnutrition'] = 1.2
        p['rp_cold_indoor_air_pollution'] = 1.05
        p['initial_prev_pneumonia_base'] = 0.2
        p['rp_pneumonia_agelt2mo'] = 1.2
        p['rp_pneumonia_age12to23mo'] = 0.8
        p['rp_pneumonia_age24to59mo'] = 0.5
        p['rp_pneumonia_HIV'] = 1.4
        p['rp_pneumonia_weight_moderate_underw'] = 1.1
        p['rp_pneumonia_weight_severe_underw'] = 1.2
        p['rp_pneumonia_malnutrition'] = 1.25
        p['rp_pneumonia_indoor_air_pollution'] = 1.1
        p['initial_prev_severe_pneumonia'] = 0.1
        p['rp_severe_pneumonia_agelt2mo'] = 1.3
        p['rp_severe_pneumonia_age12to23mo'] = 0.8
        p['rp_severe_pneumonia_age24to59mo'] = 0.5
        p['rp_severe_pneumonia_HIV'] = 1.3
        p['rp_severe_pneumonia_weight_moderate_underw'] = 1.1
        p['rp_severe_pneumonia_weight_severe_underw'] = 1.2
        p['rp_severe_pneumonia_malnutrition'] = 1.3
        p['rp_severe_pneumonia_indoor_air_pollution'] = 1.1
        p['r_progress_to_severe_pneumonia'] = 0.05
        p['rr_progress_severe_pneumonia_age_<2months'] = 1.3
        p['rr_progress_severe_pneumonia_age_12-23months'] = 0.9
        p['rr_progress_severe_pneumonia_age_24-59months'] = 0.6
        p['rr_progress_severe_pneumonia_HIV'] = 1.2
        p['rr_progress_severe_pneumonia_weight_moderate_underw'] = 1.03
        p['rr_progress_severe_pneumonia_weight_severe_underw'] = 1.08
        p['rr_progress_severe_pneumonia_malnutrition'] = 1.1
        p['rr_progress_severe_pneumonia_indoor_air_pollution'] = 1.08
        p['r_death_pneumonia'] = 0.2
        p['rr_death_pneumonia_age_<2months'] = 1.2
        p['rr_death_pneumonia_age_12-23months'] = 0.8
        p['rr_death_pneumonia_age_24-59months'] = 0.04
        p['rr_death_pneumonia_HIV'] = 1.4
        p['rr_death_pneumonia_weight_moderate_underw'] = 1.1
        p['rr_death_pneumonia_weight_severe_underw'] = 1.2
        p['rr_death_pneumonia_malnutrition'] = 1.3
        p['rr_death_pneumonia_indoor_air_pollution'] = 1.1
        p['rr_death_pneumonia_treatment_adherence'] = 1.4
        p['r_recovery_cold'] = 0.9
        p['rr_recovery_cold_age_<2months'] = 0.6
        p['rr_recovery_cold_age_12-23months'] = 1.1
        p['rr_recovery_cold_age_24-59months'] = 1.7
        p['rr_recovery_cold_HIV'] = 0.8
        p['rr_recovery_cold_malnutrition'] = 0.8
        p['rr_recovery_cold_treatment_adherence'] = 0.8
        p['r_recovery_pneumonia'] = 0.5
        p['rr_recovery_pneumonia_age_under2months'] = 0.3
        p['rr_recovery_pneumonia_age_12-23months'] = 0.7
        p['rr_recovery_pneumonia_age_24-59months'] = 0.8
        p['rr_recovery_pneumonia_HIV'] = 0.3
        p['rr_recovery_pneumonia_malnutrition'] = 0.4
        p['rr_recovery_pneumonia_treatment_adherence'] = 0.6
        p['r_recovery_severe_pneumonia'] = 0.2
        p['rr_recovery_severe_pneumonia_age_under2months'] = 0.6
        p['rr_recovery_severe_pneumonia_age_12-23months'] = 1.2
        p['rr_recovery_severe_pneumonia_age_24-59months'] = 1.5
        p['rr_recovery_severe_pneumonia_HIV'] = 0.5
        p['rr_recovery_severe_pneumonia_malnutrition'] = 0.6
        p['rr_recovery_severe_pneumonia_treatment_adherence'] = 0.7
        p['init_prop_resp_infection_status'] = [0.50, 0.2, 0.2, 0.1]

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

        # -------------------- DEFAULTS ------------------------------------------------------------

        df['ri_respiratory_infection_status'] = 'none'
        df['ri_cough']= False
        df['ri_fever'] = False
        df['ri_fast_breathing'] = False
        df['ri_chest_indraw'] = False
        df['ri_stridor'] = False
        df['ri_not_able_drink_breastfeed'] = False
        df['ri_convulsions'] = False
        df['ri_lethargic_unconscious'] = False
        df['ri_diagnosis'] = False
        df['ri_weight'] = 'normal'
        df['malnutrition'] = False
        df['has_HIV'] = False
        df['indoor_air_pollution'] = False

        # -------------------- ASSIGN VALUES OF RESPIRATORY INFECTION STATUS AT BASELINE -----------

        agelt5_idx = df.index[(df.age_years < 5) & df.is_alive]

        # create dataframe of the probabilities of ri_respiratory_infection_status for children
        # aged 2-11 months, HIV negative, normal weight, no SAM, no indoor air pollution
        p_resp_infect_stat = pd.DataFrame(data=[m.init_prop_resp_infection_status],
                                     columns=['none', 'common cold', 'pneumonia',
                                              'severe pneumonia'], index=agelt5_idx)

        # create probabilities of cold for all age under 5
        p_resp_infect_stat.loc[(df.age_exact_years < 0.1667) & df.is_alive] *= m.rp_cold_agelt2mo
        p_resp_infect_stat.loc[(df.age_exact_years >= 1) & (df.age_exact_years < 2) & df.is_alive] *= m.rp_cold_age12to23mo
        p_resp_infect_stat.loc[(df.age_exact_years >= 2) & (df.age_exact_years < 5) & df.is_alive] *= m.rp_cold_age24to59mo
        p_resp_infect_stat.loc[(df.has_hiv == True)& (df.age_years < 5) & df.is_alive] *= m.rp_cold_HIV
        p_resp_infect_stat.loc[(df.ri_weight == 'underweight') & (df.age_years < 5) & df.is_alive] *= m.rp_cold_weight_moderate_underw
        p_resp_infect_stat.loc[(df.ri_weight == 'severely underweight') & (df.age_years < 5) & df.is_alive] *= m.rp_cold_weight_severe_underw
        p_resp_infect_stat.loc[(df.malnutrition == True) & (df.age_years < 5) & df.is_alive] *= m.rp_cold_malnutrition
        p_resp_infect_stat.loc[(df.indoor_air_pollution == True) & (df.age_years < 5) & df.is_alive] *= m.rp_cold_indoor_air_pollution

        # create probabilities of pneumonia for all age under 5
        p_resp_infect_stat.loc[(df.age_exact_years < 0.1667) & df.is_alive] *= m.rp_pneumonia_agelt2mo
        p_resp_infect_stat.loc[(df.age_exact_years >= 1) & (df.age_exact_years < 2) & df.is_alive] *= m.rp_pneumonia_age12to23mo
        p_resp_infect_stat.loc[(df.age_exact_years >= 2) & (df.age_exact_years < 5) & df.is_alive] *= m.rp_pneumonia_age24to59mo
        p_resp_infect_stat.loc[(df.has_hiv == True) & (df.age_years < 5) & df.is_alive] *= m.rp_pneumonia_HIV
        p_resp_infect_stat.loc[(df.ri_weight == 'underweight') & (df.age_years < 5) & df.is_alive] *= m.rp_pneumonia_weight_moderate_underw
        p_resp_infect_stat.loc[(df.ri_weight == 'severely underweight') & (df.age_years < 5) & df.is_alive] *= m.rp_pneumonia_weight_severe_underw
        p_resp_infect_stat.loc[(df.malnutrition == True) & (df.age_years < 5) & df.is_alive] *= m.rp_pneumonia_malnutrition
        p_resp_infect_stat.loc[(df.indoor_air_pollution == True) & (df.age_years <5) & df.is_alive] *= m.rp_pneumonia_indoor_air_pollution

        # create probabilities of severe pneumonia for all age under 5
        p_resp_infect_stat.loc[(df.age_exact_years < 0.1667) & df.is_alive] *= m.rp_severe_pneumonia_agelt2mo
        p_resp_infect_stat.loc[(df.age_exact_years >= 1) & (df.age_exact_years < 2) & df.is_alive] *= m.rp_severe_pneumonia_age12to23mo
        p_resp_infect_stat.loc[(df.age_exact_years >= 2) & (df.age_exact_years < 5) & df.is_alive] *= m.rp_severe_pneumonia_age24to59mo
        p_resp_infect_stat.loc[(df.has_hiv == True) & (df.age_years < 5) & df.is_alive] *= m.rp_severe_pneumonia_HIV
        p_resp_infect_stat.loc[(df.ri_weight == 'underweight') & (df.age_years < 5) & df.is_alive] *= m.rp_severe_pneumonia_weight_moderate_underw
        p_resp_infect_stat.loc[(df.ri_weight == 'severely underweight') & (df.age_years < 5) & df.is_alive] *= m.rp_severe_pneumonia_weight_severe_underw
        p_resp_infect_stat.loc[(df.malnutrition == True) & (df.age_years < 5) & df.is_alive] *= m.rp_severe_pneumonia_malnutrition
        p_resp_infect_stat.loc[(df.indoor_air_pollution == True) & (df.age_years < 5) & df.is_alive] *= m.rp_severe_pneumonia_indoor_air_pollution

        random_draw = pd.Series(rng.random_sample(size=len(agelt5_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive])

        # create a temporary dataframe called dfx to hold values of probabilities and random draw
        dfx = pd.concat([p_resp_infect_stat, random_draw], axis=1)
        dfx.columns = ['p_none', 'p_common_cold', 'p_pneumonia', 'p_severe_pneumonia']

    def initialise_simulation(self, sim):
        """
        Get ready for simulation start.
        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        # add the basic event
        event = RespInfectionEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(weeks=1))

        # add an event to log to screen
        sim.schedule_event(RespInfectionLoggingEvent(self), sim.date + DateOffset(weeks=4))

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        This is called by the simulation whenever a new person is born.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """

        df = self.sim.population.props

        df.at[child_id, 'ri_respiratory_infection_status'] = 'none'
        df.at[child_id, 'ri_cough'] = False
        df.at[child_id, 'ri_fever'] = False
        df.at[child_id, 'ri_fast_breathing'] = False
        df.at[child_id, 'ri_chest_indraw'] = False
        df.at[child_id, 'ri_stridor'] = False
        df.at[child_id, 'ri_not_able_drink_breastfeed'] = False
        df.at[child_id, 'ri_convulsions'] = False
        df.at[child_id, 'ri_lethargic_unconscious'] = False
        df.at[child_id, 'ri_diagnosis'] = False


class Event(RegularEvent, PopulationScopeEventMixin):
    """
    Regular event that updates all Respiratory Infection properties for population
    Regular events automatically reschedule themselves at a fixed frequency,
    and thus implement discrete timestep type behaviour. The frequency is
    specified when calling the base class constructor in our __init__ method.
    """
    def __init__(self, module):
        """schedule to run every 7 weeks
        note: if change this offset from 1 week need to consider code conditioning on age.years_exact
        We need to pass the frequency at which we want to occur to the base class
        constructor using super(). We also pass the module that created this event,
        so that random number generators can be scoped per-module.
        :param module: the module that created this event
        """
        super().__init__(module, frequency=DateOffset(weeks=1))

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props
        m = self.module
        rng = m.rng

        # -------------------- UPDATING OF RESPIRATORY INFECTION STATUS OVER TIME -----------------------------------

        # updating for children under 5 with current status 'none'

        ri_current_none_idx = df.index[df.is_alive & (df.ri_resp_infection_stat == 'none')
                                                & (df.age_years < 5)]
        ri_current_none_agelt2mo_idx = df.index[df.is_alive & (df.ri_resp_infection_stat == 'none')
                                                & (df.age_exact_years < 0.1667)]
        ri_current_none_age12to23mo_idx = df.index[df.is_alive & (df.ri_resp_infection_stat == 'none') &
                                                   (df.age_exact_years >= 1) & (df.age_exact_years < 2)]
        ri_current_none_age24to59mo_idx = df.index[df.is_alive & (df.ri_resp_infection_stat == 'none') &
                                                  (df.age_exact_years >= 2) & (df.age_exact_years < 5)]
        ri_current_none_HHhandwashing_idx = df.index[df.is_alive & (df.ri_resp_infection_stat == 'none') &
                                                     (df.HHhandwashing) & (df.age_years < 5)]
        ri_current_none_HIV_idx = df.index[df.is_alive & (df.ri_resp_infection_stat == 'none') &
                                                     (df.has_hiv) & (df.age_years < 5)]
        ri_current_none_malnutrition_idx = df.index[df.is_alive & (df.ri_resp_infection_stat == 'none') &
                                                     (df.malnutrition) & (df.age_years < 5)]
        ri_current_none_siblings_idx = df.index[df.is_alive & (df.ri_resp_infection_stat == 'none') &
                                                    (df.indoor_ai_pollution) & (df.age_years < 5)]
        ri_current_none_wealth_idx = df.index[df.is_alive & (df.ri_resp_infection_stat == 'none') &
                                                    (df.li_wealth) & (df.age_years < 5)]
        eff_prob_ri_cold = pd.Series(m.r_cold_none,
                                    index=df.index[df.is_alive & (df.resp_infection_stat == 'none') & (df.age_years < 5)])

        eff_prob_ri_cold.loc[ri_current_none_agelt2mo_idx] *= m.rr_cold_agelt2mo
        eff_prob_ri_cold.loc[ri_current_none_age12to23mo_idx] *= m.rr_cold_age12to23mo
        eff_prob_ri_cold.loc[ri_current_none_age24to59mo_idx] *= m.rr_cold_age24to59mo
        eff_prob_ri_cold.loc[ri_current_none_HHhandwashing_idx] *= m.rr_cold_HHhandwashing
        eff_prob_ri_cold.loc[ri_current_none_HIV_idx] *= m.rr_cold_HIV
        eff_prob_ri_cold.loc[ri_current_none_malnutrition_idx] *= m.rr_cold_malnutrition
        eff_prob_ri_cold.loc[ri_current_none_siblings_idx] *= m.rr_cold_siblings
        eff_prob_ri_cold.loc[ri_current_none_wealth_idx] *= m.rr_cold_wealth

        random_draw = pd.Series(rng.random_sample(size=len(ri_current_none_idx)),
                                   index=df.index[(df.age_years < 5) & df.is_alive & (df.ri_resp_infection_status == 'none')])

        dfx = pd.concat([eff_prob_ri_cold, random_draw], axis=1)
        dfx.columns = ['eff_prob_ri_cold', 'random_draw']
        idx_incident_cold = dfx.index[dfx.eff_prob_ri_cold > dfx.random_draw]
        df.loc[idx_incident_cold, 'ri_respiratory_infection_status'] = 'common cold'

        eff_prob_ri_pneumonia = pd.Series(m.r_pneumonia_none,
                                     index=df.index[
                                         df.is_alive & (df.resp_infection_stat == 'pneumonia') & (df.age_years < 5)])

        eff_prob_ri_pneumonia.loc[ri_current_none_agelt2mo_idx] *= m.rr_pneumonia_agelt2mo
        eff_prob_ri_pneumonia.loc[ri_current_none_age12to23mo_idx] *= m.rr_pneumonia_age12to23mo
        eff_prob_ri_pneumonia.loc[ri_current_none_age24to59mo_idx] *= m.rr_pneumonia_age24to59mo
        eff_prob_ri_pneumonia.loc[ri_current_none_HHhandwashing_idx] *= m.rr_pneumonia_HHhandwashing
        eff_prob_ri_pneumonia.loc[ri_current_none_HIV_idx] *= m.rr_pneumonia_HIV
        eff_prob_ri_pneumonia.loc[ri_current_none_malnutrition_idx] *= m.rr_pneumonia_malnutrition
        eff_prob_ri_pneumonia.loc[ri_current_none_siblings_idx] *= m.rr_pneumonia_siblings
        eff_prob_ri_pneumonia.loc[ri_current_none_wealth_idx] *= m.rr_pneumonia_wealth

        random_draw = pd.Series(rng.random_sample(size=len(ri_current_none_idx)),
                                index=df.index[
                                    (df.age_years < 5) & df.is_alive & (df.ri_resp_infection_status == 'none')])

        dfx = pd.concat([eff_prob_ri_pneumonia, random_draw], axis=1)
        dfx.columns = ['eff_prob_ri_pneumonia', 'random_draw']
        idx_incident_pneumonia = dfx.index[dfx.eff_prob_ri_pneumonia > dfx.random_draw]
        df.loc[idx_incident_cold, 'ri_respiratory_infection_status'] = 'pneumonia'

        eff_prob_ri_severe_pneumonia = pd.Series(m.r_severe_pneumonia_none,
                                          index=df.index[
                                              df.is_alive & (df.resp_infection_stat == 'severe pneumonia') & (
                                                      df.age_years < 5)])

        eff_prob_ri_severe_pneumonia.loc[ri_current_none_agelt2mo_idx] *= m.rr_severe_pneumonia_agelt2mo
        eff_prob_ri_severe_pneumonia.loc[ri_current_none_age12to23mo_idx] *= m.rr_severe_pneumonia_age12to23mo
        eff_prob_ri_severe_pneumonia.loc[ri_current_none_age24to59mo_idx] *= m.rr_severe_pneumonia_age24to59mo
        eff_prob_ri_severe_pneumonia.loc[ri_current_none_HHhandwashing_idx] *= m.rr_severe_pneumonia_HHhandwashing
        eff_prob_ri_severe_pneumonia.loc[ri_current_none_HIV_idx] *= m.rr_severe_pneumonia_HIV
        eff_prob_ri_severe_pneumonia.loc[ri_current_none_malnutrition_idx] *= m.rr_severe_pneumonia_malnutrition
        eff_prob_ri_severe_pneumonia.loc[ri_current_none_siblings_idx] *= m.rr_severe_pneumonia_siblings
        eff_prob_ri_severe_pneumonia.loc[ri_current_none_wealth_idx] *= m.rr_severe_pneumonia_wealth

        random_draw = pd.Series(rng.random_sample(size=len(ri_current_none_idx)),
                                index=df.index[
                                    (df.age_years < 5) & df.is_alive & (df.ri_resp_infection_status == 'none')])

        dfx = pd.concat([eff_prob_ri_pneumonia, random_draw], axis=1)
        dfx.columns = ['eff_prob_ri_severe_pneumonia', 'random_draw']
        idx_incident_severe_pneumonia = dfx.index[dfx.eff_prob_ri_severe_pneumonia > dfx.random_draw]
        df.loc[idx_incident_cold, 'ri_respiratory_infection_status'] = 'severe pneumonia'


        # updating for children under 5 with current status 'pneumonia' to 'severe pneumonia'

        ri_current_pneumonia_idx = df.index[df.is_alive & (df.ri_resp_infection_stat == 'pneumonia')
                                                & (df.age_years < 5)]
        ri_current_pneumonia_agelt2mo_idx = df.index[df.is_alive & (df.ri_resp_infection_stat == 'pneumonia')
                                                & (df.age_exact_years < 0.1667)]
        ri_current_pneumonia_age12to23mo_idx = df.index[df.is_alive & (df.ri_resp_infection_stat == 'pneumonia') &
                                                   (df.age_exact_years >= 1) & (df.age_exact_years < 2)]
        ri_current_pneumonia_age24to59mo_idx = df.index[df.is_alive & (df.ri_resp_infection_stat == 'pneumonia') &
                                                   (df.age_exact_years >= 2) & (df.age_exact_years < 5)]
        ri_current_pneumonia_HHhandwashing_idx = df.index[df.is_alive & (df.ri_resp_infection_stat == 'pneumonia') &
                                                     (df.HHhandwashing) & (df.age_years < 5)]
        ri_current_pneumonia_HIV_idx = df.index[df.is_alive & (df.ri_resp_infection_stat == 'pneumonia') &
                                           (df.has_hiv) & (df.age_years < 5)]
        ri_current_pneumonia_malnutrition_idx = df.index[df.is_alive & (df.ri_resp_infection_stat == 'pneumonia') &
                                                    (df.malnutrition) & (df.age_years < 5)]
        ri_current_pneumonia_indoor_air_pollution_idx = df.index[df.is_alive & (df.ri_resp_infection_stat == 'pneumonia') &
                                                (df.indoor_air_pollution) & (df.age_years < 5)]
        ri_current_pneumonia_wealth_idx = df.index[df.is_alive & (df.ri_resp_infection_stat == 'pneumonia') &
                                              (df.li_wealth) & (df.age_years < 5)]

        eff_prob_prog_severe_pneumonia= pd.Series(m.r_progress_to_severe_penumonia,
                                             index=df.index[df.is_alive & (df.resp_infection_stat == 'pneumonia')
                                                            & (df.age_years < 5)])
        eff_prob_prog_severe_pneumonia.loc[ri_current_pneumonia_agelt2mo_idx] *= m.rr_progress_severe_pneumonia_agelt2mo
        eff_prob_prog_severe_pneumonia.loc[ri_current_pneumonia_age12to23mo_idx] *= m.rr_progress_severe_pneumonia_age_12to23mo
        eff_prob_prog_severe_pneumonia.loc[ri_current_pneumonia_age24to59mo_idx] *= m.rr_progress_severe_pneumonia_age_24to59mo
        eff_prob_prog_severe_pneumonia.loc[ri_current_pneumonia_HHhandwashing_idx] *= m.rr_progress_severe_pneumonia_HHhandwashing
        eff_prob_prog_severe_pneumonia.loc[ri_current_pneumonia_HIV_idx] *= m.rr_progress_severe_pneumonia_HIV
        eff_prob_prog_severe_pneumonia.loc[ri_current_pneumonia_malnutrition_idx] *= m.rr_progress_severe_pneumonia_malnutrition
        eff_prob_prog_severe_pneumonia.loc[ri_current_pneumonia_indoor_air_pollution_idx] *= m.rr_progress_severe_pneumonia_indoor_air_pollution
        eff_prob_prog_severe_pneumonia.loc[ri_current_pneumonia_wealth_idx] *= m.rr_progress_severe_pneumonia_wealth

        random_draw = pd.Series(rng.random_sample(size=len(ri_current_pneumonia_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive &
                                               (df.resp_infection_stat == 'pneumonia')])
        dfx = pd.concat([eff_prob_ri_severe_pneumonia, random_draw], axis=1)
        dfx.columns = ['eff_prob_prog_severe_pneumonia', 'random_draw']
        idx_r_progress_severe_pneumonia = dfx.index[dfx.eff_prob_prog_severe_pneumonia > dfx.random_draw]
        df.loc[idx_r_progress_severe_pneumonia, 'ri_respiratory_infection_status'] = 'severe pneumonia'

        # updating for people aged over 20 with current status 'high grade dysplasia'

        ca_oes_current_high_grade_dysp_idx = df.index[df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia') &
                                                     (df.age_years >= 20)]
        ca_oes_current_high_grade_dysp_treated_idx = df.index[df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia') &
                                                     (df.age_years >= 20)
                                                     & (df.ca_oesophagus_curative_treatment == 'high_grade_dysplasia')]
        eff_prob_stage1 = pd.Series(m.r_stage1_high_grade_dysp,
                                             index=df.index[df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia')
                                                            & (df.age_years >= 20)])
        eff_prob_stage1.loc[ca_oes_current_high_grade_dysp_treated_idx] \
            *= m.rr_stage1_undergone_curative_treatment
        random_draw = pd.Series(rng.random_sample(size=len(ca_oes_current_high_grade_dysp_idx)),
                                index=df.index[(df.age_years >= 20) & df.is_alive &
                                               (df.ca_oesophagus == 'high_grade_dysplasia')])
        dfx = pd.concat([eff_prob_stage1, random_draw], axis=1)
        dfx.columns = ['eff_prob_stage1', 'random_draw']
        idx_incident_stage1 = dfx.index[dfx.eff_prob_stage1 > dfx.random_draw]
        df.loc[idx_incident_stage1, 'ca_oesophagus'] = 'stage1'

        # updating for people aged over 20 with current status stage 1 oes cancer

        ca_oes_current_stage1_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage1') &
                                             (df.age_years >= 20)]
        ca_oes_current_stage1_treated_idx = df.index[
            df.is_alive & (df.ca_oesophagus == 'stage1') &
            (df.age_years >= 20)
            & (df.ca_oesophagus_curative_treatment == 'stage1')]
        eff_prob_stage2 = pd.Series(m.r_stage2_stage1,
                                    index=df.index[df.is_alive & (df.ca_oesophagus == 'stage1')
                                                   & (df.age_years >= 20)])
        eff_prob_stage2.loc[ca_oes_current_stage1_treated_idx] \
            *= m.rr_stage2_undergone_curative_treatment
        random_draw = pd.Series(rng.random_sample(size=len(ca_oes_current_stage1_idx)),
                                index=df.index[(df.age_years >= 20) & df.is_alive &
                                               (df.ca_oesophagus == 'stage1')])
        dfx = pd.concat([eff_prob_stage2, random_draw], axis=1)
        dfx.columns = ['eff_prob_stage2', 'random_draw']
        idx_incident_stage2 = dfx.index[dfx.eff_prob_stage2 > dfx.random_draw]
        df.loc[idx_incident_stage2, 'ca_oesophagus'] = 'stage2'

        # updating for people aged over 20 with current status stage 2 oes cancer

        ca_oes_current_stage2_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage2') &
                                             (df.age_years >= 20)]
        ca_oes_current_stage2_treated_idx = df.index[
            df.is_alive & (df.ca_oesophagus == 'stage2') &
            (df.age_years >= 20)
            & (df.ca_oesophagus_curative_treatment == 'stage2')]
        eff_prob_stage3 = pd.Series(m.r_stage3_stage2,
                                    index=df.index[df.is_alive & (df.ca_oesophagus == 'stage2')
                                                   & (df.age_years >= 20)])
        eff_prob_stage3.loc[ca_oes_current_stage2_treated_idx] \
            *= m.rr_stage3_undergone_curative_treatment
        random_draw = pd.Series(rng.random_sample(size=len(ca_oes_current_stage2_idx)),
                                index=df.index[(df.age_years >= 20) & df.is_alive &
                                               (df.ca_oesophagus == 'stage2')])
        dfx = pd.concat([eff_prob_stage3, random_draw], axis=1)
        dfx.columns = ['eff_prob_stage3', 'random_draw']
        idx_incident_stage3 = dfx.index[dfx.eff_prob_stage3 > dfx.random_draw]
        df.loc[idx_incident_stage3, 'ca_oesophagus'] = 'stage3'

        # updating for people aged over 20 with current status stage 3 oes cancer

        ca_oes_current_stage3_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage3') &
                                             (df.age_years >= 20)]
        ca_oes_current_stage3_treated_idx = df.index[
            df.is_alive & (df.ca_oesophagus == 'stage3') &
            (df.age_years >= 20)
            & (df.ca_oesophagus_curative_treatment == 'stage3')]
        eff_prob_stage4 = pd.Series(m.r_stage4_stage3,
                                    index=df.index[df.is_alive & (df.ca_oesophagus == 'stage3')
                                                   & (df.age_years >= 20)])
        eff_prob_stage4.loc[ca_oes_current_stage3_treated_idx] \
            *= m.rr_stage4_undergone_curative_treatment
        random_draw = pd.Series(rng.random_sample(size=len(ca_oes_current_stage3_idx)),
                                index=df.index[(df.age_years >= 20) & df.is_alive &
                                               (df.ca_oesophagus == 'stage3')])
        dfx = pd.concat([eff_prob_stage4, random_draw], axis=1)
        dfx.columns = ['eff_prob_stage4', 'random_draw']
        idx_incident_stage4 = dfx.index[dfx.eff_prob_stage4 > dfx.random_draw]
        df.loc[idx_incident_stage4, 'ca_oesophagus'] = 'stage4'

        # -------------------- UPDATING OF CA_OESOPHAGUS DIAGNOSED OVER TIME --------------------------------

        df['ca_incident_oes_cancer_diagnosis_this_3_month_period'] = False

        # update diagnosis status for undiagnosed people with low grade dysplasia

        ca_oes_current_low_grade_dysp_not_diag_idx = df.index[
            df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia') &
            (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed]
        eff_prob_diag = pd.Series(m.r_diagnosis_low_grade_dysp,
                                  index=df.index[df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia') &
                                                 (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed])
        random_draw = rng.random_sample(size=len(ca_oes_current_low_grade_dysp_not_diag_idx))
        df.loc[ca_oes_current_low_grade_dysp_not_diag_idx, 'ca_oesophagus_diagnosed'] = (random_draw < eff_prob_diag)
        df.loc[ca_oes_current_low_grade_dysp_not_diag_idx, 'ca_incident_oes_cancer_diagnosis_this_3_month_period'] \
            = (random_draw < eff_prob_diag)

        # update diagnosis status for undiagnosed people with high grade dysplasia

        ca_oes_current_high_grade_dysp_not_diag_idx = df.index[df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia') &
                                                          (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed]
        eff_prob_diag = pd.Series(m.r_diagnosis_low_grade_dysp*m.rr_diagnosis_high_grade_dysp,
                              index=df.index[df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia') &
                                             (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed])
        random_draw = rng.random_sample(size=len(ca_oes_current_high_grade_dysp_not_diag_idx))
        df.loc[ca_oes_current_high_grade_dysp_not_diag_idx, 'ca_oesophagus_diagnosed'] = (random_draw < eff_prob_diag)
        df.loc[ca_oes_current_high_grade_dysp_not_diag_idx, 'ca_incident_oes_cancer_diagnosis_this_3_month_period'] \
            = (random_draw < eff_prob_diag)

        # update diagnosis status for undiagnosed people with stage 1 oes cancer

        ca_oes_current_stage1_not_diag_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage1') &
                                                      (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed]
        eff_prob_diag = pd.Series(m.r_diagnosis_low_grade_dysp * m.rr_diagnosis_stage1,
                                  index=df.index[df.is_alive & (df.ca_oesophagus == 'stage1') &
                                                 (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed])
        random_draw = rng.random_sample(size=len(ca_oes_current_stage1_not_diag_idx))
        df.loc[ca_oes_current_stage1_not_diag_idx, 'ca_oesophagus_diagnosed'] = (random_draw < eff_prob_diag)
        df.loc[ca_oes_current_stage1_not_diag_idx, 'ca_incident_oes_cancer_diagnosis_this_3_month_period'] \
            = (random_draw < eff_prob_diag)

        # update diagnosis status for undiagnosed people with stage 2 oes cancer

        ca_oes_current_stage2_not_diag_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage2') &
                                                          (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed]
        eff_prob_diag = pd.Series(m.r_diagnosis_low_grade_dysp*m.rr_diagnosis_stage2,
                              index=df.index[df.is_alive & (df.ca_oesophagus == 'stage2') &
                                             (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed])
        random_draw = rng.random_sample(size=len(ca_oes_current_stage2_not_diag_idx))
        df.loc[ca_oes_current_stage2_not_diag_idx, 'ca_oesophagus_diagnosed'] = (random_draw < eff_prob_diag)
        df.loc[ca_oes_current_stage2_not_diag_idx, 'ca_incident_oes_cancer_diagnosis_this_3_month_period'] \
            = (random_draw < eff_prob_diag)

        # update diagnosis status for undiagnosed people with stage 3 oes cancer

        ca_oes_current_stage3_not_diag_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage3') &
                                                          (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed]
        eff_prob_diag = pd.Series(m.r_diagnosis_low_grade_dysp*m.rr_diagnosis_stage3,
                              index=df.index[df.is_alive & (df.ca_oesophagus == 'stage3') &
                                             (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed])
        random_draw = rng.random_sample(size=len(ca_oes_current_stage3_not_diag_idx))
        df.loc[ca_oes_current_stage3_not_diag_idx, 'ca_oesophagus_diagnosed'] = (random_draw < eff_prob_diag)
        df.loc[ca_oes_current_stage3_not_diag_idx, 'ca_incident_oes_cancer_diagnosis_this_3_month_period'] \
            = (random_draw < eff_prob_diag)

        # update diagnosis status for undiagnosed people with stage 4 oes cancer

        ca_oes_current_stage4_not_diag_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage4') &
                                                          (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed]
        eff_prob_diag = pd.Series(m.r_diagnosis_low_grade_dysp*m.rr_diagnosis_stage4,
                              index=df.index[df.is_alive & (df.ca_oesophagus == 'stage4') &
                                             (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed])
        random_draw = rng.random_sample(size=len(ca_oes_current_stage4_not_diag_idx))
        df.loc[ca_oes_current_stage4_not_diag_idx, 'ca_oesophagus_diagnosed'] = (random_draw < eff_prob_diag)
        df.loc[ca_oes_current_stage4_not_diag_idx, 'ca_incident_oes_cancer_diagnosis_this_3_month_period'] \
            = (random_draw < eff_prob_diag)

        # -------------------- UPDATING VALUES OF CA_OESOPHAGUS_CURATIVE_TREATMENT -------------------

        # update ca_oesophagus_curative_treatment for diagnosed, untreated people with low grade dysplasia w

        ca_oes_diag_low_grade_dysp_not_treated_idx = df.index[
            df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia') &
            (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (df.ca_oesophagus_curative_treatment == 'never')]
        eff_prob_treatment = pd.Series(m.r_curative_treatment_low_grade_dysp,
                                       index=df.index[df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia') &
                                                      (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (
                                                              df.ca_oesophagus_curative_treatment == 'never')])
        random_draw = pd.Series(rng.random_sample(size=len(ca_oes_diag_low_grade_dysp_not_treated_idx)),
                                index=df.index[df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia') &
                                               (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (
                                                       df.ca_oesophagus_curative_treatment == 'never')])
        dfx = pd.concat([eff_prob_treatment, random_draw], axis=1)
        dfx.columns = ['eff_prob_treatment', 'random_draw']
        idx_incident_treatment = dfx.index[dfx.eff_prob_treatment > dfx.random_draw]
        df.loc[idx_incident_treatment, 'ca_oesophagus_curative_treatment'] = 'low_grade_dysplasia'

        # update ca_oesophagus_curative_treatment for diagnosed, untreated people with high grade dysplasia w

        ca_oes_diag_high_grade_dysp_not_treated_idx = df.index[
            df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia') &
            (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (df.ca_oesophagus_curative_treatment == 'never')]
        eff_prob_treatment = pd.Series(m.r_curative_treatment_low_grade_dysp*m.rr_curative_treatment_high_grade_dysp,
                                       index=df.index[df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia') &
                                                      (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (
                                                              df.ca_oesophagus_curative_treatment == 'never')])
        random_draw = pd.Series(rng.random_sample(size=len(ca_oes_diag_high_grade_dysp_not_treated_idx)),
                                index=df.index[df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia') &
                                               (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (
                                                       df.ca_oesophagus_curative_treatment == 'never')])
        dfx = pd.concat([eff_prob_treatment, random_draw], axis=1)
        dfx.columns = ['eff_prob_treatment', 'random_draw']
        idx_incident_treatment = dfx.index[dfx.eff_prob_treatment > dfx.random_draw]
        df.loc[idx_incident_treatment, 'ca_oesophagus_curative_treatment'] = 'high_grade_dysplasia'

        # update ca_oesophagus_curative_treatment for diagnosed, untreated people with stage 1

        ca_oes_diag_stage1_not_treated_idx = df.index[
            df.is_alive & (df.ca_oesophagus == 'stage1') &
            (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (df.ca_oesophagus_curative_treatment == 'never')]
        eff_prob_treatment = pd.Series(m.r_curative_treatment_low_grade_dysp * m.rr_curative_treatment_stage1,
                                       index=df.index[df.is_alive & (df.ca_oesophagus == 'stage1') &
                                                      (df.age_years >= 20) & df.ca_oesophagus_diagnosed &
                                                      (df.ca_oesophagus_curative_treatment == 'never')])
        random_draw = pd.Series(rng.random_sample(size=len(ca_oes_diag_stage1_not_treated_idx)),
                                index=df.index[df.is_alive & (df.ca_oesophagus == 'stage1') &
                                               (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (
                                                   df.ca_oesophagus_curative_treatment == 'never')])
        dfx = pd.concat([eff_prob_treatment, random_draw], axis=1)
        dfx.columns = ['eff_prob_treatment', 'random_draw']
        idx_incident_treatment = dfx.index[dfx.eff_prob_treatment > dfx.random_draw]
        df.loc[idx_incident_treatment, 'ca_oesophagus_curative_treatment'] = 'stage1'

       # update ca_oesophagus_curative_treatment for diagnosed, untreated people with stage 2

        ca_oes_diag_stage2_not_treated_idx = df.index[
            df.is_alive & (df.ca_oesophagus == 'stage2') &
            (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (df.ca_oesophagus_curative_treatment == 'never')]
        eff_prob_treatment = pd.Series(m.r_curative_treatment_low_grade_dysp*m.rr_curative_treatment_stage2,
                                       index=df.index[df.is_alive & (df.ca_oesophagus == 'stage2') &
                                                      (df.age_years >= 20) & df.ca_oesophagus_diagnosed &
                                                      (df.ca_oesophagus_curative_treatment == 'never') ] )
        random_draw = pd.Series(rng.random_sample(size=len(ca_oes_diag_stage2_not_treated_idx)),
                                index=df.index[df.is_alive & (df.ca_oesophagus == 'stage2') &
                                               (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (
                                                       df.ca_oesophagus_curative_treatment == 'never')])
        dfx = pd.concat([eff_prob_treatment, random_draw], axis=1)
        dfx.columns = ['eff_prob_treatment', 'random_draw']
        idx_incident_treatment = dfx.index[dfx.eff_prob_treatment > dfx.random_draw]
        df.loc[idx_incident_treatment, 'ca_oesophagus_curative_treatment'] = 'stage2'

       # update ca_oesophagus_curative_treatment for diagnosed, untreated people with stage 3

        ca_oes_diag_stage3_not_treated_idx = df.index[
            df.is_alive & (df.ca_oesophagus == 'stage3') &
            (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (df.ca_oesophagus_curative_treatment == 'never')]
        eff_prob_treatment = pd.Series(m.r_curative_treatment_low_grade_dysp*m.rr_curative_treatment_stage3,
                                       index=df.index[df.is_alive & (df.ca_oesophagus == 'stage3') &
                                                      (df.age_years >= 20) & df.ca_oesophagus_diagnosed &
                                                      (df.ca_oesophagus_curative_treatment == 'never') ] )
        random_draw = pd.Series(rng.random_sample(size=len(ca_oes_diag_stage3_not_treated_idx)),
                                index=df.index[df.is_alive & (df.ca_oesophagus == 'stage3') &
                                               (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (
                                                       df.ca_oesophagus_curative_treatment == 'never')])
        dfx = pd.concat([eff_prob_treatment, random_draw], axis=1)
        dfx.columns = ['eff_prob_treatment', 'random_draw']
        idx_incident_treatment = dfx.index[dfx.eff_prob_treatment > dfx.random_draw]
        df.loc[idx_incident_treatment, 'ca_oesophagus_curative_treatment'] = 'stage3'

        # -------------------- DEATH FROM OESOPHAGEAL CANCER ---------------------------------------

        stage4_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage4')]
        random_draw = m.rng.random_sample(size=len(stage4_idx))
        df.loc[stage4_idx, 'ca_oesophageal_cancer_death'] = (random_draw < m.r_death_oesoph_cancer)

        # todo - this code dealth with centrally
        dead_oes_can_idx = df.index[df.ca_oesophageal_cancer_death]
        df.loc[dead_oes_can_idx, 'is_alive'] = False


class OesCancerLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """Handles lifestyle logging"""
    def __init__(self, module):
        """schedule logging to repeat every 3 months
        """
        self.repeat = 3
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        # get some summary statistics
        df = population.props

        # calculate incidence of oesophageal cancer diagnosis in people aged > 60+
        # (this includes people diagnosed with dysplasia, but diagnosis rate at this stage is very low)

        incident_oes_cancer_diagnosis_agege60_idx = df.index[df.ca_incident_oes_cancer_diagnosis_this_3_month_period
        & (df.age_years >= 60)]
        agege60_without_diagnosed_oes_cancer_idx = df.index[(df.age_years >= 60) & ~df.ca_oesophagus_diagnosed]

        incidence_per_year_oes_cancer_diagnosis = (4 * 100000 * len(incident_oes_cancer_diagnosis_agege60_idx))/\
                                                  len(agege60_without_diagnosed_oes_cancer_idx)

        incidence_per_year_oes_cancer_diagnosis = round(incidence_per_year_oes_cancer_diagnosis, 3)

 #      logger.debug('%s|person_one|%s',
 #                     self.sim.date,
 #                     df.loc[0].to_dict())

#       logger.info('%s|ca_oesophagus|%s',
#                   self.sim.date,
#                   df[df.is_alive].groupby(['ca_oesophagus']).size().to_dict())

        # note below remove is_alive
#       logger.info('%s|ca_oesophagus_death|%s',
#                   self.sim.date,
#                   df[df.age_years >= 20].groupby(['ca_oesophageal_cancer_death']).size().to_dict())


        logger.info('%s|ca_incident_oes_cancer_diagnosis_this_3_month_period|%s',
                    self.sim.date,
                    incidence_per_year_oes_cancer_diagnosis)


#       logger.info('%s|ca_oesophagus_diagnosed|%s',
#                   self.sim.date,
#                   df[df.age_years >= 20].groupby(['ca_oesophagus', 'ca_oesophagus_diagnosed']).size().to_dict())

#       logger.info('%s|ca_oesophagus|%s',
#                   self.sim.date,
#                   df[df.is_alive].groupby(['age_range', 'ca_oesophagus']).size().to_dict())
