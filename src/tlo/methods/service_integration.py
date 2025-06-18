from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DAYS_IN_YEAR, Date, DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import Metadata
from tlo.methods.hsi_generic_first_appts import GenericFirstAppointmentsMixin
from tlo.util import read_csv_files
from tlo.lm import LinearModel, LinearModelType

from tlo.methods.labour import LabourAndPostnatalCareAnalysisEvent
from tlo.methods.contraception import StartInterventions

if TYPE_CHECKING:
    from tlo.methods.hsi_generic_first_appts import HSIEventScheduler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ServiceIntegration(Module, GenericFirstAppointmentsMixin):
    """
    """

    # Declare modules that need to be registered in simulation and initialised before
    # this module
    INIT_DEPENDENCIES = {'Demography'}


    # Declare Metadata
    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
    }

    # Declare Causes of Death
    CAUSES_OF_DEATH = {}

    # Declare Causes of Disability
    CAUSES_OF_DISABILITY = {
    }

    PARAMETERS = {
        'integration_year': Parameter(Types.INT, 'year on which parameters are overwritten for integration '
                                                  'modelling'),
        'serv_integration':
            Parameter(Types.STRING,
                      'name of the integration scenario to be enacted in a given run'),
    }

    PROPERTIES = {
    }

    def __init__(self, name=None, resourcefilepath=None):
        # NB. Parameters passed to the module can be inserted in the __init__ definition.

        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        self.accepted_scenarios = ['htn', 'htn_max', 'dm', 'dm_max', 'hiv', 'hiv_max', 'tb', 'tb_max',
                                    'mal', 'mal_max', 'fp_scr', 'fp_scr_max', 'anc', 'anc_max', 'pnc',
                                    'pnc_max', 'fp_pn', 'fp_pn_max', 'epi', 'chronic_care',
                                     'chronic_care_max', 'all_screening', 'all_screening_max',
                                     'all_mch', 'all_mch_max', 'all_int', 'all_int_max']

    def read_parameters(self, resourcefilepath: Optional[Path] = None):
        parameter_dataframe = read_csv_files(resourcefilepath / 'service integration',
                                            files='parameter_values')
        self.load_parameters_from_dataframe(parameter_dataframe)

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        pass

    def initialise_simulation(self, sim):

        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        params = self.parameters

        event = ServiceIntegrationParameterUpdateEvent(self)
        sim.schedule_event(event, Date(params['integration_year'], 1, 1))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the ID for the mother for this child
        :param child_id: the ID for the new child
        """

        pass

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        pass

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        pass


class ServiceIntegrationParameterUpdateEvent(Event, PopulationScopeEventMixin):
    def __init__(self, module):
        super().__init__(module)
        assert isinstance(module, ServiceIntegration)

    def apply(self, population):
        params = self.module.parameters
        hs_params = self.sim.modules['HealthSystem'].parameters

        # TODO: make this a class of the health system module instead of its own module that needs to be registered?

        logger.info(key='event_runs', data='ServiceIntegrationParameterUpdateEvent is running')

        if params['serv_integration'] == 'no_integration':
            logger.info(key='event_cancelled', data='ServiceIntegrationParameterUpdateEvent did not run')
            return
        else:
            assert params['serv_integration'] in self.module.accepted_scenarios

        def update_cons_override_treatment_ids(treatment_ids):
            for treatment_id in treatment_ids:
                if treatment_id not in hs_params['cons_override_treatment_ids']:
                    hs_params['cons_override_treatment_ids'].append(treatment_id)

        # ---------------------------------------------- SCREENING ---------------------------------------------------
        if params['serv_integration'].startswith(("htn", "all_screening", "all_int")):
            # Probability of screening when presenting to any generic first appointment set to 100%
            self.sim.modules['CardioMetabolicDisorders'].parameters[
                'hypertension_hsi']['pr_assessed_other_symptoms'] = 1.0

            # Annual community screening in over 50s increased to 100%
            self.sim.modules['CardioMetabolicDisorders'].lms_testing['hypertension'] = \
                LinearModel(LinearModelType.MULTIPLICATIVE, 1.0)

            # Now ensure consumables are always available for the relevant treatment ids
            if params['serv_integration'].endswith('_max'):

                # TODO: this should only be those using treatment for hypertension...
                update_cons_override_treatment_ids([
                    'CardioMetabolicDisorders_Prevention_CommunityTestingForHypertension',
                     'CardioMetabolicDisorders_Investigation',
                     'CardioMetabolicDisorders_Prevention_WeightLoss'])

        if params['serv_integration'].startswith(("dm", "all_screening", "all_int")):
            # Probability of screening when presenting to any generic first appointment and not sympotmatic set to 100%
            self.sim.modules['CardioMetabolicDisorders'].parameters['diabetes_hsi'][
                'pr_assessed_other_symptoms'] = 1.0

            if params['serv_integration'].endswith('_max'):
                # TODO: this should only be those using treatment for diabetes...
                update_cons_override_treatment_ids([
                    'CardioMetabolicDisorders_Investigation',
                    'CardioMetabolicDisorders_Prevention_WeightLoss'])

        if params['serv_integration'].startswith(("fp_scr", "all_screening", "all_int")):
            # Here we use the in-built functionality of the contraception model to increase the coverage of modern
            # methods of contraception. When 'fp' is listed in params['serv_int_screening'] the probability of
            # initiation in the general female population is increased. See updates to contraception.py

            self.sim.modules['Contraception'].update_params_for_interventions(initiation=True,
                                                                              after_birth=False)

            if params['serv_integration'].endswith('_max'):
                update_cons_override_treatment_ids(['Contraception_Routine'])

        if params['serv_integration'].startswith(("mal", "all_screening", "all_int", "all_mch")):

            self.sim.modules['Stunting'].parameters['prob_stunting_diagnosed_at_generic_appt'] = 1.0

            if params['serv_integration'].endswith('_max'):
                update_cons_override_treatment_ids(['Undernutrition_Feeding'])

        if params['serv_integration'].startswith(("hiv", "all_screening", "all_int")):
            # annual testing rate used in HIV scale-up scenarios, default average (2010-2020) is 0.25
            self.sim.modules['Hiv'].parameters["hiv_testing_rates"]["annual_testing_rate_adults"] = 0.4
            # update exising linear models to use new scaled-up parameters

            if params['serv_integration'].endswith('_max'):
                update_cons_override_treatment_ids([
                    'Hiv_Test', 'Hiv_Treatment'])

        if params['serv_integration'].startswith(("tb", "all_screening", "all_int")):

            # increase treatment coverage rate used to infer rate testing for active tb, default is 0.75
            self.sim.modules['Tb'].parameters["rate_testing_active_tb"]["treatment_coverage"] = 90

            if params['serv_integration'].endswith('_max'):
                update_cons_override_treatment_ids(
                    ['Tb_Test_Screening',
                    'Tb_Test_Clinical',
                    'Tb_Test_Culture',
                    'Tb_Test_Xray',
                    'Tb_Treatment'])

        # ------------------------------------ MATERNAL AND CHILD HEALTH CLINIC ---------------------------------------
        if params['serv_integration'].startswith(("anc", "all_mch", "all_int")):
            self.sim.modules['PregnancySupervisor'].current_parameters['alternative_anc_coverage'] = True
            self.sim.modules['PregnancySupervisor'].current_parameters['anc_availability_odds'] = 9.0
            self.sim.modules['PregnancySupervisor'].update_antenatal_care_coverage_for_analysis()

            if params['serv_integration'].endswith('_max'):
                update_cons_override_treatment_ids(['AntenatalCare_Outpatient', 'AntenatalCare_FollowUp'])


        if params['serv_integration'].startswith(("pnc", "all_mch", "all_int")):
            self.sim.modules['Labour'].current_parameters['alternative_pnc_coverage'] = True
            self.sim.modules['Labour'].current_parameters['pnc_availability_odds'] = 15.0
            self.sim.modules['Labour'].update_labour_or_postnatal_coverage_for_analysis()

            if params['serv_integration'].endswith('_max'):
                update_cons_override_treatment_ids(['PostnatalCare_Neonatal', 'PostnatalCare_Maternal'])

        if params['serv_integration'].startswith(("fp_pn", "all_mch", "all_int")):
            # Here we use the in-built functionality of the contraception model to increase the coverage of modern
            # methods of contraception. When 'fp' is listed in params['serv_int_mch'] the probability of
            # initiation following birth is increased. See updates to contraception.py

            self.sim.modules['Contraception'].update_params_for_interventions(initiation=False,
                                                                              after_birth=True)
            if params['serv_integration'].endswith('_max'):
                # TODO: dont we only want those seeking postnatal contraception to have available consumables?
                update_cons_override_treatment_ids(['Contraception_Routine'])

        # no parameter governing prob of receiving vaccine
        # child's prob of vax entirely dependent on vaccine being available (cons required)
        # can manipulate this to induce 100% coverage rate - will need to look up the vaccines required for each
        if params['serv_integration'].startswith(("epi", "all_mch", "all_int")):
            update_cons_override_treatment_ids(['Epi_Childhood_Bcg',
                                                'Epi_Childhood_Opv',
                                                'Epi_Childhood_DtpHibHep',
                                                'Epi_Childhood_Rota',
                                                'Epi_Childhood_Pneumo',
                                                'Epi_Childhood_MeaslesRubella',
                                                'Epi_Adolescent_Hpv',
                                                'Epi_Pregnancy_Td'
                                                ])

        # ------------------------------------- CHRONIC CARE CLINIC ---------------------------------------------------
        # todo: currently only hiv and ncds are linked to other services (what about those presenting for depression etc)

        if params['serv_integration'].startswith(("chronic_care", "all_int")):

            self.sim.modules['Hiv'].parameters['virally_suppressed_on_art'] = 1.0
            self.sim.modules['Tb'].parameters['tb_prob_tx_success_ds'] = 0.9
            self.sim.modules['Tb'].parameters['tb_prob_tx_success_mdr'] = 0.9
            self.sim.modules['Epilepsy'].parameters[
                'prob_start_anti_epilep_when_seizures_detected_in_generic_first_appt'] = 1.0
            self.sim.modules['Depression'].parameters['pr_assessed_for_depression_in_generic_appt_level1'] = 1.0

            # commented out because tx_success higher than 0.9 already in these groups
            # self.sim.modules['Tb'].parameters['tb_prob_tx_success_0_4'] = 0.9
            # self.sim.modules['Tb'].parameters['tb_prob_tx_success_5_14'] = 0.9

            if params['serv_integration'].endswith('_max'):
                update_cons_override_treatment_ids(
                    ['CardioMetabolicDisorders_Investigation',
                     'CardioMetabolicDisorders_Prevention_WeightLoss',
                     'Hiv_Test',
                     'Hiv_Treatment',
                     'Tb_Test_Screening',
                     'Tb_Test_Clinical',
                     'Tb_Test_Culture',
                     'Tb_Test_Xray',
                     'Tb_Treatment',
                     'Depression_TalkingTherapy',
                     'Depression_Treatment',
                     'Epilepsy_Treatment_Start',
                     'Epilepsy_Treatment_Followup'])
