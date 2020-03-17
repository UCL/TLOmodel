"""
A skeleton template for disease methods.

"""

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods.healthsystem import HSI_Event

# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Skeleton(Module):
    """
    One line summary goes here...

    All disease modules need to be implemented as a class inheriting from Module.
    They need to provide several methods which will be called by the simulation
    framework:
    * `read_parameters(data_folder)`
    * `initialise_population(population)`
    * `initialise_simulation(sim)`
    * `on_birth(mother, child)`
    * `on_hsi_alert(person_id, treatment_id)` [If this is disease module]
    *  `report_daly_values()` [If this is disease module]

    """

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    PARAMETERS = {
        'parameter_a': Parameter(
            Types.REAL, 'Description of parameter a'),
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.

    # Note that all properties must have a two letter prefix that identifies them to this module.

    PROPERTIES = {
        'sk_property_a': Property(Types.BOOL, 'Description of property a'),
    }

    # Declare the non-generic symptoms that this module will use.
    # It will not be able to use any that are not declared here. They do not need to be unique to this module.
    # You should not declare symptoms that are generic here (i.e. in the generic list of symptoms)
    SYMPTOMS = {}

    def __init__(self, name=None, resourcefilepath=None):
        # NB. Parameters passed to the module can be inserted in the __init__ definition.

        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.
        To access files use: Path(self.resourcefilepath) / file_name
        """
        pass

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        If this is a disease module, register this disease module with the healthsystem:
        self.sim.modules['HealthSystem'].register_disease_module(self)

        :param population: the population of individuals
        """
        raise NotImplementedError

    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.

        """
        raise NotImplementedError

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        raise NotImplementedError

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        # To return a value of 0.0 (fully health) for everyone, use:
        # df = self.sim.popultion.props
        # return pd.Series(index=df.index[df.is_alive],data=0.0)

        raise NotImplementedError

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        raise NotImplementedError


# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
#
#   These are the events which drive the simulation of the disease. It may be a regular event that updates
#   the status of all the population of subsections of it at one time. There may also be a set of events
#   that represent disease events for particular persons.
# ---------------------------------------------------------------------------------------------------------

class Skeleton_Event(RegularEvent, PopulationScopeEventMixin):
    """A skeleton class for an event

    Regular events automatically reschedule themselves at a fixed frequency,
    and thus implement discrete timestep type behaviour. The frequency is
    specified when calling the base class constructor in our __init__ method.
    """

    def __init__(self, module):
        """One line summary here

        We need to pass the frequency at which we want to occur to the base class
        constructor using super(). We also pass the module that created this event,
        so that random number generators can be scoped per-module.

        :param module: the module that created this event
        """
        super().__init__(module, frequency=DateOffset(months=1))
        assert isinstance(module, Skeleton)

    def apply(self, population):
        """Apply this event to the population.

        :param population: the current population
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------------------------------------
#   LOGGING EVENTS
#
#   Put the logging events here. There should be a regular logger outputting current states of the
#   population. There may also be a loggig event that is driven by particular events.
# ---------------------------------------------------------------------------------------------------------

class Skeleton_LoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """Produce a summary of the numbers of people with respect to the action of this module.
        This is a regular event that can output current states of people or cumulative events since last logging event.
        """

        # run this event every year
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))
        assert isinstance(module, Skeleton)

    def apply(self, population):
        # Make some summary statitics

        dict_to_output = {
            'Metric_One': 1.0,
            'Metric_Two': 2.0
        }

        logger.info('%s|summary_12m|%s', self.sim.date, dict_to_output)


# ---------------------------------------------------------------------------------------------------------
#   HEALTH SYSTEM INTERACTION EVENTS
#
#   Here are all the different Health System Interactions Events that this module will use.
# ---------------------------------------------------------------------------------------------------------

class HSI_Skeleton_Example_Interaction(HSI_Event, IndividualScopeEventMixin):
    """This is a Health System Interaction Event. An interaction with the healthsystem are encapsulated in events
    like this.
    It must begin HSI_<Module_Name>_Description
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Skeleton)

        # Define the call on resources of this treatment event: Time of Officers (Appointments)
        #   - get an 'empty' footprint:
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        #   - update to reflect the appointments that are required
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient

        # Define the facilities at which this event can occur (only one is allowed)
        # Choose from: list(pd.unique(self.sim.modules['HealthSystem'].parameters['Facilities_For_Each_District']
        #                            ['Facility_Level']))
        the_accepted_facility_level = 0

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Skeleton_Example_Interaction'  # This must begin with the module name
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = the_accepted_facility_level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        """
        Do the action that take place in this health system interaction, in light of squeeze_factor
        Can reutrn an updated APPT_FOOTPRINT if this differs from the declaration in self.EXPECTED_APPT_FOOTPRINT
        """
        pass

    def did_not_run(self):
        """
        Do any action that is neccessary when the health system interaction is not run.
        This is called each day that the HSI is 'due' but not run due to insufficient health system capabilities.
        Return False to cause this HSI event not to be rescheduled and to therefore never be run.
        (Returning nothing or True will cause this event to be rescheduled for the next day.)
        """
        pass

class HSI_NewbornOutcomes_ReceivesCareFollowingDelivery(HSI_Event, IndividualScopeEventMixin):
    """This HSI ReceivesCareFollowingDelivery. This event is scheduled by the on_birth function. All newborns whose
    mothers delivered in a facility are automatically scheduled to this event. Newborns delivered at home, but who
    experience complications, have this event scheduled due via a care seeking equation (currently just a dummy). It is
    responsible for prophylactic treatments following delivery (i.e. cord care, breastfeeding), applying risk of
    complications in facility and referral for additional treatment. This module will be reviewed with a clinician and
     may be changed
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, NewbornOutcomes)

        self.TREATMENT_ID = 'NewbornOutcomes_ReceivesCareFollowingDelivery'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1  # Todo: review  (DUMMY)

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        params = self.module.parameters
        nci = self.module.newborn_care_info
        child = df.loc[person_id]

        logger.info('This is HSI_NewbornOutcomes_ReceivesCareFollowingDelivery, neonate %d is receiving care from a '
                    'skilled birth attendant following their birth', person_id)

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code = pd.unique(consumables.loc[consumables
                                             ['Intervention_Pkg'] == 'Clean practices and immediate essential newborn '
                                                                     'care (in facility)',
                                                                     'Intervention_Pkg_Code'])[0]
        pkg_code_bcg = pd.unique(consumables.loc[consumables[
                                                 'Intervention_Pkg'] == 'BCG vaccine', 'Intervention_Pkg_Code'])[0]
        pkg_code_polio = pd.unique(consumables.loc[consumables['Intervention_Pkg'] == 'Polio vaccine',
                                                                                      'Intervention_Pkg_Code'])[0]

        item_code_vk = pd.unique(
            consumables.loc[consumables['Items'] == 'vitamin K1  (phytomenadione) 1 mg/ml, 1 ml, inj._100_IDA',
                            'Item_Code'])[0]
        item_code_tc = pd.unique(
            consumables.loc[consumables['Items'] == 'tetracycline HCl 3% skin ointment, 15 g_10_IDA', 'Item_Code'])[0]

        consumables_needed = {
            'Intervention_Package_Code': {pkg_code: 1, pkg_code_bcg: 1, pkg_code_polio: 1},
            'Item_Code': {item_code_vk: 1, item_code_tc: 1}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed)

        # TODO: Need to ensure not double counting consumables (i.e. chlorhexidine for cord care already included in
        #  delivery kit?)
        # TODO: chlorhex should be eye drops not ointment?
        # TODO: apply effect of squeeze factor

        # We apply the effect of a number of interventions that newborns in a facility will receive including cord care,
        # vaccinations, vitamin K prophylaxis, tetracycline eyedrops, kangaroo mother care and prophylactic antibiotics

# ----------------------------------- CHLORHEXIDINE CORD CARE ----------------------------------------------------------

        nci[person_id]['cord_care'] = True

# ------------------------------------- VACCINATIONS (BCG/POLIO) -------------------------------------------------------

        # For vaccines, vitamin K and tetracycline we condition these interventions  on the availibility of the
        # consumable
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_bcg]:
            logger.debug('pkg_code_bcg is available, so use it.')
            nci[person_id]['bcg_vacc'] = True
        else:
            logger.debug('pkg_code_bcg is not available, so can' 't use it.')
            logger.debug('newborn %d did not receive a BCG vaccine as there was no stock available', person_id)
            nci[person_id]['bcg_vacc'] = False

        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_polio]:
            logger.debug('pkg_code_polio is available, so use it.')
            nci[person_id]['polio_vacc'] = True
        else:
            logger.debug('pkg_code_polio is not available, so can' 't use it.')
            logger.debug('newborn %d did not receive a BCG vaccine as there was no stock availble', person_id)
            nci[person_id]['polio_vacc'] = False

# ------------------------------------------ VITAMIN K  ----------------------------------------------------------------
        if outcome_of_request_for_consumables['Item_Code'][item_code_vk]:
            logger.debug('item_code_vk is available, so use it.')
            nci[person_id]['vit_k'] = True
        else:
            logger.debug('item_code_vk is not available, so can' 't use it.')
            logger.debug('newborn %d did not receive vitamin K prophylaxis as there was no stock available', person_id)
            nci[person_id]['vit_k'] = False

# --------------------------------------- TETRACYCLINE EYE DROPS -------------------------------------------------------
        if outcome_of_request_for_consumables['Item_Code'][item_code_tc]:
            logger.debug('item_code_tc is available, so use it.')
            nci[person_id]['tetra_eye_d'] = True
        else:
            logger.debug('item_code_tc is not available, so can' 't use it.')
            logger.debug('newborn %d did not receive tetracycline eyedrops as there was no stock availble', person_id)
            nci[person_id]['tetra_eye_d'] = False

# --------------------------------- ANTIBIOTIC PROPHYLAXIS (MATERNAL RISK FACTORS)-------------------------------------

        # TODO: Confirm the guidelines (indications) and consumables for antibiotic prophylaxis
        # nci[person_id]['proph_abx'] = True

# ----------------------------------------- KANGAROO MOTHER CARE -------------------------------------------------------

        # Here we use a probability, derived from the DHS, to determine if a woman with a low birth weight infant will
        # be encouraged to undertake KMC. Currently only 'stable' (no complication) newborns can undergo KMC. This wil
        # need review
        if (child.nb_birth_weight == 'LBW') and (~child.nb_respiratory_depression and
                                                 ~child.nb_early_onset_neonatal_sepsis
                                                 and (child.nb_encephalopathy == 'none')):

            if self.module.rng.random_sample() < params['prob_facility_offers_kmc']:
                df.at[person_id, 'nb_kmc'] = True
            # TODO: evidence suggests KMC reduces the risk of sepsis and this needs to be applied
            # TODO: Check guidelines regarding if KMC is only used in stable infants

# ------------------------------ EARLY INITIATION OF BREAST FEEDING ----------------------------------------------------

        # As with KMC we use a DHS derived probability that a woman who delivers in a facility will initiate
        # breastfeeding within one hour
        if self.module.rng.random_sample() < params['prob_early_breastfeeding_hf']:
            df.at[person_id, 'nb_early_breastfeeding'] = True
            logger.debug('Neonate %d has started breastfeeding within 1 hour of birth', person_id)
        else:
            logger.debug('Neonate %d did not start breastfeeding within 1 hour of birth', person_id)

# ------------------------------ RECALCULATE SEPSIS RISK---------------------------------------------------------------

        # Following the application of the prophylactic/therapeutic effect of these interventions we then recalculate
        # individual sepsis risk and determine if this newborn will develop sepsis

        if self.module.rng.random_sample() < nci[person_id]['ongoing_sepsis_risk']:
            df.at[person_id, 'nb_early_onset_neonatal_sepsis'] = True

            logger.debug('Neonate %d has developed early onset sepsis in a health facility on date %s', person_id,
                         self.sim.date)
            logger.debug('%s|early_onset_nb_sep_fd|%s', self.sim.date, {'person_id': person_id})
            # TODO code in effect of prophylaxis

#  ================================ SCHEDULE ADDITIONAL TREATMENT ===================================================

        # Finally, for newborns who have experienced a complication within a facility, additional treatment is scheduled
        # through other HSIs
        if child.nb_respiratory_depression:
            logger.info('This is HSI_NewbornOutcomes_ReceivesCareFollowingDelivery: scheduling resuscitation for '
                        'neonate %d who has experienced birth asphyxia following delivery', person_id)

            event = HSI_NewbornOutcomes_ReceivesNewbornResuscitation(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        if child.nb_early_onset_neonatal_sepsis:
            logger.info('This is HSI_NewbornOutcomes_ReceivesCareFollowingDelivery: scheduling treatment for neonate'
                        'person %d who has developed early onset sepsis following delivery', person_id)

            event = HSI_NewbornOutcomes_ReceivesNewbornResuscitation(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        # actual_appt_footprint['InpatientDays'] = actual_appt_footprint['InpatientDays'] * 1

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_NewbornOutcomes_ReceivesCareFollowingDelivery: did not run')
        pass


class HSI_NewbornOutcomes_ReceivesNewbornResuscitation(HSI_Event, IndividualScopeEventMixin):
    """ This is HSI ReceivesNewbornResuscitation. This event is scheduled by HSI ReceivesCareFollowingDelivery if a
    child experiences respiratory depression. This event contains the intervention basic newborn resuscitation. This
    event is unfinished and will need to schedule very sick neonates for additional inpatient care"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, NewbornOutcomes)

        self.TREATMENT_ID = 'NewbornOutcomes_ReceivesNewbornResuscitation'

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1  # Todo: review  (DUMMY)

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        params = self.module.parameters

        logger.info('This is HSI_NewbornOutcomes_ReceivesNewbornResuscitation, neonate %d is receiving newborn '
                    'resuscitation following birth ', person_id)

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code = pd.unique(consumables.loc[consumables['Intervention_Pkg'] == 'Neonatal resuscitation '
                                                                                '(institutional)',
                                             'Intervention_Pkg_Code'])[0]

        consumables_needed = {'Intervention_Package_Code': {pkg_code: 1}, 'Item_Code': {}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed)

        # TODO: apply effect of squeeze factor

        # The block of code below is the intervention, newborns can only be resuscitated if the consumables are
        # available, so the effect is conditioned on this
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code]:
            logger.debug('resuscitation equipment is available, so use it.')
            if self.module.rng.random_sample() < params['prob_successful_resuscitation']:
                df.at[person_id, 'nb_respiratory_depression'] = False
                logger.info('Neonate %d has been successfully resuscitated after delivery with birth asphyxia',
                            person_id)

        else:
            logger.debug('PkgCode1 is not available, so can' 't use it.')
            # TODO: apply a probability of death without resuscitation here?
            # TODO: schedule additional care for very sick newborns

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        # actual_appt_footprint['InpatientDays'] = actual_appt_footprint['InpatientDays'] * 1

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_NewbornOutcomes_ReceivesCareFollowingDelivery: did not run')
        pass


class HSI_NewbornOutcomes_ReceivesTreatmentForSepsis(HSI_Event, IndividualScopeEventMixin):
    """ This is HSI ReceivesTreatmentForSepsis. This event is scheduled by HSI ReceivesCareFollowingDelivery if a
       child experiences early on sent neonatal sepsis. This event contains the intervention intravenous antibiotics.
       This  event is unfinished and will need to schedule very sick neonates for additional inpatient care"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, NewbornOutcomes)

        self.TREATMENT_ID = 'NewbornOutcomes_ReceivesTreatmentForSepsis'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1  # Todo: review  (DUMMY)

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        params = self.module.parameters

        logger.info('This is HSI_NewbornOutcomes_ReceivesTreatmentForSepsis, neonate %d is receiving treatment '
                    'for early on set neonatal sepsis following birth ', person_id)

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code_sep = pd.unique(consumables.loc[consumables[
                                                       'Intervention_Pkg'] == 'Treatment of local infections (newborn)',
                                                 'Intervention_Pkg_Code'])[0]

        consumables_needed = {'Intervention_Package_Code': {pkg_code_sep: 1}, 'Item_Code': {}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed)
        # TODO: Condition intervention on availability of antibiotics, considering 1st/2nd/3rd line and varying
        #  efficacy?

        # Here we use the treatment effect to determine if the newborn remains septic. This logic will need to be
        # changed to reflect need for inpatient admission and longer course of antibiotic
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_sep]:
            logger.debug('pkg_code_sepsis is available, so use it.')
            if params['prob_cure_antibiotics'] > self.module.rng.random_sample():
                df.at[person_id, 'nb_early_onset_neonatal_sepsis'] = False
        else:
            logger.debug('pkg_code_sepsis is not available, so can' 't use it.')

        # TODO: septic newborns will receive a course of ABX as an inpatient, this needs to be scheduled here

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        # actual_appt_footprint['InpatientDays'] = actual_appt_footprint['InpatientDays'] * 1

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_NewbornOutcomes_ReceivesCareFollowingDelivery: did not run')
        pass
