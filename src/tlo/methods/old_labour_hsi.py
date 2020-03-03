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
class HSI_Labour_PresentsForSkilledAttendanceInLabour(HSI_Event, IndividualScopeEventMixin):
    """This is the HSI PresentsForSkilledAttendanceInLabour. This event is scheduled by the LabourOnset Event. This
    event manages initial care around the time of delivery including prophylactic interventions (i.e. clean birth
    practices). This event uses a womans stored risk of complications, which may be manipulated by treatment effects to
    determines if they will experience a complication during their labour in hospital. It is responsible for scheduling
    treatment HSIs for those complications. In the event that this HSI will not run, women are scheduled back to the
    LabourAtHomeEvent. This event is not finalised an both interventions and referral are subject to change"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        self.TREATMENT_ID = 'Labour_PresentsForSkilledAttendanceInLabour'
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['NormalDelivery'] = 1

        # Define the necessary information for an HSI
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        mni = self.module.mother_and_newborn_info

        # magic number 0.8 is an arbitrary squeeze factor threshold
        if squeeze_factor > 0.8:
            self.did_not_run()
            logger.debug(
                'person %d sought care for a facility delivery but HSI_Labour_PresentsForSkilledAttendanceInLabour'
                'could not run, they will now deliver at home', person_id)
            mni[person_id]['delivery_setting'] = 'home_birth'
            self.sim.schedule_event(LabourAtHomeEvent(self.module, person_id), self.sim.date)

        logger.info('This is HSI_Labour_PresentsForSkilledAttendanceInLabour: Providing initial skilled attendance '
                    'at birth for person %d on date %s', person_id, self.sim.date)

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code_sba_uncomp = pd.unique(consumables.loc[consumables[
                                                      'Intervention_Pkg'] ==
                                                  'Vaginal delivery - skilled attendance',
                                                  'Intervention_Pkg_Code'])[0]

        consumables_needed = {
            'Intervention_Package_Code': {pkg_code_sba_uncomp: 1},
            'Item_Code': {}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed)

        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_sba_uncomp]:
            logger.debug('PkgCode1 is available, so use it.')
        else:
            logger.debug('PkgCode1 is not available, so can' 't use it.')
            # TODO: If delivery pack isn't available then birth will still occur but should have risk of sepsis?

        df = self.sim.population.props
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

        # TODO: Discuss with Tim H the best way to capture which HCP will be attending this delivery and how that may
        #  affect outcomes?

    # ============================ CLEAN DELIVERY PRACTICES AT BIRTH ==================================================

        # First we apply the estimated impact of clean birth practices on maternal and newborn risk of sepsis

        adjusted_maternal_sepsis_risk = mni[person_id]['risk_ip_sepsis'] * \
            params['rr_maternal_sepsis_clean_delivery']
        mni[person_id]['risk_ip_sepsis'] = adjusted_maternal_sepsis_risk

        adjusted_newborn_sepsis_risk = mni[person_id]['risk_newborn_sepsis'] * \
            params['rr_newborn_sepsis_clean_delivery']
        mni[person_id]['risk_newborn_sepsis'] = adjusted_newborn_sepsis_risk

# =============================== SKILLED BIRTH ATTENDANCE EFFECT ======================================================
        # Then we apply the estimated effect of fetal surveillance methods on maternal/newborn outcomes

        # TODO: Discuss again with Tim C if it is worth trying to quantify this, as many evaluative studies look at
        #  provision of BEmOC etc. so will include the impact of the interventions

# ================================== INTERVENTIONS FOR PRE-EXSISTING CONDITIONS =====================================
        # Here we apply the effect of any 'upstream' interventions that may reduce the risk of intrapartum
        # complications/poor newborn outcomes

        # ------------------------------------- PROM ------------------------------------------------------------------
        # First we apply a risk reduction in likelihood of sepsis for women with PROM as they have received prophylactic
        # antibiotics
        if mni[person_id]['PROM']:
            treatment_effect = params['rr_sepsis_post_abx_prom']
            new_sepsis_risk = mni[person_id]['risk_ip_sepsis'] * treatment_effect
            mni[person_id]['risk_ip_sepsis'] = new_sepsis_risk

        # ------------------------------------- PREMATURITY # TODO: care bundle for women in preterm labour (pg 49)----
        # Here we apply the effect of interventions to improve outcomes of neonates born preterm

        # Antibiotics for group b strep prophylaxis: (are we going to apply this to all cause sepsis?)
        if mni[person_id]['labour_state'] == 'EPTL' or mni[person_id]['labour_state'] == 'LPTL':
            mni[person_id]['risk_newborn_sepsis'] = mni[person_id]['risk_newborn_sepsis'] * \
                                                    params['rr_newborn_sepsis_proph_abx']

        # TODO: STEROIDS - effect applied directly to newborns (need consumables here) store within maternal MNI
        # TOXOLYTICS !? - WHO advises against

# ===================================  COMPLICATIONS OF LABOUR ========================================================

        # Here, using the adjusted risks calculated following 'in-labour' interventions to determine which complications
        # a woman may experience and store those in the data frame

        self.module.set_complications_during_facility_birth(person_id, complication='obstructed_labour',
                                                            labour_stage='ip',
                                                            treatment_hsi=HSI_Labour_ReceivesCareForObstructedLabour
                                                            (self.module, person_id=person_id))
        if df.at[person_id, 'la_obstructed_labour']:
            mni[person_id]['labour_is_currently_obstructed'] = True
            mni[person_id]['labour_has_previously_been_obstructed'] = True

        # (htn_treatment variable made so line would conform to flake8 formatting)
        htn_treatment = HSI_Labour_ReceivesCareForHypertensiveDisordersOfPregnancy
        self.module.set_complications_during_facility_birth(person_id, complication='eclampsia', labour_stage='ip',
                                                            treatment_hsi=htn_treatment(self.module,
                                                                                        person_id=person_id))

        self.module.set_complications_during_facility_birth(person_id, complication='antepartum_haem',
                                                            labour_stage='ip',
                                                            treatment_hsi=HSI_Labour_ReceivesCareForMaternalHaemorrhage
                                                            (self.module, person_id=person_id))

        self.module.set_complications_during_facility_birth(person_id, complication='sepsis', labour_stage='ip',
                                                            treatment_hsi=HSI_Labour_ReceivesCareForMaternalSepsis
                                                            (self.module, person_id=person_id))

        self.module.set_complications_during_facility_birth(person_id, complication='uterine_rupture',
                                                            labour_stage='ip',
                                                            treatment_hsi=HSI_Labour_ReceivesCareForObstructedLabour
                                                            (self.module, person_id=person_id))

        # TODO: issue- if we apply risk of both UR and OL here then we will negate the effect of OL treatment on
        #  reduction of incidence of UR

        # DUMMY ... (risk factors?)
        if self.module.rng.random_sample() < params['prob_cord_prolapse']:
            mni[person_id]['cord_prolapse'] = True

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT

        if mni[person_id]['uterine_rupture'] or mni[person_id]['eclampsia'] or mni[person_id]['antepartum_haem'] or\
           mni[person_id]['sepsis'] or mni[person_id]['labour_is_currently_obstructed']:
            actual_appt_footprint['NormalDelivery'] = actual_appt_footprint['CompDelivery'] * 1

        return actual_appt_footprint

    def did_not_run(self):
        return True


class HSI_Labour_ReceivesCareForObstructedLabour(HSI_Event, IndividualScopeEventMixin):
    """This is the HSI ReceivesCareForObstructedLabour. This event is scheduled by the the HSI
    PresentsForSkilledAttendanceInLabour if a woman experiences obstructed Labour. This event contains the intervetions
    around assisted vaginal delivery. This event schedules HSI ReferredForSurgicalCareInLabour when assisted delivery
    fails. This event is not finalised an both interventions and referral are subject to change"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_ReceivesCareForObstructedLabour'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1  # TODO: to be discussed with TH/TC best appt footprint to use

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2  # check this?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        # TODO: apply squeeze factor

        logger.info('This is HSI_Labour_ReceivesCareForObstructedLabour, management of obstructed labour for '
                    'person %d on date %s',
                    person_id, self.sim.date)

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code_obst_lab = pd.unique(consumables.loc[consumables[
                                                      'Intervention_Pkg'] ==
                                                      'Antibiotics for pPRoM',  # TODO: Obs labour package not working
                                                      'Intervention_Pkg_Code'])[0]

        consumables_needed = {
            'Intervention_Package_Code': {pkg_code_obst_lab: 1},
            'Item_Code': {}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed)

        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_obst_lab]:
            logger.debug('pkg_code_obst_lab is available, so use it.')
        else:
            logger.debug('pkg_code_obst_lab is not available, so can' 't use it.')
            # TODO: This will need to be equipment by equipment- i.e. if no vacuum then forceps if none then caesarean?
            # TODO: add equipment to lines on consumable chart?

        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

# =====================================  OBSTRUCTED LABOUR TREATMENT ==================================================

        # TODO: Differentiate between CPD and other?

        # For women in obstructed labour delivery is first attempted by vacuum, delivery mode is stored
        if mni[person_id]['labour_is_currently_obstructed']:
            if params['prob_deliver_ventouse'] > self.module.rng.random_sample():
                # df.at[person_id, 'la_obstructed_labour'] = False
                mni[person_id]['labour_is_currently_obstructed'] = False
                mni[person_id]['mode_of_delivery'] = 'AVDV'
                # add here effect of antibiotics?
            else:
                # If the vacuum is unsuccessful we apply the probability of successful forceps delivery
                if params['prob_deliver_forceps'] > self.module.rng.random_sample():
                    # df.at[person_id, 'la_obstructed_labour'] = False
                    mni[person_id]['labour_is_currently_obstructed'] = False
                    mni[person_id]['mode_of_delivery'] = 'AVDF'  # add here effect of antibiotics?

                # Finally if assisted vaginal delivery fails then CS is scheduled
                else:
                    event = HSI_Labour_ReferredForSurgicalCareInLabour(self.module, person_id=person_id)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                        priority=0,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset(days=1))

                    logger.info(
                        'This is HSI_Labour_ReceivesCareForObstructedLabour referring person %d for a caesarean section'
                        ' following unsuccessful attempts at assisted vaginal delivery ', person_id)

                    # Symphysiotomy??? Does this happen in malawi

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        #    actual_appt_footprint['ConWithDCSA'] = actual_appt_footprint['ConWithDCSA'] * 2

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_Labour_PresentsForInductionOfLabour: did not run')
        pass


class HSI_Labour_ReceivesCareForMaternalSepsis(HSI_Event, IndividualScopeEventMixin):
    """This is the HSI ReceivesCareForMaternalSepsis. This event is scheduled by the either HSI
    PresentsForSkilledAttendanceInLabour or HSI ReceivesCareForPostpartumPeriod if a woman experiences maternal sepsis.
    This event deliveries antibiotics as an intervention. This event is not finalised an both interventions and referral
    are subject to change"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_ReceivesCareForMaternalSepsis'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2  # check this?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        # TODO: Apply squeeze factor

        logger.info('This is HSI_Labour_ReceivesCareForMaternalSepsis, management of maternal sepsis for '
                    'person %d on date %s',
                    person_id, self.sim.date)

        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code_sepsis = pd.unique(consumables.loc[consumables[
                                                        'Intervention_Pkg'] ==
                                                    'Maternal sepsis case management',
                                                    'Intervention_Pkg_Code'])[0]

        consumables_needed = {'Intervention_Package_Code': {pkg_code_sepsis: 1}, 'Item_Code': {}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed)

        # TODO: consider 1st line/2nd line and efficacy etc etc

        # Treatment can only be delivered if appropriate antibiotics are available, if they're not the woman isn't
        # treated
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_sepsis]:
            logger.debug('pkg_code_sepsis is available, so use it.')
            if params['prob_cure_antibiotics'] > self.module.rng.random_sample():
                mni[person_id]['sepsis'] = False
        else:
            logger.debug('pkg_code_sepsis is not available, so can' 't use it.')

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        #    actual_appt_footprint['ConWithDCSA'] = actual_appt_footprint['ConWithDCSA'] * 2

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_Labour_ReceivesCareForMaternalSepsis: did not run')
        pass


class HSI_Labour_ReceivesCareForHypertensiveDisordersOfPregnancy(HSI_Event, IndividualScopeEventMixin):
    """This is the HSI ReceivesCareForHypertensiveDisordersOfPregnancy. This event is scheduled by the either HSI
    PresentsForSkilledAttendanceInLabour or HSI ReceivesCareForPostpartumPeriod if a woman develops or has presented in
    labour with any of the hypertensive disorders of pregnancy. This event delivers interventions including
    anti-convulsants and anti-hypertensives. Currently this event only manages interventions for Eclampsia and therefore
    is incomplete and subject to change"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_ReceivesCareForHypertensiveDisorder'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1  # TODO: to be discussed with TH/TC best appt footprint to use

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2  # check this?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

        logger.info('This is HSI_Labour_ReceivesCareForHypertensiveDisordersOfPregnancy, management of hypertensive '
                    'disorders of pregnancy for person %d on date %s',
                    person_id, self.sim.date)

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        pkg_code_eclampsia = pd.unique(consumables.loc[consumables[
                                                               'Intervention_Pkg'] ==
                                                       'Management of eclampsia',
                                                       'Intervention_Pkg_Code'])[0]

        item_code_nf = pd.unique(
            consumables.loc[consumables['Items'] == 'nifedipine retard 20 mg_100_IDA', 'Item_Code'])[0]
        item_code_hz = pd.unique(
            consumables.loc[consumables['Items'] == 'Hydralazine hydrochloride 20mg/ml, 1ml_each_CMST', 'Item_Code'])[0]

        # TODO: Methyldopa not being recognised?
        # item_code_md = pd.unique(
        #     consumables.loc[consumables['Items'] == 'Methyldopa 250mg_1000_CMS','Item_Code'])[0]

        item_code_hs = pd.unique(
            consumables.loc[consumables['Items'] == "Ringer's lactate (Hartmann's solution), 500 ml_20_IDA",
                            'Item_Code'])[0]

        consumables_needed = {
            'Intervention_Package_Code': {pkg_code_eclampsia: 1},
            'Item_Code': {item_code_nf: 2, item_code_hz:  2, item_code_hs: 1}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed)

        # TODO: Again determine how to reflect 1st/2nd line choice
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_eclampsia]:
            logger.debug('pkg_code_eclampsia is available, so use it.')
        else:
            logger.debug('pkg_code_eclampsia is not available, so can' 't use it.')


# =======================================  HYPERTENSION TREATMENT ======================================================
        # tbc

# =======================================  SEVERE PRE-ECLAMPSIA TREATMENT ==============================================
        # tbc

# =======================================  ECLAMPSIA TREATMENT ========================================================

        # Here we apply the treatment algorithm, unsuccessful control of seizures leads to caesarean delivery
        if mni[person_id]['eclampsia'] or mni[person_id]['eclampsia_pp']:
            if params['prob_cure_mgso4'] > self.module.rng.random_sample():
                mni[person_id]['eclampsia'] = False
            else:
                if params['prob_cure_mgso4'] > self.module.rng.random_sample():
                    mni[person_id]['eclampsia'] = False
                else:
                    if params['prob_cure_diazepam'] > self.module.rng.random_sample():
                        mni[person_id]['eclampsia'] = False
                    else:
                        event = HSI_Labour_ReferredForSurgicalCareInLabour(self.module, person_id=person_id)
                        self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                            priority=0,
                                                                            topen=self.sim.date,
                                                                            tclose=self.sim.date + DateOffset(days=1))

                        logger.info(
                            'This is HSI_Labour_ReceivesCareForHypertensiveDisordersOfPregnancy referring person %d '
                            'for a caesarean section following uncontrolled eclampsia', person_id)

        # Guidelines suggest assisting with second stage of labour via AVD - consider including this?
        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        #    actual_appt_footprint['ConWithDCSA'] = actual_appt_footprint['ConWithDCSA'] * 2

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_Labour_ReceivesCareForMaternalSepsis: did not run')
        pass


class HSI_Labour_ReceivesCareForMaternalHaemorrhage(HSI_Event, IndividualScopeEventMixin):
    """This is the HSI ReceivesCareForMaternalHaemorrhage. This event is scheduled by the either HSI
    PresentsForSkilledAttendanceInLabour or HSI ReceivesCareForPostpartumPeriod if a woman experiences an antepartum or
    postpartum haemorrhage.This event delivers a cascade of interventions to arrest bleeding. In the event of treatment
    failure this event schedules HSI referredForSurgicalCareInLabour. This event is not finalised an both interventions
    and referral are subject to change """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_ReceivesCareForMaternalHaemorrhage'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2  # check this?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        # TODO: squeeze factor AND consider having APH, PPH and RPP as separate treatment events??

        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

        logger.info('This is HSI_Labour_ReceivesCareForMaternalHaemorrhage, management of obstructed labour for person '
                    '%d on date %s', person_id, self.sim.date)

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        pkg_code_pph = pd.unique(consumables.loc[consumables[
                                                         'Intervention_Pkg'] ==
                                                 'Treatment of postpartum hemorrhage',
                                                 'Intervention_Pkg_Code'])[0]

        item_code_aph1 = pd.unique(consumables.loc[consumables['Items'] == 'Blood, one unit', 'Item_Code'])[0]
        item_code_aph2 = pd.unique(consumables.loc[consumables['Items'] == 'Lancet, blood, disposable', 'Item_Code'])[0]
        item_code_aph3 = pd.unique(consumables.loc[consumables['Items'] == 'Test, hemoglobin', 'Item_Code'])[0]
        item_code_aph4 = pd.unique(consumables.loc[consumables['Items'] == 'IV giving/infusion set, with needle',
                                                   'Item_Code'])[0]

        consumables_needed = {
            'Intervention_Package_Code': {pkg_code_pph: 1},
            'Item_Code': {item_code_aph1: 2, item_code_aph2: 1, item_code_aph3: 1, item_code_aph4: 1}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed)

        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_pph]:
            logger.debug('pkg_code_pph is available, so use it.')
        else:
            logger.debug('pkg_code_pph is not available, so can' 't use it.')

        # TODO: Again determine how to use outcome of consumable request

# ===================================  ANTEPARTUM HAEMORRHAGE TREATMENT ===============================================
        # TODO: consider severity grading?

        # Here we determine the etiology of the bleed, which will determine treatment algorithm
        etiology = ['PA', 'PP']  # May need to move this to allow for risk factors?
        probabilities = [0.67, 0.33]  # DUMMY
        mni[person_id]['source_aph'] = self.module.rng.choice(etiology, size=1, p=probabilities)

        # Storing as high chance of SB in severe placental abruption
        # TODO: Needs to be dependent on blood availability and establish how we're quantifying effect
        mni[person_id]['units_transfused'] = 2

        event = HSI_Labour_ReferredForSurgicalCareInLabour(self.module, person_id=person_id)
        self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                            priority=0,
                                                            topen=self.sim.date,
                                                            tclose=self.sim.date + DateOffset(days=1))

# ======================================= POSTPARTUM HAEMORRHAGE ======================================================
        # First we use a probability weighted random draw to determine the underlying etiology of this womans PPH

        if mni[person_id]['postpartum_haem']:
            etiology = ['UA', 'RPP']
            probabilities = [0.67, 0.33]  # dummy
            mni[person_id]['source_pph'] = self.module.rng.choice(etiology, size=1, p=probabilities)

            # Todo: add a level of severity of PPH -yes

            # ======================== TREATMENT CASCADE FOR ATONIC UTERUS:=============================================

            # Here we use a treatment cascade adapted from Malawian Obs/Gynae guidelines
            # Women who are bleeding due to atonic uterus first undergo medical management, oxytocin IV, misoprostol PR
            # and uterine massage in an attempt to stop bleeding

            if mni[person_id]['source_pph'] == 'UA':
                if params['prob_cure_oxytocin'] > self.module.rng.random_sample():
                    mni[person_id]['postpartum_haem'] = False
                else:
                    if params['prob_cure_misoprostol'] > self.module.rng.random_sample():
                        mni[person_id]['postpartum_haem'] = False
                    else:
                        if params['prob_cure_uterine_massage'] > self.module.rng.random_sample():
                            mni[person_id]['postpartum_haem'] = False
                        else:
                            if params['prob_cure_uterine_tamponade'] > self.module.rng.random_sample():
                                mni[person_id]['postpartum_haem'] = False

            # Todo: consider the impact of oxy + miso + massage as ONE value, Discuss with expert

            # ===================TREATMENT CASCADE FOR RETAINED PRODUCTS/PLACENTA:====================================
            if mni[person_id]['source_pph'] == 'RPP':
                if params['prob_cure_manual_removal'] > self.module.rng.random_sample():
                    mni[person_id]['postpartum_haem'] = False
                    # blood?

            # In the instance of uncontrolled bleeding a woman is referred on for surgical care
            if mni[person_id]['postpartum_haem']:
                event = HSI_Labour_ReferredForSurgicalCareInLabour(self.module, person_id=person_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                    priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1))

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        #    actual_appt_footprint['ConWithDCSA'] = actual_appt_footprint['ConWithDCSA'] * 2

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_Labour_ReceivesCareForMaternalHaemorrhage: did not run')
        pass


class HSI_Labour_ReceivesCareForPostpartumPeriod(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    This event manages the Health System Interaction for women who receive post partum care following delivery
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_ReceivesCareForPostpartumPeriod'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2  # check this?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        # TODO: Squeeze factor
        logger.info('This is HSI_Labour_ReceivesCareForPostpartumPeriod: Providing skilled attendance following birth '
                    'for person %d', person_id)

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code_am = pd.unique(consumables.loc[consumables[
                                                            'Intervention_Pkg'] ==
                                                'Active management of the 3rd stage of labour',
                                                'Intervention_Pkg_Code'])[0]

        consumables_needed = {
            'Intervention_Package_Code': {pkg_code_am: 1}, 'Item_Code': {}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed
        )

        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

    #  =========================  ACTIVE MANAGEMENT OF THE THIRD STAGE  ===============================================

        # Here we apply a risk reduction of post partum bleeding following active management of the third stage of
        # labour (additional oxytocin, uterine massage and controlled cord traction)
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_am]:
            logger.debug('pkg_code_am is available, so use it.')
            mni[person_id]['risk_pp_postpartum_haem'] = mni[person_id]['risk_pp_postpartum_haem'] * \
                params['rr_pph_amtsl']
        else:
            logger.debug('pkg_code_am is not available, so can' 't use it.')
            logger.debug('woman %d did not receive active management of the third stage of labour due to resource '
                         'constraints')
    # ===============================  POSTPARTUM COMPLICATIONS ========================================================

        # TODO: link eclampsia/sepsis diagnosis in SBA and PPC

        # As with the SkilledBirthAttendance HSI we recalculate risk of complications in light of preventative
        # interventions

        htn_treatment = HSI_Labour_ReceivesCareForHypertensiveDisordersOfPregnancy
        self.module.\
            set_complications_during_facility_birth(person_id, complication='eclampsia', labour_stage='pp',
                                                    treatment_hsi=htn_treatment(self.module, person_id=person_id))

        self.module.set_complications_during_facility_birth(person_id, complication='postpartum_haem',
                                                            labour_stage='pp',
                                                            treatment_hsi=HSI_Labour_ReceivesCareForMaternalHaemorrhage
                                                            (self.module, person_id=person_id))

        self.module.set_complications_during_facility_birth(person_id, complication='sepsis', labour_stage='pp',
                                                            treatment_hsi=HSI_Labour_ReceivesCareForMaternalSepsis
                                                            (self.module, person_id=person_id))

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # TODO: modify based on complications?
        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_Labour_ReceivesCareForPostpartumPeriod: did not run')
        pass


class HSI_Labour_ReferredForSurgicalCareInLabour(HSI_Event, IndividualScopeEventMixin):
    """This is the HSI ReferredForSurgicalCareInLabour.Currently this event is scheduled HSIs:
    PresentsForSkilledAttendanceInLabour and ReceivesCareForMaternalHaemorrhage, but this is not finalised.
    This event delivers surgical management for haemorrhage, uterine rupture and performs caesarean section. This event
    is not finalised and may need to be restructured to better simulate referral """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_ReferredForSurgicalCareInLabour'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()

        #   the_appt_footprint['MajorSurg'] = 1  # this appt could be used for uterine repair/pph management
        the_appt_footprint['Csection'] = 1

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2  # check this?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        #  TODO: squeeze factor and consider splitting HSIs for different surgeries

        df = self.sim.population.props
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

        logger.info('This is HSI_Labour_ReferredForSurgicalCareInLabour,providing surgical care during labour and the'
                    ' postpartum period for person %d on date %s', person_id, self.sim.date)

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        pkg_code_cs = pd.unique(consumables.loc[consumables[
                                                    'Intervention_Pkg'] ==
                                                'Cesearian Section with indication (with complication)',  # or without?
                                                'Intervention_Pkg_Code'])[0]

        consumables_needed = {
            'Intervention_Package_Code': {pkg_code_cs: 1},
            'Item_Code': {}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed)

        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_cs]:
            logger.debug('pkg_code_cs is available, so use it.')
        else:
            logger.debug('pkg_code_cs is not available, so can' 't use it.')

        # TODO: consumables

# ====================================== EMERGENCY CAESAREAN SECTION ==================================================

        if df.at[person_id, 'is_alive'] and (mni[person_id]['uterine_rupture']) or (mni[person_id]['antepartum_haem']) \
            or (mni[person_id]['eclampsia']) or (mni[person_id]['labour_is_currently_obstructed']) or \
           (df.at[person_id, 'la_current_labour_successful_induction'] == 'failed_induction'):
            # Consider all indications (elective)
            mni[person_id]['antepartum_haem'] = False
            # reset eclampsia status?
            mni[person_id]['labour_is_currently_obstructed'] = False
            mni[person_id]['mode_of_delivery'] = 'CS'
            df.at[person_id, 'la_previous_cs_delivery'] = True
            # apply risk of death from CS?

# ====================================== UTERINE REPAIR ==============================================================

        # For women with UR we determine if the uterus can be repaired surgically
        if mni[person_id]['uterine_rupture']:
            if params['prob_cure_uterine_repair'] > self.module.rng.random_sample():
                # df.at[person_id, 'la_uterine_rupture'] = False
                mni[person_id]['uterine_rupture'] = False

            # In the instance of failed surgical repair, the woman undergoes a hysterectomy
            else:
                if params['prob_cure_hysterectomy'] > self.module.rng.random_sample():
                    # df.at[person_id, 'la_uterine_rupture'] = False
                    mni[person_id]['uterine_rupture'] = False

# ================================== SURGERY FOR UNCONTROLLED POSTPARTUM HAEMORRHAGE ==================================

        # If a woman has be referred for surgery for uncontrolled post partum bleeding we use the treatment algorithm to
        # determine if her bleeding can be controlled surgically
        if mni[person_id]['postpartum_haem']:
            if params['prob_cure_uterine_ligation'] > self.module.rng.random_sample():
                mni[person_id]['postpartum_haem'] = False
            else:
                if params['prob_cure_b_lynch'] > self.module.rng.random_sample():
                    mni[person_id]['postpartum_haem'] = False
                else:
                    # Todo: similarly consider bunching surgical interventions
                    if params['prob_cure_hysterectomy'] > self.module.rng.random_sample():
                        mni[person_id]['postpartum_haem'] = False

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        #  TODO: modify based on complications?
        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_Labour_ReferredForSurgicalCareInLabour: did not run')
        pass


class HSI_Labour_PresentsForInductionOfLabour(HSI_Event, IndividualScopeEventMixin):
    """ This is the HSI PresentsForInductionOfLabour. Currently it IS NOT scheduled, but will be scheduled by the
    AntenatalCare module pending its completion. It will be responsible for the intervention of induction for women who
    are postterm. As this intervention will start labour- it will schedule either the LabourOnsetEvent or the caesarean
    section event depending on success"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)
        # TODO: await antenatal care module to schedule induction
        # TODO: women with praevia, obstructed labour, should be induced
        self.TREATMENT_ID = 'Labour_PresentsForInductionOfLabour'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1
        # TODO: review appt footprint as this wont include midwife time?

        # Define the necessary information for an HSI
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug('This is HSI_Labour_PresentsForInductionOfLabour, person %d is attending a health facility to have'
                     'their labour induced on date %s', person_id, self.sim.date)

        # TODO: discuss squeeze factors/ consumable back up with TC

        df = self.sim.population.props
        params = self.module.parameters

        # Initial request for consumables needed
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code_induction = pd.unique(consumables.loc[consumables[
                                                       'Intervention_Pkg'] ==
                                                       'Induction of labour (beyond 41 weeks)',
                                                       'Intervention_Pkg_Code'])[0]
        # TODO: review induction guidelines to confirm appropriate 1st line/2nd line consumable use

        consumables_needed = {'Intervention_Package_Code': {pkg_code_induction: 1},
                              'Item_Code': {}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self, cons_req_as_footprint=consumables_needed)

        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_induction]:
            logger.debug('pkg_code_induction is available, so use it.')
            # TODO: reschedule if consumables aren't available at this point in time?
        else:
            logger.debug('PkgCode1 is not available, so can' 't use it.')

        # We use a random draw to determine if this womans labour will be successfully induced
        # Indications: Post term, eclampsia, severe pre-eclampsia, mild pre-eclampsia at term, PROM > 24 hrs at term
        # or PPROM > 34 weeks EGA, and IUFD.

        if self.module.rng.random_sample() < params['prob_successful_induction']:
            logger.info('Person %d has had her labour successfully induced', person_id)
            df.at[person_id, 'la_current_labour_successful_induction'] = 'successful_induction'

            self.sim.schedule_event(LabourOnsetEvent(self.module, person_id), self.sim.date)
            # TODO: scheduling
        # For women whose induction fails they will undergo caesarean section
        else:
            logger.info('Persons %d labour has been unsuccessful induced', person_id)
            df.at[person_id, 'la_current_labour_successful_induction'] = 'failed_induction'
            # TODO: schedule CS or second attempt induction? -- will need to lead to labour event

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
    #    actual_appt_footprint['ConWithDCSA'] = actual_appt_footprint['ConWithDCSA'] * 2

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_Labour_PresentsForInductionOfLabour: did not run')
        pass

    #  TODO: 3 HSIs, for each facility level (interventions in functions)??
