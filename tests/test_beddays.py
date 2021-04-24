"""Test file for the bed-days"""

import os
from pathlib import Path

import pandas as pd

from tlo import Date, Module, Simulation
# 1) Core functionality of the BedDays module
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import Metadata, bed_days, demography, healthsystem
from tlo.methods.healthsystem import HSI_Event

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 200

"""Suite of tests to examine the use of BedDays module when abstracted away from the HealthSystem Module"""


def test_beddays_in_isolation():
    sim = Simulation(start_date=start_date)
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        bed_days.BedDays(resourcefilepath=resourcefilepath)
    )
    bd = sim.modules['BedDays']

    # Update BedCapacity data with a simple table:
    default_facility_id = 0
    cap_bedtype1 = 100
    cap_bedtype2 = 100

    sim.modules['BedDays'].parameters['BedCapacity'] = pd.DataFrame(
        index=[0],
        data={
            'Facility_ID': default_facility_id,
            'bedtype1': cap_bedtype1,
            'bedtype2': cap_bedtype2
        }
    )

    # Create a ten day simulation
    days_sim = 10
    sim.make_initial_population(n=100)
    sim.simulate(end_date=start_date + pd.DateOffset(days=days_sim))

    # 1) impose a footprint
    person_id = 0
    dur_bedtype1 = 5
    footprint = {'bedtype1': dur_bedtype1, 'bedtype2': 0}

    sim.date = start_date
    bd.impose_beddays_footprint(person_id=person_id, footprint=footprint)
    tracker = bd.bed_tracker['bedtype1'][default_facility_id]
    assert ([cap_bedtype1 - 1] * dur_bedtype1 + [cap_bedtype1] * (days_sim + 1 - dur_bedtype1) == tracker.values).all()

    # 2) cause someone to die and relieve their footprint from the bed-days tracker
    bd.remove_beddays_footprint(person_id)
    assert ([cap_bedtype1] * (days_sim + 1) == tracker.values).all()

    # 3) check that removing bed-days from a person without bed-days does nothing
    bd.remove_beddays_footprint(2)
    assert ([cap_bedtype1] * (days_sim + 1) == tracker.values).all()


"""Suite of tests to examine the use of BedDays module when used as it is during a simulation - i.e. through the
HealthSystem Module"""


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def check_bed_days_basics(hs_disable):
    """Check all the basic functionality about bed-days footprints and capacity management by the health-system"""

    class DummyModule(Module):
        METADATA = {Metadata.USES_HEALTHSYSTEM}

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            pass

    # Create a dummy HSI event with No Bed-days specified
    class HSI_Dummy_NoBedDaysSpec(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = 'Dummy'
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
            self.ACCEPTED_FACILITY_LEVEL = 1
            self.ALERT_OTHER_DISEASES = []

        def apply(self, person_id, squeeze_factor):
            print(f'squeeze-factor is {squeeze_factor}')
            print(f'Bed-days allocated to this event: {self.bed_days_allocated_to_this_event}')

    # Create a dummy HSI with both-types of Bed Day specified
    class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = 'Dummy'
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
            self.ACCEPTED_FACILITY_LEVEL = 2
            self.ALERT_OTHER_DISEASES = []
            self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({
                'high_dependency_bed': 10,
                'general_bed': 5
            })

        def apply(self, person_id, squeeze_factor):
            print(f'squeeze-factor is {squeeze_factor}')
            print(f'Bed-days allocated to this event: {self.bed_days_allocated_to_this_event}')

    # Create simulation with the healthsystem and DummyModule
    sim = Simulation(start_date=start_date, seed=0)
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=hs_disable),
        bed_days.BedDays(resourcefilepath=resourcefilepath),
        DummyModule()
    )
    sim.make_initial_population(n=100)
    sim.simulate(end_date=start_date + pd.DateOffset(days=100))
    hs = sim.modules['HealthSystem']
    bd = sim.modules['BedDays']

    # 0) Create instances of the HSI's for a person
    person_id = 0
    hsi_nobd = HSI_Dummy_NoBedDaysSpec(module=sim.modules['DummyModule'], person_id=person_id)
    hsi_bd = HSI_Dummy(module=sim.modules['DummyModule'], person_id=person_id)

    # 1) Check that HSI_Event come with correctly formatted bed-days footprints, whether explicitly defined or not.
    bd.check_beddays_footrpint_format(hsi_nobd.BEDDAYS_FOOTPRINT)
    bd.check_beddays_footrpint_format(hsi_bd.BEDDAYS_FOOTPRINT)

    # 2) Check that helper-function to make footprints works as expected:
    assert {'non_bed_space': 0, 'general_bed': 0, 'high_dependency_bed': 0} \
           == hsi_nobd.make_beddays_footprint({})
    assert {'non_bed_space': 0, 'general_bed': 4, 'high_dependency_bed': 1} \
           == hsi_nobd.make_beddays_footprint({'general_bed': 4, 'high_dependency_bed': 1})

    # 3) Check that can schedule an HSI with a bed-day footprint
    hs.schedule_hsi_event(hsi_event=hsi_nobd, topen=sim.date, tclose=sim.date + pd.DateOffset(days=1), priority=0)
    hs.schedule_hsi_event(hsi_event=hsi_bd, topen=sim.date, tclose=sim.date + pd.DateOffset(days=1), priority=0)

    # 4) Check that HSI can be run by the health system with the number of bed-days provided being passed to the HSI:
    #       - if the health-system does update the '_received_info_about_bed_days' property:
    info_sent_to_hsi = {k: int(hsi_bd.BEDDAYS_FOOTPRINT[k] * 0.5) for k in hsi_bd.BEDDAYS_FOOTPRINT}
    hsi_bd._received_info_about_bed_days = info_sent_to_hsi
    hsi_bd.apply(person_id=0, squeeze_factor=0.0)
    assert info_sent_to_hsi == hsi_bd.bed_days_allocated_to_this_event

    #       - confirm that if the `_received_info_about_bed_days` is not written to, it defaults to the full footprint
    #       (this it what happens when the event is from inside the HSIEventWrapper)
    hsi_bd_a = HSI_Dummy(module=sim.modules['DummyModule'], person_id=0)
    hsi_bd_a.apply(person_id=0, squeeze_factor=0.0)
    assert hsi_bd_a.bed_days_allocated_to_this_event == hsi_bd_a.BEDDAYS_FOOTPRINT
    assert hsi_bd_a.is_all_beddays_allocated()

    # 4) Check that footprint can be correctly recorded in the tracker after the HSI event is run and that
    #  '''bd_is_patient''' is updated. (All when the days fall safely inside the period of the simulation)

    # store copy of the original tracker
    import copy
    orig = copy.deepcopy(bd.bed_tracker)

    # check that person is not an in-patient before the HSI event's postapply hook is run.
    df = sim.population.props
    assert not df.at[person_id, 'bd_is_inpatient']

    # impose the footprint:
    sim.date = start_date + pd.DateOffset(days=5)
    hsi_bd.post_apply_hook()

    # check that person is an in-patient now
    assert df.at[person_id, 'bd_is_inpatient']  # should be flagged as in-patient

    # check imposition works:
    footprint = hsi_bd.bed_days_allocated_to_this_event
    the_facility_id = 0  # <-- default id for the facility_id

    diff = pd.DataFrame()
    for bed_type in hsi_bd.BEDDAYS_FOOTPRINT:
        diff[bed_type] = - (
            bd.bed_tracker[bed_type].loc[:, the_facility_id] - orig[bed_type].loc[:, the_facility_id]
        )

    first_day = diff[diff.sum(axis=1) > 0].index.min()
    last_day = diff[diff.sum(axis=1) > 0].index.max()

    assert diff.sum().sum() == sum(footprint.values())
    assert (diff.sum(axis=1) <= 1).all()
    assert first_day == sim.date
    assert last_day == sim.date + pd.DateOffset(days=sum(footprint.values()) - 1)
    assert (1 == diff.loc[
        (diff.index >= first_day) &
        (diff.index <= last_day)].sum(axis=1)).all()

    # check that beds timed to be used in the order specified (descending order of intensiveness):
    for i, bed_type in enumerate(sim.modules['BedDays'].bed_types):
        d = diff[diff.columns[i]]
        this_bed_type_starts_on = d.loc[d > 0].index.min()
        if i > 0:
            d_last_bed_type = diff[diff.columns[i - 1]]
            last_bed_type_ends_on = d_last_bed_type.loc[d_last_bed_type > 0].index.max()
            if not (pd.isnull(last_bed_type_ends_on) or pd.isnull(this_bed_type_starts_on)):
                assert this_bed_type_starts_on > last_bed_type_ends_on

    # - Check the same but when the days implied in the footprint extend beyond the period of the simulation:
    bd.initialise_beddays_tracker()

    # store copy of the original tracker
    orig = copy.deepcopy(bd.bed_tracker)

    # impose the footprint (that will extend past end of the simulation): should not error and should not extend df
    sim.date = sim.end_date - pd.DateOffset(days=1)
    hsi_bd.post_apply_hook()

    # check that additional row have not been added
    for bed_type in bd.bed_tracker:
        assert all(orig[bed_type].index == bd.bed_tracker[bed_type].index)

    # compute difference in the bed-tracker compared to its initial state
    diff = pd.DataFrame()
    for bed_type in hsi_bd.BEDDAYS_FOOTPRINT:
        diff[bed_type] = - (
            bd.bed_tracker[bed_type].loc[:, the_facility_id] - orig[bed_type].loc[:, the_facility_id]
        )

    # tracker should show only the 2 days in the high-dependency bed that occur before end of simulation
    assert orig['non_bed_space'].equals(bd.bed_tracker['non_bed_space'])
    assert orig['general_bed'].equals(bd.bed_tracker['general_bed'])
    assert all(
        [0] * 99 + [1] * 2 == (
            orig['high_dependency_bed'].loc[:, the_facility_id] -
            bd.bed_tracker['high_dependency_bed'].loc[:, the_facility_id]
        ).values
    )


def check_bed_days_property_is_inpatient(hs_disable):
    """Check that the is_inpatient property is controlled correctly and kept in sync with the bed-tracker"""

    class DummyModule(Module):
        METADATA = {Metadata.USES_HEALTHSYSTEM}

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            # Schedule event that will query the status of the property 'is_inpatient' each day
            self.sim.schedule_event(
                QueryInPatientStatus(self),
                self.sim.date
            )
            self.in_patient_status = pd.DataFrame(
                index=pd.date_range(self.sim.start_date, self.sim.end_date),
                columns=[0, 1, 2],
                data=False
            )

            # Schedule person_id=0 to attend care on day 2
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Dummy(self, person_id=0),
                topen=self.sim.date + pd.DateOffset(days=2),
                tclose=None,
                priority=0)
            # Schedule person_id=1 to attend care on day 5
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Dummy(self, person_id=1),
                topen=self.sim.date + pd.DateOffset(days=5),
                tclose=None,
                priority=0)

            # Schedule person_id=2 to attend care on day 12, and then again on day 14 [overlapping in-patient durations]
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Dummy(self, person_id=2),
                topen=self.sim.date + pd.DateOffset(days=12),
                tclose=None,
                priority=0)
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Dummy(self, person_id=2),
                topen=self.sim.date + pd.DateOffset(days=14),
                tclose=None,
                priority=0)

    class QueryInPatientStatus(RegularEvent, PopulationScopeEventMixin):
        def __init__(self, module):
            super().__init__(module, frequency=pd.DateOffset(days=1))

        def apply(self, population):
            self.module.in_patient_status.loc[self.sim.date.normalize()] = \
                population.props.loc[[0, 1, 2], 'bd_is_inpatient'].values

    # Create a dummy HSI with both-types of Bed Day specified
    class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = 'Dummy'
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
            self.ACCEPTED_FACILITY_LEVEL = 2
            self.ALERT_OTHER_DISEASES = []
            self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 5})

        def apply(self, person_id, squeeze_factor):
            pass

    # Create simulation with the health system and DummyModule
    sim = Simulation(start_date=start_date, seed=0)
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=hs_disable),
        bed_days.BedDays(resourcefilepath=resourcefilepath),
        DummyModule()
    )
    sim.make_initial_population(n=100)
    sim.simulate(end_date=start_date + pd.DateOffset(days=20))

    # check that the daily checks on 'is_inpatient' are as expected:
    assert all([False] * 2 + [True] * 5 + [False] * 14 ==
               sim.modules['DummyModule'].in_patient_status[0].values
               )
    assert all([False] * 5 + [True] * 5 + [False] * 11 ==
               sim.modules['DummyModule'].in_patient_status[1].values
               )
    assert all([False] * 12 + [True] * 7 + [False] * 2 ==
               sim.modules['DummyModule'].in_patient_status[2].values
               )

    # check that in-patient status is consistent with recorded usage of beds
    tot_time_as_in_patient = sim.modules['DummyModule'].in_patient_status.sum(axis=1)
    tracker = sim.modules['BedDays'].bed_tracker['general_bed']
    beds_occupied = tracker.sum(axis=1)[0] - tracker.sum(axis=1)
    assert (beds_occupied == tot_time_as_in_patient).all()

    check_dtypes(sim)


def check_bed_days_released_on_death(hs_disable):
    """Check that bed-days scheduled to be occupied are released upon the death of the person"""

    class DummyModule(Module):
        METADATA = {Metadata.USES_HEALTHSYSTEM}

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            # Schedule event that will query the status of the property 'is_inpatient' each day
            self.sim.schedule_event(
                QueryInPatientStatus(self),
                self.sim.date
            )
            self.in_patient_status = pd.DataFrame(
                index=pd.date_range(self.sim.start_date, self.sim.end_date),
                columns=[0, 1],
                data=False
            )

            # Schedule person_id=0 and person_id=1 to attend care on day 2 for 10 days
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Dummy(self, person_id=0),
                topen=self.sim.date + pd.DateOffset(days=2),
                tclose=None,
                priority=0)

            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Dummy(self, person_id=1),
                topen=self.sim.date + pd.DateOffset(days=2),
                tclose=None,
                priority=0)

            # Schedule person_id=0 to die on day 5
            self.sim.schedule_event(
                demography.InstantaneousDeath(self.sim.modules['Demography'], 0, ''),
                self.sim.date + pd.DateOffset(days=5)
            )

    class QueryInPatientStatus(RegularEvent, PopulationScopeEventMixin):
        def __init__(self, module):
            super().__init__(module, frequency=pd.DateOffset(days=1))

        def apply(self, population):
            self.module.in_patient_status.loc[self.sim.date.normalize()] = \
                population.props.loc[[0, 1], 'bd_is_inpatient'].values

    # Create a dummy HSI with both-types of Bed Day specified
    class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = 'Dummy'
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
            self.ACCEPTED_FACILITY_LEVEL = 2
            self.ALERT_OTHER_DISEASES = []
            self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 10})

        def apply(self, person_id, squeeze_factor):
            pass

    # Create simulation with the health system and DummyModule
    sim = Simulation(start_date=start_date, seed=0)
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=hs_disable),
        bed_days.BedDays(resourcefilepath=resourcefilepath),
        DummyModule()
    )
    sim.make_initial_population(n=100)
    sim.simulate(end_date=start_date + pd.DateOffset(days=20))

    # Test that all bed-days released when person dies
    assert not sim.population.props.at[0, 'is_alive']  # person 0 has died
    assert sim.population.props.at[1, 'is_alive']  # person 1 is alive

    tracker = sim.modules['BedDays'].bed_tracker['general_bed']
    bed_occupied = tracker.sum(axis=1)[0] - tracker.sum(axis=1)
    assert all([0] * 2 + [2] * 3 + [1] * 7 + [0] * 9 == bed_occupied.values)


def test_bed_days_if_healthsystem_not_disabled():
    check_bed_days_basics(hs_disable=False)
    check_bed_days_property_is_inpatient(hs_disable=False)
    check_bed_days_released_on_death(hs_disable=False)


def test_bed_days_if_healthsystem_is_disabled():
    # check_bed_days_basics(hs_disable=True)
    check_bed_days_property_is_inpatient(hs_disable=True)
    check_bed_days_released_on_death(hs_disable=True)
