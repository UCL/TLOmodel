"""Test file for the bed-days class"""
import copy
import os
from pathlib import Path
from typing import Dict

import pandas as pd
import pytest

from tlo import Date, Module, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import Metadata, demography, healthsystem
from tlo.methods.bed_days import BedDays
from tlo.methods.hsi_event import HSI_Event

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

bed_types = list(pd.read_csv(
    resourcefilepath / "healthsystem" / "infrastructure_and_equipment" / "ResourceFile_Bed_Capacity.csv").set_index(
    'Facility_ID').columns)

start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 200

"""Suite of tests to examine the use of BedDays class when initialised by the HealthSystem Module"""


def test_bed_days_resourcefile_defines_non_bed_space():
    """Check that "non_bed_space" is defined as the lowest type of bed. """
    assert 'non_bed_space' == bed_types[-1]


def test_beddays_in_isolation(tmpdir, seed):
    """Test the functionalities of BedDays class in the absence of HSI_Events"""
    sim = Simulation(start_date=start_date, seed=seed)
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
    )

    # Update BedCapacity data with a simple table:
    level2_facility_ids = [128, 129, 130]  # <-- the level 2 facilities for each region
    cap_bedtype1 = 5
    cap_bedtype2 = 100

    # create a simple bed capacity dataframe
    hs = sim.modules['HealthSystem']
    hs.parameters['BedCapacity'] = pd.DataFrame(
        data={
            'Facility_ID': level2_facility_ids,
            'bedtype1': cap_bedtype1,
            'bedtype2': cap_bedtype2
        }
    )

    # Create a 21-day simulation
    days_sim = 21
    sim.make_initial_population(n=100)
    sim.simulate(end_date=start_date + pd.DateOffset(days=days_sim))

    # reset bed days tracker to the start_date of the simulation
    hs.bed_days.initialise_beddays_tracker()

    # 1) impose a footprint
    person_id = 0
    dur_bedtype1 = 2
    footprint = {'bedtype1': dur_bedtype1, 'bedtype2': 0}

    sim.date = start_date
    hs.bed_days.impose_beddays_footprint(person_id=person_id, footprint=footprint)
    tracker = hs.bed_days.bed_tracker['bedtype1'][hs.bed_days.get_facility_id_for_beds(person_id)][0: days_sim]

    # check if impose footprint works as expected
    assert ([cap_bedtype1 - 1] * dur_bedtype1 + [cap_bedtype1] * (days_sim - dur_bedtype1) == tracker.values).all()

    # 2) cause someone to die and relieve their footprint from the bed-days tracker
    hs.bed_days.remove_beddays_footprint(person_id)
    assert ([cap_bedtype1] * days_sim == tracker.values).all()

    # 3) check that removing bed-days from a person without bed-days does nothing
    hs.bed_days.remove_beddays_footprint(2)
    assert ([cap_bedtype1] * days_sim == tracker.values).all()


def test_beddays_allocation_resolution(tmpdir, seed):
    sim = Simulation(start_date=start_date, seed=seed)
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
    )

    # Update BedCapacity data with a simple table:
    level2_facility_ids = [128, 129, 130]  # <-- the level 2 facilities for each region
    # This ensures over-allocations have to be properly resolved
    cap_bedtype1 = 10
    cap_bedtype2 = 10
    cap_bedtype3 = 10

    # create a simple bed capacity dataframe
    hs = sim.modules["HealthSystem"]
    hs.parameters["BedCapacity"] = pd.DataFrame(
        data={
            "Facility_ID": level2_facility_ids,
            "bedtype1": cap_bedtype1,
            "bedtype2": cap_bedtype2,
            "bedtype3": cap_bedtype3,
        }
    )

    sim.make_initial_population(n=100)
    sim.simulate(end_date=start_date)

    # reset bed days tracker to the start_date of the simulation
    hs.bed_days.initialise_beddays_tracker()

    def assert_footprint_matches_expected(
        footprint: Dict[str, int], expected_footprint: Dict[str, int]
    ):
        """
        Asserts that two footprints are identical.
        The footprint provided as the 2nd argument is assumed to be the footprint
        that we want to match, and the 1st as the result of the program attempting
        to resolve over-allocations.
        """
        assert len(footprint) == len(
            expected_footprint
        ), "Bed type footprints did not return same allocations."
        for bed_type, expected_days in expected_footprint.items():
            allocated_days = footprint[bed_type]
            assert expected_days == allocated_days, (
                f"Bed type {bed_type} was allocated {allocated_days} upon combining, "
                f"but expected it to get {expected_days}."
            )

    # Check that combining footprints for a person returns the expected output

    # SIMPLE 2-bed days case
    # Test uses example fail case given in https://github.com/UCL/TLOmodel/issues/1399
    # Person p has: bedtyp1 for 2 days, bedtype2 for 0 days.
    # Person p then assigned: bedtype1 for 1 days, bedtype2 for 6 days.
    # EXPECT: p's footprints are combined into bedtype1 for 2 days, bedtype2 for 5 days.
    existing_footprint = {"bedtype1": 2, "bedtype2": 0, "bedtype3": 0}
    incoming_footprint = {"bedtype1": 1, "bedtype2": 6, "bedtype3": 0}
    expected_resolution = {"bedtype1": 2, "bedtype2": 5, "bedtype3": 0}
    allocated_footprint = hs.bed_days.combine_footprints_for_same_patient(
        existing_footprint, incoming_footprint
    )
    assert_footprint_matches_expected(allocated_footprint, expected_resolution)

    # TEST case involve 3 different bed-types.
    # Person p has: bedtype1 for 2 days, then bedtype3 for 4 days.
    # p is assigned: bedtype1 for 1 day, bedtype2 for 3 days, and bedtype3 for 1 day.
    # EXPECT: p spends 2 days in each bedtype;
    # - Day 1 needs bedtype1 for both footprints
    # - Day 2 existing footprint at bedtype1 overwrites incoming at bedtype2
    # - Day 3 & 4 incoming footprint at bedtype2 overwrites existing allocation to bedtype3
    # - Day 5 both footprints want bedtype3
    # - Day 6 existing footprint needs bedtype3, whilst incoming footprint is over.s
    existing_footprint = {"bedtype1": 2, "bedtype2": 0, "bedtype3": 4}
    incoming_footprint = {"bedtype1": 1, "bedtype2": 3, "bedtype3": 1}
    expected_resolution = {"bedtype1": 2, "bedtype2": 2, "bedtype3": 2}
    allocated_footprint = hs.bed_days.combine_footprints_for_same_patient(
        existing_footprint, incoming_footprint
    )
    assert_footprint_matches_expected(allocated_footprint, expected_resolution)

def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def test_bed_days_basics(tmpdir, seed):
    """Check all the basic functionality about bed-days footprints and capacity management by the health-system"""

    req_for_beds = {k: i for i, k in enumerate(bed_types, start=10) if k != 'non_bed_space'}

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
            self.ACCEPTED_FACILITY_LEVEL = '1a'
            self.ALERT_OTHER_DISEASES = []

        def apply(self, person_id, squeeze_factor):
            print(f'squeeze-factor is {squeeze_factor}')
            print(f'Bed-days allocated to this event: {self.bed_days_allocated_to_this_event}')

    # Create a dummy HSI with two types of Bed Day specified
    class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = 'Dummy'
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
            self.ACCEPTED_FACILITY_LEVEL = '2'
            self.ALERT_OTHER_DISEASES = []
            self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint(req_for_beds)

        def apply(self, person_id, squeeze_factor):
            print(f'squeeze-factor is {squeeze_factor}')
            print(f'Bed-days allocated to this event: {self.bed_days_allocated_to_this_event}')

    # Create simulation:
    sim = Simulation(start_date=start_date, seed=seed, log_config={
        'filename': 'bed_days',
        'directory': tmpdir,
        'custom_levels': {
            'tlo.methods.healthsystem': logging.INFO}
    })
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
        DummyModule()
    )
    sim.make_initial_population(n=100)
    sim.simulate(end_date=start_date + pd.DateOffset(days=100))
    hs = sim.modules['HealthSystem']

    # 0) Check that structure of the log is as expected (if the healthsystem was not disabled)
    log = parse_log_file(sim.log_filepath)['tlo.methods.healthsystem']
    assert set([f"bed_tracker_{bed}" for bed in hs.bed_days.bed_types]).issubset(set(log.keys()))

    for bed_type in [f"bed_tracker_{bed}" for bed in hs.bed_days.bed_types]:
        # Check dates are as expected:
        dates_in_log = pd.to_datetime(log[bed_type]['date'])
        # Default behaviour of date_range is to include both start and end date in range
        # therefore offset end by minus one day to get all days up to but not including
        # end date. closed / inclusive kwarg avoided here to keep compatibility across
        # Pandas versions
        date_range = pd.date_range(
            sim.start_date, sim.end_date - pd.DateOffset(days=1), freq='D'
        )
        assert set(date_range) == set(dates_in_log)

        # Check columns (for each facility_ID) are as expected:
        assert ([str(x) for x in hs.parameters['BedCapacity']['Facility_ID'].values] ==
                log[bed_type].columns.drop(['date', 'date']).values).all()

    # 1) Create instances of the HSIs for a person
    person_id = 0
    hsi_nobd = HSI_Dummy_NoBedDaysSpec(module=sim.modules['DummyModule'], person_id=person_id)
    hsi_bd = HSI_Dummy(module=sim.modules['DummyModule'], person_id=person_id)

    # 2) Check that HSI_Event come with correctly formatted bed-days footprints, whether explicitly defined or not.
    hs.bed_days.check_beddays_footprint_format(hsi_nobd.BEDDAYS_FOOTPRINT)
    hs.bed_days.check_beddays_footprint_format(hsi_bd.BEDDAYS_FOOTPRINT)

    # 3) Check that helper-function to make footprints works as expected:
    assert {k: 0 for k in bed_types} == hsi_nobd.BEDDAYS_FOOTPRINT
    assert {**req_for_beds, **{"non_bed_space": 0}} == hsi_bd.BEDDAYS_FOOTPRINT

    # 4) Check that can schedule an HSI with a bed-day footprint
    hs.schedule_hsi_event(hsi_event=hsi_nobd, topen=sim.date, tclose=sim.date + pd.DateOffset(days=1), priority=0)
    hs.schedule_hsi_event(hsi_event=hsi_bd, topen=sim.date, tclose=sim.date + pd.DateOffset(days=1), priority=0)

    # 5) Check that HSI can be run by the health system with the number of bed-days provided being passed to the HSI:
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

    # 6) Check that footprint can be correctly recorded in the tracker after the HSI event is run and that
    #  '''bd_is_patient''' is updated. (All when the days fall safely inside the period of the simulation)
    # store copy of the original tracker
    orig = copy.deepcopy(hs.bed_days.bed_tracker)

    # check that person is not an in-patient before the HSI event's postapply hook is run.
    df = sim.population.props
    assert not df.at[person_id, 'hs_is_inpatient']

    # impose the footprint:
    hsi_bd._run_after_hsi_event()

    # check that person is an in-patient now
    assert df.at[person_id, 'hs_is_inpatient']

    # check imposition works:
    footprint = hsi_bd.bed_days_allocated_to_this_event

    diff = pd.DataFrame()
    for bed_type in hsi_bd.BEDDAYS_FOOTPRINT:
        diff[bed_type] = - (
            hs.bed_days.bed_tracker[bed_type].loc[:, hs.bed_days.get_facility_id_for_beds(person_id)] -
            orig[bed_type].loc[:, hs.bed_days.get_facility_id_for_beds(person_id)]
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
    for bed_type in footprint:
        assert diff[bed_type].sum() == footprint[bed_type]

    # check that beds timed to be used in the order specified (descending order of intensiveness):
    for i, bed_type in enumerate(hs.bed_days.bed_types):
        d = diff[diff.columns[i]]
        this_bed_type_starts_on = d.loc[d > 0].index.min()
        if i > 0:
            d_last_bed_type = diff[diff.columns[i - 1]]
            last_bed_type_ends_on = d_last_bed_type.loc[d_last_bed_type > 0].index.max()
            if not (pd.isnull(last_bed_type_ends_on) or pd.isnull(this_bed_type_starts_on)):
                assert this_bed_type_starts_on > last_bed_type_ends_on

    check_dtypes(sim)


def test_bed_days_property_is_inpatient(tmpdir, seed):
    """Check that the is_inpatient property is controlled correctly and kept in sync with the bed-tracker"""

    _bed_type = bed_types[0]

    class DummyModule(Module):
        METADATA = {Metadata.USES_HEALTHSYSTEM}

        def read_parameters(self, data_folder):
            # set 21 day bed days tracking period
            self.parameters['tracking_period'] = 21

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            # Schedule event that will query the status of the property 'is_inpatient' each day
            self.sim.schedule_event(
                QueryInPatientStatus(self),
                self.sim.date + pd.DateOffset(days=1)
            )
            self.in_patient_status = pd.DataFrame(
                index=pd.date_range(self.sim.start_date, self.sim.start_date + pd.DateOffset(days=self.parameters[
                    'tracking_period'])),
                columns=[0, 1, 2],
                data=False
            )

            # Schedule person_id=0 to attend care on day 3rd January
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Dummy(self, person_id=0),
                topen=Date(2010, 1, 3),
                tclose=None,
                priority=0)

            # Schedule person_id=1 to attend care on day 6th January
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Dummy(self, person_id=1),
                topen=Date(2010, 1, 6),
                tclose=None,
                priority=0)

            # Schedule person_id=2 to attend care on 13th Jan, and then again on 15th Jan
            # [overlapping in-patient durations]
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Dummy(self, person_id=2),
                topen=Date(2010, 1, 13),
                tclose=None,
                priority=0)

            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Dummy(self, person_id=2),
                topen=Date(2010, 1, 15),
                tclose=None,
                priority=0)

    class QueryInPatientStatus(RegularEvent, PopulationScopeEventMixin):
        def __init__(self, module):
            super().__init__(module, frequency=pd.DateOffset(days=1))

        def apply(self, population):
            # This event occurs _before_ the HealthSystemScheduler event, which determines changes in status. Therefore,
            # this event is reporting on the status that existed following "yesterday's" HealthSystemScheduler. So,
            # the "reporting date" is yesterday.
            reporting_date = self.sim.date - pd.DateOffset(days=1)
            self.module.in_patient_status.loc[reporting_date] = \
                population.props.loc[[0, 1, 2], 'hs_is_inpatient'].values

    # Create a dummy HSI with one particular type of bed needed fo4 5 days
    class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = 'Dummy'
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
            self.ACCEPTED_FACILITY_LEVEL = '2'
            self.ALERT_OTHER_DISEASES = []
            self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({_bed_type: 5})

        def apply(self, person_id, squeeze_factor):
            pass

    # Create simulation with the health system and DummyModule
    sim = Simulation(start_date=start_date, seed=seed, log_config={
        'filename': 'temp',
        'directory': tmpdir,
        'custom_levels': {
            'tlo.methods.healthsystem': logging.INFO,
        }
    })
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, beds_availability='all'),
        DummyModule()
    )
    sim.make_initial_population(n=100)
    sim.simulate(end_date=start_date + pd.DateOffset(days=100))
    check_dtypes(sim)

    # Load the logged tracker for general beds
    log = parse_log_file(sim.log_filepath)['tlo.methods.healthsystem']
    tracker = log[f'bed_tracker_{_bed_type}'].set_index('date')
    tracker.index = pd.to_datetime(tracker.index)

    # Load the in-patient status store:
    ips = sim.modules['DummyModule'].in_patient_status
    ips.index = pd.to_datetime(ips.index)

    # check that the daily checks on 'is_inpatient' are as expected:
    false_ser = pd.Series(index=pd.date_range(sim.start_date, sim.end_date - pd.DateOffset(days=2), freq='D'),
                          data=False)

    # person 0
    person0 = false_ser.copy()
    person0.loc[pd.date_range(Date(2010, 1, 3), Date(2010, 1, 7))] = True
    assert ips[0].equals(person0)

    # person 1
    person1 = false_ser.copy()
    person1.loc[pd.date_range(Date(2010, 1, 6), Date(2010, 1, 10))] = True
    assert ips[1].equals(person1)

    # person 2
    person2 = false_ser.copy()
    person2.loc[pd.date_range(Date(2010, 1, 13), Date(2010, 1, 19))] = True
    assert ips[2].equals(person2)

    # check that in-patient status is consistent with recorded usage of beds
    tot_time_as_in_patient = ips.sum(axis=1)
    beds_occupied = tracker.sum(axis=1)[0] - tracker.sum(axis=1)

    def assert_two_series_are_the_same_where_index_overlaps(a, b):
        return pd.concat([tot_time_as_in_patient, beds_occupied], axis=1)\
            .dropna().apply(lambda row: row[0] == row[1], axis=1).all()

    assert assert_two_series_are_the_same_where_index_overlaps(beds_occupied, tot_time_as_in_patient)


def test_bed_days_released_on_death(tmpdir, seed):
    """Check that bed-days scheduled to be occupied are released upon the death of the person"""
    _bed_type = 'general_bed'
    days_simulation_duration = 20

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
                index=pd.date_range(self.sim.start_date,
                                    self.sim.start_date + pd.DateOffset(days=days_simulation_duration)
                                    ),
                columns=[0, 1],
                data=False
            )

            # Schedule person_id=0 and person_id=1 to attend care on 3rd January for 10 days
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Dummy(self, person_id=0),
                topen=Date(2010, 1, 3),
                tclose=None,
                priority=0)

            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Dummy(self, person_id=1),
                topen=Date(2010, 1, 3),
                tclose=None,
                priority=0)

            # Schedule person_id=0 to die on 6th January
            self.sim.schedule_event(
                demography.InstantaneousDeath(self.sim.modules['Demography'], 0, 'Other'),
                Date(2010, 1, 6)
            )

    class QueryInPatientStatus(RegularEvent, PopulationScopeEventMixin):
        def __init__(self, module):
            super().__init__(module, frequency=pd.DateOffset(days=1))

        def apply(self, population):
            self.module.in_patient_status.loc[self.sim.date] = \
                population.props.loc[[0, 1], 'hs_is_inpatient'].values

    # Create a dummy HSI with both-types of Bed Day specified
    class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = 'Dummy'
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
            self.ACCEPTED_FACILITY_LEVEL = '2'
            self.ALERT_OTHER_DISEASES = []
            self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({_bed_type: 10})

        def apply(self, person_id, squeeze_factor):
            pass

    # Create simulation with the health system and DummyModule
    sim = Simulation(start_date=start_date, seed=seed, log_config={
        'filename': 'temp',
        'directory': tmpdir,
        'custom_levels': {
            'tlo.methods.healthsystem': logging.INFO,
        }
    })
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, beds_availability='all'),
        DummyModule()
    )
    sim.make_initial_population(n=100)
    sim.simulate(end_date=start_date + pd.DateOffset(days=days_simulation_duration))
    check_dtypes(sim)

    # Test that all bed-days released when person dies
    assert not sim.population.props.at[0, 'is_alive']  # person 0 has died
    assert sim.population.props.at[1, 'is_alive']  # person 1 is alive

    # Load the logged tracker for general beds
    log = parse_log_file(sim.log_filepath)['tlo.methods.healthsystem']
    tracker = log[f'bed_tracker_{_bed_type}'].set_index('date')
    tracker.index = pd.to_datetime(tracker.index)

    # compute beds occupied
    beds_occupied = tracker.sum(axis=1)[0] - tracker.sum(axis=1)

    expected_beds_occupied = pd.Series(
        index=pd.date_range(sim.start_date, sim.end_date - pd.DateOffset(days=1), freq='D'), data=0
    ).add(
        # two persons occupy beds from 3rd Jan for 10 days
        pd.Series(index=pd.date_range(Date(2010, 1, 3), Date(2010, 1, 12)), data=2), fill_value=0
    ).add(
        # death of one person, releases bed from 6th January - 12th January:
        pd.Series(index=pd.date_range(Date(2010, 1, 6), Date(2010, 1, 12)), data=-1), fill_value=0
    )

    assert beds_occupied.astype(int).equals(expected_beds_occupied.astype(int))


def test_bed_days_basics_with_healthsystem_disabled(seed):
    """Check basic functionality of bed-days class when the health-system has been disabled"""
    _bed_type = bed_types[0]

    class DummyModule(Module):
        METADATA = {Metadata.USES_HEALTHSYSTEM}

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            # Schedule person_id=0 to attend care on 3rd January
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Dummy(self, person_id=0),
                topen=Date(2010, 1, 3),
                tclose=None,
                priority=0)

            # Schedule person_id=0 to die on 6th January
            self.sim.schedule_event(
                demography.InstantaneousDeath(self.sim.modules['Demography'], 0, 'Other'),
                Date(2010, 1, 6)
            )

    # Create a dummy HSI with two types of Bed Day specified
    class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):

        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.this_ran = False
            self.TREATMENT_ID = 'Dummy'
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
            self.ACCEPTED_FACILITY_LEVEL = '2'
            self.ALERT_OTHER_DISEASES = []
            self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({
                _bed_type: 10,
            })

        def apply(self, person_id, squeeze_factor):
            self.this_ran = (self.bed_days_allocated_to_this_event == self.BEDDAYS_FOOTPRINT)
            print(f'squeeze-factor is {squeeze_factor}')
            print(f'Bed-days allocated to this event: {self.bed_days_allocated_to_this_event}')

    # Create simulation:
    sim = Simulation(start_date=start_date, seed=seed)
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                  disable=True),
        DummyModule(),
    )

    hs = sim.modules['HealthSystem']

    # Update BedCapacity data with a simple table:
    hs.parameters['BedCapacity'] = pd.DataFrame(
        data={
            'Facility_ID': [128, 129, 130],  # <-- the level 2 facilities for each region,
            _bed_type: 0,
        }
    )

    sim.make_initial_population(n=100)
    sim.simulate(end_date=start_date + pd.DateOffset(days=100))

    person_id = 0

    # ensure we can run a hsi event without errors
    hsi_bd = HSI_Dummy(module=sim.modules['DummyModule'], person_id=person_id)
    hsi_bd.apply(person_id=person_id, squeeze_factor=0.0)
    assert hsi_bd.this_ran

    assert not sim.population.props.at[0, 'is_alive']  # person 0 has died


def test_the_use_of_beds_from_multiple_facilities(seed):
    """Test the functionalities of BedDays class when multiple facilities are defined"""
    sim = Simulation(start_date=start_date, seed=seed)
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
    )

    # get shortcut to HealthSystem Module
    hs = sim.modules['HealthSystem']

    # Create a simple bed capacity dataframe with capacity designated for two regions
    hs.parameters['BedCapacity'] = pd.DataFrame(
        data={
            'Facility_ID': [129, 130],  # <-- facility_id for level 2 facilities in Northern (129) and Central (130)
            'bedtype1': 50,
            'bedtype2': 100
        }
    )

    # Create a 21-day simulation
    days_sim = 21
    sim.make_initial_population(n=100)
    sim.simulate(end_date=start_date + pd.DateOffset(days=days_sim))

    # Define the district and the facility_id to which the person will have beddays.
    person_info = [
        ("Chitipa", 129),    # <-- in the Northern region, so use facility_id 129 (for which capacity is defined)
        ("Kasungu", 130),    # <-- in the Central region, so use facility_id 130 (for which capacity is defined)
        ("Machinga", 128)    # <-- in the Southern region, so use facility_id 128 (for which no capacity is defined)
    ]

    df = sim.population.props
    for _person_id, _info in enumerate(person_info):
        df.loc[_person_id, "district_of_residence"] = _info[0]

    # reset bed days tracker to the start_date of the simulation
    hs.bed_days.initialise_beddays_tracker()

    # 1) create a footprint
    bedtype1_dur = 4
    footprint = {'bedtype1': bedtype1_dur, 'bedtype2': 0}

    sim.date = start_date
    bedtype1_capacity = hs.parameters['BedCapacity']['bedtype1']

    # impose bed days footprint for each of the persons in Northern and Central districts
    for _person_id in [0, 1]:
        hs.bed_days.impose_beddays_footprint(person_id=_person_id, footprint=footprint)

        # get facility_id that should be receive this footprint
        _fac_id = person_info[_person_id][1]

        # check if impose footprint works as expected
        tracker = hs.bed_days.bed_tracker['bedtype1'][_fac_id][0: days_sim]

        assert (
            [bedtype1_capacity.loc[bedtype1_capacity.index[_person_id]] - 1] * bedtype1_dur
            + [bedtype1_capacity.loc[bedtype1_capacity.index[_person_id]]] * (days_sim - bedtype1_dur)
            == tracker.values
        ).all()

    # -- Check that there is an error if there is demand for beddays in a region for which no capacity is defined
    # person 2 is in the Southern region for which no beddays capacity is defimed
    with pytest.raises(KeyError):
        hs.bed_days.impose_beddays_footprint(person_id=2, footprint=footprint)


def test_bed_days_allocation_to_HSI(seed):
    """Checks the functionality of `issue_bed_days_according_to_availability` in providing the best possible footprint
     to the HSI, given the requested footprint and the current state of the trackers"""

    facility_id = 0  # Arbitrary facility_id

    def prepare_sim():
        """Create and run simulation"""
        sim = Simulation(start_date=start_date, seed=seed)
        sim.register(
            demography.Demography(resourcefilepath=resourcefilepath),
            healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
        )

        # Update BedCapacity parameter with a simple table:
        hs = sim.modules['HealthSystem']
        hs.parameters['BedCapacity'] = pd.DataFrame(
            data={
                'Facility_ID': [facility_id],  # The level 2 facility that will be used,
                'bed_A': 1,  # Only one bed of the each of the required type at the facility.
                'bed_B': 1,
            }
        )

        # Simulate for 0 days to get everything initialised
        sim.make_initial_population(n=1)
        sim.simulate(end_date=start_date)
        return sim

    def check_footprint_against_expectation(footprint_requested,
                                            fn_edit_bed_tracker,
                                            expected_footprint_sent_to_hsi
                                            ):
        _sim = prepare_sim()

        # Edit tracker
        fn_edit_bed_tracker(_sim.modules['HealthSystem'].bed_days.bed_tracker)

        # See what footprint can be provided to HSI
        footprint_provided = _sim.modules['HealthSystem'].bed_days.issue_bed_days_according_to_availability(
            facility_id=facility_id, footprint=footprint_requested)

        # Check:
        assert expected_footprint_sent_to_hsi == footprint_provided

    # A: When only requesting a bed of the lowest tier:
    # A1) ... when the bed is available for all requested days:
    def make_no_changes(_bed_tracker):
        return _bed_tracker

    check_footprint_against_expectation(
        footprint_requested={'bed_A': 0, 'bed_B': 3},
        fn_edit_bed_tracker=make_no_changes,
        expected_footprint_sent_to_hsi={'bed_A': 0, 'bed_B': 3}
    )

    # A2) ... when the bed is available for only the first of the days requested;
    def make_bed_b_available_on_first_day_only(_bed_tracker):
        _bed_tracker['bed_B'][facility_id].values[1:] = 0
        return _bed_tracker

    check_footprint_against_expectation(
        footprint_requested={'bed_A': 0, 'bed_B': 3},
        fn_edit_bed_tracker=make_bed_b_available_on_first_day_only,
        expected_footprint_sent_to_hsi={'bed_A': 0, 'bed_B': 1}
    )

    # A3) ... when the bed is not available on the first day but is available on later days;
    def make_bed_b_not_available_on_first_day(_bed_tracker):
        _bed_tracker['bed_B'][facility_id].values[0] = 0
        return _bed_tracker

    check_footprint_against_expectation(
        footprint_requested={'bed_A': 0, 'bed_B': 3},
        fn_edit_bed_tracker=make_bed_b_not_available_on_first_day,
        expected_footprint_sent_to_hsi={'bed_A': 0, 'bed_B': 0}
    )

    # B: When requesting a bed of the highest tier only:
    # B1) ... when the bed is available for all requested days:
    check_footprint_against_expectation(
        footprint_requested={'bed_A': 3, 'bed_B': 0},
        fn_edit_bed_tracker=make_no_changes,
        expected_footprint_sent_to_hsi={'bed_A': 3, 'bed_B': 0}
    )

    # B2) ... when the bed is available for only the first of the days requested;
    def make_bed_a_available_on_first_day_only(_bed_tracker):
        _bed_tracker['bed_A'][facility_id].values[1:] = 0
        return _bed_tracker

    check_footprint_against_expectation(
        footprint_requested={'bed_A': 3, 'bed_B': 0},
        fn_edit_bed_tracker=make_bed_a_available_on_first_day_only,
        expected_footprint_sent_to_hsi={'bed_A': 1, 'bed_B': 2}
    )

    # B3) ... when the bed is not available on the first day but is available on later days;
    def make_bed_a_not_available_on_first_day(_bed_tracker):
        _bed_tracker['bed_A'][facility_id].values[0] = 0
        return _bed_tracker

    check_footprint_against_expectation(
        footprint_requested={'bed_A': 3, 'bed_B': 0},
        fn_edit_bed_tracker=make_bed_a_not_available_on_first_day,
        expected_footprint_sent_to_hsi={'bed_A': 0, 'bed_B': 3}
    )

    # C: When requesting a bed of the highest and lowest tier:
    # C1) ... when the bed of the higher tier is only available on first day, but the lower tier is always available
    check_footprint_against_expectation(
        footprint_requested={'bed_A': 3, 'bed_B': 3},
        fn_edit_bed_tracker=make_bed_a_available_on_first_day_only,
        expected_footprint_sent_to_hsi={'bed_A': 1, 'bed_B': 5}
    )

    # C2) ... when the bed of the higher tier is only available on first day, and the lower tier is available only on
    # 1st and 2nd day
    def make_bed_a_available_only_on_first_day_and_bed_b_available_on_first_and_second_day_only(_bed_tracker):
        _bed_tracker = make_bed_a_available_on_first_day_only(_bed_tracker)
        _bed_tracker['bed_B'][facility_id].values[2:] = 0
        return _bed_tracker

    check_footprint_against_expectation(
        footprint_requested={'bed_A': 3, 'bed_B': 3},
        fn_edit_bed_tracker=make_bed_a_available_only_on_first_day_and_bed_b_available_on_first_and_second_day_only,
        expected_footprint_sent_to_hsi={'bed_A': 1, 'bed_B': 1}
    )

    # C3) ... when the bed of the higher tier and the lower tier are only available on the 1st day
    def make_bed_a_and_b_available_on_first_day_only(_bed_tracker):
        return make_bed_b_available_on_first_day_only(
            make_bed_a_available_on_first_day_only(_bed_tracker))

    check_footprint_against_expectation(
        footprint_requested={'bed_A': 3, 'bed_B': 3},
        fn_edit_bed_tracker=make_bed_a_and_b_available_on_first_day_only,
        expected_footprint_sent_to_hsi={'bed_A': 1, 'bed_B': 0}
    )


def test_bed_days_allocation_information_is_provided_to_HSI(seed):
    """Checks the HSI is "informed" of the bed days footprint provided to it"""

    district_of_residence = 'Zomba'   # Where person_id=0 is resident: Zomba district is in in the Southern region
    facility_id = 128  # Facility that will provide the beds (Referral Hospital_Southern)
    days_of_simulation = 1
    footprint_requested = {'bed_A': 3, 'bed_B': 3}

    class DummyModule(Module):
        METADATA = {Metadata.USES_HEALTHSYSTEM}

        def __init__(self):
            super().__init__()

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            # Edit the HealthSystem bed tracker so that the bed of the higher tier is not available after
            # 1st day, but the lower tier is.
            self.sim.modules['HealthSystem'].bed_days.bed_tracker['bed_A'][facility_id].values[1] = 0

            # Make HSI event (and hold pointer to it) and schedule it to occur on the first day of the simulation.
            self.hsi_event = HSI_Dummy(module=self, person_id=0)
            self.sim.modules['HealthSystem'].schedule_hsi_event(self.hsi_event, topen=self.sim.date, priority=0)

    class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):

        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = 'Dummy'
            self.ACCEPTED_FACILITY_LEVEL = '2'
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
            self.BEDDAYS_FOOTPRINT = footprint_requested

        def apply(self, person_id, squeeze_factor):
            pass

    sim = Simulation(start_date=start_date, seed=seed)

    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
        DummyModule(),
    )

    # Update BedCapacity parameter with a simple table:
    hs = sim.modules['HealthSystem']
    hs.parameters['BedCapacity'] = pd.DataFrame(
        data={
            'Facility_ID': [facility_id],  # The level 2 facility that will be used,
            'bed_A': 1,  # Only one bed of the each of the required type at the facility.
            'bed_B': 1,
        }
    )

    # Simulate
    sim.make_initial_population(n=1)
    sim.population.props.loc[0, 'district_of_residence'] = district_of_residence
    sim.simulate(end_date=start_date + pd.DateOffset(days=days_of_simulation))

    # Return the information provided to the HSI
    assert {'bed_A': 1, 'bed_B': 5} == sim.modules['DummyModule'].hsi_event._received_info_about_bed_days


def test_in_patient_admission_included_in_appt_footprint_if_any_bed_days():
    """Check that helper function works which adds the in-patient admission appointment type to the APPT_FOOTPRINT. """
    from tlo.methods.bed_days import (
        IN_PATIENT_ADMISSION,
        IN_PATIENT_DAY_FIRST_DAY,
        IN_PATIENT_DAY_SUBSEQUENT_DAYS,
    )

    footprint = {'Under5OPD': 1}
    footprint_with_correct_inpatient_admission_and_inpatient_day = {
        **footprint, **IN_PATIENT_DAY_FIRST_DAY, **IN_PATIENT_ADMISSION
    }

    add_first_day_inpatient_appts_to_footprint = BedDays(hs_module=None).add_first_day_inpatient_appts_to_footprint

    # If in-patient admission appointment is present already, no change is made:
    assert footprint_with_correct_inpatient_admission_and_inpatient_day == \
           add_first_day_inpatient_appts_to_footprint(footprint_with_correct_inpatient_admission_and_inpatient_day)

    # If in-patient admission or in-patient appointment is not present or is incomplete, is it added:
    assert footprint_with_correct_inpatient_admission_and_inpatient_day == \
           add_first_day_inpatient_appts_to_footprint(footprint)
    assert footprint_with_correct_inpatient_admission_and_inpatient_day == \
           add_first_day_inpatient_appts_to_footprint({**footprint, **IN_PATIENT_ADMISSION})
    assert footprint_with_correct_inpatient_admission_and_inpatient_day == \
           add_first_day_inpatient_appts_to_footprint({**footprint, **IN_PATIENT_DAY_SUBSEQUENT_DAYS})

    # If the in-patient admission is wrong, then it is corrected:
    assert footprint_with_correct_inpatient_admission_and_inpatient_day == \
           add_first_day_inpatient_appts_to_footprint({'Under5OPD': 1, 'IPAdmission': 99, 'InpatientDays': 99})

    # If the footprint is blank, then the bed-days appointments are added:
    assert {**IN_PATIENT_DAY_FIRST_DAY, **IN_PATIENT_ADMISSION} == add_first_day_inpatient_appts_to_footprint({})


def test_in_patient_appt_included_and_logged(tmpdir, seed):
    """Check that in-patient appointments (admission and in-patients) are used correctly for in-patients when succ."""
    from tlo.methods.bed_days import IN_PATIENT_ADMISSION, IN_PATIENT_DAY_SUBSEQUENT_DAYS

    # Create and run a simulation that includes in-patients
    _bed_type = bed_types[0]
    date_of_admission = Date(2010, 1, 3)
    dur_stay_in_days = 5
    date_of_discharge = date_of_admission + pd.DateOffset(days=dur_stay_in_days - 1)
    footprint = {'Over5OPD': 1}
    num_persons = 3

    class DummyModule(Module):
        METADATA = {Metadata.USES_HEALTHSYSTEM}

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            # Schedule some persons to attend care on `date_of_admission`
            for person_id in range(num_persons):
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    HSI_Dummy(self, person_id=person_id),
                    topen=date_of_admission,
                    tclose=None,
                    priority=0)

    # Create a dummy HSI with one type of Bed Day specified - but no inpatient admission/care appointments specified.
    class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = 'Dummy'
            self.ACCEPTED_FACILITY_LEVEL = '2'
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint(footprint)
            self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({_bed_type: dur_stay_in_days})

        def apply(self, person_id, squeeze_factor):
            pass

    # Create simulation with the health system and DummyModule
    sim = Simulation(start_date=start_date, seed=seed, log_config={
        'filename': 'temp',
        'directory': tmpdir,
        'custom_levels': {
            "tlo.methods.healthsystem": logging.DEBUG,
        }
    })
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
        DummyModule()
    )
    sim.make_initial_population(n=100)
    sim.simulate(end_date=start_date + pd.DateOffset(days=100))
    check_dtypes(sim)

    # Load the logged tracker for general beds
    log_hsi = parse_log_file(
        sim.log_filepath, logging.DEBUG
    )['tlo.methods.healthsystem']['HSI_Event']
    log_hsi.index = pd.to_datetime(log_hsi.date)
    appts_freq_by_date = log_hsi[
        'Number_By_Appt_Type_Code'].apply(pd.Series).fillna(0).astype(int).groupby(level=0).sum()

    # Check that what is logged equals what is expected
    expectation = pd.concat(
        [
            pd.DataFrame(index=[date_of_admission],
                         data={k: num_persons * v for k, v in IN_PATIENT_ADMISSION.items()}),
            pd.DataFrame(index=pd.date_range(date_of_admission + pd.DateOffset(days=1), date_of_discharge),
                         data={k: num_persons * v for k, v in IN_PATIENT_DAY_SUBSEQUENT_DAYS.items()}),
            pd.DataFrame(index=[date_of_admission],
                         data={k: num_persons * v for k, v in footprint.items()}),
        ], axis=1).fillna(0).astype(int)

    pd.testing.assert_frame_equal(appts_freq_by_date, expectation,
                                  check_dtype=False, check_names=False, check_freq=False)

    # Check that the facility_id is included for each entry in the `HSI_Events` log, including HSI Events for
    # in-patient appointments.
    assert not (log_hsi['Facility_ID'] == -99).any()

def test_beddays_availability_switch(seed):
    """
    Test that calling bed_days.switch_beddays_availability correctly updates the
    bed capacities and adjusts the existing trackers to reflect the new capacities.
    """
    sim = Simulation(start_date=start_date, seed=seed)
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
    )

    # get shortcut to HealthSystem Module
    hs: healthsystem.HealthSystem = sim.modules["HealthSystem"]

    # As obtained from the resource file
    facility_id_with_patient =  128
    facility_id_without_patient = 129
    bedtype1_init_capacity = 5
    bedtype2_init_capacity = 10

    # Create a simple bed capacity dataframe with capacity designated for two regions
    hs.parameters["BedCapacity"] = pd.DataFrame(
        data={
            "Facility_ID": [
                facility_id_with_patient, #<-- patient 0 is admitted here
                facility_id_without_patient,
            ],
            "bedtype1": bedtype1_init_capacity,
            "bedtype2": bedtype2_init_capacity,
        }
    )
    sim.make_initial_population(n=100)
    sim.simulate(end_date=start_date)

    day_2 = start_date + pd.DateOffset(days=1)
    day_3 = start_date + pd.DateOffset(days=2)
    day_4 = start_date + pd.DateOffset(days=3)

    bed_days = hs.bed_days
    # Reset the bed occupancies
    bed_days.initialise_beddays_tracker()
    # Have a patient occupy a bed at the start of the simulation
    bed_days.impose_beddays_footprint(person_id=0, footprint={"bedtype1": 3, "bedtype2": 0})

    # Have the bed_days availability switch to "none" on the 2nd simulation day
    bed_days.switch_beddays_availability("none", effective_on_and_from=day_2)

    # We should now see that the scaled capacities are all zero
    assert (
        not bed_days._scaled_capacity.any().any()
    ), "At least one bed capacity was not set to 0"
    # We should also see that bedtype1 should have -1 beds available for days 2 and 3 of the simulation,
    # due to the existing occupancy and the new capacity of 0.
    # It should have 4 beds available on the first day (since the original capacity was 5 and the availability
    # switch happens day 2).
    # It should then have 0 beds available after (not including) day 3
    bedtype1: pd.DataFrame = bed_days.bed_tracker["bedtype1"]
    bedtype2: pd.DataFrame = bed_days.bed_tracker["bedtype2"]

    assert (
        bedtype1.loc[start_date, facility_id_with_patient] == bedtype1_init_capacity - 1
        and bedtype1.loc[start_date, facility_id_without_patient]
        == bedtype1_init_capacity
    ), "Day 1 capacities were incorrectly affected"
    assert (bedtype1.loc[day_2:day_3, facility_id_with_patient] == -1).all() and (
        bedtype1.loc[day_2:day_3, facility_id_without_patient] == 0
    ).all(), "Day 2 & 3 capacities were not updated correctly"
    assert (
        (bedtype1.loc[day_4:, :] == 0).all().all()
    ), "Day 4 onwards did not have correct capacity"

    # Bedtype 2 should have also have been updated, but there is no funny business here.
    assert (
        (bedtype2.loc[day_2:, :] == 0).all().all()
    ), "Bedtype 2 was not updated correctly"
    assert (
        (bedtype2.loc[start_date, :] == bedtype2_init_capacity).all().all()
    ), "Bedtype 2 had capacity updated on the incorrect dates"
