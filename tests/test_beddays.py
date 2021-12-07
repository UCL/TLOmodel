"""Test file for the bed-days class"""
import copy
import os
from pathlib import Path

import pandas as pd
import pytest

from tlo import Date, Module, Simulation, logging
# 1) Core functionality of the BedDays module
from tlo.analysis.utils import parse_log_file
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import Metadata, demography, healthsystem
from tlo.methods.healthsystem import HSI_Event

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 200

"""Suite of tests to examine the use of BedDays class when initialised by the HealthSystem Module"""


def test_beddays_in_isolation(tmpdir):
    """Test the functionalities of BedDays class in the absence of HSI_Events"""
    sim = Simulation(start_date=start_date)
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
    )

    # call HealthSystem Module to initialise BedDays class
    hs = sim.modules['HealthSystem']

    # Update BedCapacity data with a simple table:
    level2_facility_ids = [128, 129, 130]  # <-- the level 2 facilities for each region
    cap_bedtype1 = 5
    cap_bedtype2 = 100

    # create a simple bed capacity dataframe
    hs.parameters['BedCapacity'] = pd.DataFrame(
        data={
            'Facility_ID': level2_facility_ids,
            'bedtype1': cap_bedtype1,
            'bedtype2': cap_bedtype2
        }
    )

    # Create a 21 day simulation
    days_sim = hs.bed_days.days_until_last_day_of_bed_tracker
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
    tracker = hs.bed_days.bed_tracker['bedtype1'][hs.bed_days.get_facility_id_for_beds(person_id)]

    # check if impose footprint works as expected
    assert ([cap_bedtype1 - 1] * dur_bedtype1 + [cap_bedtype1] * (days_sim + 1 - dur_bedtype1) == tracker.values).all()

    # 2) cause someone to die and relieve their footprint from the bed-days tracker
    hs.bed_days.remove_beddays_footprint(person_id)
    assert ([cap_bedtype1] * (days_sim + 1) == tracker.values).all()

    # 3) check that removing bed-days from a person without bed-days does nothing
    hs.bed_days.remove_beddays_footprint(2)
    assert ([cap_bedtype1] * (days_sim + 1) == tracker.values).all()


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def test_bed_days_basics(tmpdir):
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
            self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({
                'high_dependency_bed': 10,
                'general_bed': 5
            })

        def apply(self, person_id, squeeze_factor):
            print(f'squeeze-factor is {squeeze_factor}')
            print(f'Bed-days allocated to this event: {self.bed_days_allocated_to_this_event}')

    # Create simulation:
    sim = Simulation(start_date=start_date, seed=0, log_config={
        'filename': 'bed_days',
        'directory': tmpdir,
        'custom_levels': {
            "BedDays": logging.INFO}
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
        dates_in_log = pd.to_datetime(log[bed_type]['date_of_bed_occupancy'])
        date_range = pd.date_range(sim.start_date, sim.end_date, freq='D', closed='left')
        assert set(date_range) == set(dates_in_log)

        # Check columns (for each facility_ID) are as expected:
        assert ([str(x) for x in hs.parameters['BedCapacity']['Facility_ID'].values] ==
                log[bed_type].columns.drop(['date', 'date_of_bed_occupancy']).values).all()

    # 1) Create instances of the HSIs for a person
    person_id = 0
    hsi_nobd = HSI_Dummy_NoBedDaysSpec(module=sim.modules['DummyModule'], person_id=person_id)
    hsi_bd = HSI_Dummy(module=sim.modules['DummyModule'], person_id=person_id)

    # 2) Check that HSI_Event come with correctly formatted bed-days footprints, whether explicitly defined or not.
    hs.bed_days.check_beddays_footprint_format(hsi_nobd.BEDDAYS_FOOTPRINT)
    hs.bed_days.check_beddays_footprint_format(hsi_bd.BEDDAYS_FOOTPRINT)

    # 3) Check that helper-function to make footprints works as expected:
    assert {'non_bed_space': 0, 'general_bed': 0, 'high_dependency_bed': 0} == hsi_nobd.BEDDAYS_FOOTPRINT
    assert {'non_bed_space': 0, 'general_bed': 5, 'high_dependency_bed': 10} == hsi_bd.BEDDAYS_FOOTPRINT

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
    import copy
    orig = copy.deepcopy(hs.bed_days.bed_tracker)

    # check that person is not an in-patient before the HSI event's postapply hook is run.
    df = sim.population.props
    assert not df.at[person_id, 'hs_is_inpatient']

    # impose the footprint:
    hsi_bd.post_apply_hook()

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


def test_bed_days_property_is_inpatient(tmpdir):
    """Check that the is_inpatient property is controlled correctly and kept in sync with the bed-tracker"""

    class DummyModule(Module):
        METADATA = {Metadata.USES_HEALTHSYSTEM}

        def read_parameters(self, data_folder):
            # get 21 day bed days tracking period
            self.parameters['tracking_period'] = 21

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            # Schedule event that will query the status of the property 'is_inpatient' each day
            self.sim.schedule_event(
                QueryInPatientStatus(self),
                self.sim.date
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
            self.module.in_patient_status.loc[self.sim.date] = \
                population.props.loc[[0, 1, 2], 'hs_is_inpatient'].values

    # Create a dummy HSI with both-types of Bed Day specified
    class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = 'Dummy'
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
            self.ACCEPTED_FACILITY_LEVEL = '2'
            self.ALERT_OTHER_DISEASES = []
            self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 5})

        def apply(self, person_id, squeeze_factor):
            pass

    # Create simulation with the health system and DummyModule
    sim = Simulation(start_date=start_date, seed=0, log_config={
        'filename': 'temp',
        'directory': tmpdir,
        'custom_levels': {
            "BedDays": logging.INFO,
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
    log = parse_log_file(sim.log_filepath)['tlo.methods.healthsystem']
    tracker = log['bed_tracker_general_bed'].drop(columns={'date'}).set_index('date_of_bed_occupancy')
    tracker.index = pd.to_datetime(tracker.index)

    # Load the in-patient status store:
    ips = sim.modules['DummyModule'].in_patient_status
    ips.index = pd.to_datetime(ips.index)

    # check that the daily checks on 'is_inpatient' are as expected:
    false_ser = pd.Series(index=pd.date_range(sim.start_date, sim.end_date - pd.DateOffset(days=1), freq='D'),
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

    assert beds_occupied.equals(tot_time_as_in_patient)


def test_bed_days_released_on_death(tmpdir):
    """Check that bed-days scheduled to be occupied are released upon the death of the person"""
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
            self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 10})

        def apply(self, person_id, squeeze_factor):
            pass

    # Create simulation with the health system and DummyModule
    sim = Simulation(start_date=start_date, seed=0, log_config={
        'filename': 'temp',
        'directory': tmpdir,
        'custom_levels': {
            "BedDays": logging.INFO,
        }
    })
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
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
    tracker = log['bed_tracker_general_bed'].drop(columns={'date'}).set_index('date_of_bed_occupancy')
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


def test_bed_days_basics_with_healthsystem_disabled():
    """Check basic functionality of bed-days class when the health-system has been disabled"""

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
                'high_dependency_bed': 10,
                'general_bed': 5
            })

        def apply(self, person_id, squeeze_factor):
            self.this_ran = (self.bed_days_allocated_to_this_event == self.BEDDAYS_FOOTPRINT)
            print(f'squeeze-factor is {squeeze_factor}')
            print(f'Bed-days allocated to this event: {self.bed_days_allocated_to_this_event}')

    # Create simulation:
    sim = Simulation(start_date=start_date, seed=0)
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
            'high_dependency_bed': 0,
            'general_bed': 0
        }
    )

    sim.make_initial_population(n=100)
    sim.simulate(end_date=start_date + pd.DateOffset(days=100))

    person_id = 0

    # ensure we can run an hsi event without errors
    hsi_bd = HSI_Dummy(module=sim.modules['DummyModule'], person_id=person_id)
    hsi_bd.apply(person_id=person_id, squeeze_factor=0.0)
    assert hsi_bd.this_ran

    assert not sim.population.props.at[0, 'is_alive']  # person 0 has died


def test_the_use_of_beds_from_multiple_facilities():
    """Test the functionalities of BedDays class when multiple facilities are defined"""
    sim = Simulation(start_date=start_date)
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

    # Create a simulation that has the same duration as the window of the tracker
    days_sim = hs.bed_days.days_until_last_day_of_bed_tracker
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
        tracker = hs.bed_days.bed_tracker['bedtype1'][_fac_id]

        assert ([bedtype1_capacity.loc[bedtype1_capacity.index[_person_id]] - 1] * bedtype1_dur + [
            bedtype1_capacity.loc[bedtype1_capacity.index[_person_id]]] * (
                    days_sim + 1 - bedtype1_dur) == tracker.values).all()

    # -- Check that there is an error if there is demand for beddays in a region for which no capacity is defined
    # person 2 is in the Southern region for which no beddays capacity is defimed
    with pytest.raises(KeyError):
        hs.bed_days.impose_beddays_footprint(person_id=2, footprint=footprint)


def test_bed_days_allocation_to_one_bed_type():
    """This test checks the integrity of bed days allocation algorithm with only one bed-type defined and given
     different scenarios i.e. when the bed is available for all requested days, available for the first but
     not all of the days requested, not available on the first day, but available on later days and available
     on 1st, 3rd and 5th day"""

    class DummyModule(Module):
        METADATA = {Metadata.USES_HEALTHSYSTEM}

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            pass

    # Create a dummy HSI with one type of Bed Day specified
    class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):

        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.this_ran = False
            self.TREATMENT_ID = 'Dummy'
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
            self.ACCEPTED_FACILITY_LEVEL = 2
            self.ALERT_OTHER_DISEASES = []
            self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({
                'high_dependency_bed': 5
            })

        def apply(self, person_id, squeeze_factor):
            self.this_ran = (self.bed_days_allocated_to_this_event == self.BEDDAYS_FOOTPRINT)
            print(f'squeeze-factor is {squeeze_factor}')
            print(f'Bed-days allocated to this event: {self.bed_days_allocated_to_this_event}')

    # Create simulation:
    sim = Simulation(start_date=start_date, seed=0)
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
        DummyModule(),
    )

    hs = sim.modules['HealthSystem']

    # Update BedCapacity data with a simple table:
    hs.parameters['BedCapacity'] = pd.DataFrame(
        data={
            'Facility_ID': [128, 129, 130],  # <-- the level 2 facilities for each region,
            'high_dependency_bed': 0,     # make bed-type unavailable - to be reset later based on the given scenario
        }
    )

    # Create a 21 days simulation
    days_sim = 21
    sim.make_initial_population(n=100)
    sim.simulate(end_date=start_date + pd.DateOffset(days=days_sim))

    # reset bed days tracker to the start_date of the simulation
    hs.bed_days.initialise_beddays_tracker()

    sim.date = start_date

    """ 1) test bed available for all requested days"""
    person_id = 0  # individual id in population

    # get bed days dataframe
    beds_days_df = hs.bed_days.bed_tracker["high_dependency_bed"]

    # initialise bed available days
    bed_available_days = [sim.date + pd.DateOffset(days=_d) for _d in range(5)]

    # check bed is not available before reset tracker
    assert all(0 == beds_days_df[hs.bed_days.get_facility_id_for_beds(person_id)].loc[bed_available_days])

    # reset bed tracker to make bed available for all requested days - 5 days
    beds_days_df[hs.bed_days.get_facility_id_for_beds(person_id)].loc[bed_available_days] = 1

    # check all requested days are now available
    assert all(1 == beds_days_df[hs.bed_days.get_facility_id_for_beds(person_id)].loc[bed_available_days])

    # run HSI_Dummy Event and check whether days are allocated as expected - all days should be allocated
    hsi_bd = HSI_Dummy(module=sim.modules['DummyModule'], person_id=person_id)

    hsi_bd.apply(person_id=person_id, squeeze_factor=0.0)
    assert hsi_bd.this_ran, "the event did not run"  # check the event run

    # check bed days are allocated as expected - 1. check all days of bed-type A are allocated
    #                                            2. check total allocation equates to 5 (bed-type A allocation)
    assert 5 == hsi_bd.bed_days_allocated_to_this_event['high_dependency_bed'], "not all days are allocated"
    assert 5 == sum(hsi_bd.bed_days_allocated_to_this_event.values()), "not all days are allocated"

    # impose footprint
    hsi_bd.post_apply_hook()

    # check impose footprint works
    assert 0 == beds_days_df[hs.bed_days.get_facility_id_for_beds(person_id)].sum(), 'equating different values'

    """ 2) test Bed available for the first but not all of the days requested """
    # ------------- reset variables --------------
    person_id = 1  # person id
    hs.bed_days.initialise_beddays_tracker()  # bed tracker
    beds_days_df = hs.bed_days.bed_tracker["high_dependency_bed"]  # bed days dataframe

    # initialise bed available days
    bed_available_days = [sim.date]
    # --------------------------------------------

    # check bed is not available before reset tracker
    assert all(0 == beds_days_df[hs.bed_days.get_facility_id_for_beds(person_id)].loc[bed_available_days])

    # reset bed tracker to make bed available only on first day
    beds_days_df[hs.bed_days.get_facility_id_for_beds(person_id)].loc[bed_available_days] = 1

    # check tracker is reset as expected - only 1 day should be available and it should be sim.date
    assert 1 == beds_days_df[hs.bed_days.get_facility_id_for_beds(person_id)].loc[sim.date]
    assert 1 == beds_days_df[hs.bed_days.get_facility_id_for_beds(person_id)].sum(), "bed tracker is not properly reset"

    # run HSI_Dummy Event and check whether days are allocated as expected
    hsi_bd = HSI_Dummy(module=sim.modules['DummyModule'], person_id=person_id)

    hsi_bd.apply(person_id=person_id, squeeze_factor=0.0)
    assert hsi_bd.this_ran, "the event did not run"  # check the event run

    # check bed days are allocated as expected - 1. check only one day of bed-type A is allocated
    #                                            2. check total allocation equates to 1 (bed-type A allocation)
    assert 1 == hsi_bd.bed_days_allocated_to_this_event['high_dependency_bed'], "not all days are allocated"
    assert 1 == sum(hsi_bd.bed_days_allocated_to_this_event.values()), "days are not properly allocated"

    # impose footprint on person_id 1 starting from the first day of simulation
    hsi_bd.post_apply_hook()

    # check impose footprint works
    assert 0 == beds_days_df[hs.bed_days.get_facility_id_for_beds(person_id)].sum(), 'equating different values'

    """ 3) test Bed not available on the first day, but available on later days """
    # ------------- reset variables --------------
    person_id = 2  # person id
    hs.bed_days.initialise_beddays_tracker()  # bed tracker
    beds_days_df = hs.bed_days.bed_tracker["high_dependency_bed"]  # bed days dataframe

    # initialise bed available days
    bed_available_days = [sim.date + pd.DateOffset(days=_d) for _d in range(1, days_sim)]
    # --------------------------------------------

    # check bed is not available before reset tracker
    assert all(0 == beds_days_df[hs.bed_days.get_facility_id_for_beds(person_id)].loc[bed_available_days])

    # make bed unavailable on the first day but available on following days
    beds_days_df[hs.bed_days.get_facility_id_for_beds(person_id)].loc[bed_available_days] = 1

    # check tracker is reset as expected (1. the bed unavailable date should be the start_date or sim.date
    #                                     2. all other days should be available
    #                                     3. a sum of available days should be equal to 20 (days_sim which is 21 - 1)
    assert 0 == beds_days_df[hs.bed_days.get_facility_id_for_beds(person_id)].loc[sim.date]
    assert all(1 == beds_days_df[hs.bed_days.get_facility_id_for_beds(person_id)].loc[bed_available_days])
    assert 20 == beds_days_df[
        hs.bed_days.get_facility_id_for_beds(person_id)].sum(), "bed tracker is not properly reset"

    # run HSI_Dummy Event and check whether days are allocated as expected
    hsi_bd = HSI_Dummy(module=sim.modules['DummyModule'], person_id=person_id)

    hsi_bd.apply(person_id=person_id, squeeze_factor=0.0)
    assert hsi_bd.this_ran, "the event did not run"  # check the event run

    # check bed days are allocated as expected - 1. check no day of bed-type A is allocated
    #                                            2. check total allocation equates to 0 (no day is allocated)
    assert 0 == hsi_bd.bed_days_allocated_to_this_event['high_dependency_bed'], "no day should be allocated"
    assert 0 == sum(hsi_bd.bed_days_allocated_to_this_event.values()), "days are not properly allocated"

    # impose footprint on person_id 2 starting from the first day of simulation
    hsi_bd.post_apply_hook()

    # check that nothing happens - 1. check that the start date is still unavailable
    #                              2. check that all other days remain available as before
    assert 0 == beds_days_df[hs.bed_days.get_facility_id_for_beds(person_id)].loc[sim.date]
    assert all(1 == beds_days_df[hs.bed_days.get_facility_id_for_beds(person_id)].loc[bed_available_days])

    """4) test Bed available on 1st, 3rd and 5th day"""
    # ------------- reset variables --------------
    person_id = 3  # person id
    hs.bed_days.initialise_beddays_tracker()  # bed tracker
    beds_days_df = hs.bed_days.bed_tracker["high_dependency_bed"]  # bed days dataframe

    # initialise bed available days
    bed_available_days = [sim.date + pd.DateOffset(days=_d) for _d in range(0, 5, 2)]
    # --------------------------------------------

    # check bed is not available before reset tracker
    assert all(0 == beds_days_df[hs.bed_days.get_facility_id_for_beds(person_id)].loc[bed_available_days])

    # make bed available on 1st, 3rd and 5th day
    beds_days_df[hs.bed_days.get_facility_id_for_beds(person_id)].loc[bed_available_days] = 1

    # check tracker is reset as expected - only 3 days should be available
    assert all(1 == beds_days_df[hs.bed_days.get_facility_id_for_beds(person_id)].loc[bed_available_days])
    assert 3 == beds_days_df[hs.bed_days.get_facility_id_for_beds(person_id)].sum(), "bed tracker is not properly reset"

    # run HSI_Dummy Event and check whether days are allocated as expected
    hsi_bd = HSI_Dummy(module=sim.modules['DummyModule'], person_id=person_id)

    hsi_bd.apply(person_id=person_id, squeeze_factor=0.0)
    assert hsi_bd.this_ran, "the event did not run"  # check the event run

    # check bed days are allocated as expected - 1. check only one day of bed-type A is allocated
    #                                            2. check total allocation equates to 1 (bed-type A allocation)
    assert 1 == hsi_bd.bed_days_allocated_to_this_event['high_dependency_bed'], "not all days are allocated"
    assert 1 == sum(hsi_bd.bed_days_allocated_to_this_event.values()), "days are not properly allocated"

    # impose footprint on person_id 3 starting from the first day of simulation
    hsi_bd.post_apply_hook()

    # check that impose footprint works - 1. check that the footprint is imposed only on sim.date
    #                                     2. check that the other 2 available days are still available
    #                                     3. check that sum tracker is now 2 instead of previous 3

    assert 0 == beds_days_df[hs.bed_days.get_facility_id_for_beds(person_id)].loc[sim.date]
    assert all(1 == beds_days_df[hs.bed_days.get_facility_id_for_beds(person_id)].loc[bed_available_days[1:]])
    assert 2 == beds_days_df[hs.bed_days.get_facility_id_for_beds(person_id)].sum(), "bed tracker is not properly reset"


def test_multiple_bed_types_allocation_with_lower_class_bed_always_available():
    """This test checks the integrity of bed days allocation algorithm with lower class bed always available and given
    different scenarios i.e. when bed of both types available for all days requested, bed A available for the first
    but not all of the days requested (but bed B available), bed A not available on the first day, but available on
    later days (bed B available) and bed A available on 1st, 3rd and 5th day """

    class DummyModule(Module):
        METADATA = {Metadata.USES_HEALTHSYSTEM}

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            pass

    # Create a dummy HSI with one type of Bed Day specified
    class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):

        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.this_ran = False
            self.TREATMENT_ID = 'Dummy'
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
            self.ACCEPTED_FACILITY_LEVEL = 2
            self.ALERT_OTHER_DISEASES = []
            self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({
                'high_dependency_bed': 5,
                'general_bed': 5
            })

        def apply(self, person_id, squeeze_factor):
            self.this_ran = (self.bed_days_allocated_to_this_event == self.BEDDAYS_FOOTPRINT)
            print(f'squeeze-factor is {squeeze_factor}')
            print(f'Bed-days allocated to this event: {self.bed_days_allocated_to_this_event}')

    # Create simulation:
    sim = Simulation(start_date=start_date, seed=0)
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
        DummyModule(),
    )

    hs = sim.modules['HealthSystem']

    # Update BedCapacity data with a simple table:
    hs.parameters['BedCapacity'] = pd.DataFrame(
        data={
            'Facility_ID': [128, 129, 130],  # <-- the level 2 facilities for each region,
            'high_dependency_bed': 0,
            'general_bed': 1    # make general_bed always available
        }
    )

    # Create a 21 days simulation
    days_sim = 21
    sim.make_initial_population(n=100)
    sim.simulate(end_date=start_date + pd.DateOffset(days=days_sim))

    sim.date = start_date

    """1) test Bed of both types available for all days requested (bed-type B available)"""
    person_id = 0  # person id in population

    # reset bed days tracker to the start_date of the simulation
    hs.bed_days.initialise_beddays_tracker()

    # get bed days dataframe
    beds_days_df = hs.bed_days.bed_tracker

    # initialise bed available days
    bed_available_days = [sim.date + pd.DateOffset(days=_d) for _d in range(5)]

    # check bed type A is not available before tracker is reset
    assert all(0 == beds_days_df["high_dependency_bed"][
        hs.bed_days.get_facility_id_for_beds(person_id)].loc[bed_available_days])

    # reset bed tracker to make bed type A available for all requested days - 5 days
    beds_days_df['high_dependency_bed'][hs.bed_days.get_facility_id_for_beds(person_id)].loc[bed_available_days] = 1

    # copy bed days dataframe
    tracker_before_impose_footprint = copy.deepcopy(hs.bed_days.bed_tracker)

    # check bed type A is now available after tracker is reset
    assert all(1 == beds_days_df["high_dependency_bed"][
        hs.bed_days.get_facility_id_for_beds(person_id)].loc[bed_available_days])

    # run HSI_Dummy Event and check whether days are allocated as expected
    hsi_bd = HSI_Dummy(module=sim.modules['DummyModule'], person_id=person_id)

    hsi_bd.apply(person_id=person_id, squeeze_factor=0.0)
    # check HSI_Dummy event run
    assert hsi_bd.this_ran, "the event did not run"

    # check bed days are allocated as expected - 1. check all days of bed-type A are allocated
    #                                            2. check all days of bed-type B are allocated
    #                                            3. check total allocated days equates to a total of bed type A plus B
    assert 5 == hsi_bd.bed_days_allocated_to_this_event['high_dependency_bed']
    assert 5 == hsi_bd.bed_days_allocated_to_this_event['general_bed']
    assert 10 == sum(hsi_bd.bed_days_allocated_to_this_event.values()), "not all days are allocated"

    # impose footprint
    hsi_bd.post_apply_hook()

    tracker_after_impose_footprint = hs.bed_days.bed_tracker

    # check impose footprint works - 1. check all days of bed-type A are imposed
    #                                2. check all days of bed-type B are imposed
    #                                3. check total imposed days equates to a total of bed type A plus B
    total_allocation = 0
    for bed_type in hs.bed_days.bed_types:
        total_bed_allocation = tracker_before_impose_footprint[bed_type].subtract(
                tracker_after_impose_footprint[bed_type])

        total_allocation += total_bed_allocation.sum(axis=1).sum()

    assert 5 == tracker_before_impose_footprint['high_dependency_bed'].subtract(
                                                tracker_after_impose_footprint['high_dependency_bed']).sum(axis=1).sum()
    assert 5 == tracker_before_impose_footprint['general_bed'].subtract(
                                                tracker_after_impose_footprint['general_bed']).sum(axis=1).sum()
    assert 10 == total_allocation, "not all bed days were allocated"

    """2) test Bed A available for the first but not all of the days requested (bed-type B available)"""
    # ------------- reset variables --------------
    person_id = 1  # person id
    hs.bed_days.initialise_beddays_tracker()  # bed tracker
    beds_days_df = hs.bed_days.bed_tracker  # bed days dataframe

    # initialise bed available days
    bed_available_days = [sim.date]
    # --------------------------------------------

    # check bed type A is not available before tracker is reset
    assert 0 == beds_days_df["high_dependency_bed"][hs.bed_days.get_facility_id_for_beds(person_id)].sum()

    # reset bed tracker to make bed type A available for the first but not all of the days requested
    beds_days_df['high_dependency_bed'][hs.bed_days.get_facility_id_for_beds(person_id)].loc[
        bed_available_days] = 1

    # check reset bed tracker works as expected - bed type A should be available only on sim.date
    assert 1 == beds_days_df["high_dependency_bed"][hs.bed_days.get_facility_id_for_beds(person_id)].loc[sim.date]
    assert 1 == beds_days_df["high_dependency_bed"][hs.bed_days.get_facility_id_for_beds(person_id)].sum()

    # copy bed days dataframe
    tracker_before_impose_footprint = copy.deepcopy(hs.bed_days.bed_tracker)

    # run HSI_Dummy Event and check whether days are allocated as expected
    hsi_bd = HSI_Dummy(module=sim.modules['DummyModule'], person_id=person_id)

    hsi_bd.apply(person_id=person_id, squeeze_factor=0.0)
    # check HSI_Dummy event run
    assert hsi_bd.this_ran, "the event did not run"

    # check bed days are allocated as expected - 1. check only one day of bed-type A is allocated
    #                                            2. check the remaining days from bed-type A have been allocated
    #                                               to bed type B
    #                                            3. check total allocation equates to a total of bed type A plus B
    assert 1 == hsi_bd.bed_days_allocated_to_this_event['high_dependency_bed']
    assert 9 == hsi_bd.bed_days_allocated_to_this_event['general_bed']
    assert 10 == sum(hsi_bd.bed_days_allocated_to_this_event.values()), "not all days are allocated"

    # impose footprint
    hsi_bd.post_apply_hook()

    tracker_after_impose_footprint = hs.bed_days.bed_tracker

    # check impose footprint works - 1. check only one day of bed-type A is imposed
    #                                2. check the remaining days from bed-type A have been imposed to bed type B
    #                                3. check total imposed days equates to a total of bed type A plus B
    total_allocation = 0
    for bed_type in hs.bed_days.bed_types:
        total_bed_allocation = tracker_before_impose_footprint[bed_type].subtract(
            tracker_after_impose_footprint[bed_type])

        total_allocation += total_bed_allocation.sum(axis=1).sum()

    assert 1 == tracker_before_impose_footprint['high_dependency_bed'].subtract(
        tracker_after_impose_footprint['high_dependency_bed']).sum(axis=1).sum()

    assert 9 == tracker_before_impose_footprint['general_bed'].subtract(
        tracker_after_impose_footprint['general_bed']).sum(axis=1).sum()

    assert 10 == total_allocation, "not all bed days were allocated"

    """3) test Bed not available on the first day, but available on later days (bed-type B available)"""
    # ------------- reset variables --------------
    person_id = 2  # person id
    hs.bed_days.initialise_beddays_tracker()  # bed tracker
    beds_days_df = hs.bed_days.bed_tracker  # bed days dataframe

    # initialise bed available days
    bed_available_days = [sim.date + pd.DateOffset(days=_d) for _d in range(1, days_sim)]
    # --------------------------------------------

    # check bed type A is not available before tracker is reset
    assert 0 == beds_days_df["high_dependency_bed"][hs.bed_days.get_facility_id_for_beds(person_id)].sum()

    # reset bed tracker to make bed type A not available on the first day, but available on later days
    beds_days_df['high_dependency_bed'][hs.bed_days.get_facility_id_for_beds(person_id)].loc[
        bed_available_days] = 1

    # check reset bed tracker works as expected - bed type A should not be available on the first day but available
    # on later days
    assert 0 == beds_days_df["high_dependency_bed"][hs.bed_days.get_facility_id_for_beds(person_id)].loc[sim.date]
    assert all(1 == beds_days_df[
        "high_dependency_bed"][hs.bed_days.get_facility_id_for_beds(person_id)].loc[bed_available_days])

    # copy bed days dataframe
    tracker_before_impose_footprint = copy.deepcopy(hs.bed_days.bed_tracker)

    # run HSI_Dummy Event and check whether days are allocated as expected
    hsi_bd = HSI_Dummy(module=sim.modules['DummyModule'], person_id=person_id)

    hsi_bd.apply(person_id=person_id, squeeze_factor=0.0)
    # check HSI_Dummy event run
    assert hsi_bd.this_ran, "the event did not run"

    # check bed days are allocated as expected - 1. check no bed day is allocated to bed-type A
    #                                            2. check all days of bed-type A have been allocated to bed type B
    #                                            3. check total allocation equates to a total of bed type A plus B
    assert 0 == hsi_bd.bed_days_allocated_to_this_event['high_dependency_bed']
    assert 10 == hsi_bd.bed_days_allocated_to_this_event['general_bed']
    assert 10 == sum(hsi_bd.bed_days_allocated_to_this_event.values()), "not all days are allocated"

    # impose footprint
    hsi_bd.post_apply_hook()

    tracker_after_impose_footprint = hs.bed_days.bed_tracker

    # check impose footprint works - 1. check no bed day is imposed on bed-type A
    #                                2. check all days from bed-type A have been imposed to bed type B
    #                                3. check total imposed days equates to a total of bed type A plus B
    total_allocation = 0
    for bed_type in hs.bed_days.bed_types:
        total_bed_allocation = tracker_before_impose_footprint[bed_type].subtract(
            tracker_after_impose_footprint[bed_type])

        total_allocation += total_bed_allocation.sum(axis=1).sum()

    assert 0 == tracker_before_impose_footprint['high_dependency_bed'].subtract(
        tracker_after_impose_footprint['high_dependency_bed']).sum(axis=1).sum()

    assert 10 == tracker_before_impose_footprint['general_bed'].subtract(
        tracker_after_impose_footprint['general_bed']).sum(axis=1).sum()

    assert 10 == total_allocation, "not all bed days were imposed"

    """4) test Bed available on 1st, 3rd and 5th day (bed-type B available)"""
    # ------------- reset variables --------------
    person_id = 3  # person id
    hs.bed_days.initialise_beddays_tracker()  # bed tracker
    beds_days_df = hs.bed_days.bed_tracker  # bed days dataframe

    # initialise bed available days
    bed_available_days = [sim.date + pd.DateOffset(days=_d) for _d in range(0, 5, 2)]
    # --------------------------------------------

    # check bed type A is not available before tracker is reset
    assert 0 == beds_days_df["high_dependency_bed"][hs.bed_days.get_facility_id_for_beds(person_id)].sum()

    # reset bed tracker to make bed-typeA available on 1st, 3rd and 5th day
    beds_days_df['high_dependency_bed'][hs.bed_days.get_facility_id_for_beds(person_id)].loc[
        bed_available_days] = 1

    # check reset bed tracker works as expected - bed type A should be available on 1st, 3rd and 5th day
    assert all(1 == beds_days_df[
        "high_dependency_bed"][hs.bed_days.get_facility_id_for_beds(person_id)].loc[bed_available_days])

    # copy bed days dataframe
    tracker_before_impose_footprint = copy.deepcopy(hs.bed_days.bed_tracker)

    # run HSI_Dummy Event and check whether days are allocated as expected
    hsi_bd = HSI_Dummy(module=sim.modules['DummyModule'], person_id=person_id)

    hsi_bd.apply(person_id=person_id, squeeze_factor=0.0)
    # check HSI_Dummy event run
    assert hsi_bd.this_ran, "the event did not run"

    # check bed days are allocated as expected - 1. check only one day of bed-type A is allocated
    #                                            2. check the remaining days from bed-type A have been allocated
    #                                               to bed type B
    #                                            3. check total allocation equates to a total of bed type A plus B
    assert 1 == hsi_bd.bed_days_allocated_to_this_event['high_dependency_bed']
    assert 9 == hsi_bd.bed_days_allocated_to_this_event['general_bed']
    assert 10 == sum(hsi_bd.bed_days_allocated_to_this_event.values()), "not all days are allocated"

    # impose footprint
    hsi_bd.post_apply_hook()

    tracker_after_impose_footprint = hs.bed_days.bed_tracker

    # check impose footprint works - 1. check only one day of bed-type A is imposed
    #                                2. check the remaining days from bed-type A have been imposed to bed type B
    #                                3. check total imposed days equates to a total of bed type A plus B
    total_allocation = 0
    for bed_type in hs.bed_days.bed_types:
        total_bed_allocation = tracker_before_impose_footprint[bed_type].subtract(
            tracker_after_impose_footprint[bed_type])

        total_allocation += total_bed_allocation.sum(axis=1).sum()

    assert 1 == tracker_before_impose_footprint['high_dependency_bed'].subtract(
        tracker_after_impose_footprint['high_dependency_bed']).sum(axis=1).sum()

    assert 9 == tracker_before_impose_footprint['general_bed'].subtract(
        tracker_after_impose_footprint['general_bed']).sum(axis=1).sum()

    assert 10 == total_allocation, "not all bed days were allocated"


def test_multiple_bed_types_allocation_with_lower_class_bed_never_available():
    """This test checks the integrity of bed days allocation algorithm with lower class bed never available and given
        different scenarios i.e. when bed-type A is available for all days requested, bed A available for the first
        but not all of the days requested (bed-type B not available), bed A not available on the first day, but
        available on later days (bed-type B not available) and bed A available on 1st, 3rd and 5th day """

    class DummyModule(Module):
        METADATA = {Metadata.USES_HEALTHSYSTEM}

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            pass

    # Create a dummy HSI with one type of Bed Day specified
    class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):

        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.this_ran = False
            self.TREATMENT_ID = 'Dummy'
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
            self.ACCEPTED_FACILITY_LEVEL = 2
            self.ALERT_OTHER_DISEASES = []
            self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({
                'high_dependency_bed': 5,
                'general_bed': 5
            })

        def apply(self, person_id, squeeze_factor):
            self.this_ran = (self.bed_days_allocated_to_this_event == self.BEDDAYS_FOOTPRINT)
            print(f'squeeze-factor is {squeeze_factor}')
            print(f'Bed-days allocated to this event: {self.bed_days_allocated_to_this_event}')

    # Create simulation:
    sim = Simulation(start_date=start_date, seed=0)
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
        DummyModule(),
    )

    hs = sim.modules['HealthSystem']

    # Update BedCapacity data with a simple table:
    hs.parameters['BedCapacity'] = pd.DataFrame(
        data={
            'Facility_ID': [128, 129, 130],  # <-- the level 2 facilities for each region,
            'high_dependency_bed': 0,
            'general_bed': 0  # make general_bed never available
        }
    )

    # Create a 21 days simulation
    days_sim = 21
    sim.make_initial_population(n=100)
    sim.simulate(end_date=start_date + pd.DateOffset(days=days_sim))

    sim.date = start_date

    """1) test Bed-type A available for all days requested (bed-type B never available)"""
    person_id = 0  # person id in population

    # reset bed days tracker to the start_date of the simulation
    hs.bed_days.initialise_beddays_tracker()

    # get bed days dataframe
    beds_days_df = hs.bed_days.bed_tracker

    # initialise bed available days
    bed_available_days = [sim.date + pd.DateOffset(days=_d) for _d in range(5)]

    # check bed type A is not available before tracker is reset
    assert all(0 == beds_days_df["high_dependency_bed"][
        hs.bed_days.get_facility_id_for_beds(person_id)].loc[bed_available_days])

    # reset bed tracker to make bed type A available for all requested days - 5 days
    beds_days_df['high_dependency_bed'][hs.bed_days.get_facility_id_for_beds(person_id)].loc[
        bed_available_days] = 1

    # check bed type A is now available after tracker is reset
    assert all(1 == beds_days_df["high_dependency_bed"][
        hs.bed_days.get_facility_id_for_beds(person_id)].loc[bed_available_days])

    # copy bed days dataframe
    tracker_before_impose_footprint = copy.deepcopy(hs.bed_days.bed_tracker)

    # run HSI_Dummy Event and check whether days are allocated as expected
    hsi_bd = HSI_Dummy(module=sim.modules['DummyModule'], person_id=person_id)

    hsi_bd.apply(person_id=person_id, squeeze_factor=0.0)
    # check HSI_Dummy event run
    assert hsi_bd.this_ran, "the event did not run"

    # check bed days allocation is as expected - 1. check all days of bed-type A are allocated
    #                                            2. check no day of bed-type B is allocated
    #                                            3. check total allocated days equates to a total of bed type A(5)
    assert 5 == hsi_bd.bed_days_allocated_to_this_event['high_dependency_bed']
    assert 0 == hsi_bd.bed_days_allocated_to_this_event['general_bed']
    assert 5 == sum(hsi_bd.bed_days_allocated_to_this_event.values()), "not all days are allocated"

    # impose footprint
    hsi_bd.post_apply_hook()

    tracker_after_impose_footprint = hs.bed_days.bed_tracker

    # check impose footprint works - 1. check all days of bed-type A are imposed
    #                                2. check no day of bed-type B is imposed
    #                                3. check total imposed days equates to a total of bed type A(5)
    total_allocation = 0
    for bed_type in hs.bed_days.bed_types:
        total_bed_allocation = tracker_before_impose_footprint[bed_type].subtract(
            tracker_after_impose_footprint[bed_type])

        total_allocation += total_bed_allocation.sum(axis=1).sum()

    assert 5 == tracker_before_impose_footprint['high_dependency_bed'].subtract(
        tracker_after_impose_footprint['high_dependency_bed']).sum(axis=1).sum()
    assert 0 == tracker_before_impose_footprint['general_bed'].subtract(
        tracker_after_impose_footprint['general_bed']).sum(axis=1).sum()
    assert 5 == total_allocation, "not all bed days were allocated"

    """2) test Bed-type A available for the first but not all of the days requested (bed-type B never available)"""
    # ------------- reset variables --------------
    person_id = 1  # person id
    hs.bed_days.initialise_beddays_tracker()  # bed tracker
    beds_days_df = hs.bed_days.bed_tracker  # bed days dataframe

    # initialise bed available days
    bed_available_days = [sim.date]
    # --------------------------------------------

    # check bed type A is not available before tracker is reset
    assert 0 == beds_days_df["high_dependency_bed"][hs.bed_days.get_facility_id_for_beds(person_id)].sum()

    # reset bed tracker to make bed type A for the first but not all of the days requested
    beds_days_df['high_dependency_bed'][hs.bed_days.get_facility_id_for_beds(person_id)].loc[
        bed_available_days] = 1

    # check reset bed tracker works as expected - bed type A should now be available on the first day only
    assert 1 == beds_days_df["high_dependency_bed"][hs.bed_days.get_facility_id_for_beds(person_id)].loc[sim.date]
    assert 1 == beds_days_df["high_dependency_bed"][hs.bed_days.get_facility_id_for_beds(person_id)].sum()

    # copy bed days dataframe
    tracker_before_impose_footprint = copy.deepcopy(hs.bed_days.bed_tracker)

    # run HSI_Dummy Event and check whether days are allocated as expected
    hsi_bd = HSI_Dummy(module=sim.modules['DummyModule'], person_id=person_id)

    hsi_bd.apply(person_id=person_id, squeeze_factor=0.0)
    # check HSI_Dummy event run
    assert hsi_bd.this_ran, "the event did not run"

    # check bed days are allocated as expected - 1. check only one day of bed-type A is allocated
    #                                            2. check no bed day is transferred or allocated to bed-type B
    #                                            3. check total allocation equates to 1 (bed-type A allocation)
    assert 1 == hsi_bd.bed_days_allocated_to_this_event['high_dependency_bed']
    assert 0 == hsi_bd.bed_days_allocated_to_this_event['general_bed']
    assert 1 == sum(hsi_bd.bed_days_allocated_to_this_event.values()), "not all days are allocated"

    # impose footprint
    hsi_bd.post_apply_hook()

    tracker_after_impose_footprint = hs.bed_days.bed_tracker

    # check impose footprint works - 1. check only one day of bed-type A is imposed
    #                                2. check no bed day is transferred or imposed to bed-type B
    #                                3. check total imposed days equates to 1 (bed-type A allocation)
    total_allocation = 0
    for bed_type in hs.bed_days.bed_types:
        total_bed_allocation = tracker_before_impose_footprint[bed_type].subtract(
            tracker_after_impose_footprint[bed_type])

        total_allocation += total_bed_allocation.sum(axis=1).sum()

    assert 1 == tracker_before_impose_footprint['high_dependency_bed'].subtract(
        tracker_after_impose_footprint['high_dependency_bed']).sum(axis=1).sum()

    assert 0 == tracker_before_impose_footprint['general_bed'].subtract(
        tracker_after_impose_footprint['general_bed']).sum(axis=1).sum()

    assert 1 == total_allocation, "not all bed days were allocated"

    """3) test Bed-type A not available on the first day, but available on later days (bed-type B never available)"""
    # ------------- reset variables --------------
    person_id = 2  # person id
    hs.bed_days.initialise_beddays_tracker()  # bed tracker
    beds_days_df = hs.bed_days.bed_tracker  # bed days dataframe

    # initialise bed available days
    bed_available_days = [sim.date + pd.DateOffset(days=_d) for _d in range(1, days_sim)]
    # --------------------------------------------

    # check bed type A is not available before tracker is reset
    assert 0 == beds_days_df["high_dependency_bed"][hs.bed_days.get_facility_id_for_beds(person_id)].sum()

    # reset bed tracker to make bed type A not available on the first day, but available on later days
    beds_days_df['high_dependency_bed'][hs.bed_days.get_facility_id_for_beds(person_id)].loc[
        bed_available_days] = 1

    # check reset bed tracker works as expected - bed type A should not be available on the first day but available
    # on later days
    assert 0 == beds_days_df["high_dependency_bed"][hs.bed_days.get_facility_id_for_beds(person_id)].loc[sim.date]
    assert all(1 == beds_days_df[
        "high_dependency_bed"][hs.bed_days.get_facility_id_for_beds(person_id)].loc[bed_available_days])

    # copy bed days dataframe
    tracker_before_impose_footprint = copy.deepcopy(hs.bed_days.bed_tracker)

    # run HSI_Dummy Event and check whether days are allocated as expected
    hsi_bd = HSI_Dummy(module=sim.modules['DummyModule'], person_id=person_id)

    hsi_bd.apply(person_id=person_id, squeeze_factor=0.0)
    # check HSI_Dummy event run
    assert hsi_bd.this_ran, "the event did not run"

    # check bed days are allocated as expected - 1. check no bed day is allocated to bed-type A
    #                                            2. check no bed day is transferred or allocated to bed type B
    #                                            3. check total allocation equates to 0(since there's no
    #                                            allocation from both bed-types)
    assert 0 == hsi_bd.bed_days_allocated_to_this_event['high_dependency_bed']
    assert 0 == hsi_bd.bed_days_allocated_to_this_event['general_bed']
    assert 0 == sum(hsi_bd.bed_days_allocated_to_this_event.values()), "not all days are allocated"

    # impose footprint
    hsi_bd.post_apply_hook()

    tracker_after_impose_footprint = hs.bed_days.bed_tracker

    # check impose footprint works - 1. check no bed day is imposed on bed-type A
    #                                2. check no bed day is transferred or imposed to bed type B
    #                                3. check total imposed days equates to 0(since there's no
    #                                   allocation from both bed-types)
    total_allocation = 0
    for bed_type in hs.bed_days.bed_types:
        total_bed_allocation = tracker_before_impose_footprint[bed_type].subtract(
            tracker_after_impose_footprint[bed_type])

        total_allocation += total_bed_allocation.sum(axis=1).sum()

    assert 0 == tracker_before_impose_footprint['high_dependency_bed'].subtract(
        tracker_after_impose_footprint['high_dependency_bed']).sum(axis=1).sum()

    assert 0 == tracker_before_impose_footprint['general_bed'].subtract(
        tracker_after_impose_footprint['general_bed']).sum(axis=1).sum()

    assert 0 == total_allocation, "not all bed days were allocated"

    """4) test Bed-type A available on 1st, 3rd and 5th day (bed-type B never available)"""
    # ------------- reset variables --------------
    person_id = 3  # person id
    hs.bed_days.initialise_beddays_tracker()  # bed tracker
    beds_days_df = hs.bed_days.bed_tracker  # bed days dataframe

    # initialise bed available days
    bed_available_days = [sim.date + pd.DateOffset(days=_d) for _d in range(0, 5, 2)]
    # --------------------------------------------

    # check bed type A is not available before tracker is reset
    assert 0 == beds_days_df["high_dependency_bed"][hs.bed_days.get_facility_id_for_beds(person_id)].sum()

    # reset bed tracker to make bed-typeA available on 1st, 3rd and 5th day
    beds_days_df['high_dependency_bed'][hs.bed_days.get_facility_id_for_beds(person_id)].loc[
        bed_available_days] = 1

    # check reset bed tracker works as expected - bed type A should now be available on 1st, 3rd and 5th day
    assert all(1 == beds_days_df[
        "high_dependency_bed"][hs.bed_days.get_facility_id_for_beds(person_id)].loc[bed_available_days])

    # copy bed days dataframe
    tracker_before_impose_footprint = copy.deepcopy(hs.bed_days.bed_tracker)

    # run HSI_Dummy Event and check whether days are allocated as expected
    hsi_bd = HSI_Dummy(module=sim.modules['DummyModule'], person_id=person_id)

    hsi_bd.apply(person_id=person_id, squeeze_factor=0.0)
    # check HSI_Dummy event run
    assert hsi_bd.this_ran, "the event did not run"

    # check bed days are allocated as expected - 1. check only one day of bed-type A is allocated
    #                                            2. check no bed day is transferred or allocated to bed type B
    #                                            3. check total allocation equates to 1 (bed-type A allocation)
    assert 1 == hsi_bd.bed_days_allocated_to_this_event['high_dependency_bed']
    assert 0 == hsi_bd.bed_days_allocated_to_this_event['general_bed']
    assert 1 == sum(hsi_bd.bed_days_allocated_to_this_event.values()), "not all days are allocated"

    # impose footprint
    hsi_bd.post_apply_hook()

    tracker_after_impose_footprint = hs.bed_days.bed_tracker

    # check impose footprint works - 1. check only one day of bed-type A is imposed
    #                                2. check no bed day is transferred or imposed to bed type B
    #                                3. check total imposed days equates to 1 (bed-type A allocation)
    total_allocation = 0
    for bed_type in hs.bed_days.bed_types:
        total_bed_allocation = tracker_before_impose_footprint[bed_type].subtract(
            tracker_after_impose_footprint[bed_type])

        total_allocation += total_bed_allocation.sum(axis=1).sum()

    assert 1 == tracker_before_impose_footprint['high_dependency_bed'].subtract(
        tracker_after_impose_footprint['high_dependency_bed']).sum(axis=1).sum()

    assert 0 == tracker_before_impose_footprint['general_bed'].subtract(
        tracker_after_impose_footprint['general_bed']).sum(axis=1).sum()

    assert 1 == total_allocation
