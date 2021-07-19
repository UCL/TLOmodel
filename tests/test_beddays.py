"""Test file for the bed-days"""

import os
from pathlib import Path

import pandas as pd

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

    sim = Simulation(start_date=start_date)
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
    )

    # call HealthSystem Module to initialise BedDays class
    hs = sim.modules['HealthSystem']

    # Update BedCapacity data with a simple table:
    default_facility_id = 0
    cap_bedtype1 = 5
    cap_bedtype2 = 100

    # create a simple bed capacity dataframe
    hs.parameters['BedCapacity'] = pd.DataFrame(
        index=[0],
        data={
            'Facility_ID': default_facility_id,
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
    tracker = hs.bed_days.bed_tracker['bedtype1'][default_facility_id]

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
            self.ACCEPTED_FACILITY_LEVEL = 1
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
            self.ACCEPTED_FACILITY_LEVEL = 2
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
    log = parse_log_file(sim.log_filepath)['tlo.methods.bed_days']
    assert set([f"bed_tracker_{bed}" for bed in hs.bed_days.bed_types]) == set(log.keys())

    for bed_type in [f"bed_tracker_{bed}" for bed in hs.bed_days.bed_types]:
        # Check dates are as expected:
        dates_in_log = pd.to_datetime(log[bed_type]['date_of_bed_occupancy'])
        date_range = pd.date_range(sim.start_date, sim.end_date, freq='D', closed='left')
        assert set(date_range) == set(dates_in_log)

        # Check columns (for each facility_ID) are as expected:
        assert [str(x) for x in hs.parameters['BedCapacity']['Facility_ID'].values] == \
               log[bed_type].columns.drop(['date', 'date_of_bed_occupancy']).values

    # 1) Create instances of the HSIs for a person
    person_id = 0
    hsi_nobd = HSI_Dummy_NoBedDaysSpec(module=sim.modules['DummyModule'], person_id=person_id)
    hsi_bd = HSI_Dummy(module=sim.modules['DummyModule'], person_id=person_id)

    # 2) Check that HSI_Event come with correctly formatted bed-days footprints, whether explicitly defined or not.
    hs.bed_days.check_beddays_footrpint_format(hsi_nobd.BEDDAYS_FOOTPRINT)
    hs.bed_days.check_beddays_footrpint_format(hsi_bd.BEDDAYS_FOOTPRINT)

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
    the_facility_id = 0  # <-- default id for the facility_id

    diff = pd.DataFrame()
    for bed_type in hsi_bd.BEDDAYS_FOOTPRINT:
        diff[bed_type] = - (
            hs.bed_days.bed_tracker[bed_type].loc[:, the_facility_id] - orig[bed_type].loc[:, the_facility_id]
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
            self.ACCEPTED_FACILITY_LEVEL = 2
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
    log = parse_log_file(sim.log_filepath)['tlo.methods.bed_days']
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
            self.ACCEPTED_FACILITY_LEVEL = 2
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
    log = parse_log_file(sim.log_filepath)['tlo.methods.bed_days']
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


def test_bed_days_basics_with_healthsystem_disabled(tmpdir):
    """Check basic functionality of bed-days class when the health-system has been disabled"""

    # Create simulation:
    sim = Simulation(start_date=start_date, seed=0, log_config={
        'filename': 'bed_days',
        'directory': tmpdir,
        'custom_levels': {
            "BedDays": logging.INFO}
    })
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                  disable=True),
    )
    sim.make_initial_population(n=100)
    sim.simulate(end_date=start_date + pd.DateOffset(days=100))
    hs = sim.modules['HealthSystem']

    # get the number of tracked days from beddays class
    tracker_days = hs.bed_days.days_until_last_day_of_bed_tracker

    # create a datetime series from start date to a days until last day of bed tracker
    date_s = pd.date_range(start_date, start_date + pd.DateOffset(days=tracker_days))

    """ since health system is disabled, tracker index will not shift. This implies that tracker index will be equal to
     date_s variable. check that this is true """
    for bed_type in hs.bed_days.bed_types:
        assert (hs.bed_days.bed_tracker[bed_type].index == date_s).all()

    """disabling health system will also cause only start_date to be logged as a result of calling on_simulation_end
    method of the health system."""

    # Check that structure of the log is as expected
    log = parse_log_file(sim.log_filepath)['tlo.methods.bed_days']
    assert set([f"bed_tracker_{bed}" for bed in hs.bed_days.bed_types]) == set(log.keys())

    # check that only one day has been logged and that that day(date_of_bed_occupancy) is start_date.
    for bed_type in set(log.keys()):
        assert log[bed_type].index.size == 1
        assert log[bed_type].loc[log[bed_type].index[0], "date_of_bed_occupancy"] == start_date
