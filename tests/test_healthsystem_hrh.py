import heapq as hp
import os
from pathlib import Path

import pandas as pd
import pytest

from tlo import Date, Module, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.events import IndividualScopeEventMixin
from tlo.methods import (
    Metadata,
    chronicsyndrome,
    demography,
    enhanced_lifestyle,
    epi,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    mockitis,
    simplified_births,
    symptommanager,
    tb,
)
from tlo.methods.fullmodel import fullmodel
from tlo.methods.hsi_event import HSI_Event

resourcefilepath = Path(os.path.dirname(__file__)) / "../resources"

start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 200

"""
Test whether the system runs under multiple configurations of the healthsystem.

This test file is focussed on controlling the behaviour of the module in allocation its human resources:
 * Mode1
 * Mode 2 (including rescaling and clinics).
"""

def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


@pytest.mark.slow
def test_policy_and_lowest_priority_and_fasttracking_enforced(seed, tmpdir):
    """The priority set by the policy should overwrite the priority the event was scheduled with. If the priority
    is below the lowest one considered, the event will not be scheduled (call never_ran at tclose). If a TREATMENT_ID
    and a person characteristic warrant it, fast-tracking is enabled."""

    class DummyHSI(HSI_Event, IndividualScopeEventMixin):
        """HSI event that schedules another HSI_Event for the same day"""

        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = "HSI_Dummy"
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({})
            self.ACCEPTED_FACILITY_LEVEL = "1a"

        def apply(self, person_id, squeeze_factor):
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                DummyHSI(module=self.module, person_id=person_id), topen=self.sim.date, tclose=None, priority=0
            )

    class DummyModule(Module):
        """Schedules an HSI to occur on the first day of the simulation from initialise_simulation, and an event that
        will schedule the event for the same day."""

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            pass

    log_config = {
        "filename": "log",
        "directory": tmpdir,
        "custom_levels": {"tlo.methods.healthsystem": logging.DEBUG},
    }
    sim = Simulation(start_date=Date(2010, 1, 1), seed=seed, log_config=log_config, resourcefilepath=resourcefilepath)
    sim.register(
        demography.Demography(),
        healthsystem.HealthSystem(
            disable=False,
            randomise_queue=True,
            ignore_priority=False,
            mode_appt_constraints=2,
            policy_name="Test",  # Test policy enforcing lowest_priority_policy
            # assumed in this test. This allows us to check policies
            # are loaded correctly.
            cons_availability="all",
        ),
        symptommanager.SymptomManager(),
        healthseekingbehaviour.HealthSeekingBehaviour(),
        enhanced_lifestyle.Lifestyle(),
        epi.Epi(),
        hiv.Hiv(run_with_checks=False),
        tb.Tb(),
        DummyModule(),
        check_all_dependencies=False,
    )
    sim.make_initial_population(n=100)
    sim.simulate(end_date=sim.start_date + pd.DateOffset(days=5))

    sim.event_queue.queue = []  # clear the queue
    sim.modules["HealthSystem"].HSI_EVENT_QUEUE = []  # clear the queue
    # Overwrite one of the Treatments with HSI_Dummy, and assign it a policy priority
    dictio = sim.modules["HealthSystem"].priority_rank_dict
    dictio["HSI_Dummy"] = dictio["Alri_Pneumonia_Treatment_Outpatient"]
    del dictio["Alri_Pneumonia_Treatment_Outpatient"]
    dictio["HSI_Dummy"]["Priority"] = 0

    # Schedule an 'HSI_Dummy' event with priority different from policy one
    sim.modules["HealthSystem"].schedule_hsi_event(
        DummyHSI(module=sim.modules["DummyModule"], person_id=0),
        topen=sim.date + pd.DateOffset(days=sim.modules["DummyModule"].rng.randint(1, 30)),
        tclose=None,
        priority=1,
    )  # Give a priority different than the one assumed by the policy for this Treatment_ID

    assert len(sim.modules["HealthSystem"].HSI_EVENT_QUEUE) == 1
    event_prev = hp.heappop(sim.modules["HealthSystem"].HSI_EVENT_QUEUE)
    assert event_prev.priority == 0  # Check that the event's priority is the policy one

    # Make
    # i) both policy priority and scheduled priority =2,
    # ii) HSI_Dummy eligible for fast-tracking for tb_diagnosed individuals exclusively,
    # iii) person for whom HSI will be scheduled tb-positive (hence fast-tracking eligible)
    # and check that person is fast-tracked with priority=1
    dictio["HSI_Dummy"]["Priority"] = 2
    dictio["HSI_Dummy"]["FT_if_5orUnder"] = -1
    dictio["HSI_Dummy"]["FT_if_pregnant"] = -1
    dictio["HSI_Dummy"]["FT_if_Hivdiagnosed"] = -1
    dictio["HSI_Dummy"]["FT_if_tbdiagnosed"] = 1
    sim.population.props.at[0, "tb_diagnosed"] = True

    # Schedule an 'HSI_Dummy' event with priority different to that with which it is scheduled
    sim.modules["HealthSystem"].schedule_hsi_event(
        DummyHSI(module=sim.modules["DummyModule"], person_id=0),
        topen=sim.date + pd.DateOffset(days=sim.modules["DummyModule"].rng.randint(1, 30)),
        tclose=None,
        priority=2,
    )  # Give a priority below fast tracking

    assert len(sim.modules["HealthSystem"].HSI_EVENT_QUEUE) == 1
    event_prev = hp.heappop(sim.modules["HealthSystem"].HSI_EVENT_QUEUE)
    assert event_prev.priority == 1  # Check that the event priority is the fast tracking one

    # Repeat, but now assinging priority below threshold through policy, to check that the event is not scheduled.
    # Person still tb positive, so ensure fast tracking is no longer available for this treatment to tb-diagnosed.
    dictio["HSI_Dummy"]["Priority"] = 7
    dictio["HSI_Dummy"]["FT_if_tbdiagnosed"] = -1
    _tclose = sim.date + pd.DateOffset(days=35)

    # Schedule an 'HSI_Dummy' event with priority different from policy one
    sim.modules["HealthSystem"].schedule_hsi_event(
        DummyHSI(module=sim.modules["DummyModule"], person_id=0),
        topen=sim.date + pd.DateOffset(days=sim.modules["DummyModule"].rng.randint(1, 30)),
        tclose=_tclose,
        priority=1,
    )  # Give a priority different than the one assumed by the policy for this Treatment_ID

    # Check that event wasn't scheduled due to priority being below threshold
    assert len(sim.modules["HealthSystem"].HSI_EVENT_QUEUE) == 0

    # Check that event was scheduled to never run on tclose
    assert len(sim.event_queue) == 1
    ev = hp.heappop(sim.event_queue.queue)
    assert not ev[3].run_hsi
    assert ev[0] == _tclose


@pytest.mark.slow
def test_run_in_mode_1_with_capacity(tmpdir, seed):
    # All events should run with some zero squeeze factors
    # (Mode 1 -> elastic constraints)

    # Establish the simulation object
    sim = Simulation(
        start_date=start_date,
        seed=seed,
        log_config={
            "filename": "log",
            "directory": tmpdir,
            "custom_levels": {
                "tlo.methods.healthsystem": logging.DEBUG,
            },
        },
        resourcefilepath=resourcefilepath,
    )

    # Define the service availability
    service_availability = ["*"]

    # Register the core modules
    sim.register(
        demography.Demography(),
        simplified_births.SimplifiedBirths(),
        enhanced_lifestyle.Lifestyle(),
        healthsystem.HealthSystem(
            service_availability=service_availability, capabilities_coefficient=1.0, mode_appt_constraints=1
        ),
        symptommanager.SymptomManager(),
        healthseekingbehaviour.HealthSeekingBehaviour(),
        mockitis.Mockitis(),
        chronicsyndrome.ChronicSyndrome(),
    )

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # read the results
    output = parse_log_file(sim.log_filepath, level=logging.DEBUG)

    # Do the checks
    assert len(output["tlo.methods.healthsystem"]["HSI_Event"]) > 0
    assert output["tlo.methods.healthsystem"]["HSI_Event"]["did_run"].all()
    assert (output["tlo.methods.healthsystem"]["HSI_Event"]["Squeeze_Factor"] == 0.0).all()

    # Check that some mockitis cured occurred (though health system)
    assert any(sim.population.props["mi_status"] == "P")



@pytest.mark.slow
def test_policy_has_no_effect_on_mode1(tmpdir, seed):
    """Events ran in mode 1 should be identical regardless of policy assumed.
    In policy "No Services", have set all HSIs to priority below lowest_priority_considered,
    in mode 1 they should all be scheduled and delivered regardless"""

    output = []
    policy_list = ["Naive", "Test Mode 1", "", "ClinicallyVulnerable"]
    for _, policy in enumerate(policy_list):
        # Establish the simulation object
        sim = Simulation(
            start_date=start_date,
            seed=seed,
            log_config={
                "filename": "log",
                "directory": tmpdir,
                "custom_levels": {
                    "tlo.methods.healthsystem": logging.DEBUG,
                },
            },
            resourcefilepath=resourcefilepath,
        )

        # Register the core modules
        sim.register(
            *fullmodel(
                module_kwargs={
                    "HealthSystem": {"capabilities_coefficient": 1.0, "mode_appt_constraints": 1, "policy_name": policy}
                }
            )
        )

        # Run the simulation
        sim.make_initial_population(n=popsize)
        sim.simulate(end_date=end_date)
        check_dtypes(sim)

        print(type(parse_log_file(sim.log_filepath, level=logging.DEBUG)))

        # read the results
        output.append(parse_log_file(sim.log_filepath, level=logging.DEBUG))

    # Check that the outputs are the same
    for i in range(1, len(policy_list)):
        pd.testing.assert_frame_equal(
            output[0]["tlo.methods.healthsystem"]["HSI_Event"], output[i]["tlo.methods.healthsystem"]["HSI_Event"]
        )


@pytest.mark.slow
def test_rescaling_capabilities_based_on_load_factors(tmpdir, seed):
    # Capabilities should increase when a HealthSystem that has low capabilities changes mode with
    # the option `scale_to_effective_capabilities` set to `True`.

    # Establish the simulation object
    sim = Simulation(
        start_date=start_date,
        seed=seed,
        log_config={
            "filename": "log",
            "directory": tmpdir,
            "custom_levels": {
                "tlo.methods.healthsystem": logging.DEBUG,
                "tlo.methods.healthsystem.summary": logging.INFO,
            },
        },
        resourcefilepath=resourcefilepath,
    )

    n_sim_initial_population = 1000
    n_pop_2010 = 14.6e6
    # Ensure capabilities are much smaller (1/1000) than expected given initial pop size
    small_capabilities = (n_sim_initial_population / n_pop_2010) / 10000

    # Register the core modules
    # Set the year in which mode is changed to start_date + 1 year, and mode after that still 1.
    # Check that in second year, squeeze factor is smaller on average.
    sim.register(
        demography.Demography(),
        simplified_births.SimplifiedBirths(),
        enhanced_lifestyle.Lifestyle(),
        healthsystem.HealthSystem(
            capabilities_coefficient=small_capabilities,
            # This will mean that capabilities are
            # very close to 0 everywhere.
            # (If the value was 0, then it would
            # be interpreted as the officers NEVER
            # being available at a facility,
            # which would mean the HSIs should not
            # run (as opposed to running with
            # a very high squeeze factor)).
        ),
        symptommanager.SymptomManager(),
        healthseekingbehaviour.HealthSeekingBehaviour(),
        mockitis.Mockitis(),
        chronicsyndrome.ChronicSyndrome(),
    )

    # Define the "switch" from Mode 1 to Mode 1, with the rescaling
    hs_params = sim.modules["HealthSystem"].parameters
    hs_params["mode_appt_constraints"] = 1
    hs_params["mode_appt_constraints_postSwitch"] = 1
    hs_params["year_mode_switch"] = start_date.year + 1
    hs_params["scale_to_effective_capabilities"] = True

    # Run the simulation
    sim.make_initial_population(n=n_sim_initial_population)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # read the results
    output = parse_log_file(sim.log_filepath, level=logging.INFO)
    pd.set_option("display.max_columns", None)
    summary = output["tlo.methods.healthsystem.summary"]
    capacity_by_officer_and_level = summary["Capacity_By_FacID_and_Officer"]

    # Filter rows for the two years
    row_2010 = capacity_by_officer_and_level.loc[capacity_by_officer_and_level["date"] == "2010-12-31"].squeeze()
    row_2011 = capacity_by_officer_and_level.loc[capacity_by_officer_and_level["date"] == "2011-12-31"].squeeze()

    # Dictionary to store results
    results = {}

    # Check that load has significantly reduced in second year, thanks to the significant
    # rescaling of capabilities.
    # (There is some degeneracy here, in that load could also be reduced due to declining demand.
    # However it is extremely unlikely that demand for care would have dropped by a factor of 10
    # in second year, hence this is a fair test).
    for col in capacity_by_officer_and_level.columns:
        if col == "date":
            continue  # skip the date column
        if not (capacity_by_officer_and_level[col] == 0).any() and ("GenericClinic" in col):
            ratio = row_2010[col] / row_2011[col]

            results[col] = ratio > 10
            if not results[col]:
                print(f"Load for {col} did not reduce sufficiently: ratio={ratio}")

    # Ensure that this test is not passing because issue in results
    assert len(results) > 0
    assert all(results.values())



@pytest.mark.slow
def test_run_in_mode_2_with_capacity(tmpdir, seed):
    # All events should run
    # (Mode 2 -> hard constraints)

    # Establish the simulation object
    sim = Simulation(
        start_date=start_date,
        seed=seed,
        log_config={
            "filename": "log",
            "directory": tmpdir,
            "custom_levels": {
                "tlo.methods.healthsystem": logging.DEBUG,
            },
        },
        resourcefilepath=resourcefilepath,
    )

    # Define the service availability
    service_availability = ["*"]

    # Register the core modules
    sim.register(
        demography.Demography(),
        simplified_births.SimplifiedBirths(),
        enhanced_lifestyle.Lifestyle(),
        healthsystem.HealthSystem(
            service_availability=service_availability, capabilities_coefficient=1.0, mode_appt_constraints=2
        ),
        symptommanager.SymptomManager(),
        healthseekingbehaviour.HealthSeekingBehaviour(),
        mockitis.Mockitis(),
        chronicsyndrome.ChronicSyndrome(),
    )

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # read the results
    output = parse_log_file(sim.log_filepath, level=logging.DEBUG)

    # Do the checks
    assert len(output["tlo.methods.healthsystem"]["HSI_Event"]) > 0
    assert output["tlo.methods.healthsystem"]["HSI_Event"]["did_run"].all()
    assert (output["tlo.methods.healthsystem"]["HSI_Event"]["Squeeze_Factor"] == 0.0).all()

    # Check that some Mockitis cures occurred (though health system)
    assert any(sim.population.props["mi_status"] == "P")




@pytest.mark.slow
@pytest.mark.group2
def test_run_in_mode_2_with_no_capacity(tmpdir, seed):
    # No individual level events (with non-blank footprint) should run and the log should contain events with a flag
    # showing that all individual events did not run. Population level events should have run.
    # (Mode 2 -> hard constraints)

    # Establish the simulation object
    sim = Simulation(
        start_date=start_date,
        seed=seed,
        log_config={
            "filename": "log",
            "directory": tmpdir,
            "custom_levels": {
                "tlo.methods.healthsystem": logging.DEBUG,
            },
        },
        resourcefilepath=resourcefilepath,
    )

    # Define the service availability
    service_availability = ["*"]

    # Register the core modules
    sim.register(
        demography.Demography(),
        simplified_births.SimplifiedBirths(),
        enhanced_lifestyle.Lifestyle(),
        healthsystem.HealthSystem(
            service_availability=service_availability, capabilities_coefficient=0.0, mode_appt_constraints=2
        ),
        symptommanager.SymptomManager(),
        healthseekingbehaviour.HealthSeekingBehaviour(),
        mockitis.Mockitis(),
        chronicsyndrome.ChronicSyndrome(),
    )

    # Run the simulation, manually setting smaller values to decrease runtime (logfile size)
    sim.make_initial_population(n=100)
    sim.simulate(end_date=Date(2011, 1, 1))
    check_dtypes(sim)

    # read the results
    output = parse_log_file(sim.log_filepath, level=logging.DEBUG)

    # Do the checks
    hsi_events = output["tlo.methods.healthsystem"]["HSI_Event"]
    assert not (
        hsi_events.loc[
            (hsi_events["Person_ID"] >= 0) & (hsi_events["Number_By_Appt_Type_Code"] != {}), "did_run"
        ].astype(bool)
    ).any()  # not any Individual level with non-blank footprints
    assert (output["tlo.methods.healthsystem"]["Capacity"]["Frac_Time_Used_Overall"] == 0.0).all()
    assert (hsi_events.loc[hsi_events["Person_ID"] < 0, "did_run"]).astype(bool).all()  # all Population level events
    assert pd.isnull(sim.population.props["mi_date_cure"]).all()  # No cures of mockitis occurring

    # Check that no Mockitis cures occurred (though health system)
    assert not any(sim.population.props["mi_status"] == "P")




def test_mode_appt_constraints2_on_healthsystem(seed, tmpdir):
    """Test that mode_appt_constraints=2 leads to correct constraints on number of HSIs that can run,
    in particular:
    - If capabilities required to carry out an hsi at facility have been exhausted for the day, the hsi
      cannot be ran;
    - HSIs with higher priority are ran preferentially;
    - Competition for resources takes place at facility level;
    """

    # Create Dummy Module to host the HSI
    class DummyModule(Module):
        METADATA = {Metadata.DISEASE_MODULE, Metadata.USES_HEALTHSYSTEM}

        def read_parameters(self, resourcefilepath=None):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            pass

    # Create a dummy HSI event class
    class DummyHSIEvent(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id, appt_type, level):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = "DummyHSIEvent"
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({appt_type: 1})
            self.ACCEPTED_FACILITY_LEVEL = level

            self.this_hsi_event_ran = False

        def apply(self, person_id, squeeze_factor):
            self.this_hsi_event_ran = True

    log_config = {
        "filename": "log",
        "directory": tmpdir,
        "custom_levels": {"tlo.methods.healthsystem": logging.DEBUG},
    }
    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config, resourcefilepath=resourcefilepath)

    # Register the core modules and simulate for 0 days
    sim.register(
        demography.Demography(),
        healthsystem.HealthSystem(
            capabilities_coefficient=1.0,
            mode_appt_constraints=2,
            ignore_priority=False,
            randomise_queue=True,
            policy_name="",
            use_funded_or_actual_staffing="funded_plus",
        ),
        DummyModule(),
    )

    tot_population = 100
    sim.make_initial_population(n=tot_population)
    sim.simulate(end_date=sim.start_date)

    # Get pointer to the HealthSystemScheduler event
    healthsystemscheduler = sim.modules["HealthSystem"].healthsystemscheduler

    # Split individuals equally across two districts
    person_for_district = {d: i for i, d in enumerate(sim.population.props["district_of_residence"].cat.categories)}
    keys_district = list(person_for_district.keys())

    # First half of population in keys_district[0], second half in keys_district[1]
    for i in range(0, int(tot_population / 2)):
        sim.population.props.at[i, "district_of_residence"] = keys_district[0]
    for i in range(int(tot_population / 2), tot_population):
        sim.population.props.at[i, "district_of_residence"] = keys_district[1]

    # Schedule an identical appointment for all individuals, assigning priority as follows:
    # - In first district, half individuals have priority=0 and half priority=1
    # - In second district, half individuals have priority=2 and half priority=3
    for i in range(0, tot_population):
        hsi = DummyHSIEvent(module=sim.modules["DummyModule"], person_id=i, appt_type="MinorSurg", level="1a")

        sim.modules["HealthSystem"].schedule_hsi_event(
            hsi,
            topen=sim.date,
            tclose=sim.date + pd.DateOffset(days=1),
            # Assign priority as 0,1,0,1,...0,1,2,3,2,3,....2,3. In doing so, in following tests also
            # check that events are rearranged in queue based on priority and not order in which were scheduled.
            priority=int(i / int(tot_population / 2)) * 2 + i % 2,
        )

    # Now adjust capabilities available.
    # In first district, make capabilities half of what would be required to run all events
    # without squeeze:
    hsi1 = DummyHSIEvent(
        module=sim.modules["DummyModule"],
        person_id=0,  # Ensures call is on officers in first district
        appt_type="MinorSurg",
        level="1a",
    )
    hsi1.initialise()
    for k, v in hsi1.expected_time_requests.items():
        print(k, sim.modules["HealthSystem"]._daily_capabilities["GenericClinic"][k])
        sim.modules["HealthSystem"]._daily_capabilities["GenericClinic"][k] = v * (tot_population / 4)

    # In second district, make capabilities tuned to be those required to run all priority=2 events under
    # maximum squeezed allowed for this priority, which currently is zero.
    max_squeeze = 0.0
    scale = 1.0 + max_squeeze
    print("Scale is ", scale)
    hsi2 = DummyHSIEvent(
        module=sim.modules["DummyModule"],
        person_id=int(tot_population / 2),  # Ensures call is on officers in second district
        appt_type="MinorSurg",
        level="1a",
    )
    hsi2.initialise()
    for k, v in hsi2.expected_time_requests.items():
        sim.modules["HealthSystem"]._daily_capabilities["GenericClinic"][k] = (v / scale) * (tot_population / 4)

    # Run healthsystemscheduler
    healthsystemscheduler.apply(sim.population)

    # read the results
    output = parse_log_file(sim.log_filepath, level=logging.DEBUG)
    hs_output = output["tlo.methods.healthsystem"]["HSI_Event"]

    # Check that some events could run, but not all
    assert hs_output["did_run"].sum() < tot_population, "All events ran"
    assert hs_output["did_run"].sum() != 0, "No events ran"

    # Get the appointments that ran for each priority
    Nran_w_priority0 = len(hs_output[(hs_output["priority"] == 0) & (hs_output["did_run"])])
    Nran_w_priority1 = len(hs_output[(hs_output["priority"] == 1) & (hs_output["did_run"])])
    Nran_w_priority2 = len(hs_output[(hs_output["priority"] == 2) & (hs_output["did_run"])])
    Nran_w_priority3 = len(hs_output[(hs_output["priority"] == 3) & (hs_output["did_run"])])

    # Within district, check that appointments with higher priority occurred more frequently
    assert Nran_w_priority0 > Nran_w_priority1
    assert Nran_w_priority2 > Nran_w_priority3

    # Check that if capabilities ran out in one district, capabilities in different district
    # cannot be accessed, even if priority should give precedence:
    # Because competition for resources occurs by facility, priority=2 should occur more
    # frequently than priority=1.
    assert Nran_w_priority2 > Nran_w_priority1

    # SQUEEZE CHECKS

    # Check that some level of squeeze occurs:
    # Although the capabilities in first district were set to half of those required,
    # if some level of squeeze was allowed (i.e. if max squeeze allowed for priority=0 is >0)
    # more than half of appointments should have taken place in total.
    if max_squeeze > 0:
        assert Nran_w_priority0 + Nran_w_priority1 > (tot_population / 4)

    # Check that the maximum squeeze allowed is set by priority:
    # The capabilities in the second district were tuned to accomodate all priority=2
    # appointments under the maximum squeeze allowed. Check that exactly all priority=2
    # appointments were allowed and no priority=3, to verify that the maximum squeeze
    # allowed in queue given priority is correct.
    assert (Nran_w_priority2 == int(tot_population / 4)) & (Nran_w_priority3 == 0)



@pytest.mark.slow
def test_run_in_mode_2_with_capacity_with_health_seeking_behaviour(tmpdir, seed):
    # All events should run
    # (Mode 2 -> hard constraints)

    # Establish the simulation object
    sim = Simulation(
        start_date=start_date,
        seed=seed,
        log_config={
            "filename": "log",
            "directory": tmpdir,
            "custom_levels": {
                "tlo.methods.healthsystem": logging.DEBUG,
            },
        },
        resourcefilepath=resourcefilepath,
    )

    # Define the service availability
    service_availability = ["*"]

    # Register the core modules
    sim.register(
        demography.Demography(),
        simplified_births.SimplifiedBirths(),
        enhanced_lifestyle.Lifestyle(),
        healthsystem.HealthSystem(
            service_availability=service_availability, capabilities_coefficient=1.0, mode_appt_constraints=2
        ),
        symptommanager.SymptomManager(),
        healthseekingbehaviour.HealthSeekingBehaviour(),
        mockitis.Mockitis(),
        chronicsyndrome.ChronicSyndrome(),
    )

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # read the results
    output = parse_log_file(sim.log_filepath, level=logging.DEBUG)

    # Do the check for the occurrence of the GenericFirstAppt which is created by the HSB module
    assert "FirstAttendance_NonEmergency" in output["tlo.methods.healthsystem"]["HSI_Event"]["TREATMENT_ID"].values

    # Check that some mockitis cured occurred (though health system)
    assert any(sim.population.props["mi_status"] == "P")



def test_mode_2_clinics(seed, tmpdir):
    """Test that clinics work as expected in mode_appt_constraints=2. Specifically:
    - An HSI Event whose treatment id is mapped to a specific clinic runs if corresponding
    clinic capabilities are available;
    - Conversely, if the clinic specific capabilities run out, then the event DOES NOT run even if
    GenericClinic capabilities are available; this test checks that that events query
    the correct capabilities and that correct counters are run down;
    - An event whose treatment id is not mapped to a specific clinic runs if GenericClinic
    capabilities are available;
    - Conversely, an event whose treatment id is not mapped to a specific clinic does not run
    if GenericClinic capabilities are not available;
    """

    # Create a dummy HSI event class
    class DummyHSIEvent(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id, appt_type, level, treatment_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = treatment_id
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({appt_type: 1})
            self.ACCEPTED_FACILITY_LEVEL = level

        def apply(self, person_id, squeeze_factor):
            self.this_hsi_event_ran = True

    def create_simulation(tmpdir: Path, tot_population) -> Simulation:
        class DummyModuleGenericClinic(Module):
            METADATA = {Metadata.DISEASE_MODULE, Metadata.USES_HEALTHSYSTEM}

            def read_parameters(self, data_folder):
                pass

            def initialise_population(self, population):
                pass

            def initialise_simulation(self, sim):
                pass

        class DummyModuleClinic1(Module):
            METADATA = {Metadata.DISEASE_MODULE, Metadata.USES_HEALTHSYSTEM}

            def read_parameters(self, data_folder):
                pass

            def initialise_population(self, population):
                pass

            def initialise_simulation(self, sim):
                pass

        log_config = {
            "filename": "log",
            "directory": tmpdir,
            "custom_levels": {"tlo.methods.healthsystem": logging.DEBUG},
        }
        start_date = Date(2010, 1, 1)
        sim = Simulation(start_date=start_date, seed=0, log_config=log_config, resourcefilepath=resourcefilepath)

        sim.register(
            demography.Demography(),
            healthsystem.HealthSystem(
                capabilities_coefficient=1.0,
                mode_appt_constraints=2,
                ignore_priority=False,
                randomise_queue=True,
                policy_name="",
                use_funded_or_actual_staffing="funded_plus",
            ),
            DummyModuleGenericClinic(),
            DummyModuleClinic1(),
        )
        sim.make_initial_population(n=tot_population)

        sim.modules["HealthSystem"]._clinic_configuration = pd.DataFrame(
            [{"Facility_ID": 20.0, "Officer_Type_Code": "DCSA", "Clinic1": 0.6, "GenericClinic": 0.4}]
        )
        sim.modules["HealthSystem"]._clinic_mapping = pd.DataFrame(
            [{"Treatment": "DummyHSIEvent", "Clinic": "Clinic1"}]
        )
        sim.modules["HealthSystem"]._clinic_names = ["Clinic1", "GenericClinic"]
        sim.modules["HealthSystem"].setup_daily_capabilities("funded_plus")

        # Assign the entire population to the first district, so that all events are run in the same district
        col = "district_of_residence"
        s = sim.population.props[col]
        ## Not specifying the dtype explicitly here made the col a string rather than a category
        ## and that caused problems later on.
        sim.population.props[col] = pd.Series(s.cat.categories[0], index=s.index, dtype=s.dtype)

        sim.simulate(end_date=sim.start_date + pd.DateOffset(years=1))

        return sim

    def schedule_hsi_events(ngenericclinic, nclinic1, sim):
        for i in range(0, ngenericclinic):
            hsi = DummyHSIEvent(
                module=sim.modules["DummyModuleGenericClinic"],
                person_id=i,
                appt_type="ConWithDCSA",
                level="0",
                treatment_id="DummyHSIEventGenericClinic",
            )
            sim.modules["HealthSystem"].schedule_hsi_event(
                hsi, topen=sim.date, tclose=sim.date + pd.DateOffset(days=1), priority=1
            )

        for i in range(ngenericclinic, ngenericclinic + nclinic1):
            hsi = DummyHSIEvent(
                module=sim.modules["DummyModuleClinic1"],
                person_id=i,
                appt_type="ConWithDCSA",
                level="0",
                treatment_id="DummyHSIEvent",
            )
            sim.modules["HealthSystem"].schedule_hsi_event(
                hsi, topen=sim.date, tclose=sim.date + pd.DateOffset(days=1), priority=1
            )

        return sim

    tot_population = 100
    sim = create_simulation(tmpdir, tot_population)

    ## Test that capabilities are split according the proportion specified for the Facility Id
    ## and officer combination in the Resource file.
    ## 40% of capabilities are GenericClinic and 60% Clinic1 capabilities
    other_clinic = sim.modules["HealthSystem"]._daily_capabilities["GenericClinic"]
    clinic1 = sim.modules["HealthSystem"]._daily_capabilities["Clinic1"]

    # 'FacilityID_20_Officer_DCSA' is coming from the resource file
    ratio = clinic1["FacilityID_20_Officer_DCSA"] / other_clinic["FacilityID_20_Officer_DCSA"]
    expect = 0.6 / 0.4
    assert abs(ratio - expect) < 1e-7, "GenericClinic capabilities are not split correctly"

    # Schedule an identical appointment for all individuals, assigning clinic as follows:
    # half individuals have clinic_eligibility=GenericClinic and half clinic_eligibility=Clinic1
    sim = schedule_hsi_events(50, 50, sim)
    ## This hsi is only created to get the expected items; therefore the treatment_id is not important
    hsi1 = DummyHSIEvent(
        module=sim.modules["DummyModuleGenericClinic"],
        person_id=0,  # Ensures call is on officers in first district
        appt_type="ConWithDCSA",
        level="0",
        treatment_id="DummyHSIEventGenericClinic",
    )
    hsi1.initialise()

    # Now adjust capabilities available.
    # We first want to make sure there are enough capabilities available to run all events

    sim.modules["HealthSystem"]._daily_capabilities["Clinic1"] = {}
    for k, v in hsi1.expected_time_requests.items():
        sim.modules["HealthSystem"]._daily_capabilities["GenericClinic"][k] = v * tot_population
        sim.modules["HealthSystem"]._daily_capabilities["Clinic1"][k] = v * tot_population

    # Run healthsystemscheduler and read the results
    sim.modules["HealthSystem"].healthsystemscheduler.apply(sim.population)

    output = parse_log_file(sim.log_filepath, level=logging.DEBUG)
    hs_output = output["tlo.methods.healthsystem"]["HSI_Event"]
    ## All events should have run
    assert hs_output["did_run"].sum() == tot_population, "All events did not run!!"
    Nevents = hs_output.groupby("Clinic")["did_run"].value_counts()
    assert Nevents.loc[("Clinic1", True)] == tot_population // 2, "Unexpected count of Clinic1 events"
    assert Nevents.loc[("GenericClinic", True)] == tot_population // 2, "Unexpected count of GenericClinic events"

    ## Test 2: Events requiring GenericClinic capabilities do not run if those capabilities are unavailable
    sim = create_simulation(tmpdir, tot_population)
    sim = schedule_hsi_events(tot_population // 2, tot_population // 2, sim)

    sim.modules["HealthSystem"]._daily_capabilities["GenericClinic"] = {"FacilityID_20_Officer_DCSA": 0.0}
    for k, v in hsi1.expected_time_requests.items():
        sim.modules["HealthSystem"]._daily_capabilities["Clinic1"][k] = v * (tot_population)

    sim.modules["HealthSystem"].healthsystemscheduler.apply(sim.population)

    output = parse_log_file(sim.log_filepath, level=logging.DEBUG)
    hs_output = output["tlo.methods.healthsystem"]["HSI_Event"]

    assert hs_output["did_run"].sum() == tot_population // 2, "Unexpected number of events ran"
    Nevents = hs_output.groupby("Clinic")["did_run"].value_counts()
    ## No GenericClinic events should have run, but all Clinic1 ones should have
    assert Nevents.loc[("Clinic1", True)] == tot_population // 2, "Unexpected count of Clinic1 events"
    assert Nevents.loc[("GenericClinic", False)] == tot_population // 2, "Unexpected count of GenericClinic events"

    ## Test 3: Events requiring Clinic1 capabilities do not run if those capabilities are unavailable
    ## Mirror of test 2 above
    sim = create_simulation(tmpdir, tot_population)
    sim = schedule_hsi_events(tot_population // 2, tot_population // 2, sim)

    # Now adjust capabilities available using hsi2 created above
    sim.modules["HealthSystem"]._daily_capabilities["Clinic1"] = {"FacilityID_20_Officer_DCSA": 0.0}
    for k, v in hsi1.expected_time_requests.items():
        sim.modules["HealthSystem"]._daily_capabilities["GenericClinic"][k] = v * (tot_population)

    sim.modules["HealthSystem"].healthsystemscheduler.apply(sim.population)

    output = parse_log_file(sim.log_filepath, level=logging.DEBUG)
    hs_output = output["tlo.methods.healthsystem"]["HSI_Event"]

    assert hs_output["did_run"].sum() == tot_population // 2, "Half of the events ran"
    Nevents = hs_output.groupby("Clinic")["did_run"].value_counts()
    ## No more non-fungible events should have run, but all GenericClinic ones should have
    assert Nevents.loc[("Clinic1", False)] == tot_population // 2
    assert Nevents.loc[("GenericClinic", True)] == tot_population // 2

    ## Test 4: Queue up GenericClinic/Clinic1/GenericClinic; have Clinic1 capabilities run out
    ## and ensure GenericClinic events still run.
    sim = create_simulation(tmpdir, tot_population)
    sim = schedule_hsi_events(25, 0, sim)
    sim = schedule_hsi_events(0, 25, sim)
    sim = schedule_hsi_events(25, 0, sim)
    sim = schedule_hsi_events(0, 25, sim)

    # Now adjust capabilities available.
    sim.modules["HealthSystem"]._daily_capabilities["Clinic1"] = {"FacilityID_20_Officer_DCSA": 0.0}
    for k, v in hsi1.expected_time_requests.items():
        sim.modules["HealthSystem"]._daily_capabilities["GenericClinic"][k] = v * (tot_population / 2)

    sim.modules["HealthSystem"].healthsystemscheduler.apply(sim.population)

    output = parse_log_file(sim.log_filepath, level=logging.DEBUG)
    hs_output = output["tlo.methods.healthsystem"]["HSI_Event"]

    assert hs_output["did_run"].sum() == tot_population // 2, "Unexpected number of events"
    Nevents = hs_output.groupby("Clinic")["did_run"].value_counts()
    ## No more non-fungible events should have run, but all GenericClinic ones should have
    assert Nevents.loc[("Clinic1", False)] == tot_population // 2, "No additional NonFungible events ran"
    assert Nevents.loc[("GenericClinic", True)] == tot_population // 2, "Scheduled GenericClinic events ran"

