import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tlo import Date, Module, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.events import IndividualScopeEventMixin
from tlo.methods import (
    Metadata,
    demography,
    healthsystem,
)
from tlo.methods.consumables import Consumables, create_dummy_data_for_cons_availability
from tlo.methods.hsi_event import HSI_Event

resourcefilepath = Path(os.path.dirname(__file__)) / "../resources"

start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 200

"""
Test whether the system runs under multiple configurations of the healthsystem.

This test file is focussed on the logging of the module.
"""


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()

@pytest.mark.slow
def test_two_loggers_in_healthsystem(seed, tmpdir):
    """Check that two different loggers used by the HealthSystem for more/less detailed logged information are
    consistent with one another."""

    # Create a dummy disease module (to be the parent of the dummy HSI)
    class DummyModule(Module):
        METADATA = {Metadata.DISEASE_MODULE}

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Dummy(self, person_id=0), topen=self.sim.date, tclose=None, priority=0
            )

    # Create a dummy HSI event:
    class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = "Dummy"
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1, "Under5OPD": 1})
            self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({"general_bed": 2})
            self.ACCEPTED_FACILITY_LEVEL = "1a"

        def apply(self, person_id, squeeze_factor):
            # Request a consumable (either 0 or 1)
            self.get_consumables(item_codes=self.module.rng.choice((0, 1), p=(0.5, 0.5)))

            # Schedule another occurrence of itself in three days.
            sim.modules["HealthSystem"].schedule_hsi_event(
                self, topen=self.sim.date + pd.DateOffset(days=3), tclose=None, priority=0
            )

    # Set up simulation:
    sim = Simulation(
        start_date=start_date,
        seed=seed,
        log_config={
            "filename": "tmpfile",
            "directory": tmpdir,
            "custom_levels": {
                "tlo.methods.healthsystem": logging.DEBUG,
                "tlo.methods.healthsystem.summary": logging.INFO,
            },
        },
        resourcefilepath=resourcefilepath,
    )

    sim.register(
        demography.Demography(),
        healthsystem.HealthSystem(
            mode_appt_constraints=1,
            capabilities_coefficient=1e-10,  # <--- to give non-trivial squeeze-factors
        ),
        DummyModule(),
        sort_modules=False,
        check_all_dependencies=False,
    )
    sim.make_initial_population(n=1000)

    # Replace consumables class with version that declares only one consumable, available with probability 0.5
    mfl = pd.read_csv(resourcefilepath / "healthsystem" / "organisation" / "ResourceFile_Master_Facilities_List.csv")
    all_fac_ids = set(mfl.loc[mfl.Facility_Level != "5"].Facility_ID)

    sim.modules["HealthSystem"].consumables = Consumables(
        availability_data=create_dummy_data_for_cons_availability(
            intrinsic_availability={0: 0.5, 1: 0.5}, months=list(range(1, 13)), facility_ids=list(all_fac_ids)
        ),
        rng=sim.modules["HealthSystem"].rng,
        availability="default",
    )

    sim.simulate(end_date=start_date + pd.DateOffset(years=2))
    log = parse_log_file(sim.log_filepath, level=logging.DEBUG)

    # Standard log:
    detailed_hsi_event = log["tlo.methods.healthsystem"]["HSI_Event"]
    detailed_capacity = log["tlo.methods.healthsystem"]["Capacity"]
    detailed_consumables = log["tlo.methods.healthsystem"]["Consumables"]

    assert {
        "date",
        "Clinic",
        "TREATMENT_ID",
        "did_run",
        "Squeeze_Factor",
        "priority",
        "Number_By_Appt_Type_Code",
        "Person_ID",
        "Facility_Level",
        "Facility_ID",
        "Event_Name",
        "Equipment",
    } == set(detailed_hsi_event.columns)
    assert {
        "date",
        "Clinic",
        "Frac_Time_Used_Overall",
        "Frac_Time_Used_By_Facility_ID",
        "Frac_Time_Used_By_OfficerType",
    } == set(detailed_capacity.columns)
    assert {"date", "TREATMENT_ID", "Item_Available", "Item_NotAvailable", "Item_Used"} == set(
        detailed_consumables.columns
    )

    bed_types = sim.modules["HealthSystem"].bed_days.bed_types
    detailed_beddays = {bed_type: log["tlo.methods.healthsystem"][f"bed_tracker_{bed_type}"] for bed_type in bed_types}

    # Summary log:
    summary_hsi_event = log["tlo.methods.healthsystem.summary"]["HSI_Event"]
    summary_capacity = log["tlo.methods.healthsystem.summary"]["Capacity"]
    summary_consumables = log["tlo.methods.healthsystem.summary"]["Consumables"]
    summary_beddays = log["tlo.methods.healthsystem.summary"]["BedDays"]

    def dict_all_close(dict_1, dict_2):
        return (dict_1.keys() == dict_2.keys()) and all(np.isclose(dict_1[k], dict_2[k]) for k in dict_1.keys())

    # Check correspondence between the two logs
    #  - Counts of TREATMENT_ID (total over entire period of log)
    summary_treatment_id_counts = summary_hsi_event["TREATMENT_ID"].apply(pd.Series).sum().to_dict()
    detailed_treatment_id_counts = detailed_hsi_event.groupby("TREATMENT_ID").size().to_dict()
    assert dict_all_close(summary_treatment_id_counts, detailed_treatment_id_counts)

    # Average of squeeze-factors for each TREATMENT_ID (by each year)
    summary_treatment_id_mean_squeeze_factors = (
        summary_hsi_event["squeeze_factor"]
        .apply(pd.Series)
        .groupby(by=summary_hsi_event.date.dt.year)
        .sum()
        .unstack()
        .to_dict()
    )
    detailed_treatment_id_mean_squeeze_factors = (
        detailed_hsi_event.assign(
            treatment_id_hsi_name=lambda df: df["TREATMENT_ID"],
            year=lambda df: df.date.dt.year,
        )
        .groupby(by=["treatment_id_hsi_name", "year"])["Squeeze_Factor"]
        .mean()
        .to_dict()
    )
    assert dict_all_close(summary_treatment_id_mean_squeeze_factors, detailed_treatment_id_mean_squeeze_factors)

    #  - Appointments (total over entire period of the log)
    assert (
        summary_hsi_event["Number_By_Appt_Type_Code"].apply(pd.Series).sum().to_dict()
        == detailed_hsi_event["Number_By_Appt_Type_Code"].apply(pd.Series).sum().to_dict()
    )

    #  - Average fraction of HCW time used (year by year)
    summary_capacity_indexed = summary_capacity.set_index(pd.to_datetime(summary_capacity.date).dt.year)
    for clinic in sim.modules["HealthSystem"]._clinic_names:
        summary_clinic_capacity = summary_capacity_indexed["average_Frac_Time_Used_Overall"].apply(
            lambda x: x.get(clinic, None)
        )
        detailed_clinic_capacity = detailed_capacity[detailed_capacity["Clinic"] == clinic]
        assert (
            summary_clinic_capacity.round(4).to_dict()
            == detailed_clinic_capacity.set_index(pd.to_datetime(detailed_clinic_capacity.date).dt.year)[
                "Frac_Time_Used_Overall"
            ]
            .groupby(level=0)
            .mean()
            .round(4)
            .to_dict()
        )

    #  - Consumables (total over entire period of log that are available / not available)  # add _Item_
    assert (
        summary_consumables["Item_Available"].apply(pd.Series).sum().to_dict()
        == detailed_consumables["Item_Available"]
        .apply(lambda x: {f"{k}": v for k, v in eval(x).items()})
        .apply(pd.Series)
        .sum()
        .to_dict()
    )
    assert (
        summary_consumables["Item_NotAvailable"].apply(pd.Series).sum().to_dict()
        == detailed_consumables["Item_NotAvailable"]
        .apply(lambda x: {f"{k}": v for k, v in eval(x).items()})
        .apply(pd.Series)
        .sum()
        .to_dict()
    )
    assert (
        summary_consumables["Item_Used"].apply(pd.Series).sum().to_dict()
        == detailed_consumables["Item_Used"]
        .apply(lambda x: {f"{k}": v for k, v in eval(x).items()})
        .apply(pd.Series)
        .sum()
        .to_dict()
    )

    #  - Bed-Days (bed-type by bed-type and year by year)
    for _bed_type in bed_types:
        # Detailed:
        tracker = (
            detailed_beddays[_bed_type]
            .assign(year=pd.to_datetime(detailed_beddays[_bed_type].date).dt.year)
            .set_index("year")
            .drop(columns=["date"])
            .T
        )
        tracker.index = tracker.index.astype(int)
        capacity = sim.modules["HealthSystem"].bed_days._scaled_capacity[_bed_type]
        detail_beddays_used = tracker.sub(capacity, axis=0).mul(-1).sum().groupby(level=0).sum().to_dict()

        # Summary: total bed-days used by year
        summary_beddays_used = (
            summary_beddays.assign(year=pd.to_datetime(summary_beddays.date).dt.year)
            .set_index("year")[_bed_type]
            .to_dict()
        )

        assert detail_beddays_used == summary_beddays_used

    # Check the count of appointment type (total) matches the count split by level
    counts_of_appts_by_level = (
        pd.concat(
            {
                idx: pd.DataFrame.from_dict(mydict)
                for idx, mydict in summary_hsi_event["Number_By_Appt_Type_Code_And_Level"].items()
            }
        )
        .unstack()
        .fillna(0.0)
        .astype(int)
    )

    assert (
        summary_hsi_event["Number_By_Appt_Type_Code"].apply(pd.Series).sum().to_dict()
        == counts_of_appts_by_level.groupby(axis=1, level=1).sum().sum().to_dict()
    )


@pytest.mark.slow
def test_summary_logger_for_never_ran_hsi_event(seed, tmpdir):
    """Check that under a mode_appt_constraints = 2 with zero resources, HSIs with a tclose
    soon after topen will be correctly recorded in the summary logger, and that this can
    be parsed correctly when a different set of HSI are never ran."""

    # Create a dummy disease module (to be the parent of the dummy HSI)
    class DummyModule(Module):
        METADATA = {Metadata.DISEASE_MODULE}

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            # In 2010: Dummy1 only
            sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Dummy1(self, person_id=0),
                topen=self.sim.date,
                tclose=self.sim.date + pd.DateOffset(days=2),
                priority=0,
            )
            # In 2011: Dummy2 & Dummy3
            sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Dummy2(self, person_id=0),
                topen=self.sim.date + pd.DateOffset(years=1),
                tclose=self.sim.date + pd.DateOffset(years=1) + pd.DateOffset(days=2),
                priority=0,
            )
            sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Dummy3(self, person_id=0),
                topen=self.sim.date + pd.DateOffset(years=1),
                tclose=self.sim.date + pd.DateOffset(years=1) + pd.DateOffset(days=2),
                priority=0,
            )

    # Create two different dummy HSI events:
    class HSI_Dummy1(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = "Dummy1"
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
            self.ACCEPTED_FACILITY_LEVEL = "1a"

        def apply(self, person_id, squeeze_factor):
            pass

    class HSI_Dummy2(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = "Dummy2"
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
            self.ACCEPTED_FACILITY_LEVEL = "1a"

        def apply(self, person_id, squeeze_factor):
            pass

    class HSI_Dummy3(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = "Dummy3"
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
            self.ACCEPTED_FACILITY_LEVEL = "1b"

        def apply(self, person_id, squeeze_factor):
            pass

    # Set up simulation:
    sim = Simulation(
        start_date=start_date,
        seed=seed,
        log_config={
            "filename": "tmpfile",
            "directory": tmpdir,
            "custom_levels": {
                "tlo.methods.healthsystem": logging.DEBUG,
                "tlo.methods.healthsystem.summary": logging.INFO,
            },
        },
        resourcefilepath=resourcefilepath,
    )

    sim.register(
        demography.Demography(),
        healthsystem.HealthSystem(
            mode_appt_constraints=2,
            capabilities_coefficient=0.0,  # <--- Ensure all events postponed
        ),
        DummyModule(),
        sort_modules=False,
        check_all_dependencies=False,
    )
    sim.make_initial_population(n=1000)

    sim.simulate(end_date=start_date + pd.DateOffset(years=2))
    log = parse_log_file(sim.log_filepath, level=logging.DEBUG)

    # Summary log:
    summary_hsi_event = log["tlo.methods.healthsystem.summary"]["Never_ran_HSI_Event"]
    # In 2010, should have recorded one instance of Dummy1 having never ran
    assert summary_hsi_event.loc[summary_hsi_event["date"] == Date(2010, 12, 31), "TREATMENT_ID"][0] == {"Dummy1": 1}
    # In 2011, should have recorded one instance of Dummy2 and one of Dummy3 having never ran
    assert summary_hsi_event.loc[summary_hsi_event["date"] == Date(2011, 12, 31), "TREATMENT_ID"][1] == {
        "Dummy2": 1,
        "Dummy3": 1,
    }


@pytest.mark.slow
def test_summary_logger_for_hsi_event_squeeze_factors(seed, tmpdir):
    """Check that the summary logger can be parsed correctly when a different set of HSI occur in different years."""

    # Create a dummy disease module (to be the parent of the dummy HSI)
    class DummyModule(Module):
        METADATA = {Metadata.DISEASE_MODULE}

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            # In 2010: Dummy1 only
            sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Dummy1(self, person_id=0), topen=self.sim.date, tclose=None, priority=0
            )
            # In 2011: Dummy2 & Dummy3
            sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Dummy2(self, person_id=0), topen=self.sim.date + pd.DateOffset(years=1), tclose=None, priority=0
            )
            sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Dummy3(self, person_id=0), topen=self.sim.date + pd.DateOffset(years=1), tclose=None, priority=0
            )

            # In 2011: to-do.....

    # Create two different dummy HSI events:
    class HSI_Dummy1(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = "Dummy1"
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
            self.ACCEPTED_FACILITY_LEVEL = "1a"

        def apply(self, person_id, squeeze_factor):
            pass

    class HSI_Dummy2(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = "Dummy2"
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
            self.ACCEPTED_FACILITY_LEVEL = "1a"

        def apply(self, person_id, squeeze_factor):
            pass

    class HSI_Dummy3(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = "Dummy3"
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
            self.ACCEPTED_FACILITY_LEVEL = "1a"

        def apply(self, person_id, squeeze_factor):
            pass

    # Set up simulation:
    sim = Simulation(
        start_date=start_date,
        seed=seed,
        log_config={
            "filename": "tmpfile",
            "directory": tmpdir,
            "custom_levels": {
                "tlo.methods.healthsystem": logging.DEBUG,
                "tlo.methods.healthsystem.summary": logging.INFO,
            },
        },
        resourcefilepath=resourcefilepath,
    )

    sim.register(
        demography.Demography(),
        healthsystem.HealthSystem(
            mode_appt_constraints=1,
            capabilities_coefficient=1e-10,  # <--- to give non-trivial squeeze-factors
        ),
        DummyModule(),
        sort_modules=False,
        check_all_dependencies=False,
    )
    sim.make_initial_population(n=1000)

    sim.simulate(end_date=start_date + pd.DateOffset(years=2))
    log = parse_log_file(sim.log_filepath, level=logging.DEBUG)

    # Standard log:
    detailed_hsi_event = log["tlo.methods.healthsystem"]["HSI_Event"]

    # Summary log:
    summary_hsi_event = log["tlo.methods.healthsystem.summary"]["HSI_Event"]

    #  - The squeeze-factors that applied for each TREATMENT_ID
    assert (
        summary_hsi_event.set_index(summary_hsi_event["date"].dt.year)["squeeze_factor"]
        .apply(pd.Series)
        .unstack()
        .dropna()
        .to_dict()
        == detailed_hsi_event.assign(
            treatment_id_hsi_name=lambda df: df["TREATMENT_ID"],
            year=lambda df: df.date.dt.year,
        )
        .groupby(by=["treatment_id_hsi_name", "year"])["Squeeze_Factor"]
        .mean()
        .to_dict()
    )


@pytest.mark.slow
def test_summary_logger_generated_in_year_long_simulation(seed, tmpdir):
    """Check that the summary logger is created when the simulation lasts exactly one year."""

    def summary_logger_is_present(end_date_of_simulation):
        """Returns True if the summary logger is present when using the specified end_date for the simulation."""

        # Create a dummy disease module (to be the parent of the dummy HSI)
        class DummyModule(Module):
            METADATA = {Metadata.DISEASE_MODULE}

            def read_parameters(self, data_folder):
                pass

            def initialise_population(self, population):
                pass

            def initialise_simulation(self, sim):
                sim.modules["HealthSystem"].schedule_hsi_event(
                    HSI_Dummy(self, person_id=0), topen=self.sim.date, tclose=None, priority=0
                )

        # Create a dummy HSI event:
        class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):
            def __init__(self, module, person_id):
                super().__init__(module, person_id=person_id)
                self.TREATMENT_ID = "Dummy"
                self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1, "Under5OPD": 1})
                self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({"general_bed": 2})
                self.ACCEPTED_FACILITY_LEVEL = "1a"

            def apply(self, person_id, squeeze_factor):
                # Request a consumable (either 0 or 1)
                self.get_consumables(item_codes=self.module.rng.choice((0, 1), p=(0.5, 0.5)))

                # Schedule another occurrence of itself in three days.
                sim.modules["HealthSystem"].schedule_hsi_event(
                    self, topen=self.sim.date + pd.DateOffset(days=3), tclose=None, priority=0
                )

        # Set up simulation:
        sim = Simulation(
            start_date=start_date,
            seed=seed,
            log_config={
                "filename": "tmpfile",
                "directory": tmpdir,
                "custom_levels": {
                    "tlo.methods.healthsystem": logging.DEBUG,
                    "tlo.methods.healthsystem.summary": logging.INFO,
                },
            },
            resourcefilepath=resourcefilepath,
        )

        sim.register(
            demography.Demography(),
            healthsystem.HealthSystem(),
            DummyModule(),
            sort_modules=False,
            check_all_dependencies=False,
        )
        sim.make_initial_population(n=1000)

        sim.simulate(end_date=end_date_of_simulation)
        log = parse_log_file(sim.log_filepath)

        return ("tlo.methods.healthsystem.summary" in log) and len(log["tlo.methods.healthsystem.summary"])

    assert summary_logger_is_present(start_date + pd.DateOffset(years=1))


def test_logging_of_only_hsi_events_with_non_blank_footprints(tmpdir):
    """Run the simulation with an HSI_Event that may have a blank_footprint and examine the healthsystem.summary logger.
    * If the footprint is blank, the HSI event should be recorded in the usual loggers but not the 'no_blank' logger
    * If the footprint is non-blank, the HSI event should be recorded in the usual and the 'no_blank' loggers.
    """

    def run_simulation_and_return_healthsystem_summary_log(tmpdir: Path, blank_footprint: bool) -> dict:
        """Return the `healthsystem.summary` logger for a simulation. In that simulation, there is HSI_Event run on the
        first day of the simulation and its `EXPECTED_APPT_FOOTPRINT` may or may not be blank. The simulation is run for
        one year in order that the summary logger is active (it runs annually)."""

        class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):
            def __init__(self, module, person_id, _is_footprint_blank):
                super().__init__(module, person_id=person_id)
                self.TREATMENT_ID = "Dummy"
                self.ACCEPTED_FACILITY_LEVEL = "0"
                self.EXPECTED_APPT_FOOTPRINT = (
                    self.make_appt_footprint({}) if blank_footprint else self.make_appt_footprint({"ConWithDCSA": 1})
                )

            def apply(self, person_id, squeeze_factor):
                pass

        class DummyModule(Module):
            METADATA = {Metadata.DISEASE_MODULE}

            def read_parameters(self, data_folder):
                pass

            def initialise_population(self, population):
                pass

            def initialise_simulation(self, sim):
                hsi_event = HSI_Dummy(module=self, person_id=0, _is_footprint_blank=blank_footprint)
                sim.modules["HealthSystem"].schedule_hsi_event(hsi_event=hsi_event, topen=sim.date, priority=0)

        start_date = Date(2010, 1, 1)
        sim = Simulation(
            start_date=start_date,
            seed=0,
            log_config={"filename": "tmp", "directory": tmpdir},
            resourcefilepath=resourcefilepath,
        )
        sim.register(
            demography.Demography(),
            healthsystem.HealthSystem(mode_appt_constraints=1),
            DummyModule(),
            # Disable sorting + checks to avoid error due to missing dependencies
            sort_modules=False,
            check_all_dependencies=False,
        )
        sim.make_initial_population(n=100)
        sim.simulate(end_date=sim.start_date + pd.DateOffset(years=1))

        return parse_log_file(sim.log_filepath)["tlo.methods.healthsystem.summary"]

    # When the footprint is blank:
    log = run_simulation_and_return_healthsystem_summary_log(tmpdir, blank_footprint=True)
    assert log["HSI_Event"]["TREATMENT_ID"].iloc[0] == {"Dummy": 1}  # recorded in usual logger
    assert log["HSI_Event_non_blank_appt_footprint"]["TREATMENT_ID"].iloc[0] == {}  # not recorded in 'non-blank' logger

    # When the footprint is non-blank:
    log = run_simulation_and_return_healthsystem_summary_log(tmpdir, blank_footprint=False)
    assert not log["HSI_Event"].empty
    assert "TREATMENT_ID" in log["HSI_Event"].columns
    assert "TREATMENT_ID" in log["HSI_Event_non_blank_appt_footprint"].columns
    assert (
        log["HSI_Event"]["TREATMENT_ID"].iloc[0]
        == log["HSI_Event_non_blank_appt_footprint"]["TREATMENT_ID"].iloc[0]
        == {"Dummy": 1}
        # recorded in both the usual and the 'non-blank' logger
    )

