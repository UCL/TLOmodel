from _paths import TLO_ROOT, TLO_OUTPUT_DIR

# Parameters to pass to scale_run
scale_run_parameters = {
    "years": 0,
    "months": 1,
    "initial_population": 50000,
    "tlo_dir": TLO_ROOT,
    "output_dir": TLO_OUTPUT_DIR,
    "log_filename": "scale_run_benchmark",
    "log_level": "DEBUG",
    "parse_log_file": False,
    "show_progress_bar": True,
    "seed": 0,
    "disable_health_system": False,
    "disable_spurious_symptoms": False,
    "capabilities_coefficient": None,
    "mode_appt_constraints": 2,
    "save_final_population": False,
    "record_hsi_event_details": False,
}
