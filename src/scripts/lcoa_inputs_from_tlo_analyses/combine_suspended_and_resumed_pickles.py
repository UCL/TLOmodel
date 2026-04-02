"""CLI helper to combine suspended and resumed pickle outputs."""

# python src/scripts/lcoa_inputs_from_tlo_analyses/combine_suspended_and_resumed_pickles.py --suspended_results_folder outputs/s.bhatia@imperial.ac.uk/effect_of_each_treatment_id-2026-02-12T120859Z --resumed_results_folder outputs/s.bhatia@imperial.ac.uk/effect_of_each_treatment_id-2026-02-16T154500Z_folder --output_folder outputs/s.bhatia@imperial.ac.uk/effect_of_each_treatment_id-combined


import argparse
import pickle
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

def _validate_input_output_paths(
    suspended_results_folder: Path,
    resumed_results_folder: Path,
    output_folder: Path,
) -> None:
    """Validate input/output path constraints for pickle combination helper."""
    suspended_resolved = suspended_results_folder.resolve()
    resumed_resolved = resumed_results_folder.resolve()
    output_resolved = output_folder.resolve()

    if output_resolved == suspended_resolved or output_resolved == resumed_resolved:
        raise ValueError(
            "output_folder must be different from both suspended_results_folder and resumed_results_folder."
        )

def _combine_pickled_objects(suspended_obj: Any, resumed_obj: Any, context: str = "root") -> Any:
    """Combine suspended and resumed objects with suspended object first."""
    if suspended_obj is None and resumed_obj is None:
        return None
    if isinstance(suspended_obj, dict) and isinstance(resumed_obj, dict):
        combined = {}
        for key, suspended_value in suspended_obj.items():
            if key in resumed_obj:
                combined[key] = _combine_pickled_objects(
                    suspended_value, resumed_obj[key], context=f"{context}.{key}"
                )
            else:
                combined[key] = suspended_value
        for key, resumed_value in resumed_obj.items():
            if key not in combined:
                combined[key] = resumed_value
        return combined
    if isinstance(suspended_obj, pd.DataFrame) and isinstance(resumed_obj, pd.DataFrame):
        return pd.concat([suspended_obj, resumed_obj], axis=0)
    if isinstance(suspended_obj, pd.Series) and isinstance(resumed_obj, pd.Series):
        return pd.concat([suspended_obj, resumed_obj], axis=0)
    if isinstance(suspended_obj, list) and isinstance(resumed_obj, list):
        return suspended_obj + resumed_obj
    if isinstance(suspended_obj, tuple) and isinstance(resumed_obj, tuple):
        return suspended_obj + resumed_obj
    try:
        return suspended_obj + resumed_obj
    except TypeError as exc:
        raise TypeError(
            f"Unsupported combine operation at {context}: "
            f"{type(suspended_obj).__name__} and {type(resumed_obj).__name__}."
        ) from exc


def combine_suspended_and_resumed_pickles(
    suspended_results_folder: Path,
    resumed_results_folder: Path,
    output_folder: Path,
) -> None:
    """Combine corresponding suspended and resumed pickles into output folder."""
    _validate_input_output_paths(suspended_results_folder, resumed_results_folder, output_folder)

    draw_dirs = sorted([p for p in resumed_results_folder.iterdir() if p.is_dir()], key=lambda p: p.name)
    for draw_dir in draw_dirs:
        print(f"Processing draw directory: {draw_dir}...")
        run_dirs = sorted([p for p in draw_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
        for run_dir in run_dirs:
            print(f"  Processing run directory: {run_dir}...")
            pickles = sorted(run_dir.glob("*.pickle"), key=lambda p: p.name)
            for resumed_pickle_path in pickles:
                print(f"    Processing pickle file: {resumed_pickle_path}...")
                with resumed_pickle_path.open("rb") as resumed_file:
                    resumed_obj = pickle.load(resumed_file)

                suspended_pickle_path = (
                    suspended_results_folder / "0" / run_dir.name / resumed_pickle_path.name
                )
                if suspended_pickle_path.exists():
                    with suspended_pickle_path.open("rb") as suspended_file:
                        suspended_obj = pickle.load(suspended_file)
                    try:
                        combined_obj = _combine_pickled_objects(suspended_obj, resumed_obj)
                    except TypeError as exc:
                        raise TypeError(
                            "Could not combine pickled objects for "
                            f"{resumed_pickle_path} with types "
                            f"{type(suspended_obj).__name__} and {type(resumed_obj).__name__}."
                        ) from exc
                else:
                    warnings.warn(
                        "No suspended counterpart found for "
                        f"{resumed_pickle_path} (expected at {suspended_pickle_path}); "
                        "copying resumed object to output unchanged.",
                        stacklevel=2,
                    )
                    combined_obj = resumed_obj

                output_pickle_path = output_folder / draw_dir.name / run_dir.name / resumed_pickle_path.name
                output_pickle_path.parent.mkdir(parents=True, exist_ok=True)
                with output_pickle_path.open("wb") as output_file:
                    pickle.dump(combined_obj, output_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine suspended and resumed pickle outputs into a new output folder, "
            "with suspended content prepended where counterparts exist."
        )
    )
    parser.add_argument("suspended_results_folder", type=Path)
    parser.add_argument("resumed_results_folder", type=Path)
    parser.add_argument("output_folder", type=Path)
    args = parser.parse_args()

    combine_suspended_and_resumed_pickles(
        suspended_results_folder=args.suspended_results_folder,
        resumed_results_folder=args.resumed_results_folder,
        output_folder=args.output_folder,
    )


if __name__ == "__main__":
    main()
