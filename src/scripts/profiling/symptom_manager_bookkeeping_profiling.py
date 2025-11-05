"""
A script to register so many symptoms and run the has_what function so many times.
For use in profiling.
"""

import time
from pathlib import Path

import numpy as np

from tlo import Date, Simulation
from tlo.methods import demography, symptommanager


def setup_simulation(pop_size=100_000, num_symptoms=50):
    """Set up a simulation with population and many symptoms."""
    start_date = Date(2010, 1, 1)

    resource_dir = Path("./resources")

    sim = Simulation(
        start_date=start_date,
        seed=0,
        # log_config={"filename": "symptom_profiling", "directory": "./outputs"},
        resourcefilepath=resource_dir,
    )

    sim.register(demography.Demography(), symptommanager.SymptomManager())

    # Register symptoms
    sm = sim.modules["SymptomManager"]
    for i in range(num_symptoms):
        sm.register_symptom(symptommanager.Symptom(name=f"symptom_{i}"))

    # Initialize population - this will create the symptom properties
    sim.make_initial_population(n=pop_size)

    return sim


def assign_random_symptoms(sim, symptom_prob=0.1):
    """Assign random symptoms to the population."""
    df = sim.population.props
    sm = sim.modules["SymptomManager"]

    # Assign symptoms randomly
    for symptom in sm.symptom_names:
        # Random subset of population to have this symptom
        has_symptom = np.random.random(len(df)) < symptom_prob
        person_ids = df.index[has_symptom].tolist()

        if person_ids:
            sm.change_symptom(
                person_id=person_ids,
                symptom_string=symptom,
                add_or_remove="+",
                disease_module=sm,
                duration_in_days=None,
            )


def profile_has_what(sim, num_tests=1000000):
    """Profiling has_what function by calling it repeatedly."""
    df = sim.population.props
    sm = sim.modules["SymptomManager"]

    # Get random sample of person_ids to test
    test_ids = np.random.choice(df.index[df.is_alive], size=num_tests, replace=True)

    # Time the has_what function
    start_time = time.time()

    results = []
    for person_id in test_ids:
        results.append(sm.has_what(person_id))

    elapsed = time.time() - start_time
    avg_time = elapsed / num_tests

    print(f"Tested has_what() {num_tests} times")
    print(f"Total time: {elapsed:.4f} seconds")
    print(f"Average time per call: {avg_time:.6f} seconds")
    print(f"First 5 results: {results[:5]}")

    return avg_time


print("Setting up simulation...")
sim = setup_simulation(pop_size=100_000, num_symptoms=50)

print("Assigning random symptoms...")
assign_random_symptoms(sim, symptom_prob=0.6)

print("\nProfiling has_what...")
avg_time = profile_has_what(sim, num_tests=1000000)

print("\nProfiling complete!")
print(f"Average time per has_what call: {avg_time:.6f} seconds")
