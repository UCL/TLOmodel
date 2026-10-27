"""
Dummy nested-logit healthcare-seeking model

Run:
    python dummy_nested_logit_healthcare.py
i
The model has three nests:
    1. no care          - singleton nest
    2. self-care        - singleton nest
    3. formal care      - health centre and hospital

The model is deliberately small so that you can change the inputs and see
how distance, cost, availability, symptoms, and the nesting parameter affect
choice probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import exp, log
from random import Random
from typing import Dict, List, Tuple


@dataclass
class Person:
    age: int = 35
    severe_symptoms: int = 0
    wealth: float = 0.0


@dataclass
class Facility:
    name: str
    facility_type: str
    distance_km: float
    cost: float
    availability: float  # 0 = unavailable, 1 = fully available
    wait_hours: float


# A nest is represented by its name, a list of alternatives, and lambda.
# Singleton nests have one alternative and lambda = 1.
NEST_LAMBDAS = {
    "no_care": 1.0,
    "self_care": 1.0,
    "formal_care": 0.60,
}


def utility_no_care(person: Person) -> float:
    """Direct utility of not seeking care."""
    return 0.30 - 0.35 * person.severe_symptoms


def utility_self_care(person: Person) -> float:
    """Direct utility of self-care/pharmacy."""
    return 0.10 - 0.20 * person.severe_symptoms + 0.05 * person.wealth


def utility_facility(person: Person, facility: Facility) -> float:
    """Systematic facility utility V_ij.

    Change these coefficients to experiment with behavioural assumptions.
    Distance, cost, and waiting time reduce utility. Availability and symptom
    suitability increase it. Severe symptoms make hospital care more valuable.
    """
    utility = 1.00
    utility -= 0.18 * facility.distance_km
    utility -= 0.08 * facility.cost
    utility -= 0.12 * facility.wait_hours
    utility += 1.40 * facility.availability

    if facility.facility_type == "health_centre":
        utility += 0.35
        utility += 0.15 * (1 - person.severe_symptoms)
    elif facility.facility_type == "hospital":
        utility += 0.20
        utility += 1.20 * person.severe_symptoms

    utility += 0.08 * person.wealth
    return utility


def stable_exp(x: float) -> float:
    """Exponentiation for this small demonstration model."""
    return exp(max(-700.0, min(700.0, x)))


def probabilities(
    person: Person,
    facilities: List[Facility],
    nest_direct_utilities: Dict[str, float] | None = None,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Return (overall probabilities, nest probabilities, facility utilities).

    The formal-care nest uses:
        P(j | g) = exp(V_j / lambda_g) / sum_l exp(V_l / lambda_g)
        IV_g     = log(sum_l exp(V_l / lambda_g))
        A_g      = W_g + lambda_g * IV_g
        P(g)     = exp(A_g) / sum_h exp(A_h)
        P(j)     = P(g) * P(j | g)
    """
    W = {
        "no_care": 0.0,
        "self_care": 0.0,
        "formal_care": 0.0,
    }
    if nest_direct_utilities:
        W.update(nest_direct_utilities)

    facility_utilities = {
        facility.name: utility_facility(person, facility)
        for facility in facilities
    }

    # Build nest-level inclusive values and systematic nest utilities A_g.
    nest_alternatives = {
        "no_care": [("no_care", utility_no_care(person))],
        "self_care": [("self_care", utility_self_care(person))],
        "formal_care": [
            (facility.name, facility_utilities[facility.name])
            for facility in facilities
        ],
    }

    nest_indices: Dict[str, float] = {}
    conditional: Dict[str, Dict[str, float]] = {}

    for nest, alternatives in nest_alternatives.items():
        lam = NEST_LAMBDAS[nest]
        scaled = {
            name: value / lam for name, value in alternatives
        }
        denominator = sum(stable_exp(value) for value in scaled.values())
        conditional[nest] = {
            name: stable_exp(value) / denominator
            for name, value in scaled.items()
        }

        inclusive_value = log(denominator)
        nest_indices[nest] = W[nest] + lam * inclusive_value

    nest_denominator = sum(
        stable_exp(index) for index in nest_indices.values()
    )
    nest_probabilities = {
        nest: stable_exp(index) / nest_denominator
        for nest, index in nest_indices.items()
    }

    overall: Dict[str, float] = {}
    for nest, conditional_probabilities in conditional.items():
        for alternative, probability in conditional_probabilities.items():
            overall[alternative] = nest_probabilities[nest] * probability

    return overall, nest_probabilities, facility_utilities


def draw_choice(
    overall_probabilities: Dict[str, float],
    rng: Random,
) -> str:
    """Draw one choice from the unconditional probabilities."""
    threshold = rng.random()
    cumulative = 0.0
    for choice, probability in overall_probabilities.items():
        cumulative += probability
        if threshold <= cumulative:
            return choice
    return list(overall_probabilities)[-1]


def print_results(
    title: str,
    overall: Dict[str, float],
    nests: Dict[str, float],
    utilities: Dict[str, float],
) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print("Facility utilities V_ij:")
    for name, value in utilities.items():
        print(f"  {name:18s} {value:8.3f}")
    print("\nNest probabilities:")
    for name, probability in nests.items():
        print(f"  {name:18s} {probability:8.3%}")
    print("\nOverall choice probabilities:")
    for name, probability in overall.items():
        print(f"  {name:18s} {probability:8.3%}")


def run_experiment(
    person: Person,
    facilities: List[Facility],
    label: str,
) -> None:
    overall, nests, utilities = probabilities(person, facilities)
    print_results(label, overall, nests, utilities)


def simulate_choices(
    person: Person,
    facilities: List[Facility],
    n: int = 10_000,
    seed: int = 123,
) -> Dict[str, int]:
    """Draw repeated choices to compare simulated shares with probabilities."""
    rng = Random(seed)
    counts: Dict[str, int] = {}
    for _ in range(n):
        overall, _, _ = probabilities(person, facilities)
        choice = draw_choice(overall, rng)
        counts[choice] = counts.get(choice, 0) + 1
    return counts


def main() -> None:
    person = Person(age=35, severe_symptoms=0, wealth=0.0)

    facilities = [
        Facility(
            name="health_centre",
            facility_type="health_centre",
            distance_km=3.0,
            cost=1.0,
            availability=0.90,
            wait_hours=1.0,
        ),
        Facility(
            name="hospital",
            facility_type="hospital",
            distance_km=12.0,
            cost=3.0,
            availability=0.75,
            wait_hours=4.0,
        ),
    ]

    run_experiment(person, facilities, "Baseline: mild symptoms")

    # Experiment 1: severe symptoms make hospital care more valuable.
    severe_person = replace(person, severe_symptoms=1)
    run_experiment(severe_person, facilities, "Experiment 1: severe symptoms")

    # Experiment 2: the health centre becomes more distant.
    distant_health_centre = replace(facilities[0], distance_km=10.0)
    facilities_distance = [distant_health_centre, facilities[1]]
    run_experiment(
        person,
        facilities_distance,
        "Experiment 2: health centre is farther away",
    )

    # Experiment 3: the health centre is unavailable.
    unavailable_health_centre = replace(facilities[0], availability=0.0)
    facilities_unavailable = [unavailable_health_centre, facilities[1]]
    run_experiment(
        person,
        facilities_unavailable,
        "Experiment 3: health centre unavailable",
    )

    # Experiment 4: add a new clinic to show the value of more options.
    new_clinic = Facility(
        name="new_clinic",
        facility_type="health_centre",
        distance_km=5.0,
        cost=1.5,
        availability=0.95,
        wait_hours=0.5,
    )
    run_experiment(
        person,
        facilities + [new_clinic],
        "Experiment 4: a new clinic opens",
    )

    # Experiment 5: change lambda to see the effect of within-nest similarity.
    print("\nExperiment 5: changing the formal-care lambda")
    print("---------------------------------------------")
    original_lambda = NEST_LAMBDAS["formal_care"]
    for lambda_value in (1.0, 0.75, 0.60, 0.30):
        NEST_LAMBDAS["formal_care"] = lambda_value
        overall, nests, _ = probabilities(person, facilities)
        print(
            f"lambda={lambda_value:4.2f} | "
            f"formal nest={nests['formal_care']:.3%} | "
            f"health centre={overall['health_centre']:.3%} | "
            f"hospital={overall['hospital']:.3%}"
        )
    NEST_LAMBDAS["formal_care"] = original_lambda

    # Monte Carlo check: simulated choices should approximately match probabilities.
    overall, _, _ = probabilities(person, facilities)
    counts = simulate_choices(person, facilities, n=10_000)
    print("\nMonte Carlo check: 10,000 simulated choices")
    print("--------------------------------------------")
    for choice, probability in overall.items():
        simulated_share = counts.get(choice, 0) / 10_000
        print(
            f"{choice:18s} model={probability:8.3%}  "
            f"simulated={simulated_share:8.3%}"
        )


if __name__ == "__main__":
    main()
