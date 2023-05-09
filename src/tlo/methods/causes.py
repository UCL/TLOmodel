"""
Classes and functions that support declarations of causes of death and disability.
"""

from collections import defaultdict
from typing import Union

import pandas as pd

from tlo.methods import Metadata


class Cause:
    """Data structure to store information about a Cause (of Death or Disability) used in the model
    'gbd_causes': set of strings for causes in the GBD datasets to which this cause is equivalent.
    'cause_of_death': the (single) category to which this cause belongs and should be labelled in output statistics.
    """
    def __init__(self, label: str, gbd_causes: Union[set, str] = None):
        """Do basic type checking."""
        assert (type(label) is str) and (label != '')
        self.label = label

        if gbd_causes is None:
            gbd_causes = set()

        if gbd_causes:
            gbd_causes = set(gbd_causes) if type(gbd_causes) in (list, set) else {gbd_causes}
            assert all([(type(c) is str) and (c != '') for c in gbd_causes])
        self.gbd_causes = gbd_causes


def collect_causes_from_disease_modules(all_modules, collect, acceptable_causes: set = None):
    """Helper function used by Demography and HealthBurden modules to look through Disease Modules and register
    declarations for either 'CAUSES_OF_DEATH' or 'CAUSES_OF_DISABILITY'.
    It will check that a gbd_cause is not associated with more than label (a requirement of this scheme).
    Optionally, for each cause that is collected, it will check the gbd_causes defined within are in a set of
    acceptable causes.
     """

    def check_cause(_cause: Cause, _acceptable_causes: set):
        """Helper function to check that a 'Cause' has been defined in a way that is acceptable."""
        # 0) Check type
        assert isinstance(_cause, Cause)

        # 1) Check that the declared gbd_cause is among the acceptable causes.
        for _c in _cause.gbd_causes:
            assert _c in _acceptable_causes, f'The declared gbd_cause {_c} is not among the acceptable causes.'

    collected_causes = dict()
    for m in all_modules:
        if Metadata.DISEASE_MODULE in m.METADATA:
            assert hasattr(m, collect), f'Disease module {m.name} must declare {collect} (even if empty)'
            declaration_in_module = getattr(m, collect)
            assert type(declaration_in_module) is dict

            for tlo_cause, cause in declaration_in_module.items():
                if (acceptable_causes is not None) and cause.gbd_causes:
                    check_cause(_cause=cause, _acceptable_causes=acceptable_causes)

                # Prevent over-writing of causes: throw error if the name is already in use but the new Cause is not
                # the same as that already registered.
                if tlo_cause in collected_causes:
                    assert (
                        (collected_causes[tlo_cause].gbd_causes == cause.gbd_causes) and
                        (collected_causes[tlo_cause].label == cause.label)
                    ), \
                        f"Conflict in declared cause {tlo_cause} by {m.name}. " \
                        f"A different specification has already been registered."

                # If ok, update these causes to the master dict of all causes of death
                collected_causes.update({tlo_cause: cause})

    # Check that each gbd_cause is not defined in respect of more than one label
    gbd_causes = dict()  # dict(<gbd_cause: label>)
    for c in collected_causes.values():
        for g in c.gbd_causes:
            if g in gbd_causes:
                assert gbd_causes[g] == c.label, f"The gbd cause {g} is defined under more than one label: " \
                                                 f"{gbd_causes[g]} and {c.label}."
            else:
                gbd_causes[g] = c.label

    return collected_causes


def get_gbd_causes_not_represented_in_disease_modules(causes: dict, gbd_causes: set):
    """
    Find the causes in the GBD datasets (`gbd_causes`) that are *not* represented within the Causes defined in `causes`
    :return: set of gbd_causes that are not represented.
    """
    all_gbd_causes_in_sim = set()
    for c in causes.values():
        all_gbd_causes_in_sim.update(c.gbd_causes)

    return gbd_causes - all_gbd_causes_in_sim


def create_mappers_from_causes_to_label(causes: dict, all_gbd_causes: set = None):
    """Helper function to create two mapping dicts to map to (1) tlo_cause --> label; and (2) gbd_cause --> label.
    Optionally, can provide checking that the mapping from gbd_causes exhaustively covers all gbd_causes.

    :causes: is a dict of the form {<tlo_name>: <Cause object>}
    :all_gbd_causes: is a set of strings for all possible gbd_causes.

    Note that this is specific to a run of the simulation as the configuration of modules determine which causes of
    death are counted under the tlo_cause named "Other".

    Nomeclectgure:
    'label' is the commmon category in which any type of death is classified (for ouput in statistics etc);
    'tlo_cause' is the name of cause of death used by the module;
    'gbd_cause' is the name of cause of death in the GBD dataset.
    """

    # 1) Reorganise the causes so that we have a dict
    # lookup: dict(<label> : dict(<tlo_causes>:<list of tlo_strings>, <gbd_causes>: <list_of_gbd_causes))
    lookup = defaultdict(lambda: {'tlo_causes': set(), 'gbd_causes': set()})

    for tlo_cause_name, cause in causes.items():
        label = cause.label
        list_of_gbd_causes = cause.gbd_causes
        lookup[label]['tlo_causes'].add(tlo_cause_name)
        for gbd_cause in list_of_gbd_causes:
            lookup[label]['gbd_causes'].add(gbd_cause)

    # 2) Create dicts for mapping (gbd_cause --> label) and (tlo_cause --> label)
    lookup_df = pd.DataFrame.from_dict(lookup, orient='index').applymap(lambda x: list(x))

    # Sort the lists and sort the index to provide reliable structure
    lookup_df = lookup_df.applymap(sorted).sort_index()

    #  - from tlo_cause --> label (key=tlo_cause, value=label)
    mapper_from_tlo_causes = dict((v, k) for k, v in (
        lookup_df.tlo_causes.apply(pd.Series).stack().reset_index(level=1, drop=True)
    ).items())

    #  - from gbd_cause --> label (key=gbd_cause, value=label)
    mapper_from_gbd_causes = dict((v, k) for k, v in (
        lookup_df.gbd_causes.apply(pd.Series).stack().reset_index(level=1, drop=True)
    ).items())

    # -- checks
    assert set(mapper_from_tlo_causes.keys()) == set(causes.keys())
    assert set(mapper_from_gbd_causes.values()).issubset(mapper_from_tlo_causes.values())
    if all_gbd_causes:
        assert set(mapper_from_gbd_causes.keys()) == all_gbd_causes

    return mapper_from_tlo_causes, mapper_from_gbd_causes
