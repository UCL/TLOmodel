import pytest

from tlo.dependencies import (
    get_all_dependencies,
    get_all_required_dependencies,
    get_dependencies_and_initialise,
)

from .test_module_dependencies import (
    module_class_map,
    parameterize_module_class,
    register_modules_and_simulate,
    resourcefilepath,
)
from .test_module_dependencies import sim as get_simulation  # to silence unused import warning

sim = get_simulation   # pytest fixture

@pytest.mark.slow
@parameterize_module_class
def test_module_dependencies_complete(sim, module_class):
    """Check declared dependencies are sufficient for successful (short) simulation.

    Dependencies here refers to the union of INIT_DEPENDENCIES and
    ADDITIONAL_DEPENDENCIES.
    """
    try:
        # If this module is an 'alternative' to one or more other modules, exclude these
        # modules from being selected to avoid clashes with this module
        excluded_module_classes = {
            module_class_map[module_name] for module_name in module_class.ALTERNATIVE_TO
        }
        register_modules_and_simulate(
            sim,
            get_dependencies_and_initialise(
                module_class,
                module_class_map=module_class_map,
                get_dependencies=get_all_dependencies,
                excluded_module_classes=excluded_module_classes,
                resourcefilepath=resourcefilepath
            ),
            check_all_dependencies=True
        )
    except Exception:
        all_dependencies = get_all_required_dependencies(module_class)
        pytest.fail(
            f"Module {module_class.__name__} appears to be missing dependencies "
            f"required to run simulation in the union of the INIT_DEPENDENCIES and "
            f"ADDITIONAL_DEPENDENCIES class attributes which is currently "
            f"{{{', '.join(all_dependencies)}}}."
        )
