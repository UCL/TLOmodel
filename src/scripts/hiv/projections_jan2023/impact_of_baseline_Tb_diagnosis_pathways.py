"""
Analyses baseline impact of Tb diagnosis pathways

It can be submitted on Azure Batch by running:
 tlo batch-submit src/scripts/hiv/projections_jan2023/impact_of_baseline_TB_diagnosis_pathways.py

or locally using:
tlo scenario-run src/scripts/hiv/projections_jan2023/impact_of_baseline_TB_diagnosis_pathways.py


"""
import warnings
from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


# Ignore warnings to avoid cluttering output from simulation - generally you do not
# need (and generally shouldn't) do this as warnings can contain useful information
# we will do so here for the purposes of this example to keep things simple.
warnings.simplefilter("ignore", (UserWarning, RuntimeWarning))


class ImpactBaselineTbDiagnosisPathways(BaseScenario):
    def __init__(self):
        super().__init__(
            seed=5001,
            start_date=Date(2010, 1, 1),
            end_date=Date(2033, 12, 31),
            initial_population_size=50_000,
            number_of_draws=2,
            runs_per_draw=1,
        )

    def log_configuration(self):
        return {
            'filename': 'impact_of_baseline_Tb_diagnosis_pathways',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.healthburden': logging.INFO,
                'tlo.methods.Tb': logging.INFO,
                'tlo.methods.Hiv': logging.INFO,
                'tlo.methods.healthsystem.summary': logging.INFO,
            }
        }
    def modules(self):
        return fullmodel(resourcefilepath=self.resources)

    def draw_parameters(self, draw_number, rng):
        return {
            'Tb': {
                'scenario': 0
            },
        }
if __name__ == '__main__':

    from tlo.cli import scenario_run

    scenario_run([__file__])

################################################################################################

