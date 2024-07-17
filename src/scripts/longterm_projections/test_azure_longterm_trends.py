import datetime
import time
from pathlib import Path

from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo
from tlo.scenario import BaseScenario
from tlo.methods import (
    alri, bladder_cancer, breast_cancer, cardio_metabolic_disorders,
    copd, demography, depression, diarrhoea, enhanced_lifestyle, epi,
    epilepsy, healthburden, healthseekingbehaviour, healthsystem, hiv,
    malaria, measles, oesophagealcancer, other_adult_cancers, prostate_cancer,
    rti, schisto, simplified_births, stunting, symptommanager, tb, wasting
)
from tlo.methods.scenario_switcher import ImprovedHealthSystemAndCareSeekingScenarioSwitcher

# Start time for the simulation
start_time = time.time()

# Paths for outputs and resources
output_path = Path("./outputs/longterm_trends")
resource_file_path = Path("./resources")

# Date-stamp for labeling log files and other outputs
date_stamp = datetime.date.today().strftime("__%Y_%m_%d")

class LongRun(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2060, 12, 31)
        self.pop_size = 1000
        self.number_of_draws = 1
        self.runs_per_draw = 1

    def log_configuration(self):
        return {
            'filename': 'longterm_trends_all_diseases',
            'directory': './outputs/longterm_trends',
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.demography.detail': logging.WARNING,
                'tlo.methods.healthburden': logging.INFO,
                'tlo.methods.healthsystem': logging.INFO,
                'tlo.methods.healthsystem.summary': logging.INFO,
                'tlo.methods.contraception': logging.INFO,
            }
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources),
            simplified_births.SimplifiedBirths(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthsystem.HealthSystem(
                resourcefilepath=resource_file_path,
                service_availability=["*"],
                mode_appt_constraints=0,
                cons_availability="all",
                ignore_priority=False,
                beds_availability='all',
                equip_availability='all',
                capabilities_coefficient=1.0,
                use_funded_or_actual_staffing="funded",
                disable=False,
                disable_and_reject_all=False,
            ),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            epi.Epi(resourcefilepath=self.resources),
            hiv.Hiv(resourcefilepath=self.resources),
            tb.Tb(resourcefilepath=self.resources),
            malaria.Malaria(resourcefilepath=self.resources),
            alri.Alri(resourcefilepath=self.resources),
            diarrhoea.Diarrhoea(resourcefilepath=self.resources),
            stunting.Stunting(resourcefilepath=self.resources),
            wasting.Wasting(resourcefilepath=self.resources),
            measles.Measles(resourcefilepath=self.resources),
            schisto.Schisto(resourcefilepath=self.resources),
            bladder_cancer.BladderCancer(resourcefilepath=self.resources),
            breast_cancer.BreastCancer(resourcefilepath=self.resources),
            oesophagealcancer.OesophagealCancer(resourcefilepath=self.resources),
            other_adult_cancers.OtherAdultCancer(resourcefilepath=self.resources),
            prostate_cancer.ProstateCancer(resourcefilepath=self.resources),
            cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=self.resources),
            rti.RTI(resourcefilepath=self.resources),
            copd.Copd(resourcefilepath=self.resources),
            depression.Depression(resourcefilepath=self.resources),
            epilepsy.Epilepsy(resourcefilepath=self.resources),
            ImprovedHealthSystemAndCareSeekingScenarioSwitcher(resourcefilepath=self.resources)
        ]

    def draw_parameters(self, draw_number, rng):
        return get_parameters_for_status_quo()

if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
