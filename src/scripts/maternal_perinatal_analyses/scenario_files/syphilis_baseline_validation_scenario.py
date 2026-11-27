from tlo import Date, DateOffset, Module, logging
from tlo.events import PopulationScopeEventMixin, Priority, RegularEvent
from tlo.methods import syphilis
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario

logger = logging.getLogger('tlo.analysis.syphilis_prevalence_validation')
logger.setLevel(logging.INFO)


class MaternalSyphilisPrevalenceObserver(Module):
    """Passive validation observer for maternal syphilis prevalence in pregnancy."""

    INIT_DEPENDENCIES = {'Contraception', 'Demography', 'PregnancySupervisor', 'Syphilis'}
    METADATA = {}
    PARAMETERS = {}
    PROPERTIES = {}

    def read_parameters(self, data_folder):
        pass

    def initialise_population(self, population):
        pass

    def initialise_simulation(self, sim):
        sim.schedule_event(
            MaternalSyphilisPrevalenceLoggingEvent(self),
            sim.date
        )

    def on_birth(self, mother_id, child_id):
        pass


class MaternalSyphilisPrevalenceLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """Log point prevalence of active maternal syphilis among currently pregnant women."""

    def __init__(self, module):
        super().__init__(
            module,
            frequency=DateOffset(months=1),
            priority=Priority.END_OF_DAY
        )

    def apply(self, population):
        df = population.props

        pregnant = df.is_alive & df.is_pregnant
        active_syphilis = pregnant & df.ps_syphilis_state.isin(syphilis.DETECTABLE_STAGES)

        number_pregnant = int(pregnant.sum())
        number_active_syphilis = int(active_syphilis.sum())
        prevalence = number_active_syphilis / number_pregnant if number_pregnant else float('nan')

        stage_counts = {
            stage: int((pregnant & (df.ps_syphilis_state == stage)).sum())
            for stage in syphilis.DETECTABLE_STAGES
        }

        logger.info(
            key='maternal_syphilis_prevalence',
            data={
                'number_pregnant': number_pregnant,
                'number_active_syphilis_pregnant': number_active_syphilis,
                'prevalence': prevalence,
                'prevalence_percent': prevalence * 100,
                **stage_counts,
                'active_syphilis_treated': int((active_syphilis & df.ps_syphilis_treated).sum()),
                'treated_cured_pregnant': int(
                    (pregnant & (df.ps_syphilis_state == 'none') & df.ps_syphilis_treated).sum()
                ),
            },
            description='Point prevalence of active syphilis among alive currently pregnant women'
        )


class SyphilisBaselineValidationScenario(BaseScenario):
    """Baseline scenario for validating maternal syphilis prevalence from 2020 onward."""

    def __init__(self):
        super().__init__()
        self.seed = 661184
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2031, 1, 1)
        self.pop_size = 10_000
        self.number_of_draws = 1
        self.runs_per_draw = 1

    def log_configuration(self):
        return {
            'filename': 'syphilis_baseline_validation_scenario',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.analysis.syphilis_prevalence_validation': logging.INFO,
                'tlo.methods.care_of_women_during_pregnancy': logging.INFO,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.healthsystem.summary': logging.INFO,
                'tlo.methods.labour': logging.INFO,
                'tlo.methods.newborn_outcomes': logging.INFO,
                'tlo.methods.pregnancy_supervisor': logging.INFO,
            }
        }

    def modules(self):
        modules = fullmodel(
            use_simplified_births=False,
            module_kwargs={
                'HealthSystem': {
                    'mode_appt_constraints': 1,
                    'cons_availability': 'default',
                }
            }
        )
        return [*modules, MaternalSyphilisPrevalenceObserver()]


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
