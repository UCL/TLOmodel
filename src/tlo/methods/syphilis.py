from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from tlo import Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin
from tlo.methods import Metadata, pregnancy_helper_functions
from tlo.util import read_csv_files

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DETECTABLE_STAGES = ['primary', 'secondary', 'early_latent', 'late_latent']
CONGENITAL_SYPHILIS_STILLBIRTH_CHECKPOINTS = [27, 31, 35, 40]


class Syphilis(Module):
    """Pregnancy syphilis natural history and congenital syphilis outcomes.

    This module owns the syphilis-specific biology while preserving the pregnancy episode
    timing, ANC workflow, and MNH logging surfaces in the existing pregnancy modules.
    """

    def __init__(self, name=None):
        super().__init__(name)
        self.current_parameters = dict()

    INIT_DEPENDENCIES = {'Demography', 'PregnancySupervisor'}

    METADATA = {
        Metadata.DISEASE_MODULE,
    }

    PARAMETERS = {
        'prob_syphilis_during_pregnancy': Parameter(
            Types.LIST, 'probability that this womans will develop syphilis during her pregnancy'),
        'prob_pre_existing_syphilis_at_pregnancy_start': Parameter(
            Types.LIST, 'probability that a woman enters pregnancy with untreated syphilis infection'),
        'pre_existing_syphilis_stage_distribution': Parameter(
            Types.LIST, 'probability distribution across primary, secondary, early latent, and late latent syphilis '
                        'for untreated infections present at the start of pregnancy'),
        'duration_primary_weeks': Parameter(
            Types.INT, 'typical duration of primary syphilis in weeks'),
        'duration_secondary_weeks': Parameter(
            Types.INT, 'typical duration of secondary syphilis in weeks'),
        'duration_early_latent_weeks': Parameter(
            Types.INT, 'typical duration from early latent to late latent syphilis in weeks'),
        'prob_congenital_transmission_primary': Parameter(
            Types.LIST, 'probability of fetal infection if mother has untreated primary syphilis'),
        'prob_congenital_transmission_secondary': Parameter(
            Types.LIST, 'probability of fetal infection if mother has untreated secondary syphilis'),
        'prob_congenital_transmission_early_latent': Parameter(
            Types.LIST, 'probability of fetal infection if mother has untreated early latent syphilis'),
        'prob_congenital_transmission_late_latent': Parameter(
            Types.LIST, 'probability of fetal infection if mother has untreated late latent syphilis'),
        'prob_stillbirth_from_congenital_syphilis': Parameter(
            Types.LIST, 'probability of stillbirth if fetus has congenital syphilis (untreated)'),
    }

    PROPERTIES = {
        # The ps_ names are preserved for output/API compatibility with the existing pregnancy modules.
        'ps_syphilis_state': Property(
            Types.CATEGORICAL,
            'Current stage of syphilis infection during pregnancy',
            categories=['none', 'primary', 'secondary', 'early_latent', 'late_latent', 'tertiary']
        ),
        'ps_syphilis_treated': Property(
            Types.BOOL,
            'Whether syphilis treatment has been received during the current pregnancy'
        ),
        'ps_congenital_syphilis': Property(
            Types.BOOL,
            'Whether the fetus has been infected with congenital syphilis'
        ),
    }

    def read_parameters(self, resourcefilepath: Optional[Path] = None):
        parameter_dataframe = read_csv_files(resourcefilepath / 'ResourceFile_Syphilis', files='parameter_values')
        self.load_parameters_from_dataframe(parameter_dataframe)

    def initialise_population(self, population):
        df = population.props
        df.loc[df.is_alive, 'ps_syphilis_state'] = 'none'
        df.loc[df.is_alive, 'ps_syphilis_treated'] = False
        df.loc[df.is_alive, 'ps_congenital_syphilis'] = False

    def initialise_simulation(self, sim):
        pregnancy_helper_functions.update_current_parameter_dictionary(self, list_position=0)

    def on_birth(self, mother_id, child_id):
        reset_pregnancy_syphilis_properties(self, child_id)

    def reset_pregnancy_syphilis_properties(self, id_or_index):
        reset_pregnancy_syphilis_properties(self, id_or_index)

    def schedule_syphilis_progression_event(self, individual_id):
        schedule_syphilis_progression_event(self, individual_id)

    def set_pre_existing_syphilis_stage_onset_date(self, individual_id):
        set_pre_existing_syphilis_stage_onset_date(self, individual_id)

    def initialise_pre_existing_syphilis_for_pregnancy(self, individual_id):
        initialise_pre_existing_syphilis_for_pregnancy(self, individual_id)

    def schedule_incident_syphilis_infections_for_new_pregnancies(self):
        schedule_incident_syphilis_infections_for_new_pregnancies(self)

    def apply_risk_of_congenital_syphilis(self, gestation_of_interest):
        apply_risk_of_congenital_syphilis(self, gestation_of_interest)

    def apply_assigned_congenital_syphilis_stillbirths(self, gestation_of_interest):
        apply_assigned_congenital_syphilis_stillbirths(self, gestation_of_interest)

    def treat_maternal_syphilis(self, person_id):
        treat_maternal_syphilis(self, person_id)

    def do_newborn_congenital_syphilis_handoff(self, mother_id, child_id):
        do_newborn_congenital_syphilis_handoff(self, mother_id, child_id)


def _pregnancy_supervisor(module):
    return module.sim.modules['PregnancySupervisor']


def _current_parameters(module):
    return module.current_parameters


def _rng(module):
    # Preserve pre-refactor stochastic behaviour by continuing to use the pregnancy RNG stream.
    return _pregnancy_supervisor(module).rng


def _mni(module):
    return _pregnancy_supervisor(module).mother_and_newborn_info


def _counter(module):
    return _pregnancy_supervisor(module).mnh_outcome_counter


def reset_pregnancy_syphilis_properties(module, id_or_index):
    df = module.sim.population.props
    df.loc[id_or_index, 'ps_syphilis_state'] = 'none'
    df.loc[id_or_index, 'ps_syphilis_treated'] = False
    df.loc[id_or_index, 'ps_congenital_syphilis'] = False


def schedule_syphilis_progression_event(module, individual_id):
    """Schedule the next biologically timed syphilis stage transition for a pregnant woman."""
    df = module.sim.population.props
    mni = _mni(module)
    params = _current_parameters(module)

    if individual_id not in mni:
        return

    current_stage = df.at[individual_id, 'ps_syphilis_state']
    if current_stage == 'primary':
        next_stage = 'secondary'
        duration = params['duration_primary_weeks']
    elif current_stage == 'secondary':
        next_stage = 'early_latent'
        duration = params['duration_secondary_weeks']
    elif current_stage == 'early_latent':
        if mni[individual_id]['syphilis_infection_origin'] != 'pre_existing':
            return
        next_stage = 'late_latent'
        duration = params['duration_early_latent_weeks']
    else:
        return

    stage_onset = mni[individual_id]['syphilis_stage_onset_date']
    if pd.isnull(stage_onset):
        stage_onset = module.sim.date
        mni[individual_id]['syphilis_stage_onset_date'] = stage_onset

    progression_date = stage_onset + pd.Timedelta(weeks=int(duration))
    if progression_date < module.sim.date:
        progression_date = module.sim.date

    mni[individual_id]['syphilis_next_progression_date'] = progression_date
    module.sim.schedule_event(
        SyphilisProgressionEvent(module, individual_id, from_stage=current_stage, to_stage=next_stage),
        progression_date
    )


def set_pre_existing_syphilis_stage_onset_date(module, individual_id):
    """Backdate stage onset for infections that were already present when pregnancy began."""
    df = module.sim.population.props
    mni = _mni(module)
    params = _current_parameters(module)

    current_stage = df.at[individual_id, 'ps_syphilis_state']
    max_stage_duration_weeks = {
        'primary': params['duration_primary_weeks'],
        'secondary': params['duration_secondary_weeks'],
        'early_latent': params['duration_early_latent_weeks'],
    }.get(current_stage)

    if max_stage_duration_weeks is None:
        mni[individual_id]['syphilis_stage_onset_date'] = module.sim.date
        return

    max_days_in_stage = max(1, int(max_stage_duration_weeks) * 7)
    days_since_stage_onset = _rng(module).randint(0, max_days_in_stage)
    mni[individual_id]['syphilis_stage_onset_date'] = module.sim.date - pd.Timedelta(days=days_since_stage_onset)


def initialise_pre_existing_syphilis_for_pregnancy(module, individual_id):
    """Initialise pregnancy-specific syphilis metadata for infections that pre-date pregnancy."""
    df = module.sim.population.props
    mni = _mni(module)
    params = _current_parameters(module)
    rng = _rng(module)

    if individual_id not in mni:
        return

    if df.at[individual_id, 'ps_syphilis_state'] == 'none':
        has_pre_existing_syphilis = (
            rng.random_sample() < params['prob_pre_existing_syphilis_at_pregnancy_start']
        )
        if not has_pre_existing_syphilis:
            return

        stages = ['primary', 'secondary', 'early_latent', 'late_latent']
        stage_distribution = np.asarray(params['pre_existing_syphilis_stage_distribution'], dtype=float)
        if len(stage_distribution) != len(stages):
            logger.info(key='error', data='pre_existing_syphilis_stage_distribution has invalid length')
            return
        if stage_distribution.sum() <= 0:
            return
        stage_distribution = stage_distribution / stage_distribution.sum()
        df.at[individual_id, 'ps_syphilis_state'] = rng.choice(stages, p=stage_distribution)

    df.at[individual_id, 'ps_syphilis_treated'] = False

    mni[individual_id]['syphilis_infection_origin'] = 'pre_existing'
    if pd.isnull(mni[individual_id]['syphilis_stage_onset_date']):
        set_pre_existing_syphilis_stage_onset_date(module, individual_id)
    schedule_syphilis_progression_event(module, individual_id)


def schedule_incident_syphilis_infections_for_new_pregnancies(module):
    """Schedule incident syphilis infections for women entering pregnancy."""
    df = module.sim.population.props
    params = _current_parameters(module)
    rng = _rng(module)
    mni = _mni(module)

    at_risk_women = (
        df.is_alive & df.is_pregnant &
        (df.ps_gestational_age_in_weeks == 3) &
        (df.ps_ectopic_pregnancy == 'none') &
        (df.ps_syphilis_state == 'none')
    )

    syphilis_risk = pd.Series(
        rng.random_sample(len(at_risk_women.loc[at_risk_women])) < params['prob_syphilis_during_pregnancy'],
        index=at_risk_women.loc[at_risk_women].index
    )

    for person in syphilis_risk.loc[syphilis_risk].index:
        onset_day = rng.randint(0, 280)
        mni[person]['pred_syph_infect'] = module.sim.date + pd.Timedelta(days=onset_day)
        module.sim.schedule_event(
            SyphilisInPregnancyEvent(module, person),
            module.sim.date + pd.Timedelta(days=onset_day)
        )


def apply_risk_of_congenital_syphilis(module, gestation_of_interest):
    """Apply vertical transmission risk from maternal syphilis to fetal congenital syphilis."""
    df = module.sim.population.props
    params = _current_parameters(module)
    rng = _rng(module)
    mni = _mni(module)

    at_risk = df.is_alive & df.is_pregnant & \
              (df.ps_gestational_age_in_weeks == gestation_of_interest) & \
              (df.ps_syphilis_state.isin(DETECTABLE_STAGES)) & \
              (~df.ps_congenital_syphilis) & \
              (df.ps_ectopic_pregnancy == 'none')

    if not at_risk.any():
        return

    def get_transmission_prob(row):
        state = row['ps_syphilis_state']
        if state == 'primary':
            return params['prob_congenital_transmission_primary']
        elif state == 'secondary':
            return params['prob_congenital_transmission_secondary']
        elif state == 'early_latent':
            return params['prob_congenital_transmission_early_latent']
        else:
            return params['prob_congenital_transmission_late_latent']

    transmission_probs = df.loc[at_risk].apply(get_transmission_prob, axis=1)
    random_draws = pd.Series(
        rng.random_sample(len(at_risk.loc[at_risk])),
        index=at_risk.loc[at_risk].index
    )
    infected = random_draws < transmission_probs

    future_stillbirth_weeks = [
        week for week in CONGENITAL_SYPHILIS_STILLBIRTH_CHECKPOINTS
        if week >= gestation_of_interest
    ]
    congenital_stillbirths = pd.Series(
        rng.random_sample(len(infected.loc[infected])) < params['prob_stillbirth_from_congenital_syphilis'],
        index=infected.loc[infected].index
    )

    for person in infected.loc[infected].index:
        df.at[person, 'ps_congenital_syphilis'] = True
        pregnancy_helper_functions.store_dalys_in_mni(
            person, mni, 'congenital_syphilis_onset', module.sim.date
        )
        if congenital_stillbirths.at[person] and future_stillbirth_weeks:
            mni[person]['congenital_syphilis_stillbirth_week'] = future_stillbirth_weeks[0]
        _counter(module)['congenital_syphilis_infection'] += 1


def apply_assigned_congenital_syphilis_stillbirths(module, gestation_of_interest):
    """Apply congenital syphilis stillbirth outcomes assigned when fetal infection was established."""
    df = module.sim.population.props
    mni = _mni(module)
    ps = _pregnancy_supervisor(module)

    at_risk_congenital = df.is_alive & df.is_pregnant & \
                         (df.ps_gestational_age_in_weeks == gestation_of_interest) & \
                         df.ps_congenital_syphilis & \
                         (df.ps_ectopic_pregnancy == 'none') & \
                         (df.ac_admitted_for_immediate_delivery == 'none') & \
                         ~df.la_currently_in_labour & ~df.ps_emergency_event

    congenital_stillbirths = [
        person for person in at_risk_congenital.loc[at_risk_congenital].index
        if mni.get(person, {}).get('congenital_syphilis_stillbirth_week') == gestation_of_interest
    ]

    if congenital_stillbirths:
        congenital_stillbirths = pd.Series(True, index=congenital_stillbirths)
        ps.update_variables_post_still_birth_for_data_frame(congenital_stillbirths)

        for person in congenital_stillbirths.index:
            _counter(module)['congenital_syphilis_stillbirth'] += 1


def treat_maternal_syphilis(module, person_id):
    """Cure maternal syphilis while retaining the current-pregnancy treatment record."""
    df = module.sim.population.props
    mni = _mni(module)

    df.at[person_id, 'ps_syphilis_state'] = 'none'
    df.at[person_id, 'ps_syphilis_treated'] = True

    if person_id in mni:
        mni[person_id]['syphilis_treatment_date'] = module.sim.date
        mni[person_id]['syphilis_next_progression_date'] = pd.NaT


def do_newborn_congenital_syphilis_handoff(module, mother_id, child_id):
    """Transfer fetal congenital syphilis status to the live-born newborn."""
    df = module.sim.population.props
    df.at[child_id, 'nb_congenital_syphilis'] = bool(df.at[mother_id, 'ps_congenital_syphilis'])
    if df.at[child_id, 'nb_congenital_syphilis']:
        _counter(module)['congenital_syphilis_live_birth'] += 1


class SyphilisInPregnancyEvent(Event, IndividualScopeEventMixin):
    """Onset incident syphilis during pregnancy at the pre-scheduled infection date."""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        mni = _mni(self.module)

        if (not df.at[individual_id, 'is_alive'] or
            not df.at[individual_id, 'is_pregnant'] or
            (individual_id not in mni) or
            (not (mni[individual_id]['pred_syph_infect'] == self.sim.date))):
            return

        df.at[individual_id, 'ps_syphilis_state'] = 'primary'
        df.at[individual_id, 'ps_syphilis_treated'] = False
        pregnancy_helper_functions.store_dalys_in_mni(
            individual_id, mni, 'primary_syphilis_onset', self.sim.date
        )
        mni[individual_id]['syphilis_infection_origin'] = 'incident_pregnancy'
        mni[individual_id]['syphilis_stage_onset_date'] = self.sim.date
        schedule_syphilis_progression_event(self.module, individual_id)

        _counter(self.module)['syphilis'] += 1


class SyphilisProgressionEvent(Event, IndividualScopeEventMixin):
    """Progress syphilis by one biologically timed stage if the expected untreated state still holds."""

    def __init__(self, module, individual_id, from_stage, to_stage):
        super().__init__(module, person_id=individual_id)
        self.from_stage = from_stage
        self.to_stage = to_stage

    def apply(self, individual_id):
        df = self.sim.population.props
        mni = _mni(self.module)

        if (not df.at[individual_id, 'is_alive'] or
            not df.at[individual_id, 'is_pregnant'] or
            (individual_id not in mni) or
            df.at[individual_id, 'ps_syphilis_treated'] or
            (df.at[individual_id, 'ps_syphilis_state'] != self.from_stage) or
            (mni[individual_id]['syphilis_next_progression_date'] != self.sim.date)):
            return

        if self.from_stage == 'early_latent' and mni[individual_id]['syphilis_infection_origin'] != 'pre_existing':
            return

        df.at[individual_id, 'ps_syphilis_state'] = self.to_stage
        mni[individual_id]['syphilis_stage_onset_date'] = self.sim.date
        mni[individual_id]['syphilis_next_progression_date'] = pd.NaT

        if self.to_stage == 'secondary':
            pregnancy_helper_functions.store_dalys_in_mni(
                individual_id, mni, 'secondary_syphilis_onset', self.sim.date
            )
            _counter(self.module)['syphilis_progression_primary_secondary'] += 1
        elif self.to_stage == 'early_latent':
            pregnancy_helper_functions.store_dalys_in_mni(
                individual_id, mni, 'early_latent_syphilis_onset', self.sim.date
            )
            _counter(self.module)['syphilis_progression_secondary_latent'] += 1
        elif self.to_stage == 'late_latent':
            _counter(self.module)['syphilis_progression_late_latent'] += 1

        schedule_syphilis_progression_event(self.module, individual_id)
