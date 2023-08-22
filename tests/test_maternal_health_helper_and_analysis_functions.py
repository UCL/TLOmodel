import os
from pathlib import Path

import pandas as pd

from tlo import Date, Simulation
from tlo.methods import (
    care_of_women_during_pregnancy,
    labour,
    newborn_outcomes,
    pregnancy_helper_functions,
)
from tlo.methods.fullmodel import fullmodel
from tlo.methods.healthsystem import FacilityInfo

start_date = Date(2010, 1, 1)

# The resource files
try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = 'resources'


def get_dummy_hsi(sim, mother_id, id, fl):
    """create dummy HSI to test that consumables truly are unavailable when using standard method"""
    from tlo.events import IndividualScopeEventMixin
    from tlo.methods.healthsystem import HSI_Event

    class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)

            self.TREATMENT_ID = 'Dummy'
            self.EXPECTED_APPT_FOOTPRINT = sim.modules['HealthSystem'].get_blank_appt_footprint()
            self.ACCEPTED_FACILITY_LEVEL = fl
            self.ALERT_OTHER_DISEASES = []

        def apply(self, person_id, squeeze_factor):
            pass

    # Pass some facility info to the HSI so it will run
    hsi_event = HSI_Dummy(module=sim.modules['CareOfWomenDuringPregnancy'], person_id=mother_id)
    hsi_event.facility_info = FacilityInfo(id=id,
                                           name=f'Facility_Level_{fl}_Balaka',
                                           level=fl,
                                           region='Southern')

    return hsi_event


def test_analysis_analysis_events_run_as_expected_and_update_parameters(seed):
    """Test that the analysis events run when scheduled and that they update the correct parameters as expected
    when they run"""
    sim = Simulation(start_date=start_date, seed=seed)
    sim.register(*fullmodel(resourcefilepath=resourcefilepath))
    sim.make_initial_population(n=100)
    lparams = sim.modules['Labour'].parameters
    pparams = sim.modules['PregnancySupervisor'].parameters

    # set the events to run 1/1/2010
    lparams['analysis_year'] = 2010
    pparams['analysis_year'] = 2010

    # set some availability probability
    new_avail_prob = 0.5

    # set variables that trigger updates within analysis events
    pparams['alternative_anc_coverage'] = True
    pparams['alternative_anc_quality'] = True
    pparams['anc_availability_probability'] = new_avail_prob
    lparams['alternative_bemonc_availability'] = True
    lparams['alternative_cemonc_availability'] = True
    lparams['bemonc_availability'] = new_avail_prob
    lparams['cemonc_availability'] = new_avail_prob
    lparams['alternative_pnc_coverage'] = True
    lparams['alternative_pnc_quality'] = True
    lparams['pnc_availability_probability'] = new_avail_prob

    # store the parameters determining care seeking before the change
    unchanged_odds_anc = pparams['odds_early_init_anc4'][0]
    unchanged_odds_pnc = lparams['odds_will_attend_pnc'][0]

    # run the model for 1 day
    sim.simulate(end_date=Date(2010, 1, 2))

    p_current_params = sim.modules['PregnancySupervisor'].current_parameters
    c_current_params = sim.modules['CareOfWomenDuringPregnancy'].current_parameters

    # Check antenatal parameters correctly updated
    assert p_current_params['prob_anc1_months_2_to_4'] == [1.0, 0, 0]
    assert p_current_params['prob_late_initiation_anc4'] == 0
    assert p_current_params['odds_early_init_anc4'] != unchanged_odds_anc

    for parameter in ['prob_intervention_delivered_urine_ds', 'prob_intervention_delivered_bp',
                      'prob_intervention_delivered_ifa', 'prob_intervention_delivered_llitn',
                      'prob_intervention_delivered_llitn', 'prob_intervention_delivered_tt',
                      'prob_intervention_delivered_poct', 'prob_intervention_delivered_syph_test',
                      'prob_intervention_delivered_iptp', 'prob_intervention_delivered_gdm_test']:
        assert c_current_params[parameter] == new_avail_prob

    # Check labour/postnatal/newborn parameters updated
    l_current_params = sim.modules['Labour'].current_parameters
    pn_current_params = sim.modules['PostnatalSupervisor'].current_parameters
    nbparams = sim.modules['NewbornOutcomes'].current_parameters

    assert l_current_params['squeeze_threshold_for_delay_three_bemonc'] == 10_000
    assert l_current_params['squeeze_threshold_for_delay_three_cemonc'] == 10_000
    assert l_current_params['squeeze_threshold_for_delay_three_pn'] == 10_000
    assert nbparams['squeeze_threshold_for_delay_three_nb_care'] == 10_000

    assert l_current_params['odds_will_attend_pnc'] != unchanged_odds_pnc
    assert l_current_params['prob_careseeking_for_complication_pn'] == new_avail_prob
    assert l_current_params['prob_timings_pnc'] == [1.0, 0.0]

    assert nbparams['prob_pnc_check_newborn'] == new_avail_prob
    assert nbparams['prob_care_seeking_for_complication'] == new_avail_prob

    assert pn_current_params['prob_care_seeking_postnatal_emergency_neonate'] == new_avail_prob
    assert pn_current_params['prob_care_seeking_postnatal_emergency'] == new_avail_prob


def test_analysis_events_force_availability_of_consumables_when_scheduled_in_anc(seed):
    """Test that when analysis is being conducted during a simulation that consumable availability is determined
    via some pre-defined analysis parameter and not via the health system within the ANC HSIs"""
    sim = Simulation(start_date=start_date, seed=seed)
    sim.register(*fullmodel(resourcefilepath=resourcefilepath))
    sim.make_initial_population(n=100)

    # Set the analysis event to run on the first day of the simulation
    pparams = sim.modules['PregnancySupervisor'].parameters
    pparams['analysis_year'] = 2010
    pparams['alternative_anc_quality'] = True
    pparams['anc_availability_probability'] = 1.0

    cparams = sim.modules['CareOfWomenDuringPregnancy'].parameters
    cparams['sensitivity_blood_test_syphilis'] = [1.0, 1.0]
    cparams['specificity_blood_test_syphilis'] = [1.0, 1.0]

    sim.simulate(end_date=Date(2010, 1, 2))

    # check the event ran
    assert sim.modules['PregnancySupervisor'].current_parameters['ps_analysis_in_progress']

    df = sim.population.props
    women_repro = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
    mother_id = women_repro.index[0]

    # Set key pregnancy variables so ANC will run
    df.at[mother_id, 'is_pregnant'] = True
    df.at[mother_id, 'date_of_last_pregnancy'] = start_date
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 26
    df.at[mother_id, 'ps_date_of_anc1'] = start_date + pd.DateOffset(days=2)
    df.at[mother_id, 'li_bmi'] = 1
    df.at[mother_id, 'la_parity'] = 0
    df.at[mother_id, 'ps_syphilis'] = True

    pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], mother_id)
    pregnancy_helper_functions.update_mni_dictionary(sim.modules['Labour'], mother_id)

    for params in ['prob_intervention_delivered_ifa', 'prob_adherent_ifa', 'prob_intervention_delivered_syph_test']:
        sim.modules['CareOfWomenDuringPregnancy'].current_parameters[params] = 1.0

    # Override the availability of the consumables within the health system- set to 0. If analysis was not running no
    # interventions requiring these consumable would run
    module = sim.modules['CareOfWomenDuringPregnancy']
    iron = module.item_codes_preg_consumables['iron_folic_acid']
    protein = module.item_codes_preg_consumables['balanced_energy_protein']
    calcium = module.item_codes_preg_consumables['calcium']
    syph_test = module.item_codes_preg_consumables['syphilis_test']
    syph_treat = module.item_codes_preg_consumables['syphilis_treatment']

    for cons in iron, protein, calcium, syph_test, syph_treat:
        sim.modules['HealthSystem'].override_availability_of_consumables(
            {cons[0]: 0.0})

    # refresh the consumables
    sim.modules['HealthSystem'].consumables._refresh_availability_of_consumables(date=sim.date)

    # create dummy HSI to test that consumables truly are unavailable when using standard method
    hsi_event = get_dummy_hsi(sim, mother_id, id=0, fl=0)

    # check that none of the consumables are available
    for cons in iron, protein, calcium, syph_test, syph_treat:
        available = hsi_event.get_consumables(item_codes=cons)
        assert not available

    # Then run normal ANC event to check that the analysis logic ensure that consumable availability has been set to 1.0
    # as described above - meaning all interventions will be delivered
    first_anc = care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareContact(
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=mother_id)
    first_anc.facility_info = FacilityInfo(id=1,
                                           name='Facility_Level_1a_Balaka',
                                           level='1a',
                                           region='Southern')
    first_anc.apply(person_id=mother_id, squeeze_factor=0.0)

    # Check that the function in pregnancy_helper_functions has correctly circumnavigated the availability of the
    # consumable and the intervention has been delivered
    assert df.at[mother_id, 'ac_receiving_iron_folic_acid']
    assert df.at[mother_id, 'ac_receiving_bep_supplements']
    assert df.at[mother_id, 'ac_receiving_calcium_supplements']
    assert not df.at[mother_id, 'ps_syphilis']


def test_analysis_events_force_availability_of_consumables_for_sba_analysis(seed):
    """Test that when analysis is being conducted during a simulation that consumable availability is determined
    via some pre-defined analysis parameter and not via the health system within the SBA HSIs"""
    sim = Simulation(start_date=start_date, seed=seed)
    sim.register(*fullmodel(resourcefilepath=resourcefilepath))
    sim.make_initial_population(n=100)

    # Set the analysis event to run at simulation start
    lparams = sim.modules['Labour'].parameters
    lparams['analysis_year'] = 2010
    lparams['alternative_bemonc_availability'] = True
    lparams['alternative_cemonc_availability'] = True

    # Set availability
    lparams['bemonc_availability'] = 1.0
    lparams['cemonc_availability'] = 1.0

    sim.simulate(end_date=Date(2010, 1, 2))

    params = sim.modules['Labour'].current_parameters
    assert params['la_analysis_in_progress']

    df = sim.population.props
    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info

    women_repro = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
    mother_id = women_repro.index[0]

    # Set key pregnancy variables so SBA will run
    df.at[mother_id, 'is_pregnant'] = True
    df.at[mother_id, 'la_currently_in_labour'] = True
    df.at[mother_id, 'date_of_last_pregnancy'] = start_date - pd.DateOffset(weeks=33)
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 35

    pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], mother_id)
    pregnancy_helper_functions.update_mni_dictionary(sim.modules['Labour'], mother_id)

    # Add some complications so that treatment should be delivered
    df.at[mother_id, 'ps_premature_rupture_of_membranes'] = True
    df.at[mother_id, 'ps_htn_disorders'] = 'severe_pre_eclamp'
    df.at[mother_id, 'la_obstructed_labour'] = True
    df.at[mother_id, 'la_sepsis'] = True

    mni[mother_id]['cpd'] = False
    mni[mother_id]['new_onset_spe'] = True
    mni[mother_id]['labour_state'] = 'late_preterm_labour'

    params['prob_progression_severe_pre_eclamp'] = 0.0

    # Override the availability of the consumables within the health system. set to 0.
    module = sim.modules['Labour']
    abx_prom = module.item_codes_lab_consumables['abx_for_prom']
    steroids = module.item_codes_lab_consumables['antenatal_steroids']
    cbp = module.item_codes_lab_consumables['delivery_core']
    ol = module.item_codes_lab_consumables['vacuum']
    mag_sulf = module.item_codes_lab_consumables['magnesium_sulfate']
    htns = module.item_codes_lab_consumables['iv_antihypertensives']
    seps = module.item_codes_lab_consumables['maternal_sepsis_core']
    resus = module.item_codes_lab_consumables['resuscitation']

    for cons in abx_prom, steroids, cbp, ol, mag_sulf, htns, seps, resus:
        for item in cons:
            sim.modules['HealthSystem'].override_availability_of_consumables(
                {item: 0.0})

    # refresh the consumables
    sim.modules['HealthSystem'].consumables._refresh_availability_of_consumables(date=sim.date)

    # create dummy HSI to test that consumables truly are unavailable when using standard method
    hsi_event = get_dummy_hsi(sim, mother_id, id=3, fl=2)
    for cons in abx_prom, steroids, cbp, ol, mag_sulf, htns, seps, resus:
        available = hsi_event.get_consumables(item_codes=cons)
        assert not available

    # Next define the actual HSI of interest
    sba = labour.HSI_Labour_ReceivesSkilledBirthAttendanceDuringLabour(
        module=sim.modules['Labour'], person_id=mother_id, facility_level_of_this_hsi=2)
    sba.facility_info = FacilityInfo(id=3,
                                     name='Facility_Level_2_Balaka',
                                     level='2',
                                     region='Southern')

    sba.apply(person_id=mother_id, squeeze_factor=0.0)

    # Check that the interventions are delivered as expected
    assert mni[mother_id]['abx_for_prom_given']
    assert mni[mother_id]['corticosteroids_given']
    assert df.at[mother_id, 'la_severe_pre_eclampsia_treatment']
    assert df.at[mother_id, 'la_maternal_hypertension_treatment']
    assert mni[mother_id]['mode_of_delivery'] == 'instrumental'
    assert df.at[mother_id, 'la_sepsis_treatment']
    assert mni[mother_id]['clean_birth_practices']
    assert mni[mother_id]['neo_will_receive_resus_if_needed']

    # Now repeat to test the CEmONC event
    params['success_rate_uterine_repair'] = 1.0

    df.at[mother_id, 'la_uterine_rupture'] = True
    mni[mother_id]['referred_for_surgery'] = True
    mni[mother_id]['referred_for_blood'] = True
    mni[mother_id]['referred_for_cs'] = True
    mni[mother_id]['cs_indication'] = 'ur'

    cs = module.item_codes_lab_consumables['caesarean_delivery_core']
    blood = module.item_codes_lab_consumables['blood_transfusion']

    for cons in cs, blood:
        for item in cons:
            sim.modules['HealthSystem'].override_availability_of_consumables(
                {item: 0.0})

    # refresh the consumables
    sim.modules['HealthSystem'].consumables._refresh_availability_of_consumables(date=sim.date)

    for cons in cs, blood:
        available = hsi_event.get_consumables(item_codes=cons)
        assert not available

    cemonc = labour.HSI_Labour_ReceivesComprehensiveEmergencyObstetricCare(
        module=sim.modules['Labour'], person_id=mother_id, timing='intrapartum')
    cemonc.facility_info = FacilityInfo(id=3,
                                        name='Facility_Level_2_Balaka',
                                        level='2',
                                        region='Southern')

    cemonc.apply(person_id=mother_id, squeeze_factor=0.0)

    assert mni[mother_id]['mode_of_delivery'] == 'caesarean_section'
    assert df.at[mother_id, 'la_uterine_rupture_treatment']
    assert mni[mother_id]['received_blood_transfusion']


def test_analysis_events_force_availability_of_consumables_for_pnc_analysis(seed):
    """Test that when analysis is being conducted during a simulation that consumable availability is determined
    via some pre-defined analysis parameter and not via the health system within the PNC HSIs"""
    sim = Simulation(start_date=start_date, seed=seed)
    sim.register(*fullmodel(resourcefilepath=resourcefilepath))
    sim.make_initial_population(n=100)

    # Set the analysis event to run at simulation start
    lparams = sim.modules['Labour'].parameters
    lparams['analysis_year'] = 2010
    lparams['alternative_pnc_coverage'] = True
    lparams['alternative_pnc_quality'] = True

    # Set availability
    lparams['pnc_availability_probability'] = 1.0

    sim.simulate(end_date=Date(2010, 1, 2))

    params = sim.modules['Labour'].current_parameters
    assert params['la_analysis_in_progress']

    df = sim.population.props
    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info

    women_repro = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
    mother_id = women_repro.index[0]

    # Set key pregnancy variables so SBA will run
    pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], mother_id)
    pregnancy_helper_functions.update_mni_dictionary(sim.modules['Labour'], mother_id)
    df.at[mother_id, 'la_is_postpartum'] = True
    df.at[mother_id, 'la_date_most_recent_delivery'] = sim.date
    mni[mother_id]['will_receive_pnc'] = 'early'
    mni[mother_id]['pnc_date'] = sim.date

    # set some complications
    df.at[mother_id, 'la_postpartum_haem'] = True
    mni[mother_id]['uterine_atony'] = True
    params['prob_haemostatis_uterotonics'] = 1.0

    module = sim.modules['Labour']
    pph = module.item_codes_lab_consumables['pph_core']

    for item in pph:
        sim.modules['HealthSystem'].override_availability_of_consumables({item: 0.0})

    # refresh the consumables
    sim.modules['HealthSystem'].consumables._refresh_availability_of_consumables(date=sim.date)

    hsi_event = get_dummy_hsi(sim, mother_id, id=3, fl=2)
    for cons in pph:
        available = hsi_event.get_consumables(item_codes=cons)
        assert not available

    # Next define the actual HSI of interest
    pnc = labour.HSI_Labour_ReceivesPostnatalCheck(
            module=sim.modules['Labour'], person_id=mother_id)
    pnc.facility_info = FacilityInfo(id=3,
                                     name='Facility_Level_2_Balaka',
                                     level='2',
                                     region='Southern')

    pnc.apply(person_id=mother_id, squeeze_factor=0.0)

    assert not df.at[mother_id, 'la_postpartum_haem']
    assert not mni[mother_id]['uterine_atony']


def test_analysis_events_force_availability_of_consumables_for_newborn_hsi(seed):
    """Test that when analysis is being conducted during a simulation that consumable availability is determined
    via some pre-defined analysis parameter and not via the health system within the newborn HSIs"""
    sim = Simulation(start_date=start_date, seed=seed)
    sim.register(*fullmodel(resourcefilepath=resourcefilepath))
    sim.make_initial_population(n=100)

    # Set the analysis event to run at simulation start
    lparams = sim.modules['Labour'].parameters
    lparams['analysis_date'] = Date(2010, 1, 1)
    lparams['alternative_bemonc_availability'] = True
    lparams['alternative_pnc_coverage'] = True
    lparams['alternative_pnc_quality'] = True

    # Set availability
    lparams['pnc_availability_probability'] = 1.0
    lparams['bemonc_availability'] = 1.0

    sim.simulate(end_date=Date(2010, 1, 2))

    df = sim.population.props
    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info

    mother_id = df.loc[df.is_alive & (df.sex == "F") & (df.age_years > 14) & (df.age_years < 50)].index[0]
    df.at[mother_id, 'date_of_last_pregnancy'] = sim.date
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 38
    df.at[mother_id, 'is_pregnant'] = True
    df.at[mother_id, 'co_contraception'] = "not_using"

    # Populate the minimum set of keys within the mni dict so the on_birth function will run
    pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], mother_id)
    pregnancy_helper_functions.update_mni_dictionary(sim.modules['Labour'], mother_id)

    # Set the variable that the mother has delivered at home
    mni[mother_id]['delivery_setting'] = 'hospital'
    child_id = sim.do_birth(mother_id)
    sim.modules['NewbornOutcomes'].on_birth(mother_id, child_id)

    # set comps
    df.at[child_id, 'nb_encephalopathy'] = 'severe_enceph'

    # refresh the consumables
    sim.modules['HealthSystem'].consumables._refresh_availability_of_consumables(date=sim.date)
    hsi_event = get_dummy_hsi(sim, mother_id, id=3, fl=2)

    # set postnatal comps
    df.at[child_id, 'pn_sepsis_early_neonatal'] = True

    sepsis_care = sim.modules['NewbornOutcomes'].item_codes_nb_consumables['sepsis_supportive_care_core']
    for item in sepsis_care:
        sim.modules['HealthSystem'].override_availability_of_consumables({item: 0.0})

    sim.modules['HealthSystem'].consumables._refresh_availability_of_consumables(date=sim.date)

    available = hsi_event.get_consumables(item_codes=sepsis_care)
    assert not available

    # Next define the actual HSI of interest
    nb_pnc = newborn_outcomes.HSI_NewbornOutcomes_ReceivesPostnatalCheck(
        module=sim.modules['NewbornOutcomes'], person_id=child_id)
    nb_pnc.facility_info = FacilityInfo(id=3,
                                        name='Facility_Level_2_Balaka',
                                        level='2',
                                        region='Southern')

    nb_pnc.apply(person_id=mother_id, squeeze_factor=0.0)

    assert df.at[child_id, 'nb_supp_care_neonatal_sepsis']


def test_analysis_events_circumnavigates_sf_and_competency_parameters(seed):
    """Test that the analysis event correctly overrides the parameters which controle whether the B/CEmONC signal
    functions can run"""
    sim = Simulation(start_date=start_date, seed=seed)
    sim.register(*fullmodel(resourcefilepath=resourcefilepath))
    sim.make_initial_population(n=100)

    # Set the analysis event to run at simulation start
    lparams = sim.modules['Labour'].parameters
    lparams['analysis_year'] = 2010
    lparams['alternative_bemonc_availability'] = True
    lparams['alternative_cemonc_availability'] = True

    # Set availability
    lparams['bemonc_availability'] = 1.0
    lparams['cemonc_availability'] = 1.0

    sim.simulate(end_date=Date(2010, 1, 2))

    params = sim.modules['Labour'].current_parameters
    assert params['la_analysis_in_progress']

    df = sim.population.props
    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info

    women_repro = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
    mother_id = women_repro.index[0]

    # Set key pregnancy variables so SBA will run
    df.at[mother_id, 'is_pregnant'] = True
    df.at[mother_id, 'la_currently_in_labour'] = True
    df.at[mother_id, 'date_of_last_pregnancy'] = start_date - pd.DateOffset(weeks=33)
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 35

    pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], mother_id)
    pregnancy_helper_functions.update_mni_dictionary(sim.modules['Labour'], mother_id)

    # Add some complications so that treatment should be delivered
    df.at[mother_id, 'ps_premature_rupture_of_membranes'] = True
    df.at[mother_id, 'ps_htn_disorders'] = 'severe_pre_eclamp'
    df.at[mother_id, 'la_obstructed_labour'] = True
    df.at[mother_id, 'la_sepsis'] = True

    mni[mother_id]['cpd'] = False
    mni[mother_id]['new_onset_spe'] = True
    mni[mother_id]['labour_state'] = 'late_preterm_labour'

    params['prob_progression_severe_pre_eclamp'] = 0.0

    # Override sf parameters which would block intervention delivery if analysis wasnt being conducted
    params['prob_hcw_avail_iv_abx'] = 0.0
    params['prob_hcw_avail_anticonvulsant'] = 0.0
    params['prob_hcw_avail_avd'] = 0.0
    params['mean_hcw_competence_hp'] = 0.0
    params['mean_hcw_competence_hc'] = 0.0

    # Next define the actual HSI of interest
    from tlo.methods.healthsystem import FacilityInfo

    # run the event and check the interventions were delivered as expected
    sba = labour.HSI_Labour_ReceivesSkilledBirthAttendanceDuringLabour(
        module=sim.modules['Labour'], person_id=mother_id, facility_level_of_this_hsi=2)
    sba.facility_info = FacilityInfo(id=3,
                                     name='Facility_Level_2_Balaka',
                                     level='2',
                                     region='Southern')

    sba.apply(person_id=mother_id, squeeze_factor=0.0)

    assert df.at[mother_id, 'la_severe_pre_eclampsia_treatment']
    assert mni[mother_id]['mode_of_delivery'] == 'instrumental'
    assert df.at[mother_id, 'la_sepsis_treatment']
