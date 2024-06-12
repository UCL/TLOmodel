# Comparison of time spent running tests

Durations <0.01s not listed. Breakdown is provided where applicable.

| Name of test | Module | Pre shared sim | Post shared sim | Worst case max |
|-|-|-|-|-|
| `stunted_and_correctly_diagnosed` | Stunting | 1.34 | 1.60 (setup) | 1.62 |
| `stunted_but_no_checking` | Stunting | 1.29 | - | 0.03 |
| `not_stunted` | Stunting | 1.33 | - | 0.03 |
| `diarrhoea_severe_dehydration` | Diarrhoea | 1.93 | 2.18 (setup) + 0.01 (teardown) | 2.20 |
| `diarrhoea_severe_dehydration_dxtest_notfunctional` | Diarrhoea | 2.08 | 0.01 (teardown) | 0.03 |
| `diarrhoea_non_severe_dehydration` | Diarrhoea | 1.90 | 0.01 (teardown) | 0.03 |
| `test_run_each_of_the_HSI` | Diarrhoea | 2.00 | 0.01 (call) + 0.01 (teardown) | 0.03 |
| `test_effect_of_vaccine` | Diarrhoea | 2.86 | - | 0.03 |
| `perfect_treatment_leads_to_zero_risk_of_death` | Diarrhoea | 2.82 | 0.06 (call) | 0.08 |
| `treatment_for_those_that_will_not_die` | Diarrhoea | 3.02 | 0.02 (call) + 0.01 (teardown) | 0.04 |

1. `test_routine_assessment_for_chronic_undernutrition_if_stunted_and_correctly_diagnosed`
1. `test_routine_assessment_for_chronic_undernutrition_if_stunted_but_no_checking`
1. `test_routine_assessment_for_chronic_undernutrition_if_not_stunted`
1. `test_do_when_presentation_with_diarrhoea_severe_dehydration`
1. `test_do_when_presentation_with_diarrhoea_severe_dehydration_dxtest_notfunctional`
1. `test_do_when_presentation_with_diarrhoea_non_severe_dehydration`
1. `test_run_each_of_the_HSI`
1. `test_effect_of_vaccine`
1. `test_check_perfect_treatment_leads_to_zero_risk_of_death`
1. `test_do_treatment_for_those_that_will_not_die`
