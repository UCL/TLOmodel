'''
This will print out the draw number for which searched parameters set was used
'''
import itertools
# update the parameters set for the job
parameters = {
    "base_death_rate_untreated_sam": [0.1, 0.05, 0.03, 0.01],
    "mod_wast_incidence__coef": [1.0, 0.6, 0.2],
    "progression_to_sev_wast__coef": [0.5, 0.75, 1.0, 1.5, 2.0, 2.3],
    "prob_death_after_SAMcare__as_prop_of_death_rate_untreated_sam": [0.1, 0.4, 0.7]
}
# what parameters set are you looking for?
pars_set_searched = [0.03, 0.6, 1.0, 0.1]

##################################################
def find_pars_draw_nmb(in_parameters, in_pars_set_searched):
    base_death_rate_untreated_sam__draws = parameters["base_death_rate_untreated_sam"]
    mod_wast_incidence__coef = parameters["mod_wast_incidence__coef"]
    progression_to_sev_wast__coef = parameters["progression_to_sev_wast__coef"]
    prob_death_after_SAMcare__as_prop_of_death_rate_untreated_sam = parameters["prob_death_after_SAMcare__as_prop_of_death_rate_untreated_sam"]

    pars_combinations = list(itertools.product(
        base_death_rate_untreated_sam__draws,
        mod_wast_incidence__coef,
        progression_to_sev_wast__coef,
        prob_death_after_SAMcare__as_prop_of_death_rate_untreated_sam
    ))
    if tuple(in_pars_set_searched) in pars_combinations:
        print(f"Position of searched parameters: {pars_combinations.index(tuple(in_pars_set_searched))}")
    else:
        print(f"Searched parameters {in_pars_set_searched} not found in combinations: {pars_combinations}")

find_pars_draw_nmb(parameters, pars_set_searched)
