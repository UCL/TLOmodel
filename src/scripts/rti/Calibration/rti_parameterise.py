from pathlib import Path

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    symptommanager,
    rti,
)
import numpy as np
from matplotlib import pyplot as plt

# The Resource files [NB. Working directory must be set to the root of TLO: TLOmodel]
resourcefilepath = Path('./resources')
# Establish the simulation object
yearsrun = 5
start_date = Date(year=2010, month=1, day=1)
end_date = Date(year=(2010 + yearsrun), month=1, day=1)
pop_size = 10000
iterations = 10000
output_for_different_incidence = dict()
service_availability = []
sim_age_range = []
females = 0
males = 0
percent_males_in_rti = 0.590655439
base_1m_prob_rti = 0.5
rr_injrti_male = 1
rr_injrti_age018 = 1
rr_injrti_age1829 = 1
rr_injrti_age3039 = 1
rr_injrti_age4049 = 1
rr_injrti_age5059 = 1
rr_injrti_age6069 = 1
rr_injrti_age7079 = 1
rr_injrti_excessalcohol = 1
sim = Simulation(start_date=start_date)

sim.register(demography.Demography(resourcefilepath=resourcefilepath),
             enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
             )
# create and run the simulation
sim.make_initial_population(n=pop_size)
param_dict = dict()
for i in range(0, iterations):
    eq = LinearModel(LinearModelType.MULTIPLICATIVE,
                     base_1m_prob_rti,
                     Predictor('sex').when('M', rr_injrti_male),
                     Predictor('age_years').when('.between(0,18)', rr_injrti_age018),
                     Predictor('age_years').when('.between(18,29)', rr_injrti_age1829),
                     Predictor('age_years').when('.between(30,39)', rr_injrti_age3039),
                     Predictor('age_years').when('.between(40,49)', rr_injrti_age4049),
                     Predictor('age_years').when('.between(50,59)', rr_injrti_age5059),
                     Predictor('age_years').when('.between(60,69)', rr_injrti_age6069),
                     Predictor('age_years').when('.between(70,79)', rr_injrti_age7079),
                     Predictor('li_ex_alc').when(True, rr_injrti_excessalcohol)
                     )
    pred = eq.predict(sim.population.props)
    random_draw_in_rti = sim.rng.random_sample(size=len(sim.population.props))
    selected_for_rti = sim.population.props[pred > random_draw_in_rti]
    males = len(selected_for_rti.loc[selected_for_rti['sex'] == 'M'])
    percent_males_pred = males / len(selected_for_rti)
    if percent_males_pred < percent_males_in_rti:
        rr_injrti_male = rr_injrti_male + 0.001
    else:
        param_dict.update({'rr_injrti_male': rr_injrti_male,
                           'pred_gender_ratio': percent_males_pred,
                           'difference_in_gender_ratio': percent_males_pred-percent_males_in_rti})
        break


for i in range(0, iterations):
    age_range = []
    eq = LinearModel(LinearModelType.MULTIPLICATIVE,
                     base_1m_prob_rti,
                     Predictor('sex').when('M', rr_injrti_male),
                     Predictor('age_years').when('.between(0,18)', rr_injrti_age018),
                     Predictor('age_years').when('.between(18,29)', rr_injrti_age1829),
                     Predictor('age_years').when('.between(30,39)', rr_injrti_age3039),
                     Predictor('age_years').when('.between(40,49)', rr_injrti_age4049),
                     Predictor('age_years').when('.between(50,59)', rr_injrti_age5059),
                     Predictor('age_years').when('.between(60,69)', rr_injrti_age6069),
                     Predictor('age_years').when('.between(70,79)', rr_injrti_age7079),
                     Predictor('li_ex_alc').when(True, rr_injrti_excessalcohol)
                     )
    pred = eq.predict(sim.population.props)
    random_draw_in_rti = sim.rng.random_sample(size=len(sim.population.props))
    selected_for_rti = sim.population.props[pred > random_draw_in_rti]
    ages = selected_for_rti.age_years.tolist()
    zero_to_five = len([i for i in ages if i < 6])
    six_to_ten = len([i for i in ages if 6 <= i < 11])
    eleven_to_fifteen = len([i for i in ages if 11 <= i < 16])
    sixteen_to_twenty = len([i for i in ages if 16 <= i < 21])
    twenty1_to_twenty5 = len([i for i in ages if 21 <= i < 26])
    twenty6_to_thirty = len([i for i in ages if 26 <= i < 31])
    thirty1_to_thirty5 = len([i for i in ages if 31 <= i < 36])
    thirty6_to_forty = len([i for i in ages if 36 <= i < 41])
    forty1_to_forty5 = len([i for i in ages if 41 <= i < 46])
    forty6_to_fifty = len([i for i in ages if 46 <= i < 51])
    fifty1_to_fifty5 = len([i for i in ages if 51 <= i < 56])
    fifty6_to_sixty = len([i for i in ages if 56 <= i < 61])
    sixty1_to_sixty5 = len([i for i in ages if 61 <= i < 66])
    sixty6_to_seventy = len([i for i in ages if 66 <= i < 71])
    seventy1_to_seventy5 = len([i for i in ages if 71 <= i < 76])
    seventy6_to_eighty = len([i for i in ages if 76 <= i < 81])
    eighty1_to_eighty5 = len([i for i in ages if 81 <= i < 86])
    eighty6_to_ninety = len([i for i in ages if 86 <= i < 91])
    ninety_plus = len([i for i in ages if 90 < i])
    age_list = [zero_to_five, six_to_ten, eleven_to_fifteen, sixteen_to_twenty, twenty1_to_twenty5,
                twenty6_to_thirty, thirty1_to_thirty5, thirty6_to_forty, forty1_to_forty5, forty6_to_fifty,
                fifty1_to_fifty5, fifty6_to_sixty, sixty1_to_sixty5, sixty6_to_seventy, seventy1_to_seventy5,
                seventy6_to_eighty, eighty1_to_eighty5, eighty6_to_ninety, ninety_plus]
    prob_age_range = np.divide(age_list, sum(age_list))



print(param_dict)
