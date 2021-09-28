import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

# Here I am trying to create a curve that will be used to predict the number of injuries
# people receive from road traffic accidents. I am using curve fitting from scipy to fit to data
# from a number of sources
# create a number of potential injuries
number_of_potential_injuries = [1, 2, 3, 4, 5, 6, 7, 8]
nsample = 100000
# create data used to generate injuries from a 65.6-34.4 single multiple injury split as seen in Madubueze et al. 2010
percent_multiple = [34.4, 21.9, 24.85, 18.91, 20.8, 20.6]
percent_multiple_as_decimal = np.divide(percent_multiple, 100)
sources = ['Madubueze et al.', 'Sanyang et al.', 'Qi et al. 2006', 'Ganveer & Tiwani', 'Thani & Kehinde',
           'Akinpea et al.']


def exponentialdecay(x, a, k):
    y = a * np.exp(k * x)
    return y


probability_distributions = []
for percentage in percent_multiple_as_decimal:

    data_dict = {'Ninj': [1, 2, 9],
                 'dist': [(1 - percentage),  percentage / 2 + 0.04, 0]}
    data = pd.DataFrame(data_dict)

    xdata = data['Ninj']
    ydata = data['dist']
    popt, pcov = curve_fit(exponentialdecay, xdata, ydata, p0=[1, -1])
    exponential_prediction = []
    allnumb = range(1, 10, 1)
    for i in allnumb:
        exponential_prediction.append(exponentialdecay(i, *popt))
    # Normalize the 70-30 distribution
    exponential_prediction = exponential_prediction[:-1]
    exponential_prediction = list(np.divide(exponential_prediction, sum(exponential_prediction)))
    probability_distributions.append(exponential_prediction)
# sample from the fitted 70-30 distribution
average_injuries_per_dist = []
for distribution in probability_distributions:
    predicted_number_of_injuries = []
    for i in range(0, nsample):
        predicted_number_of_injuries.append(np.random.choice(number_of_potential_injuries, p=distribution))
    average_n_injuries = sum(predicted_number_of_injuries) / nsample
    average_injuries_per_dist.append(average_n_injuries)

# Find best prediction from each of the estimates
difference_to_malawi_data = [abs(result - 7057 / 4776) for result in average_injuries_per_dist]
best_fit = min(difference_to_malawi_data)
best_fit_index = difference_to_malawi_data.index(best_fit)
best_fit_source = sources[best_fit_index]
best_fit_distribution = probability_distributions[best_fit_index]
# plot the resulting average number of injuries per person
xpos = np.arange(len(probability_distributions))
plt.bar(xpos, average_injuries_per_dist, color='lightsteelblue')
plt.xticks(xpos, ['curve fit to \n' + source for source in sources], rotation=45)
plt.bar(xpos[best_fit_index], average_injuries_per_dist[best_fit_index], color='lightsalmon')
plt.ylabel('Average number of injuries per person')
plt.title('Average number of injuries predicted from curves fit to different sources')
plt.savefig('C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/'
            'curvefitting_n_inj_comparison.png', bbox_inches='tight')
for i in range(0, len(average_injuries_per_dist)):
    plt.subplot(1, 2, 1)
    plt.plot(number_of_potential_injuries, probability_distributions[i])
    plt.xlabel('Number of Injuries')
    plt.ylabel('Probability')
    plt.title(f'Exponential decay curve\n fit to {sources[i]}')
    plt.subplot(1, 2, 2)
    plt.bar(np.arange(2), [average_injuries_per_dist[i], 7057 / 4776], color=['lightsalmon', 'lightsteelblue'])
    plt.xticks(np.arange(2), ['Est. av. \n n. of injuries \n per person',
                              'Av. \n n. of injuries\n per person\n from \nSundet et al.'])
    plt.ylabel('Number of injuries')
    plt.title('Average number of injuries \nper person from curve \nfitting and Malawi data')
    plt.savefig(f'C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/'
                f'curvefitting_{sources[i]}.png', bbox_inches='tight')
    plt.clf()
    print(average_injuries_per_dist[i], 7057 / 4776, probability_distributions[i])

injury_vibes_distribution = [10829, 5241, 2886, 1936, 1194, 736, 368, 253]
injury_vibes_distribution = np.divide(injury_vibes_distribution, sum(injury_vibes_distribution))
predicted_number_of_injuries = []
for i in range(0, nsample):
    predicted_number_of_injuries.append(np.random.choice(number_of_potential_injuries, p=injury_vibes_distribution))
    average_n_injuries = sum(predicted_number_of_injuries) / nsample
plt.subplot(1, 2, 1)
plt.plot(number_of_potential_injuries, injury_vibes_distribution)
plt.xlabel('Number of Injuries')
plt.ylabel('Probability')
plt.title('Distribution of \ninjuries from \nthe Injury Vibes study')
plt.subplot(1, 2, 2)
plt.bar(np.arange(2), [average_n_injuries, 7057 / 4776])
plt.xticks(np.arange(2), ['Est. av. \n n. of injuries \n per person', 'Av. \n n. of injuries\n per person'])
plt.ylabel('Number of injuries')
plt.title('Average number of injuries \nper person from the injury VIBES study \n and Malawi data')
plt.savefig('C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/'
            'curvefitting_vibes.png', bbox_inches='tight')
plt.clf()
