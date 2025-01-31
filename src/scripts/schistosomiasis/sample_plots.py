import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Scenario labels
scenarios = ['No MDA', 'MDA (SAC)', 'MDA (PSAC+SAC)', 'MDA (All Age Groups)']

# Create some synthetic data for DALYs Incurred vs Averted
# Simulating DALYs for 4 scenarios, 3 categories (schistosomiasis, other waterborne, indirect health gains)
categories = ['Schistosomiasis', 'Other Waterborne Diseases', 'Indirect Health Gains']
dalys_data = pd.DataFrame({
    'Scenario': np.repeat(scenarios, len(categories)),
    'Category': categories * len(scenarios),
    'DALYs Averted': np.random.rand(len(scenarios) * len(categories)) * 1000,
    'DALYs Incurred': np.random.rand(len(scenarios) * len(categories)) * 1000
})

# Plot 1: DALYs Incurred vs Averted (Stacked Bar Chart)
plt.figure(figsize=(10, 6))
sns.barplot(x='Scenario', y='DALYs Averted', hue='Category', data=dalys_data, ci=None)
sns.barplot(x='Scenario', y='DALYs Incurred', hue='Category', data=dalys_data, ci=None, alpha=0.5)
plt.title('DALYs Incurred vs Averted for Different Scenarios')
plt.ylabel('DALYs (disability-adjusted life years)')
plt.xlabel('Intervention Scenario')
plt.legend(title='Category', loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 2: Create trends in Person-Years Infected (2010-2040)
years = np.arange(2010, 2041)
baseline_infected = np.random.rand(len(years)) * 50000  # Baseline infection trend
mda_only_infected = baseline_infected * 0.8  # 20% reduction with MDA only
mda_wash_infected = baseline_infected * 0.6  # 40% reduction with MDA + WASH

# Add uncertainty intervals (95% confidence)
uncertainty_upper = mda_wash_infected + np.random.rand(len(years)) * 5000
uncertainty_lower = mda_wash_infected - np.random.rand(len(years)) * 5000

# Plot 2: Trends in Person-Years Infected (Line Chart with Uncertainty)
plt.figure(figsize=(10, 6))
plt.plot(years, baseline_infected, label='No Intervention (Baseline)', color='gray', linestyle='--')
plt.plot(years, mda_only_infected, label='MDA Only', color='blue')
plt.plot(years, mda_wash_infected, label='MDA + WASH', color='green')

# Shaded region for uncertainty
plt.fill_between(years, uncertainty_lower, uncertainty_upper, color='green', alpha=0.2, label='95% CI for MDA+WASH')

plt.title('Trends in Person-Years Infected (2010-2040)')
plt.xlabel('Year')
plt.ylabel('Person-Years Infected with Schistosomiasis')
plt.legend(title='Scenario')
plt.tight_layout()
plt.show()

# Step 3: Simulate cost-effectiveness analysis data (Cost vs DALYs averted)
costs = np.random.rand(4) * 5000000  # Incremental cost (USD) for each scenario
dalys_averted = np.random.rand(4) * 100000  # DALYs averted for each scenario
strategies = ['No MDA', 'MDA (SAC)', 'MDA (PSAC+SAC)', 'MDA (All Age Groups)']

# Plot 3: Cost-Effectiveness Plane (Scatter Plot)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=dalys_averted, y=costs, hue=strategies, palette='Set1', s=100)
plt.axvline(x=50000, color='r', linestyle='--', label='WTP Threshold')
plt.title('Cost-Effectiveness Plane for MDA + WASH Scenarios')
plt.xlabel('DALYs Averted')
plt.ylabel('Incremental Cost (USD)')
plt.legend(title='Strategy')
plt.tight_layout()
plt.show()

# Step 4: Create Cost-Effectiveness Acceptability Curve (CEAC)
wtp_thresholds = np.linspace(0, 100000, 50)  # WTP threshold range (USD per DALY averted)
probabilities = np.exp(-wtp_thresholds / 5000)  # Simulating probability of cost-effectiveness

# Plot 4: Cost-Effectiveness Acceptability Curve
plt.figure(figsize=(8, 6))
plt.plot(wtp_thresholds, probabilities, label='MDA + WASH', color='green')
plt.plot(wtp_thresholds, np.exp(-wtp_thresholds / 10000), label='No MDA', color='gray', linestyle='--')
plt.title('Cost-Effectiveness Acceptability Curve (CEAC)')
plt.xlabel('WTP Threshold (USD per DALY Averted)')
plt.ylabel('Probability of MDA being Cost-Effective')
plt.legend(title='Strategy')
plt.tight_layout()
plt.show()

