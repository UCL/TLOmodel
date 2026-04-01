import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load files
df_standard = pd.read_csv("duration_time_standardRTI.csv", sep=r",")
df_emulated = pd.read_csv("duration_time_emulatedRTI.csv", sep=r",")
df_noRTI = pd.read_csv("duration_time_noRTIPoll.csv", sep=r",")

# Remove first column
df_standard = df_standard.iloc[:, 1:]
df_emulated = df_emulated.iloc[:, 1:]
df_noRTI = df_noRTI.iloc[:, 1:]

# Extract the new first column
standard = df_standard.iloc[:, 0]
emulated = df_emulated.iloc[:, 0]
noRTI = df_noRTI.iloc[:, 0]
print(df_standard)

# Compute boost taking mean of 'noRTI' scenario, to ensure the only source of variance is RTI modelling approach
mean_noRTI = noRTI.mean()
boost_mean_noRTI = 1.0 - ((emulated - mean_noRTI)/(standard - mean_noRTI))

# Inspect boost
print(boost_mean_noRTI)

# Get mean, median and std
print('mean boost no RTI mean', boost_mean_noRTI.mean())
print('median boost no RTI mean', boost_mean_noRTI.median())
print('std boost no RTI mean', boost_mean_noRTI.std())

column_name = "boost_mean_noRTI"
df_boost_mean_noRTI = boost_mean_noRTI.to_frame(name=column_name)
n_bootstrap = 10000 #10000  # number of bootstrap samples
confidence_level = 0.95

# Bootstrap sampling
bootstrap_means = []
for _ in range(n_bootstrap):
    sample = df_boost_mean_noRTI[column_name].sample(n=len(df_boost_mean_noRTI), replace=True)
    bootstrap_means.append(sample.mean())

assert len(bootstrap_means) == n_bootstrap

# Compute confidence interval
alpha = 1 - confidence_level
lower = np.percentile(bootstrap_means, 100.0 * (alpha / 2.0))
upper = np.percentile(bootstrap_means, 100.0 * (1.0 - alpha / 2.0))

# Results
print(f"{int(confidence_level*100)}% bootstrap CI for the mean: ({lower:.3f}, {upper:.3f})")
print(f"Bootstrap mean estimate: {np.mean(bootstrap_means):.3f}")
