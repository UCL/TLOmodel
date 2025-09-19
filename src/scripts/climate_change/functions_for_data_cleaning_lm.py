import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm


def build_model(X, y, poisson=False, log_y=False, feature_selection=False, k_best=None):
    epsilon = 1

    # Log-transform y with clipping for positivity
    if log_y:
        y = np.log(np.clip(y, epsilon, None))

        # Apply mask to filter valid data
    mask = (~np.isnan(X).any(axis=1) & ~np.isnan(y) & (y <= 1e4))
    X_filtered, y_filtered = X[mask], y[mask]

    # Feature selection step (optional)
    if feature_selection:
        if poisson:
            raise ValueError("Feature selection using f_regression is only compatible with OLS regression.")
        selector = SelectKBest(score_func=f_regression, k=k_best or 'all')
        X_filtered = selector.fit_transform(X_filtered, y_filtered)
        selected_features = selector.get_support()
    else:
        selected_features = np.ones(X.shape[1], dtype=bool)  # Keep all features if no selection

    # Build the model
    model = GLM(y_filtered, X_filtered, family=NegativeBinomial(), method='nm') if poisson else sm.OLS(y_filtered,
                                                                                                       X_filtered)
    model_fit = model.fit()

    return model_fit, model_fit.predict(X_filtered), mask, selected_features



def create_binary_feature(threshold, weather_data_df, recent_months):
    binary_feature_list = []
    for facility in weather_data_df.columns:
        facility_data = weather_data_df[facility]
        for i in range(len(facility_data)):
            facility_threshold = threshold[i] if hasattr(threshold, "__len__") else threshold

            if i >= recent_months: # only count for recent months, and have to discount the data kept in for this purpose. Also, first 12 months have no data to check back to
                last_x_values = facility_data[i - recent_months:i]
                binary_feature_list.append(1 if (last_x_values > facility_threshold).any() else 0)

    return binary_feature_list

def stepwise_selection(X, y, log_y, poisson, p_value_threshold=0.05):
    included = []
    current_aic = np.inf

    while True:
        changed = False

        # Step 1: Try adding each excluded predictor and select the best one by AIC if significant
        excluded = list(set(range(X.shape[1])) - set(included))
        new_aic = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            subset_X = X[:, included + [new_column]]
            results, y_pred, mask_ANC_data, _ = build_model(subset_X, y, poisson, log_y=log_y)
            new_aic[new_column] = results.aic

        # Add the predictor with the best AIC if it's better than the current model's AIC
        if not new_aic.empty and new_aic.min() < current_aic:
            best_feature = new_aic.idxmin()
            included.append(best_feature)
            current_aic = new_aic.min()
            changed = True
        print(current_aic)


        # Exit if no changes were made in this iteration
        if not changed:
            break
    included.sort()
    results, y_pred, mask_ANC_data, _ = build_model(X[:, included], y, poisson, log_y=log_y)

    return included, results, y_pred, mask_ANC_data

def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

def repeat_info(info, num_facilities, year_range):
    # Repeat facilities in alternating order for each month and year
    repeated_info = [info[i % len(info)] for i in range(len(year_range) * 12 * num_facilities)]
    return repeated_info


def repeat_info(info, num_facilities, year_range, historical):
    # Repeat facilities in alternating order for each month and year
    repeated_info = [info[i % len(info)] for i in range(len(year_range) * 12 * num_facilities)]

    if historical:
        return repeated_info[:-4 * num_facilities]  # Exclude final 4 months for all facilities
    else:
        return repeated_info
#
