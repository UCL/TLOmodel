"""Produce plots to show the health impact (deaths, dalys) each the healthcare system (overall health impact) when
running under different MODES and POLICIES (scenario_impact_of_capabilities_expansion_scaling.py)"""

# short tclose -> ideal case
# long tclose -> status quo
import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
from tlo import Date
from tlo.analysis.utils import extract_results, make_calendar_period_type, make_calendar_period_lookup, make_age_grp_lookup, make_age_grp_types
import matplotlib.pyplot as plt

# Archive of outputs
# Batch based on /Users/mm2908/Desktop/EmuIBM/Save_With_WellPerforming/emulators/latest_CTGANSynthesizer_epochs500_dsF_batch_size500_num_k_folds10_Nsubsample10000_InAndOutC_test_k_folding_UniformEncoder_CTGANtest3_repeat_seed42_k_fold0.pkl
# 'standard_RTI': 'outputs/test_rti_emulator-2025-08-12T205454Z'
# 'emulated_RTI': 'outputs/test_rti_emulator-2025-08-13T080302Z'
# 'standard_RTI_x250': 'outputs/test_rti_emulator-2025-09-06T055808Z'
# 'emulated_RTI_x250': 'outputs/test_rti_emulator-2025-09-05T085159Z'
# 'no_RTI': 'outputs/test_rti_emulator-2025-08-13T143342Z'

outputs = {
            'standard_RTI': {'results_folder' : Path('outputs/test_rti_emulator-2025-08-12T205454Z'), 'data': {}},
            'emulated_RTI': {'results_folder' : Path('outputs/test_rti_emulator-2025-08-13T080302Z'), 'data' : {}},
            'standard_RTI_x250': {'results_folder' : Path('outputs/test_rti_emulator-2025-09-06T055808Z'), 'data': {}},
            'emulated_RTI_x250': {'results_folder' : Path('outputs/test_rti_emulator-2025-09-05T085159Z'), 'data' : {}},
            'no_RTI': {'results_folder' : Path('outputs/test_rti_emulator-2025-08-13T143342Z'), 'data' : {}},
        }


def apply(results_folder: Path) -> Tuple:

    tag = results_folder.name

    def get_parameter_names_from_scenario_file() -> Tuple[str]:
        """Get the tuple of names of the scenarios from `Scenario` class used to create the results."""
        from scripts.healthsystem.impact_of_const_capabilities_expansion.scenario_impact_of_capabilities_expansion_scaling import (
            ImpactOfHealthSystemMode,
        )
        e = ImpactOfHealthSystemMode()
        return tuple(e._scenarios.keys())
    
    def set_param_names_as_column_index_level_0(_df):
        """Set the columns index (level 0) as the param_names."""
        ordered_param_names_no_prefix = {i: x for i, x in enumerate(param_names)}
        names_of_cols_level0 = [ordered_param_names_no_prefix.get(col) for col in _df.columns.levels[0]]
        assert len(names_of_cols_level0) == len(_df.columns.levels[0])
        _df.columns = _df.columns.set_levels(names_of_cols_level0, level=0)
        return _df
        
    def get_counts_of_death_by_year(df):
        df["year"] = df["date"].dt.year
        result = df.groupby(by=["year", "label"])["person_id"].count()
        return result
        
    def get_num_dalys_by_year(_df):
        drop_cols = ["date", "sex", "age_range"]
        result = _df.drop(columns=drop_cols).groupby("year", as_index=True).sum().stack()
        result.index.set_names(["year", "label"], inplace=True)
        return result
        
    def get_dalys_by_period_sex_agegrp_label(df):
        """Sum the dalys by period, sex, age-group and label"""
        _, calperiodlookup = make_calendar_period_lookup()

        df['age_grp'] = df['age_range'].astype(make_age_grp_types())
        df["period"] = df["year"].map(calperiodlookup).astype(make_calendar_period_type())
        df = df.drop(columns=['date', 'age_range', 'year'])
        df = df.groupby(by=["period", "sex", "age_grp"]).sum().stack()
        df.index = df.index.set_names('label', level=3)
        return df

    def rename_index_levels(df):
        rename_dict = {}
        for level in df.index.names:
            if level == 'date':
                rename_dict['date'] = 'year'
            elif level == 'label':
                rename_dict['label'] = 'cause'

        if rename_dict:
            df = df.rename_axis(index=rename_dict)
            
        if "year" in df.index.names:
            years = df.index.get_level_values("year")
            mask = (years >= 2010) & (years < 2020)
            df = df[mask]
        return df
        
    param_names = get_parameter_names_from_scenario_file()

    num_deaths_by_year_and_cause = rename_index_levels(extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=get_counts_of_death_by_year,
        do_scaling=True
    ))
    
    num_dalys_by_year_and_cause = rename_index_levels(extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked",
        custom_generate_series=get_num_dalys_by_year,
        do_scaling=True
    ))
    
    num_individuals = extract_results(results_folder,
                            module="tlo.methods.demography",
                            key="population",
                            column="total",
                            index="date",
                            do_scaling=True
                            )
    num_individuals.index = num_individuals.index.year
    num_individuals = rename_index_levels(num_individuals)
  
    cfr = num_deaths_by_year_and_cause.div(num_individuals, level="year")
    
    dalys_by_sex_and_age_time = extract_results(
                                results_folder,
                                module="tlo.methods.healthburden",
                                key="dalys_stacked_by_age_and_time",  # <-- for DALYS stacked by age and time
                                custom_generate_series=get_dalys_by_period_sex_agegrp_label,
                                do_scaling=True
                                )
    # divide by five to give the average number of deaths per year within the five year period:
    dalys_by_sex_and_age_time = dalys_by_sex_and_age_time.div(5.0)
    dalys_by_sex_and_age = (dalys_by_sex_and_age_time.loc['2010-2014'] + dalys_by_sex_and_age_time.loc['2015-2019'])/2

    # Collect all data for this output
    data = {'deaths' : num_deaths_by_year_and_cause, 'dalys' : num_dalys_by_year_and_cause, 'pop' : num_individuals, 'cfr' : cfr, 'dalys_by_sex_and_age' : dalys_by_sex_and_age}

    return data

        
# Extract and calculate summary statistics
def compute_summary_stats(df):
    df_mean = df.mean(axis=1)
    df[('0','mean')] = df_mean
    df_lower = df.quantile((1-0.682)/2.0,axis=1)
    df_upper = df.quantile((1-0.682)/2.0+0.682,axis=1)
    df[('0','mean')] = df_mean
    df[('0','lower')] = df_lower
    df[('0','upper')] = df_upper
    return df


def compare_and_make_plots_age_sex_distr(outputs, first_scenario, second_scenario, third_scenario, fourth_scenario, target, compare_to_other_x250=True):
    # Load from outputs
    df_map = {
        "normal":   outputs[first_scenario]['data'][target],
        "emulated": outputs[second_scenario]['data'][target],
        "normal_x250":   outputs[third_scenario]['data'][target],
        "emulated_x250": outputs[fourth_scenario]['data'][target],
    }
    labels = [first_scenario, second_scenario, third_scenario, fourth_scenario]

    # Focus cause
    cause = 'Transport Injuries'
    for k, df in df_map.items():
        df_map[k] = compute_summary_stats(df.loc[(slice(None), slice(None), cause)])

    # Age group mapping
    age_grps = df_map["normal"].index.get_level_values('age_grp').unique()
    age_grp_mapping = {age: i for i, age in enumerate(age_grps)}

    # Init p-values
    pvalues = pd.DataFrame(columns=['p-value ks', 'p-value t', 'p-value w'],
                           index=pd.MultiIndex.from_tuples([], names=['case', 'sex', 'age_grp']))
    """
    # ---- Helper: compute p-values
    def run_tests(case, df1, df2):
        for sex in ['F', 'M']:
            for age in age_grps:
                data1, data2 = df1.loc[(sex, age)].values, df2.loc[(sex, age)].values
                p_ks = ks_2samp(data1, data2).pvalue
                p_t  = stats.ttest_ind(data1, data2).pvalue
                p_w  = stats.wilcoxon(data1, data2).pvalue
                pvalues.loc[(case, sex, age), :] = [p_ks, p_t, p_w]

    run_tests("other_x1",   df_map["normal"],   df_map["emulated"])
    if compare_to_other_x250:
        run_tests("other_x250", df_map["normal_x250"], df_map["emulated_x250"])
        """

    # ---- Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    for ax in axes: ax.set_ylim(-10, 120000)

    for i, sex in enumerate(['F', 'M']):
        x_pos = [age_grp_mapping[a] for a in age_grps]

        def plot_band_and_mean(df, color, marker, label, alpha_fill, alpha_line, ls):
            mean = df.loc[(sex, slice(None)), ('0', 'mean')].values
            low  = df.loc[(sex, slice(None)), ('0', 'lower')].values
            up   = df.loc[(sex, slice(None)), ('0', 'upper')].values
            axes[i].fill_between(x_pos, low, up, color=color, alpha=alpha_fill)
            axes[i].plot(x_pos, mean, marker=marker, color=color, alpha=alpha_line, linestyle=ls, label=label)
            return mean

        # Plot main (normal vs emulated)
        ls, fill, line = ('--', 0.1, 0.4) if compare_to_other_x250 else ('-', 0.3, 1.0)
        m_norm = plot_band_and_mean(df_map["normal"], "blue", "o", labels[0] if i==0 else None, fill, line, ls)
        m_emul = plot_band_and_mean(df_map["emulated"], "orange", "s", labels[1] if i==0 else None, fill, line, ls)

        # Optional comparison to x250
        if compare_to_other_x250:
            m_norm_x250 = plot_band_and_mean(df_map["normal_x250"], "blue", "o", labels[2] if i==0 else None, 0.3, 1.0, '-')
            m_emul_x250 = plot_band_and_mean(df_map["emulated_x250"], "orange", "s", labels[3] if i==0 else None, 0.3, 1.0, '-')

        """
        # Annotate KS values
        y_base = 42000 if sex=="M" else 30000
        for case, offset in [("other_x1", 7000), ("other_x250", 0)] if compare_to_other_x250 else [("other_x1", 7000)]:
            for age in age_grps:
                axes[i].text(age_grp_mapping[age], y_base+offset,
                             f"{pvalues.loc[(case, sex, age), 'p-value ks']:.2f}",
                             fontsize=8, ha="center", rotation=90)
            axes[i].text(age_grp_mapping['0-4'], y_base+offset+3000,
                         "KS test p-value" + (" (incr. other mortality)" if case=="other_x250" else ""),
                         fontsize=8, ha="left")
        """
        # Formatting
        axes[i].set_title(sex)
        axes[i].set_xlabel("Age Group")
        axes[i].set_ylabel("Averaged Yearly DALYs")
        axes[i].set_xticks(x_pos)
        axes[i].set_xticklabels(age_grps, rotation=45)
        axes[i].legend()

    plt.tight_layout()
    fname = "plots/final_DALYs_Breakdown_by_age_grp_DC.png" if compare_to_other_x250 else "plots/final_DALYs_Breakdown_by_age_grp.png"
    plt.savefig(fname)
    plt.close()

def compare_old(outputs, first_scenario, second_scenario, third_scenario, fourth_scenario, target):

    # Load output files, which include standard other mortality and other mortality rates increased by x250
    df_af_normal = pd.read_csv('DALYs_by_sex_age_' + first_data_set + '.csv', header=[0, 1], index_col=[0,1,2])
    df_af_emulated = pd.read_csv('DALYs_by_sex_age_' + second_data_set + '.csv', header=[0, 1], index_col=[0,1,2])
    df_af_normal_other_x250 = pd.read_csv('DALYs_by_sex_age_' + first_data_set + '_x250_other.csv', header=[0, 1], index_col=[0,1,2])
    df_af_emulated_other_x250 = pd.read_csv('DALYs_by_sex_age_'+ second_data_set + '_x250_other.csv', header=[0, 1], index_col=[0,1,2])

    # This is the cause that will be considered in the plots. Could also make loop over this
    cause = 'Transport Injuries'
    df_normal = df_af_normal.loc[(slice(None), slice(None), cause)]
    df_normal_other_x250 = df_af_normal_other_x250.loc[(slice(None), slice(None), cause)]
    df_emulated = df_af_emulated.loc[(slice(None), slice(None), cause)]
    df_emulated_other_x250 = df_af_emulated_other_x250.loc[(slice(None), slice(None), cause)]

    # Map 'age_grp' to numeric values for easier plotting
    age_grp_mapping = {age: i for i, age in enumerate(df_normal.index.get_level_values('age_grp').unique())}

    df_normal = compute_summary_stats(df_normal)
    df_emulated = compute_summary_stats(df_emulated)
    df_normal_other_x250 = compute_summary_stats(df_normal_other_x250)
    df_emulated_other_x250 = compute_summary_stats(df_emulated_other_x250)
    
    # Store p-values for all x-values
    index = pd.MultiIndex.from_tuples([], names=['case', 'sex', 'age_grp'])
    columns = ['p-value ks', 'p-value t', 'p-value w']
    pvalues = pd.DataFrame(index=index, columns=columns)

    for sex in ['F','M']:
        for age_group in df_normal_other_x250.index.get_level_values('age_grp').unique():
        
            # Compute statistical tests for x1 other mortality
            data_normal = df_normal_other_x250.loc[(sex,age_group)].values
            data_emulat = df_emulated_other_x250.loc[(sex,age_group)].values
            
            statistic, p_value_ks = ks_2samp(data_normal, data_emulat)
            t_stat, p_value_t = stats.ttest_ind(data_normal, data_emulat)
            res_w = stats.wilcoxon(data_normal,data_emulat)
            p_value_w = res_w.pvalue
            pvalues.loc[('other_x250',sex,age_group),:] = [p_value_ks, p_value_t, p_value_w]
            
            # Compute statistical tests for x250 other mortality
            data_normal = df_normal.loc[(sex,age_group)].values
            data_emulat = df_emulated.loc[(sex,age_group)].values
            
            statistic, p_value_ks = ks_2samp(data_normal, data_emulat)
            t_stat, p_value_t = stats.ttest_ind(data_normal, data_emulat)
            res_w = stats.wilcoxon(data_normal,data_emulat)
            p_value_w = res_w.pvalue
            pvalues.loc[('other_x1',sex,age_group),:] = [p_value_ks, p_value_t, p_value_w]
            

    # Create a figure for the subplots (one for each sex)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    for ax in axes:
        ax.set_ylim(-1000, 55000)
    
    # Plot for each sex
    for sex in ['F','M']:
        if sex == 'F':
            i = 0
        else:
            i = 1

        # Extract the age groups from the dataframe
        age_groups = df_normal.index.get_level_values('age_grp').unique()

        # Create a numeric x_pos based on age groups for easier plotting
        x_pos = [age_grp_mapping[age] for age in age_groups]

        # Prepare the values for plotting
        normal_means = df_normal.loc[(sex, slice(None)),('0','mean')].values
        emulated_means = df_emulated.loc[(sex, slice(None)),('0','mean')].values
        normal_lower = df_normal.loc[(sex, slice(None)),('0','lower')].values
        normal_upper = df_normal.loc[(sex, slice(None)),('0','upper')].values
        emulated_lower = df_emulated.loc[(sex, slice(None)),('0','lower')].values
        emulated_upper = df_emulated.loc[(sex, slice(None)),('0','upper')].values
        
        if compare_to_other_x250:
            normal_means_other_x250 = df_normal_other_x250.loc[(sex, slice(None)),('0','mean')].values
            emulated_means_other_x250 = df_emulated_other_x250.loc[(sex, slice(None)),('0','mean')].values
            normal_lower_other_x250 = df_normal_other_x250.loc[(sex, slice(None)),('0','lower')].values
            normal_upper_other_x250 = df_normal_other_x250.loc[(sex, slice(None)),('0','upper')].values
            emulated_lower_other_x250 = df_emulated_other_x250.loc[(sex, slice(None)),('0','lower')].values
            emulated_upper_other_x250 = df_emulated_other_x250.loc[(sex, slice(None)),('0','upper')].values
            fill_main_comparison = 0.1
            line_main_compariosn = 0.4
            main_linestyle = '--'
            print(main_linestyle)
        else:
            fill_main_comparison = 0.3
            line_main_compariosn = 1.0
            main_linestyle = '-'
            
        # Calculate % error: ((emulated - original) / original) * 100
        percent_error = 100 * (emulated_means - normal_means) / normal_means
        if compare_to_other_x250:
            percent_error_other_x250 = 100 * (emulated_means_other_x250 - normal_means_other_x250) / normal_means_other_x250

        # Plot the shaded area for normal data (F or M)
        axes[i].fill_between(x_pos, normal_lower, normal_upper,
                             color='blue', alpha=fill_main_comparison)
        if compare_to_other_x250:
            axes[i].fill_between(x_pos, normal_lower_other_x250, normal_upper_other_x250,
                                 color='blue', alpha=0.3)
        
        # Plot the shaded area for emulated data (F or M)
        axes[i].fill_between(x_pos, emulated_lower, emulated_upper,
                             color='orange', alpha=fill_main_comparison)
        if compare_to_other_x250:
            axes[i].fill_between(x_pos, emulated_lower_other_x250, emulated_upper_other_x250,
                                 color='orange', alpha=0.3)

        # Plot the mean values for normal and emulated data
        axes[i].plot(x_pos, normal_means, 'o-', color='blue', alpha=line_main_compariosn, linestyle=main_linestyle, label=labels[first_data_set] if i == 0 else "")  # Normal mean
        axes[i].plot(x_pos, emulated_means, 's-', color='orange', alpha=line_main_compariosn, linestyle=main_linestyle, label=labels[second_data_set] if i == 0 else "")  # Emulated mean
        if compare_to_other_x250:
            axes[i].plot(x_pos, normal_means_other_x250, 'o-', color='blue',  label=f'Original (incr. other mortality)' if i == 0 else "")  # Normal mean
            axes[i].plot(x_pos, emulated_means_other_x250, 's-', color='orange',  label=f'Emulated (incr. other mortality)' if i == 0 else "")  # Emulated mean

        # Include KS test p-values in the plot
        if sex == 'M':
            ks_yloc_other_x250 = 42000
        else:
            ks_yloc_other_x250 = 30000
        ks_yloc_other_x1 = ks_yloc_other_x250 + 7000

        for age_grp in df_normal.index.get_level_values('age_grp').unique():
            ks_value = pvalues.loc[('other_x1',sex, age_grp), 'p-value ks']
            axes[i].text(
                x=age_grp_mapping[age_grp],
                y=ks_yloc_other_x1,
                s=f"{ks_value:.2f}",
                fontsize=8,
                ha='center',
                rotation=90,
                #bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7)
            )
            
        #ks_value = pvalues.loc[(date, cause), 'p-value ks']
        leg_ypos = ks_yloc_other_x1+3000
        axes[i].text(
            x=age_grp_mapping['0-4'],
            y=leg_ypos,
            s=f"KS test p-value",
            fontsize=8,
            ha='left',
            rotation=0,
            #bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7)
        )
        if compare_to_other_x250:
            for age_grp in df_normal.index.get_level_values('age_grp').unique():
                ks_value = pvalues.loc[('other_x250',sex, age_grp), 'p-value ks']
                axes[i].text(
                    x=age_grp_mapping[age_grp],
                    y=ks_yloc_other_x250,
                    s=f"{ks_value:.2f}",
                    fontsize=8,
                    ha='center',
                    rotation=90,
                    #bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7)
                )
                
            leg_ypos = ks_yloc_other_x250+3000
            axes[i].text(
                x=age_grp_mapping['0-4'],
                y=leg_ypos,
                s=f"KS test p-value (incr. other mortality)",
                fontsize=8,
                ha='left',
                rotation=0,
                #bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7)
            )

        # Set titles and labels
        axes[i].set_title(f'{sex}')
        axes[i].set_xlabel('Age Group')
        axes[i].set_ylabel('Averaged Yearly DALYs')

        # Set x-ticks and labels
        axes[i].set_xticks(x_pos)
        axes[i].set_xticklabels(age_groups, rotation=45)

        # Add a legend
        axes[i].legend()

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plot
    if compare_to_other_x250:
        plt.savefig('plots/final_DALYs_Breakdown_by_age_grp_DC.png')
    else:
        plt.savefig('plots/final_DALYs_Breakdown_by_age_grp.png')


def compare_and_plot(outputs, first_scenario, second_scenario, target, factor=None, ylabel=None):

    if factor is None:
        df_first = outputs[first_scenario]['data'][target] # formally df
        df_second = outputs[second_scenario]['data'][target] # formally df_emulated
    else: 
        df_first = outputs[first_scenario]['data'][target]*factor # formally df
        df_second = outputs[second_scenario]['data'][target]*factor # formally df_emulated
    
    # Define label names
    labels = [first_scenario, second_scenario]

    # Iterate over causes
    causes = df_first.index.get_level_values('cause').unique()
    for cause in causes:
        # Extract relevant data for both scenarios
        data = {}
        for label_name, dataset in zip(labels, [df_first, df_second]):
            df_cause = dataset.loc[dataset.index.get_level_values('cause') == cause, ('0', ['lower', 'mean', 'upper'])]
            data[label_name] = {
                'lower': df_cause[('0', 'lower')],
                'mean': df_cause[('0', 'mean')],
                'upper': df_cause[('0', 'upper')]
            }
        
        years = data[labels[0]]['mean'].index.get_level_values('year')

        # % error
        percent_error = 100 * (data[labels[1]]['mean'].values - data[labels[0]]['mean'].values) / data[labels[0]]['mean'].values

        # Create figure with two panels
        fig, (ax_main, ax_error) = plt.subplots(2, 1, figsize=(10, 8),
                                                gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

        colors = ['blue', 'orange']
        markers = ['o', 'x']
        for i, label_name in enumerate(labels):
            ax_main.plot(years, data[label_name]['mean'], label=label_name, color=colors[i], linestyle='-', marker=markers[i])
            ax_main.fill_between(years, data[label_name]['lower'], data[label_name]['upper'], color=colors[i], alpha=0.2)

        if ylabel is None:
            ax_main.set_ylabel(f'{target}')
        else:
            ax_main.set_ylabel(f'{ylabel}')
        ax_main.set_title(f'{target} due to {cause}')
        ax_main.legend(loc='lower right')
        ax_main.grid(True)

        # % error panel
        ax_error.plot(years, percent_error, color='red', marker='o', linestyle='-', label='Mean')
        ax_error.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax_error.set_ylabel('% Error')
        ax_error.set_xlabel('Year')
        ax_error.legend(loc='upper right')
        ax_error.grid(True)

        plt.tight_layout()
        plt.savefig(f'plots/final_{target}_due_to_{cause}.png')
        plt.close()


# First collect all data
for key in outputs.keys():
    outputs[key]['data'] = apply(outputs[key]['results_folder'])
    for data_type in outputs[key]['data'].keys():
        outputs[key]['data'][data_type] = compute_summary_stats(outputs[key]['data'][data_type])


compare_and_plot(outputs, 'standard_RTI', 'emulated_RTI', 'deaths', None, 'Deaths')
compare_and_plot(outputs, 'standard_RTI', 'emulated_RTI', 'dalys', None, 'DALYs')
compare_and_plot(outputs, 'standard_RTI', 'emulated_RTI', 'cfr', 1000, 'Crude mortality rate (/year/1000 individuals)')
compare_and_make_plots_age_sex_distr(outputs, 'standard_RTI', 'emulated_RTI', 'standard_RTI_x250', 'emulated_RTI_x250', 'dalys_by_sex_and_age', compare_to_other_x250=True)


exit(-1)
    

    

TARGET_PERIOD = (Date(min_year, 1, 1), Date(max_year, 1, 1))

# Definitions of general helper functions
lambda stub: output_folder / f"{stub.replace('*', '_star_')}.png"  # noqa: E731

def target_period() -> str:
    """Returns the target period as a string of the form YYYY-YYYY"""
    return "-".join(str(t.year) for t in TARGET_PERIOD)



def get_num_deaths(_df):
    """Return total number of Deaths (total within the TARGET_PERIOD)
    """
    return pd.Series(data=len(_df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)]))

def get_num_dalys(_df):
    """Return total number of DALYs (Stacked) by label (total within the TARGET_PERIOD)"""
    return pd.Series(
        data=_df
        .loc[_df.year.between(*[i.year for i in TARGET_PERIOD])]
        .drop(columns=['date', 'sex', 'age_range', 'year'])
        .sum().sum()
    )

def get_num_dalys_by_cause(_df):
    """Return number of DALYs by cause by label (total within the TARGET_PERIOD)"""
    return pd.Series(
        data=_df
        .loc[_df.year.between(*[i.year for i in TARGET_PERIOD])]
        .drop(columns=['date', 'sex', 'age_range', 'year'])
        .sum()
    )

def set_param_names_as_column_index_level_0(_df):
    """Set the columns index (level 0) as the param_names."""
    ordered_param_names_no_prefix = {i: x for i, x in enumerate(param_names)}
    names_of_cols_level0 = [ordered_param_names_no_prefix.get(col) for col in _df.columns.levels[0]]
    assert len(names_of_cols_level0) == len(_df.columns.levels[0])
    _df.columns = _df.columns.set_levels(names_of_cols_level0, level=0)
    return _df

def find_difference_relative_to_comparison(_ser: pd.Series,
                                           comparison: str,
                                           scaled: bool = False,
                                           drop_comparison: bool = True,
                                           ):
    """Find the difference in the values in a pd.Series with a multi-index, between the draws (level 0)
    within the runs (level 1), relative to where draw = `comparison`.
    The comparison is `X - COMPARISON`."""
    return _ser \
        .unstack(level=0) \
        .apply(lambda x: (x - x[comparison]) / (x[comparison] if scaled else 1.0), axis=1) \
        .drop(columns=([comparison] if drop_comparison else [])) \
        .stack()


def get_counts_of_hsi_by_treatment_id(_df):
    """Get the counts of the short TREATMENT_IDs occurring"""
    _counts_by_treatment_id = _df \
        .loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD), 'TREATMENT_ID'] \
        .apply(pd.Series) \
        .sum() \
        .astype(int)
    return _counts_by_treatment_id.groupby(level=0).sum()
    
year_target = 2023
def get_counts_of_hsi_by_treatment_id_by_year(_df):
    """Get the counts of the short TREATMENT_IDs occurring"""
    _counts_by_treatment_id = _df \
        .loc[pd.to_datetime(_df['date']).dt.year ==year_target, 'TREATMENT_ID'] \
        .apply(pd.Series) \
        .sum() \
        .astype(int)
    return _counts_by_treatment_id.groupby(level=0).sum()

def get_counts_of_hsi_by_short_treatment_id(_df):
    """Get the counts of the short TREATMENT_IDs occurring (shortened, up to first underscore)"""
    _counts_by_treatment_id = get_counts_of_hsi_by_treatment_id(_df)
    _short_treatment_id = _counts_by_treatment_id.index.map(lambda x: x.split('_')[0] + "*")
    return _counts_by_treatment_id.groupby(by=_short_treatment_id).sum()
    
def get_counts_of_hsi_by_short_treatment_id_by_year(_df):
    """Get the counts of the short TREATMENT_IDs occurring (shortened, up to first underscore)"""
    _counts_by_treatment_id = get_counts_of_hsi_by_treatment_id_by_year(_df)
    _short_treatment_id = _counts_by_treatment_id.index.map(lambda x: x.split('_')[0] + "*")
    return _counts_by_treatment_id.groupby(by=_short_treatment_id).sum()

    
# Obtain parameter names for this scenario file
param_names = get_parameter_names_from_scenario_file()
print(param_names)


def get_counts_of_death_by_year(df):
    """Aggregate the model outputs into five-year periods for age and time"""
    #_, agegrplookup = make_age_grp_lookup()
    #_, calperiodlookup = make_calendar_period_lookup()
    df["year"] = df["date"].dt.year
    #df["age_grp"] = df["age"].map(agegrplookup).astype(make_age_grp_types())
    #df["period"] = df["year"].map(calperiodlookup).astype(make_calendar_period_type())
    return df.groupby(by=["year", "label"])["person_id"].count()


deaths_by_year_and_cause = extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="death",
    custom_generate_series=get_counts_of_death_by_year,
    do_scaling=True
).pipe(set_param_names_as_column_index_level_0)


print(deaths_by_year_and_cause)
print(list(deaths_by_year_and_cause.index))
deaths_by_year_and_cause.to_csv('ConvertedOutputs/Deaths_by_cause_with_time.csv', index=True)
exit(-1)

# ================================================================================================
# TIME EVOLUTION OF TOTAL DALYs
# Plot DALYs averted compared to the ``No Policy'' policy

year_target = 2023 # This global variable will be passed to custom function
def get_num_dalys_in_year(_df):
    """Return total number of DALYs (Stacked) by label (total within the TARGET_PERIOD)"""
    return pd.Series(
        data=_df
        .loc[_df.year == year_target]
        .drop(columns=['date', 'sex', 'age_range', 'year'])
        .sum().sum()
    )
    
ALL = {}
# Plot time trend show year prior transition as well to emphasise that until that point DALYs incurred
# are consistent across different policies
this_min_year = 2010
for year in range(this_min_year, max_year+1):
    year_target = year
    num_dalys_by_year = extract_results(
        results_folder,
        module='tlo.methods.healthburden',
        key='dalys_stacked',
        custom_generate_series=get_num_dalys_in_year,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0)
    ALL[year_target] = num_dalys_by_year
    print(num_dalys_by_year.columns)
# Concatenate the DataFrames into a single DataFrame
concatenated_df = pd.concat(ALL.values(), keys=ALL.keys())
concatenated_df.index = concatenated_df.index.set_names(['date', 'index_original'])
concatenated_df = concatenated_df.reset_index(level='index_original',drop=True)
dalys_by_year = concatenated_df
print(dalys_by_year)
dalys_by_year.to_csv('ConvertedOutputs/Total_DALYs_with_time.csv', index=True)

# ================================================================================================
# Print population under each scenario
pop_model = extract_results(results_folder,
                            module="tlo.methods.demography",
                            key="population",
                            column="total",
                            index="date",
                            do_scaling=True
                            ).pipe(set_param_names_as_column_index_level_0)

pop_model.index = pop_model.index.year
pop_model = pop_model[(pop_model.index >= this_min_year) & (pop_model.index <= max_year)]
print(pop_model)
assert dalys_by_year.index.equals(pop_model.index)
assert all(dalys_by_year.columns == pop_model.columns)
pop_model.to_csv('ConvertedOutputs/Population_with_time.csv', index=True)

# ================================================================================================
# DALYs BROKEN DOWN BY CAUSES AND YEAR
# DALYs by cause per year
# %% Quantify the health losses associated with all interventions combined.

year_target = 2023 # This global variable will be passed to custom function
def get_num_dalys_by_year_and_cause(_df):
    """Return total number of DALYs (Stacked) by label (total within the TARGET_PERIOD)"""
    return pd.Series(
        data=_df
        .loc[_df.year == year_target]
        .drop(columns=['date', 'sex', 'age_range', 'year'])
        .sum()
    )
    
ALL = {}
# Plot time trend show year prior transition as well to emphasise that until that point DALYs incurred
# are consistent across different policies
this_min_year = 2010
for year in range(this_min_year, max_year+1):
    year_target = year
    num_dalys_by_year = extract_results(
        results_folder,
        module='tlo.methods.healthburden',
        key='dalys_stacked',
        custom_generate_series=get_num_dalys_by_year_and_cause,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0)
    ALL[year_target] = num_dalys_by_year #summarize(num_dalys_by_year)

# Concatenate the DataFrames into a single DataFrame
concatenated_df = pd.concat(ALL.values(), keys=ALL.keys())

concatenated_df.index = concatenated_df.index.set_names(['date', 'cause'])

df_total = concatenated_df
df_total.to_csv('ConvertedOutputs/DALYS_by_cause_with_time.csv', index=True)

ALL = {}
# Plot time trend show year prior transition as well to emphasise that until that point DALYs incurred
# are consistent across different policies
for year in range(min_year, max_year+1):
    year_target = year
    
    hsi_delivered_by_year = extract_results(
            results_folder,
            module='tlo.methods.healthsystem.summary',
            key='HSI_Event',
            custom_generate_series=get_counts_of_hsi_by_short_treatment_id_by_year,
            do_scaling=True
        ).pipe(set_param_names_as_column_index_level_0)
    ALL[year_target] = hsi_delivered_by_year

# Concatenate the DataFrames into a single DataFrame
concatenated_df = pd.concat(ALL.values(), keys=ALL.keys())
concatenated_df.index = concatenated_df.index.set_names(['date', 'cause'])
HSI_ran_by_year = concatenated_df

del ALL

ALL = {}
# Plot time trend show year prior transition as well to emphasise that until that point DALYs incurred
# are consistent across different policies
for year in range(min_year, max_year+1):
    year_target = year
    
    hsi_not_delivered_by_year = extract_results(
            results_folder,
            module='tlo.methods.healthsystem.summary',
            key='Never_ran_HSI_Event',
            custom_generate_series=get_counts_of_hsi_by_short_treatment_id_by_year,
            do_scaling=True
        ).pipe(set_param_names_as_column_index_level_0)
    ALL[year_target] = hsi_not_delivered_by_year

# Concatenate the DataFrames into a single DataFrame
concatenated_df = pd.concat(ALL.values(), keys=ALL.keys())
concatenated_df.index = concatenated_df.index.set_names(['date', 'cause'])
HSI_never_ran_by_year = concatenated_df

HSI_never_ran_by_year = HSI_never_ran_by_year.fillna(0) #clean_df(
HSI_ran_by_year = HSI_ran_by_year.fillna(0)
HSI_total_by_year = HSI_ran_by_year.add(HSI_never_ran_by_year, fill_value=0)
HSI_ran_by_year.to_csv('ConvertedOutputs/HSIs_ran_by_area_with_time.csv', index=True)
HSI_never_ran_by_year.to_csv('ConvertedOutputs/HSIs_never_ran_by_area_with_time.csv', index=True)
print(HSI_ran_by_year)
print(HSI_never_ran_by_year)
print(HSI_total_by_year)
exit(-1)
    
"""
if __name__ == "__main__":
    rfp = Path('resources')

    parser = argparse.ArgumentParser(
        description="Produce plots to show the impact each set of treatments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-path",
        help=(
            "Directory to write outputs to. If not specified (set to None) outputs "
            "will be written to value of --results-path argument."
        ),
        type=Path,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--resources-path",
        help="Directory containing resource files",
        type=Path,
        default=Path('resources'),
        required=False,
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        help=(
            "Directory containing results from running "
            "src/scripts/healthsystem/impact_of_const_capabilities_expansion/scenario_impact_of_capabilities_expansion_scaling.py "
        ),
        default=None,
        required=False
    )
    args = parser.parse_args()
    assert args.results_path is not None
    results_path = args.results_path

    output_path = results_path if args.output_path is None else args.output_path

    apply(
        results_folder=results_path,
        output_folder=output_path,
        resourcefilepath=args.resources_path
    )
"""
