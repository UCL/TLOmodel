"""
An analysis file for the wasting module
"""
import datetime
# %% Import statements
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import compare_number_of_deaths, parse_log_file
from tlo.methods import (
    care_of_women_during_pregnancy,
    contraception,
    demography,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    symptommanager,
    tb,
    wasting,
)


class WastingAnalyses:
    """
    This class looks at plotting all important outputs from the wasting module
    """

    def __init__(self, log_file_path):
        self.__log_file_path = log_file_path
        # parse wasting logs
        self.__logs_dict = \
            parse_log_file(self.__log_file_path)['tlo.methods.wasting']

        # gender description
        self.__gender_desc = {'M': 'Males',
                              'F': 'Females'}

        # wasting types description
        self.__wasting_types_desc = {'WHZ<-3': 'severe wasting',
                                     '-3<=WHZ<-2': 'moderate wasting',
                                     'WHZ>=-2': 'not undernourished'}

    def plot_wasting_incidence(self):
        """ plot the incidence of wasting over time """
        w_inc_df = self.__logs_dict['wasting_incidence_count']
        w_inc_df.set_index(w_inc_df.date.dt.year, inplace=True)
        w_inc_df.drop(columns='date', inplace=True)
        # get age year. doesn't matter what wasting category you choose for
        # they all have same age groups
        age_years = list(w_inc_df.loc[w_inc_df.index[0], 'WHZ<-3'].keys(

        ))
        age_years.remove('5+y')

        _row_counter = 0
        _col_counter = 0
        # plot setup
        fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
        for _age in age_years:
            new_df = pd.DataFrame()
            for state in w_inc_df.columns:
                new_df[state] = \
                    w_inc_df.apply(lambda row: row[state][_age], axis=1)

            new_df = new_df.apply(lambda _row: _row / _row.sum(), axis=1)
            plotting = new_df[["WHZ<-3", "-3<=WHZ<-2"]]
            # convert into proportions
            ax = plotting.plot(kind='bar', stacked=True,
                               ax=axes[_row_counter, _col_counter],
                               title=f"incidence of wasting in {_age} infants",
                               ylim=[0, 0.05])
            ax.legend(self.__wasting_types_desc.values(), loc='lower right')
            ax.set_xlabel('year')
            ax.set_ylabel('proportions')
            # move to another row
            if _col_counter == 2:
                _row_counter += 1
                _col_counter = -1
            _col_counter += 1  # increment column counter
            fig.tight_layout()
        # plt.show()

    def plot_wasting_prevalence(self):
        w_prev_df = self.__logs_dict['wasting_prevalence_count']
        w_prev_df.set_index(w_prev_df.date.dt.year, inplace=True)
        w_prev_df.drop(columns='date', inplace=True)
        w_prev_df = w_prev_df.apply(lambda _row: _row / _row.sum(), axis=1)
        w_prev_df.plot(kind='bar', stacked=True,
                       title="Wasting prevalence in children 0-59 months",
                       ylabel='proportions',
                       xlabel='year'
                       )
        # plt.show()

    def plot_modal_gbd_deaths_by_gender(self):
        """ compare modal and GBD deaths by gender """
        death_compare = \
            compare_number_of_deaths(self.__log_file_path, resources)
        fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True)
        for _col, sex in enumerate(('M', 'F')):
            plot_df = death_compare.loc[(['2010-2014', '2015-2019'],
                                         sex, slice(None), 'Childhood Wasting'
                                         )].groupby('period').sum()
            plotting = plot_df.loc[['2010-2014', '2015-2019']]
            ax = plotting['model'].plot.bar(label='Model', ax=axs[_col], rot=0)
            ax.errorbar(x=plotting['model'].index, y=plotting.GBD_mean,
                        yerr=[plotting.GBD_lower, plotting.GBD_upper],
                        fmt='o', color='#000', label="GBD")
            ax.set_title(f'{self.__gender_desc[sex]} '
                         f'wasting deaths, 2010-2014')
            ax.set_xlabel("Time period")
            ax.set_ylabel("Number of deaths")
            ax.legend(loc=2)
        fig.tight_layout()
        fig.savefig(
            outputs / ('modal_gbd_deaths_by_gender' + datestamp + ".pdf"),
            format="pdf"
        )
        # plt.show()

if __name__ == "__main__":
    seed = 1

    # Path to the resource files used by the disease and intervention methods
    resources = Path("./resources")
    outputs = Path("./outputs")

    # create a datestamp
    datestamp = datetime.date.today().strftime("__%Y_%m_%d") + \
                datetime.datetime.now().strftime("%H_%M_%S")

    # configure logging
    log_config = {
        # output filename. A timestamp will be added to this.
        "filename": "wasting",
        "custom_levels": {  # Customise the output of specific loggers
            "tlo.methods.demography": logging.INFO,
            "tlo.methods.population": logging.INFO,
            "tlo.methods.wasting": logging.INFO,
            '*': logging.WARNING
        }
    }

    # Basic arguments required for the simulation
    start_date = Date(2010, 1, 1)
    end_date = Date(2030, 1, 2)
    pop_size = 30000

    # Create simulation instance for this run.
    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

    # Register modules for simulation
    sim.register(
        demography.Demography(resourcefilepath=resources),
        healthsystem.HealthSystem(resourcefilepath=resources,
                                  service_availability=['*'],
                                  cons_availability='default'),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resources),
        healthburden.HealthBurden(resourcefilepath=resources),
        symptommanager.SymptomManager(resourcefilepath=resources),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resources),
        labour.Labour(resourcefilepath=resources),
        care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(
            resourcefilepath=resources),
        contraception.Contraception(resourcefilepath=resources),
        pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resources),
        postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resources),
        newborn_outcomes.NewbornOutcomes(resourcefilepath=resources),
        hiv.Hiv(resourcefilepath=resources),
        tb.Tb(resourcefilepath=resources),
        epi.Epi(resourcefilepath=resources),
        wasting.Wasting(resourcefilepath=resources),
    )

    sim.make_initial_population(n=pop_size)
    sim.simulate(end_date=end_date)

    # read the results
    output_path = sim.log_filepath

    # initialise the wasting class
    wasting_analyses = WastingAnalyses(output_path)

    # plot wasting incidence
    wasting_analyses.plot_wasting_incidence()

    # plot wasting prevalence
    wasting_analyses.plot_wasting_prevalence()

    # plot wasting deaths by gender as compared to GBD deaths
    wasting_analyses.plot_modal_gbd_deaths_by_gender()
