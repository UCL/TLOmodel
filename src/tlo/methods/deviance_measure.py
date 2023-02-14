"""
Deviance measure

This module runs at the end of a simulation and calculates a weighted deviance measure
for a given set of parameters using outputs from the demography (deaths), HIV and TB modules

"""
import math
from collections import defaultdict

import pandas as pd

from tlo import Module, logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Deviance(Module):
    """
    This module reads in logged outputs from HIV, TB and demography and compares them with reported data
    a deviance measure is calculated and returned on simulation end
    """

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        self.data_dict = dict()
        self.model_dict = dict()

        # Initialise empty dict (with factory method of list) to store lists containing information about each death
        # that is used by the `Deviance` module
        self.__demog_outputs__ = defaultdict(list)

    INIT_DEPENDENCIES = {'Demography', 'Hiv', 'Tb'}

    # Declare Metadata
    METADATA = {}

    # No parameters to declare
    PARAMETERS = {}

    # No properties to declare
    PROPERTIES = {}

    def read_parameters(self, data_folder):
        pass

    def initialise_population(self, population):
        pass

    def initialise_simulation(self, sim):
        pass

    def on_birth(self, mother_id, child_id):
        pass

    def read_data_files(self):
        """Make a dict of all data to be used in calculating calibration score"""

        # # HIV read in resource files for data
        xls = pd.ExcelFile(self.resourcefilepath / "ResourceFile_HIV.xlsx")

        # MPHIA HIV data - age-structured
        data_hiv_mphia_inc = pd.read_excel(xls, sheet_name="MPHIA_incidence2015")
        data_hiv_mphia_prev = pd.read_excel(xls, sheet_name="MPHIA_prevalence_art2015")

        # hiv prevalence
        self.data_dict["mphia_prev_2015_adult"] = data_hiv_mphia_prev.loc[
            data_hiv_mphia_prev.age == "Total 15-49", "total percent hiv positive"
        ].values[
            0
        ]
        self.data_dict["mphia_prev_2015_child"] = data_hiv_mphia_prev.loc[
            data_hiv_mphia_prev.age == "Total 0-14", "total percent hiv positive"
        ].values[
            0
        ]

        # hiv incidence
        self.data_dict["mphia_inc_2015_adult"] = data_hiv_mphia_inc.loc[
            (data_hiv_mphia_inc.age == "15-49"), "total_percent_annual_incidence"
        ].values[
            0
        ]

        # DHS HIV data
        data_hiv_dhs_prev = pd.read_excel(xls, sheet_name="DHS_prevalence")
        self.data_dict["dhs_prev_2010"] = data_hiv_dhs_prev.loc[
            (data_hiv_dhs_prev.Year == 2010), "HIV prevalence among general population 15-49"
        ].values[0]
        self.data_dict["dhs_prev_2015"] = data_hiv_dhs_prev.loc[
            (data_hiv_dhs_prev.Year == 2015), "HIV prevalence among general population 15-49"
        ].values[0]

        # UNAIDS AIDS deaths data: 2010-
        data_hiv_unaids_deaths = pd.read_excel(xls, sheet_name="unaids_mortality_dalys2021")
        self.data_dict["unaids_deaths_per_100k"] = data_hiv_unaids_deaths["AIDS_mortality_per_100k"]

        # TB
        # TB WHO data: 2010-
        xls_tb = pd.ExcelFile(self.resourcefilepath / "ResourceFile_TB.xlsx")

        # TB active incidence per 100k 2010-2017
        data_tb_who = pd.read_excel(xls_tb, sheet_name="WHO_activeTB2023")
        self.data_dict["who_tb_inc_per_100k"] = data_tb_who.loc[
            (data_tb_who.year >= 2010), "incidence_per_100k"
        ]

        # TB mortality per 100k excluding HIV: 2010-2017
        self.data_dict["who_tb_deaths_per_100k"] = data_tb_who.loc[
            (data_tb_who.year >= 2010), "mortality_tb_excl_hiv_per_100k"
        ]

    def read_model_outputs(self):
        hiv = self.sim.modules['Hiv'].hiv_outputs
        tb = self.sim.modules['Tb'].tb_outputs
        demog = self.__demog_outputs__

        # get logged outputs for calibration into dict
        # population size each year
        pop = pd.Series(hiv["population"])
        pop.index = hiv['date']
        pop.index = pd.to_datetime(pop.index, format="%Y")

        # ------------------ HIV disease ------------------ #
        # HIV - prevalence among in adults aged 15-49
        self.model_dict["hiv_prev_adult_2010"] = hiv["hiv_prev_adult_1549"][0] * 100
        self.model_dict["hiv_prev_adult_2015"] = hiv["hiv_prev_adult_1549"][6] * 100

        # hiv incidence in adults aged 15-49
        self.model_dict["hiv_inc_adult_2015"] = hiv["hiv_adult_inc_1549"][6] * 100

        # hiv prevalence in children (mphia)
        self.model_dict["hiv_prev_child_2015"] = hiv["hiv_prev_child"][6] * 100

        # ------------------ TB DISEASE ------------------ #

        # tb active incidence per 100k - all ages
        self.model_dict["TB_active_inc_per100k"] = (tb["num_new_active_tb"] / pop) * 100000

        # ------------------ DEATHS ------------------ #
        # convert dict to df for easier processing
        deaths = pd.DataFrame()
        deaths['date'] = demog['date']
        deaths['age'] = demog['age']
        deaths['sex'] = demog['sex']
        deaths['cause'] = demog['cause']

        # AIDS DEATHS
        # limit to deaths among aged 15+, include HIV/TB deaths
        keep = (deaths.age >= 15) & (
            (deaths.cause == "AIDS_TB") | (deaths.cause == "AIDS_non_TB")
        )
        deaths_AIDS = deaths.loc[keep].copy()
        tot_aids_deaths = deaths_AIDS.groupby(by=["date"]).size()
        tot_aids_deaths.index = pd.to_datetime(tot_aids_deaths.index, format="%Y")

        # aids mortality rates per 1000 person-years
        self.model_dict["AIDS_mortality_per_100k"] = (tot_aids_deaths / pop) * 100000

        # TB deaths (non-hiv only, all ages)
        keep = deaths.cause == "TB"
        deaths_TB = deaths.loc[keep].copy()
        tot_tb_non_hiv_deaths = deaths_TB.groupby(by=["date"]).size()
        tot_tb_non_hiv_deaths.index = pd.to_datetime(tot_tb_non_hiv_deaths.index, format="%Y")

        # tb mortality rates per 100k person-years
        self.model_dict["TB_mortality_per_100k"] = (tot_tb_non_hiv_deaths / pop) * 100000

    def weighted_mean(self, model_dict, data_dict):
        # assert model_output is not empty

        # return calibration score (weighted mean deviance)
        # sqrt( (observed data – model output)^2 | / observed data)
        # sum these for each data item (all male prevalence over time by age-group) and divide by # items
        # then weighted sum of all components -> calibration score

        # for debugging
        # model_dict = self.model_dict
        # data_dict = self.data_dict

        # need weights for each data item
        model_weight = 0.5  # weight if "data" are modelled estimate (e.g. UNAIDS)

        def deviance_function(data, model):
            # in case there are NAs in model outputs (e.g. no deaths in given year)
            model = pd.Series(model)
            model = model.fillna(0)

            deviance = math.sqrt((data - model) ** 2) / data

            return deviance

        # ------------------ HIV ------------------ #

        # hiv prevalence in adults 15-49: dhs 2010, 2015, mphia
        hiv_prev_adult = (deviance_function(data_dict["dhs_prev_2010"], model_dict["hiv_prev_adult_2010"]) +
                          deviance_function(data_dict["dhs_prev_2015"], model_dict["hiv_prev_adult_2015"]) +
                          deviance_function(data_dict["mphia_prev_2015_adult"], model_dict["hiv_prev_adult_2015"])
                          ) / 3

        hiv_prev_child = deviance_function(
            data_dict["mphia_prev_2015_child"], model_dict["hiv_prev_child_2015"]
        )

        # hiv incidence mphia
        hiv_inc_adult = deviance_function(
            data_dict["mphia_inc_2015_adult"], model_dict["hiv_inc_adult_2015"]
        )

        # ------------------ TB ------------------ #

        # tb active incidence (WHO estimates) 2010-2021
        tb_incidence_who = (
                               deviance_function(
                                   data_dict["who_tb_inc_per_100k"].values[0], model_dict["TB_active_inc_per100k"][0]
                               ) +
                               deviance_function(
                                   data_dict["who_tb_inc_per_100k"].values[1], model_dict["TB_active_inc_per100k"][1]
                               ) +
                               deviance_function(
                                   data_dict["who_tb_inc_per_100k"].values[2], model_dict["TB_active_inc_per100k"][2]
                               ) +
                               deviance_function(
                                   data_dict["who_tb_inc_per_100k"].values[3], model_dict["TB_active_inc_per100k"][3]
                               ) +
                               deviance_function(
                                   data_dict["who_tb_inc_per_100k"].values[4], model_dict["TB_active_inc_per100k"][4]
                               ) +
                               deviance_function(
                                   data_dict["who_tb_inc_per_100k"].values[5], model_dict["TB_active_inc_per100k"][5]
                               ) +
                               deviance_function(
                                   data_dict["who_tb_inc_per_100k"].values[6], model_dict["TB_active_inc_per100k"][6]
                               ) +
                               deviance_function(
                                   data_dict["who_tb_inc_per_100k"].values[7], model_dict["TB_active_inc_per100k"][7]
                               ) +
                               deviance_function(
                                   data_dict["who_tb_inc_per_100k"].values[8], model_dict["TB_active_inc_per100k"][8]
                               ) +
                               deviance_function(
                                   data_dict["who_tb_inc_per_100k"].values[9], model_dict["TB_active_inc_per100k"][9]
                               ) +
                               deviance_function(
                                   data_dict["who_tb_inc_per_100k"].values[10], model_dict["TB_active_inc_per100k"][10]
                               ) +
                               deviance_function(
                                   data_dict["who_tb_inc_per_100k"].values[11], model_dict["TB_active_inc_per100k"][11]
                               )
                           ) / 12

        # ------------------ DEATHS ------------------ #

        # aids deaths unaids 2010-2021
        hiv_deaths_unaids = (
                                deviance_function(
                                    data_dict["unaids_deaths_per_100k"][0],
                                    model_dict["AIDS_mortality_per_100k"][0],
                                )
                                + deviance_function(
                                    data_dict["unaids_deaths_per_100k"][1],
                                    model_dict["AIDS_mortality_per_100k"][1],
                                )
                                + deviance_function(
                                    data_dict["unaids_deaths_per_100k"][2],
                                    model_dict["AIDS_mortality_per_100k"][2],
                                )
                                + deviance_function(
                                    data_dict["unaids_deaths_per_100k"][3],
                                    model_dict["AIDS_mortality_per_100k"][3],
                                )
                                + deviance_function(
                                    data_dict["unaids_deaths_per_100k"][4],
                                    model_dict["AIDS_mortality_per_100k"][4],
                                )
                                + deviance_function(
                                    data_dict["unaids_deaths_per_100k"][5],
                                    model_dict["AIDS_mortality_per_100k"][5],
                                )
                                + deviance_function(
                                    data_dict["unaids_deaths_per_100k"][6],
                                    model_dict["AIDS_mortality_per_100k"][6],
                                )
                                + deviance_function(
                                    data_dict["unaids_deaths_per_100k"][7],
                                    model_dict["AIDS_mortality_per_100k"][7],
                                )
                                + deviance_function(
                                    data_dict["unaids_deaths_per_100k"][8],
                                    model_dict["AIDS_mortality_per_100k"][8],
                                )
                                + deviance_function(
                                    data_dict["unaids_deaths_per_100k"][9],
                                    model_dict["AIDS_mortality_per_100k"][9],
                                )
                                + deviance_function(
                                    data_dict["unaids_deaths_per_100k"][10],
                                    model_dict["AIDS_mortality_per_100k"][10],
                                )
                                + deviance_function(
                                    data_dict["unaids_deaths_per_100k"][11],
                                    model_dict["AIDS_mortality_per_100k"][11],
                                )
                            ) / 12

        # tb death rate who 2010-2021
        tb_mortality_who = (
                               deviance_function(
                                   data_dict["who_tb_deaths_per_100k"].values[0],
                                   model_dict["TB_mortality_per_100k"][0],
                               ) +
                               deviance_function(
                                   data_dict["who_tb_deaths_per_100k"].values[1],
                                   model_dict["TB_mortality_per_100k"][1],
                               ) +
                               deviance_function(
                                   data_dict["who_tb_deaths_per_100k"].values[2],
                                   model_dict["TB_mortality_per_100k"][2],
                               ) +
                               deviance_function(
                                   data_dict["who_tb_deaths_per_100k"].values[3],
                                   model_dict["TB_mortality_per_100k"][3],
                               ) +
                               deviance_function(
                                   data_dict["who_tb_deaths_per_100k"].values[4],
                                   model_dict["TB_mortality_per_100k"][4],
                               ) +
                               deviance_function(
                                   data_dict["who_tb_deaths_per_100k"].values[5],
                                   model_dict["TB_mortality_per_100k"][5],
                               ) +
                               deviance_function(
                                   data_dict["who_tb_deaths_per_100k"].values[6],
                                   model_dict["TB_mortality_per_100k"][6],
                               ) +
                               deviance_function(
                                   data_dict["who_tb_deaths_per_100k"].values[8],
                                   model_dict["TB_mortality_per_100k"][8],
                               ) +
                               deviance_function(
                                   data_dict["who_tb_deaths_per_100k"].values[9],
                                   model_dict["TB_mortality_per_100k"][9],
                               ) +
                               deviance_function(
                                   data_dict["who_tb_deaths_per_100k"].values[10],
                                   model_dict["TB_mortality_per_100k"][10],
                               ) +
                               deviance_function(
                                   data_dict["who_tb_deaths_per_100k"].values[11],
                                   model_dict["TB_mortality_per_100k"][11],
                               ) +
                               deviance_function(
                                   data_dict["who_tb_deaths_per_100k"].values[12],
                                   model_dict["TB_mortality_per_100k"][12],
                               )
                           ) / 12

        # tb who is a model estimate, also tb cnr depends on estimated incidence
        calibration_score = (
            hiv_prev_adult
            + hiv_prev_child
            + hiv_inc_adult
            + (tb_incidence_who * model_weight)
            + (hiv_deaths_unaids * model_weight)
            + (tb_mortality_who * model_weight)
        )

        hiv_beta = self.sim.modules["Hiv"].parameters["beta"]
        tb_scaling_factor_WHO = self.sim.modules["Tb"].parameters["scaling_factor_WHO"]

        return_values = [calibration_score, hiv_beta, tb_scaling_factor_WHO]

        return return_values

    def record_death(self, year, age_years, sex, cause):
        """Save outputs about one death"""
        self.__demog_outputs__["date"] += [year]
        self.__demog_outputs__["age"] += [age_years]
        self.__demog_outputs__["sex"] += [sex]
        self.__demog_outputs__["cause"] += [cause]

    def on_simulation_end(self):

        if self.sim.date.year >= 2020:
            self.read_data_files()
            self.read_model_outputs()
            deviance_measure = self.weighted_mean(model_dict=self.model_dict, data_dict=self.data_dict)

            logger.info(
                key="deviance_measure",
                description="Deviance measure for HIV and TB",
                data={"deviance_measure": deviance_measure[0],
                      "hiv_transmission_rate": deviance_measure[1],
                      "tb_scaling_factor_WHO": deviance_measure[2]
                      }
            )
