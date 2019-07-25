import pandas as pd

from tlo import Module, Parameter, Property, Types

# need to import HIV, HIV_Event, ART, ART_Event, BCG vaccine

# IPT and rifampicin as separate methods


# initial pop data #
inds = pd.read_csv("Q:/Thanzi la Onse/HIV/initial_pop_dataframe2018.csv")

TBincidence = pd.read_excel(
    "Q:/Thanzi la Onse/TB/Method Template TB.xlsx", sheet_name="TB_incidence", header=0
)

latent_TB_prevalence_total = 3170000  # Houben model paper
latent_TB_prevalence_children = 525000
latent_TB_prevalence_adults = latent_TB_prevalence_total - latent_TB_prevalence_children


# this class contains all the methods required to set up the baseline population
class TB(Module):
    """ Sets up baseline TB prevalence.

    Methods required:
    * `read_parameters(data_folder)`
    * `initialise_population(population)`
    * `initialise_simulation(sim)`
    * `on_birth(mother, child)`
    """

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    PARAMETERS = {
        "prop_fast_progressor": Parameter(
            Types.REAL,
            "Proportion of infections that progress directly to active stage, Vynnycky",
        ),
        "transmission_rate": Parameter(
            Types.REAL, "TB transmission rate, estimated by Juan"
        ),
        "progression_to_active_rate": Parameter(
            Types.REAL, "Combined rate of progression/reinfection/relapse from Juan"
        ),
        "rr_TB_HIV_stages": Parameter(
            Types.REAL, "relative risk of TB hin HIV+ compared with HIV- by CD4 stage"
        ),
        "rr_TB_ART": Parameter(Types.REAL, "relative risk of TB in HIV+ on ART"),
        "rr_TB_malnourished": Parameter(
            Types.REAL, "relative risk of TB with malnourishment"
        ),
        "rr_TB_diabetes1": Parameter(
            Types.REAL, "relative risk of TB with diabetes type 1"
        ),
        "rr_TB_alcohol": Parameter(
            Types.REAL, "relative risk of TB with heavy alcohol use"
        ),
        "rr_TB_smoking": Parameter(Types.REAL, "relative risk of TB with smoking"),
        "rr_TB_pollution": Parameter(
            Types.REAL, "relative risk of TB with indoor air pollution"
        ),
        "rr_infectiousness_HIV": Parameter(
            Types.REAL, "relative infectiousness of TB in HIV+ compared with HIV-"
        ),
        "recovery": Parameter(
            Types.REAL, "combined rate of diagnosis, treatment and self-cure, from Juan"
        ),
        "TB_mortality_rate": Parameter(Types.REAL, "mortality rate with active TB"),
        "rr_TB_mortality_HIV": Parameter(
            Types.REAL, "relative risk of mortality from TB in HIV+ compared with HIV-"
        ),
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        "has_TB": Property(Types.CATEGORICAL, "TB status: Uninfected, Latent, Active"),
        "date_TB_infection": Property(Types.DATE, "Date acquired TB infection"),
        "date_TB_death": Property(
            Types.DATE, "Projected time of TB death if untreated"
        ),
        "on_treatment": Property(Types.BOOL, "Currently on treatment for TB"),
        "date_ART_treatment_start": Property(Types.DATE, "Date treatment started"),
        "date_death": Property(Types.DATE, "Date of death"),
    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.
        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """

        params = self.parameters
        params["prop_fast_progressor"] = 0.14
        params["transmission_rate"] = 7.2
        params["progression_to_active_rate"] = 0.5

        params["rr_TB_with_HIV_stages"] = [3.44, 6.76, 13.28, 26.06]
        params["rr_ART"] = 0.39
        params["rr_TB_malnourished"] = 2.1
        params["rr_TB_diabetes1"] = 3
        params["rr_TB_alcohol"] = 2.9
        params["rr_TB_smoking"] = 2.6
        params["rr_TB_pollution"] = 1.5

        params["rr_infectiousness_HIV"] = 0.52
        params["recovery"] = 2
        params["TB_mortality_rate"] = 0.15
        params["rr_TB_mortality_HIV"] = 17.1

    # baseline population
    # assign infected status using WHO prevalence 2016 by 2 x age-groups
    def prevalence_active_TB(self, df):
        """ assign cases of active TB using weights to prioritise highest risk
        """

        params = self.module.parameters

        self.HIVstage1 = 3.33  # Williams equal duration HIV stages (x4)
        self.HIVstage2 = 6.67
        self.HIVstage3 = 10

        # create a vector of probabilities depending on HIV status and time since seroconversion
        # these probabilities need to include multiple risk factors (multiply RR unless strong confounding)
        # calculate the combined relative risk for every risk factor
        # then scale as probabilities need to sum to 1
        # then sample with probabilities for all susceptible people
        # all population without active TB then randomly assigned latent TB
        df["tmp"] = 1  # baseline risk group, no risk factors

        # RR HIV
        df["tmp"][
            (df.has_HIV == 1)
            & (self.current_time - df.date_HIV_infection <= self.HIVstage1)
        ] *= params["rr_TB_with_HIV_stages"][0]
        df["tmp"][
            (df.has_HIV == 1)
            & (self.current_time - df.date_HIV_infection > self.HIVstage1)
            & (self.current_time - df.date_HIV_infection <= self.HIVstage2)
        ] *= params["rr_TB_with_HIV_stages"][1]
        df["tmp"][
            (df.has_HIV == 1)
            & (self.current_time - df.date_HIV_infection > self.HIVstage2)
            & (self.current_time - df.date_HIV_infection <= self.HIVstage3)
        ] *= params["rr_TB_with_HIV_stages"][2]
        df["tmp"][
            (df.has_HIV == 1)
            & (self.current_time - df.date_HIV_infection > self.HIVstage3)
        ] *= params["rr_TB_with_HIV_stages"][3]

        df["tmp"][df.on_ART == 1] *= params[
            "rr_ART"
        ]  # this modifies the RR by ART status

        # RR lifestyle - not implemented yet
        df["tmp"][df.is_malnourished == 1] *= params["rr_TB_malnourished"]
        df["tmp"][df.has_diabetes1 == 1] *= params["rr_TB_diabetes1"]
        df["tmp"][df.heavy_alcohol == 1] *= params["rr_TB_alcohol"]
        df["tmp"][df.smoker == 1] *= params["rr_TB_smoking"]
        df["tmp"][df.indoor_pollution == 1] *= params["rr_TB_pollution"]

        # sample from uninfected population using WHO incidence and relative risk as weights
        # male age 0-14
        tmp1 = df.sample(
            df.index[(df.age < 15) & (df.sex == "M")],
            n=int(
                (
                    TBincidence["Incident cases"][
                        (TBincidence.Year == 2016)
                        & (TBincidence.Sex == "M")
                        & (TBincidence.Age == "0_14")
                    ]
                )
            ),
            replace=False,
            weights=df.tmp[(df.age < 15) & (df.sex == "M")],
        )
        df.loc[tmp1, "has_tb"] = "A"  # change status to active infection

        # female age 0-14
        tmp2 = df.sample(
            df.index[(df.age < 15) & (df.sex == "F")],
            n=int(
                (
                    TBincidence["Incident cases"][
                        (TBincidence.Year == 2016)
                        & (TBincidence.Sex == "F")
                        & (TBincidence.Age == "0_14")
                    ]
                )
            ),
            replace=False,
            weights=df.tmp[(df.age < 15) & (df.sex == "F")],
        )
        df.loc[tmp2, "has_tb"] = "A"  # change status to infected

        # male age >=15
        tmp3 = df.sample(
            df.index[(df.age >= 15) & (df.sex == "M")],
            n=int(
                (
                    TBincidence["Incident cases"][
                        (TBincidence.Year == 2016)
                        & (TBincidence.Sex == "M")
                        & (TBincidence.Age == "15_80")
                    ]
                )
            ),
            replace=False,
            weights=df.tmp[(df.age >= 15) & (df.sex == "M")],
        )
        df.loc[tmp3, "has_tb"] = "A"  # change status to infected

        # female age >=15
        tmp4 = df.sample(
            df.index[(df.age >= 15) & (df.sex == "F")],
            n=int(
                (
                    TBincidence["Incident cases"][
                        (TBincidence.Year == 2016)
                        & (TBincidence.Sex == "F")
                        & (TBincidence.Age == "15_80")
                    ]
                )
            ),
            replace=False,
            weights=df.tmp[(df.age >= 15) & (df.sex == "F")],
        )
        df.loc[tmp4, "has_tb"] = "A"  # change status to infected

        del df[
            "tmp"
        ]  # remove temporary column of combined relative risks, will change with time

        return df

    def prevalence_latent_TB(self, df):
        """ prevalence of latent TB infection is randomly assigned to the whole susceptible
        population with the exception of those with has_tb='A
        """
        tmp = df.sample(
            df.index[(df.has_tb == "U") & (df.age < 15)],
            n=latent_TB_prevalence_children,
            replace=False,
        )
        df.loc[tmp, "has_tb"] = "L"  # change status to latent infection

        tmp2 = df.sample(
            df.index[(df.has_tb == "U") & (df.age >= 15)],
            n=latent_TB_prevalence_adults,
            replace=False,
        )
        df.loc[tmp2, "has_tb"] = "L"  # change status to latent infection

        return df


# functions to be implemented


def force_of_infection_tb(inds):
    infected = len(
        inds[(inds.tb_status == "I") & (inds.tb_treat == 0)]
    )  # number infected untreated

    # dummy values - they were missing?
    rel_infectiousness_HIV = 0
    beta = 0

    # number co-infected with HIV * relative infectiousness (lower)
    hiv_infected = rel_infectiousness_HIV * len(
        inds[(inds.tb_status == "I") & (inds.status == "I")]
    )

    total_pop = len(
        inds[(inds.status != "DH") & (inds.status != "D")]
    )  # whole population currently alive

    foi = beta * (
        (infected + hiv_infected) / total_pop
    )  # force of infection for adults

    return foi


def inf_tb(inds):
    # apply foi to uninfected pop -> latent infection

    return inds


def tb_treatment(inds):
    # apply diagnosis / treatment / self-cure combined rates

    return inds


def progression_tb(inds):
    # apply combined progression / relapse / reinfection rates to infected pop

    return inds


def recover_tb(inds):
    # apply combined diagnosis / treatment / self-cure rates to TB cases

    return inds


# TODO: isoniazid preventive therapy
# TODO: rifampicin / alternative TB treatment
