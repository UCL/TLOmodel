import numpy as np
import pandas as pd

from tlo import Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent
from tlo.logging import INFO, getLogger

logger = getLogger("tlo.testing.loggernaires")
logger.setLevel(INFO)


class LoggerNaires(Module):
    PARAMETERS = {
        'test': Parameter(Types.REAL, 'this is a test')
    }

    PROPERTIES = {
        'ln_a': Property(Types.REAL, 'numeric a'),
        'ln_b': Property(Types.REAL, 'numeric b'),
        'ln_c': Property(Types.REAL, 'numeric c'),
        'ln_date': Property(Types.DATE, 'date'),

    }

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)

    def on_birth(self, mother, child):
        pass

    def read_parameters(self, data_folder):
        pass

    def initialise_simulation(self, sim):
        sim.schedule_event(MockLogEvent(self), sim.date + pd.DateOffset(months=0))

    def initialise_population(self, population):
        df = population.props
        for name, _type in self.PROPERTIES.items():
            if name == "ln_date":
                df[name] = self.rng.randint(1400, 1600, population.initial_size) * 1_000_000_000_000_000
                df[name] = df[name].astype('datetime64[ns]')
            else:
                df[name] = self.rng.randint(0, 100, population.initial_size)


class MockLogEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        super().__init__(module, frequency=pd.DateOffset(weeks=4))

    def apply(self, population):
        df = population.props
        a_over_50 = sum(df.ln_a > 50)
        b_over_50 = sum(df.ln_b > 50)
        c_over_50 = sum(df.ln_c > 50)

        # the preferred way to log, because it maps naturally to a row in a dataframe
        logger.info(key="each_group_over_50_unscaled",
                    data={"count_a_over_50": a_over_50, "count_b_over_50": b_over_50,
                          "count_c_over_50": c_over_50 / 3},
                    description="count over 50 for each group")

        logger.info(key="each_group_over_50_scaled",
                    data={"count_a_over_50": a_over_50, "count_b_over_50": b_over_50,
                          "count_c_over_50": c_over_50 / 3},
                    description="count over 50 for each group; a and b are raw numbers, c is normalised",
                    scale_me=['count_a_over_50', 'count_c_over_50'])

        logger.info(key="a_fixed_length_list",
                    data=[a_over_50 / 2, b_over_50 / 3, c_over_50 / 4],
                    description="divide a, b, c by 2, 3, 4 respectively")

        logger.info(key="a_variable_length_list",
                    data={"list_head": list(df.loc[0:self.module.rng.randint(2, 8), "ln_a"])},
                    description="the first few interesting items from property a, random selection")

        logger.info(key="counting_but_string",
                    data="we currently have %d total count over 50" % (a_over_50 + b_over_50 + c_over_50),
                    description="total count of loggernaires over 50, but as a string")

        logger.info(key="single_individual",
                    data=df.loc[[0]],
                    description="entire record for person 0")

        logger.info(key="three_people",
                    data=df.loc[[0, 1, 2]],
                    description="three people (0-2, inclusive), flattened to single row")

        logger.info(key="nested_dictionary",
                    data={
                        "count_over_50":
                            {"a": a_over_50,
                             "b": b_over_50,
                             "c": c_over_50 / 3
                             },
                    },
                    description="count over 50 for each group")

        logger.info(key="set_in_dict",
                    data={"count_over_50": set([a_over_50, b_over_50, c_over_50 / 3])},
                    description="count over 50 for each group")

        logger.info(key="with_every_type",
                    data={"a_over_50": a_over_50,
                          "mostly_nan": np.nan,
                          "c_over_50_div_2": c_over_50 / 2,
                          "b_over_50_as_list": [b_over_50],
                          "random_date": df.loc[self.module.rng.randint(0, len(df)), "ln_date"],
                          "count_over_50_as_dict": {"a": a_over_50, "b": b_over_50, "c": c_over_50 / 3},
                          "count_over_50_as_set": {a_over_50, b_over_50, c_over_50 / 3}
                          },
                    description="including a little bit of everything, columns have different types")
