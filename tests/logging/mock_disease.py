from pandas import DateOffset, np

from tlo import Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent
from tlo.logging import INFO, getLogger

logger = getLogger("tlo.testing.MockModule")
logger.setLevel(INFO)


class MockModule(Module):
    PARAMETERS = {
        'test': Parameter(Types.REAL, 'this is a test')
    }

    PROPERTIES = {
        'mm_a': Property(Types.REAL, 'numeric a'),
        'mm_b': Property(Types.REAL, 'numeric b'),
        'mm_c': Property(Types.REAL, 'numeric c'),
        'mm_date': Property(Types.DATE, 'date'),

    }

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)

    def on_birth(self, mother, child):
        pass

    def read_parameters(self, data_folder):
        pass

    def initialise_simulation(self, sim):
        sim.schedule_event(MockLogEvent(self), sim.date + DateOffset(months=0))

    def initialise_population(self, population):
        df = population.props
        for name, _type in self.PROPERTIES.items():
            df[name] = self.rng.randint(0, 100, population.initial_size)


class MockLogEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(weeks=4))

    def apply(self, population):
        df = population.props
        a_over_50 = sum(df.mm_a > 50)
        b_over_50 = sum(df.mm_b > 50)
        c_over_50 = sum(df.mm_c > 50)

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
                    data={"list_head": list(df.loc[0:self.module.rng.randint(2, 8), "mm_a"])},
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

        logger.info(key="with_every_type",
                    data={"a_over_50": a_over_50,
                          "mostly_nan": np.nan,
                          "c_over_50_div_2": c_over_50 / 2,
                          "b_over_50_as_list": [b_over_50],
                          "random_date": df.loc[self.module.rng.randint(0, len(df)), "mm_date"]
                          },
                    description="including a little bit of everything, columns have different types")
