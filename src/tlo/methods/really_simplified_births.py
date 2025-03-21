"""This is the ReallySimplifiedBirths Module. It aims causes pregnancy, deliveries and births to occur to
balance deaths that have occurred. It subsumes the functions of several other modules (contraception, labour, pregnant supervisor,
 postnatal supervisor, newborn outcomes) , allowing for faster runnning when these are not required. The main assumption
 is that every pregnancy results in a birth.

 It is called through do_death() in Demography module"""

import json
from pathlib import Path

import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import PopulationScopeEventMixin, RegularEvent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ReallySimplifiedBirths(Module):
    """
    A simplified births module responsible for generating births in a simplified way and assign mother ids to newborns.
    """
    INIT_DEPENDENCIES = {'Demography'}

    ALTERNATIVE_TO = {
        'Contraception',
        'Labour',
        'NewbornOutcomes',
        'PostnatalSupervisor',
        'PregnancySupervisor'
    }

    METADATA = {}

    PARAMETERS = {
        'months_between_pregnancy_and_delivery': Parameter(
            Types.INT, 'number of whole months that elapse between pregnancy and delivery'),
        'prob_breastfeeding_type': Parameter(
            Types.LIST, 'probabilities that a woman is: 1) not breastfeeding (none); 2) non-exclusively breastfeeding '
                        '(non_exclusive); 3)exclusively breastfeeding at birth (until 6 months) (exclusive)')
    }

    PROPERTIES = {
        # (Internal property)
        'si_date_of_last_delivery': Property(Types.DATE,
                                             'Date of delivery for the most recent pregnancy for this individual (if '
                                             'has ever been pregnant). Maybe in the future if is currently pregnant.'),

        # (Internal property)
        'si_breastfeeding_status_6mo_to_23mo': Property(Types.CATEGORICAL,
                                                        'How this neonate is breastfeed during ages 6mo to 23 months',
                                                        categories=['none', 'non_exclusive', 'exclusive']),

        # (Mocked property, usually handled by Contraception module)
        'is_pregnant': Property(Types.BOOL,
                                'Whether this individual is currently pregnant'),

        # (Mocked property, usually handled by Contraception module)
        'date_of_last_pregnancy': Property(Types.DATE,
                                           'Date of the onset of the last pregnancy of this individual (if has ever '
                                           'been pregnant).'),

        # (Mocked property, usually managed by Newborn_outcomes module)
        'nb_low_birth_weight_status': Property(Types.CATEGORICAL, 'temporary property',
                                               categories=['extremely_low_birth_weight', 'very_low_birth_weight',
                                                           'low_birth_weight', 'normal_birth_weight', 'macrosomia']),

        # (Mocked property, managed by Newborn_outcomes module)
        'nb_size_for_gestational_age': Property(Types.CATEGORICAL, 'temporary property',
                                                categories=['small_for_gestational_age', 'average_for_gestational_age',
                                                            'large_for_gestational_age']),

        # (Mocked property, usually managed by Newborn_outcomes module)
        'nb_late_preterm': Property(Types.BOOL, 'temporary property'),

        # (Mocked property, usually managed by Newborn_outcomes module)
        'nb_early_preterm': Property(Types.BOOL, 'temporary property'),

        # (Mocked property, usually managed by Newborn_outcomes module)
        'nb_breastfeeding_status': Property(Types.CATEGORICAL, 'temporary property',
                                            categories=['none', 'non_exclusive', 'exclusive']),
    }

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.asfr = dict()

        # Define defaults for properties:
        self.default_properties = {
            'si_date_of_last_delivery': pd.NaT,
            'si_breastfeeding_status_6mo_to_23mo': 'none',
            'is_pregnant': False,
            'date_of_last_pregnancy': pd.NaT,
            'nb_low_birth_weight_status': 'normal_birth_weight',
            'nb_size_for_gestational_age': 'average_for_gestational_age',
            'nb_late_preterm': False,
            'nb_early_preterm': False,
            'nb_breastfeeding_status': 'none',
        }

    def read_parameters(self, data_folder):
        """Load parameters for probability of pregnancy/birth and breastfeeding status for newborns"""

        self.parameters['months_between_pregnancy_and_delivery'] = 9

        # Breastfeeding status for newborns (importing from the Newborn resourcefile)
        rf = pd.read_csv(Path(self.resourcefilepath) / 'ResourceFile_NewbornOutcomes/parameter_values.csv')
        param_as_string = rf.loc[rf.parameter_name == 'prob_breastfeeding_type']['value'].iloc[0]
        parameter = json.loads(param_as_string)[0]
        self.parameters['prob_breastfeeding_type'] = parameter

    def initialise_population(self, population):
        """Set property values to their defaults for the initial population."""
        df = population.props
        df.loc[df.is_alive, self.default_properties.keys()] = self.default_properties.values()

    def initialise_simulation(self, sim):
        """Schedule the SimplifiedBirthsPoll and the SimplifiedBirthEvent to occur every month."""
        sim.schedule_event(SimplifiedBirthsPoll(self), sim.date)

        # Check that the parameters loaded are ok
        assert 1.0 == sum(self.parameters['prob_breastfeeding_type'])

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual to their defaults."""
        df = self.sim.population.props
        params = self.parameters

        # Mock 'nb_breastfeeding_status' by:
        # (i) having an initial value sampled probabilistically
        # (ii) storing the status for 6-23mo and updating this later in the `SimplifiedBirthPoll`. (Those who were
        # initially breastfed at all, switch down to either non_exclusive or none (equal probability of each)).

        initial_breastfeeding_status = self.rng.choice(
            ('none', 'non_exclusive', 'exclusive'), p=params['prob_breastfeeding_type']
        )

        breastfeeding_status_6mo_to_23mo = 'none' if (initial_breastfeeding_status == 'none') \
            else self.rng.choice(('none', 'non_exclusive'), p=[0.5, 0.5])

        # Other properties will be set to their defaults
        properties = {**self.default_properties,
                      **{
                          'nb_breastfeeding_status': initial_breastfeeding_status,
                          'si_breastfeeding_status_6mo_to_23mo': breastfeeding_status_6mo_to_23mo,
                          }
                      }
        df.loc[child_id, properties.keys()] = properties.values()

    def make_pregnancy(self):
        # find one eligible woman who is not currently pregnant

        df = self.sim.population.props  # get the population dataframe

        eligible = df.loc[
            (df.sex == 'F') & df.is_alive & ~df.is_pregnant & df.age_years.between(15, 49)
            ].index

        new_pregnancy = self.rng.choice(eligible, size=1)[0]

        # updating properties for women who will get pregnant
        df.at[new_pregnancy, 'is_pregnant'] = True
        df.at[new_pregnancy, 'date_of_last_pregnancy'] = self.sim.date
        df.at[new_pregnancy, 'si_date_of_last_delivery'] = \
            self.sim.date + pd.DateOffset(months=self.parameters['months_between_pregnancy_and_delivery'])


class SimplifiedBirthsPoll(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        self.months_between_polls = 1
        super().__init__(module, frequency=DateOffset(months=self.months_between_polls))

    def apply(self, population):

        # Do the delivery
        self.do_deliveries()

        # Update breastfeeding status at six months
        self.update_breastfed_status()

    def do_deliveries(self):
        """Checks to see if the date-of-delivery for pregnant women has been reached and implement births where
        appropriate."""

        df = self.sim.population.props  # get the population dataframe

        # find the women who are due to have delivered their babies before now
        females_to_give_birth = df.loc[
            (df.sex == 'F') &
            df.is_alive &
            df.is_pregnant &
            (df.si_date_of_last_delivery <= self.sim.date)
        ].index

        if len(females_to_give_birth) > 0:
            # update properties in df:
            df.loc[females_to_give_birth, 'is_pregnant'] = False

            # do the births:
            for mother_id in females_to_give_birth:
                self.sim.do_birth(mother_id)

    def update_breastfed_status(self):
        """Update the bread_fed status of newborns, mirroring the functionality provided by the Newborn module"""
        df = self.sim.population.props

        # 1) Update for those aged 6-23 months (set probabilistically at on_birth)
        aged_6mo_to_23mo = df.is_alive & (df.age_exact_years >= 0.5) & (df.age_exact_years < 2.0)

        df.loc[aged_6mo_to_23mo, 'nb_breastfeeding_status'] = \
            df.loc[aged_6mo_to_23mo, 'si_breastfeeding_status_6mo_to_23mo']

        # 2) Update for those aged 24+ months ('none' for all, per the Newborn module)
        df.loc[df.is_alive & (df.age_exact_years >= 2.0), 'nb_breastfeeding_status'] = 'none'
