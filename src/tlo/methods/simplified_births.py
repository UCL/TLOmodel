"""This is the SimplifiedBirths Module. It aims causes pregnancy, deliveries and births to occur to match WPP estimates of
total births. It subsumes the functions of several other modules (contraception, labour, pregnant supervisor, postnatal
supervisor, newborn outcomes) , allowing for faster runnning when these are not required."""

import pandas as pd
from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import PopulationScopeEventMixin, RegularEvent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Simplifiedbirths(Module):
    """
    A simplified births module responsible for generating births in a simplified way and assign mother ids to newborns.
    """

    METADATA = {}

    PARAMETERS = {
        'pregnancy_prob': Parameter(
            Types.REAL, 'probability of pregnancies in a month'),
        'min_age': Parameter(
            Types.REAL, 'minimum age of individuals who are at risk of becoming pregnant'),
        'max_age': Parameter(
            Types.REAL, 'maximum age of individuals who are at risk of becoming pregnant'),
        'months_between_pregnancy_and_delivery': Parameter(
            Types.INT, 'number of whole months that elapase betweeen pregnancy and delivery'),
        'prob_breastfeeding_type': Parameter(
            Types.LIST, 'probabilities for breastfeeding status: none, non_exclusive, exclusive')
    }

    PROPERTIES = {
        # (Core property, usually handled by Contraception module)
        'is_pregnant': Property(Types.BOOL,
                                'Whether this individual is currently pregnant'),

        # (Core property, usually handled by Contraception module)
        'date_of_last_pregnancy': Property(Types.DATE,
                                           'Date of the onset of the last pregnancy of this individual '
                                           '(if has ever been pregnant).'),

        # (Internal property)
        'si_date_of_last_delivery': Property(Types.DATE,
                                        'Date of delivery for the most recent pregnancy for this individual '
                                        '(if has ever been pregnant). Maybe in the future if is currently pregnant.'),

        # (Property usually managed by Newborn_outcomes module)
        'nb_breastfeeding_status': Property(Types.CATEGORICAL,
                                            'Breastfeeding status of a newborn',
                                            categories=['none', 'non_exclusive', 'exclusive']),
    }

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    def read_parameters(self, data_folder):
        """Load parameters for probability of pregnancy/birth and breastfeeding status for newborns"""

        param = self.parameters
        param['pregnancy_prob'] = 0.01
        param['min_age'] = 15
        param['max_age'] = 49

        param['months_between_pregnancy_and_delivery'] = 9

        # Breastfeeding status for newborns; #todo - @mnjowe: what is source of these parameters?
        param['prob_breastfeeding_type'] = [0.101, 0.289, 0.61]

    def initialise_population(self, population):
        """Set our property values for the initial population."""
        df = population.props

        df.loc[df.is_alive, 'is_pregnant'] = False
        df.loc[df.is_alive, 'date_of_last_pregnancy'] = pd.NaT
        df.loc[df.is_alive, 'si_date_of_last_delivery'] = pd.NaT
        df.loc[df.is_alive, 'nb_breastfeeding_status'] = 'none'

    def initialise_simulation(self, sim):
        """Schedule the SimplifiedPregnancyEvent and the SimplifiedBirthEvent to occur every month."""
        sim.schedule_event(SimplifiedPregnancyEvent(self), sim.date)
        sim.schedule_event(SimplifiedBirthsEvent(self), sim.date)

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual."""
        df = self.sim.population.props
        params = self.parameters

        # Assign the date of last delivery to a newborn
        df.at[child_id, 'is_pregnant'] = False
        df.at[child_id, 'date_of_last_pregnancy'] = pd.NaT
        df.at[child_id, 'si_date_of_last_delivery'] = pd.NaT

        # Assign breastfeeding status to newborns
        df.at[child_id, 'nb_breastfeeding_status'] = self.rng.choice(
            ('none', 'non_exclusive', 'exclusive'), p=params['prob_breastfeeding_type']
        )


class SimplifiedPregnancyEvent(RegularEvent, PopulationScopeEventMixin):
    """ A event for making women pregnant"""

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        self.age_low = module.parameters['min_age']  # min preferred age
        self.age_high = module.parameters['max_age']  # max preferred age
        self.pregnancy_prob = module.parameters['pregnancy_prob']

    def apply(self, population):
        df = self.sim.population.props  # get the population dataframe

        # select females from dataframe who are not pregnant and of age between 15 - 49
        selected_women = df.loc[(df.sex == 'F') &
                                df.is_alive &
                                ~df.is_pregnant &
                                df.age_years.between(self.age_low, self.age_high)
                                ]

        # determine which woman will get pregnant
        pregnant_women_ids = selected_women.index[
            (self.module.rng.random_sample(size=len(selected_women.index)) < self.pregnancy_prob)
        ]

        # updating properties for women who will get pregnant
        df.loc[pregnant_women_ids, 'is_pregnant'] = True
        df.loc[pregnant_women_ids, 'date_of_last_pregnancy'] = self.sim.date
        df.loc[pregnant_women_ids, 'si_date_of_last_delivery'] = \
            self.sim.date + pd.DateOffset(months=self.module.parameters['months_between_pregnancy_and_delivery'])


class SimplifiedBirthsEvent(RegularEvent, PopulationScopeEventMixin):
    """This event checks to see if the date-of-delivery for pregnant women has been reached and implement births where
    appropriate."""

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        df = self.sim.population.props  # get the population dataframe

        # find the women who are due to have delivered their babies before now
        females_to_give_birth = df.loc[
            (df.sex == 'F') & \
            df.is_alive &  \
            df.is_pregnant & \
            (df.si_date_of_last_delivery <= self.sim.date)
        ].index

        if len(females_to_give_birth) > 0:
            # update properties in df:
            df.loc[females_to_give_birth, 'is_pregnant'] = False

            # do the births:
            for mother_id in females_to_give_birth:
                self.sim.do_birth(mother_id)
