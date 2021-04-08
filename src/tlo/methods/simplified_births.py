"""This is the SimplifiedBirths Module. It aims causes pregnancy, deliveries and births to occur to match WPP estimates
 of total births. It subsumes the functions of several other modules (contraception, labour, pregnant supervisor,
 postnatal supervisor, newborn outcomes) , allowing for faster runnning when these are not required. The main assumption
 is that every pregnancy results in a birth."""

import pandas as pd
from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import PopulationScopeEventMixin, RegularEvent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SimplifiedBirths(Module):
    """
    A simplified births module responsible for generating births in a simplified way and assign mother ids to newborns.
    """

    METADATA = {}

    PARAMETERS = {
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
                                           'Date of the onset of the last pregnancy of this individual (if has ever '
                                           'been pregnant).'),

        # (Internal property)
        'si_date_of_last_delivery': Property(Types.DATE,
                                             'Date of delivery for the most recent pregnancy for this individual (if '
                                             'has ever been pregnant). Maybe in the future if is currently pregnant.'),

        # (Property usually managed by Newborn_outcomes module)
        'nb_breastfeeding_status': Property(Types.CATEGORICAL,
                                            'Breastfeeding status of a newborn',
                                            categories=['none', 'non_exclusive', 'exclusive']),
    }

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.asfr = dict()

    def read_parameters(self, data_folder):
        """Load parameters for probability of pregnancy/birth and breastfeeding status for newborns"""

        # Read in the data and format for quick look-up in simulation in self.asfr (dict(year, mapping-for-age-ranges))
        dat = pd.read_csv(self.resourcefilepath / 'ResourceFile_ASFR_WPP.csv')
        dat['Period-Start'] = dat['Period'].str.split('-').str[0].astype(int)
        dat['Period-End'] = dat['Period'].str.split('-').str[1].astype(int)

        years = range(min(dat['Period-Start'].values), 1 + max(dat['Period-End'].values))

        for year in years:
            self.asfr[year] = dat.loc[
                (year >= dat['Period-Start']) & (year <= dat['Period-End'])
                ].set_index('Age_Grp')['asfr'].to_dict()

        # Specifiy parameters
        self.parameters['months_between_pregnancy_and_delivery'] = 9

        # Breastfeeding status for newborns; #todo - @mnjowe: say what is source of these parameters? and access them
        #  ... from a resourcefile directly rather than hard-coding
        self.parameters['prob_breastfeeding_type'] = [0.101, 0.289, 0.61]
        assert 1.0 == sum(self.parameters['prob_breastfeeding_type'])

    def initialise_population(self, population):
        """Set our property values for the initial population."""
        df = population.props

        df.loc[df.is_alive, 'is_pregnant'] = False
        df.loc[df.is_alive, 'date_of_last_pregnancy'] = pd.NaT
        df.loc[df.is_alive, 'si_date_of_last_delivery'] = pd.NaT
        df.loc[df.is_alive, 'nb_breastfeeding_status'] = 'none'

    def initialise_simulation(self, sim):
        """Schedule the PregnancyEvent and the SimplifiedBirthEvent to occur every month."""
        sim.schedule_event(PregnancyEvent(self), sim.date)
        sim.schedule_event(BirthsEvent(self), sim.date)

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


class PregnancyEvent(RegularEvent, PopulationScopeEventMixin):
    """A event for making women pregnant. Rate of doing so is based on age-specific fertility rates under assumption
    that every pregnancy results in a birth."""

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        self.pregnancy_prob = module.parameters['pregnancy_prob']
        self.asfr = self.module.asfr

    def apply(self, population):
        df = self.sim.population.props  # get the population dataframe

        # find probability of becoming pregnant (using asfr for the year, limiting to alive, non-pregnant females)
        prob_preg = df.loc[
            (df.sex == 'F') & df.is_alive & ~df.is_pregnant
            ]['age_range'].map(self.asfr[self.sim.date.year]).fillna(0)

        # determine which woman will get pregnant
        pregnant_women_ids = prob_preg.index[
            (self.module.rng.random_sample(size=len(prob_preg)) < prob_preg)
        ]

        # updating properties for women who will get pregnant
        df.loc[pregnant_women_ids, 'is_pregnant'] = True
        df.loc[pregnant_women_ids, 'date_of_last_pregnancy'] = self.sim.date
        df.loc[pregnant_women_ids, 'si_date_of_last_delivery'] = \
            self.sim.date + pd.DateOffset(months=self.module.parameters['months_between_pregnancy_and_delivery'])


class BirthsEvent(RegularEvent, PopulationScopeEventMixin):
    """This event checks to see if the date-of-delivery for pregnant women has been reached and implement births where
    appropriate."""

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
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
