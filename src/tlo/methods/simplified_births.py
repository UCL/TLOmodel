"""This is the SimplifiedBirths Module. It aims causes pregnancy, deliveries and births to occur to match WPP estimates
 of total births. It subsumes the functions of several other modules (contraception, labour, pregnant supervisor,
 postnatal supervisor, newborn outcomes) , allowing for faster runnning when these are not required. The main assumption
 is that every pregnancy results in a birth."""

import json

import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import PopulationScopeEventMixin, RegularEvent
from tlo.methods.contraception import get_medium_variant_asfr_from_wpp_resourcefile

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SimplifiedBirths(Module):
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
        'age_specific_fertility_rates': Parameter(
            Types.DATA_FRAME, 'Data table from official source (WPP) for age-specific fertility rates and calendar '
                              'period'),
        'months_between_pregnancy_and_delivery': Parameter(
            Types.INT, 'number of whole months that elapase betweeen pregnancy and delivery'),
        'prob_breastfeeding_type': Parameter(
            Types.LIST, 'probabilities that a woman is: 1) not breastfeeding (none); 2) non-exclusively breastfeeding '
                        '(non_exclusive); 3)exclusively breastfeeding at birth (until 6 months) (exclusive)')
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
                                            'How this neonate is being breastfed currently',
                                            categories=['none', 'non_exclusive', 'exclusive']),
        # (Internal property)
        'si_breastfeeding_status_6mo_to_23mo': Property(Types.CATEGORICAL,
                                                        'How this neonate is breastfeed during ages 6mo to 23 months',
                                                        categories=['none', 'non_exclusive', 'exclusive']),

        # (Property usually managed by Newborn_outcomes module)
        'nb_early_init_breastfeeding': Property(Types.BOOL,
                                                'whether this neonate initiated breastfeeding within 1 hour of birth'
                                                'This is always FALSE in this module.')
    }

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.asfr = dict()

    def read_parameters(self, data_folder):
        """Load parameters for probability of pregnancy/birth and breastfeeding status for newborns"""

        self.parameters['age_specific_fertility_rates'] = \
            pd.read_csv(self.resourcefilepath / 'demography' / 'ResourceFile_ASFR_WPP.csv')

        self.parameters['months_between_pregnancy_and_delivery'] = 9

        # Breastfeeding status for newborns (importing from the Newborn resourcefile)
        rf = pd.read_excel(self.resourcefilepath / 'ResourceFile_NewbornOutcomes.xlsx')
        param_as_string = rf.loc[rf.parameter_name == 'prob_breastfeeding_type']['value'].iloc[0]
        parameter = json.loads(param_as_string)[0]
        self.parameters['prob_breastfeeding_type'] = parameter

    def initialise_population(self, population):
        """Set our property values for the initial population."""
        df = population.props

        df.loc[df.is_alive, 'is_pregnant'] = False
        df.loc[df.is_alive, 'date_of_last_pregnancy'] = pd.NaT
        df.loc[df.is_alive, 'si_date_of_last_delivery'] = pd.NaT
        df.loc[df.is_alive, 'nb_breastfeeding_status'] = 'none'
        df.loc[df.is_alive, 'si_breastfeeding_status_6mo_to_23mo'] = 'none'
        df.loc[df.is_alive, 'nb_early_init_breastfeeding'] = False

    def initialise_simulation(self, sim):
        """Schedule the SimplifiedBirthsPoll and the SimplifiedBirthEvent to occur every month."""
        sim.schedule_event(SimplifiedBirthsPoll(self), sim.date)

        # Check that the parameters loaded are ok
        assert 1.0 == sum(self.parameters['prob_breastfeeding_type'])

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual."""
        df = self.sim.population.props
        params = self.parameters

        # Determine breastfeeding characteristics
        initial_breastfeeding_status = 'none'

        # Assign breastfeeding status to newborns:

        # 1) Newborn modules assigned initial status probablistically from ['prob_breastfeeding_type']
        initial_breastfeeding_status = self.rng.choice(
            ('none', 'non_exclusive', 'exclusive'), p=params['prob_breastfeeding_type']
        )

        # 2) Breastfeeding for those aged 6-23months: those who were initially breastfed at all, switch down to
        #  either non_exclusive or none (equal probability of each)
        if initial_breastfeeding_status == 'none':
            breastfeeding_status_6mo_to_23mo = 'none'
        else:
            breastfeeding_status_6mo_to_23mo = self.rng.choice(
                ('none', 'non_exclusive'), p=[0.5, 0.5]
            )

        # Assign properties for the newborns
        df.at[child_id, ['is_pregnant',
                         'date_of_last_pregnancy',
                         'si_date_of_last_delivery',
                         'nb_breastfeeding_status',
                         'si_breastfeeding_status_6mo_to_23mo',
                         'nb_early_init_breastfeeding']] = (
            False,
            pd.NaT,
            pd.NaT,
            initial_breastfeeding_status,
            breastfeeding_status_6mo_to_23mo,
            False
        )


class SimplifiedBirthsPoll(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        self.months_between_polls = 1
        super().__init__(module, frequency=DateOffset(months=self.months_between_polls))
        self.asfr = get_medium_variant_asfr_from_wpp_resourcefile(
            dat=self.module.parameters['age_specific_fertility_rates'], months_exposure=self.months_between_polls)

    def apply(self, population):
        # Set new pregnancies:
        self.set_new_pregnancies()

        # Do the delivery
        self.do_deliveries()

        # Update breastfeeding status at six months
        self.update_breastfed_status()

    def set_new_pregnancies(self):
        """Making women pregnant. Rate of doing so is based on age-specific fertility rates under assumption that every
        pregnancy results in a birth."""

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
        """Update the bread_fed status of newborns, mirroing the functionality provided by the Newborn module"""
        df = self.sim.population.props

        # 1) Update for those aged 6-23 months (set probabilistically at on_birth)
        aged_6mo_to_23mo = df.is_alive & (df.age_exact_years >= 0.5) & (df.age_exact_years < 2.0)

        df.loc[aged_6mo_to_23mo, 'nb_breastfeeding_status'] = \
            df.loc[aged_6mo_to_23mo, 'si_breastfeeding_status_6mo_to_23mo']

        # 2) Update for those aged 24+ months ('none' for all, per the Newborn module)
        df.loc[df.is_alive & (df.age_exact_years >= 2.0), 'nb_breastfeeding_status'] = 'none'
