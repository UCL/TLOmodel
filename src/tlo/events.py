"""Support for creating different kinds of events."""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from tlo import DateOffset, logging

if TYPE_CHECKING:
    from tlo import Simulation

import pandas as pd

from tlo.util import FACTOR_POP_DICT

import copy

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger_chain = logging.getLogger('tlo.simulation')
logger_chain.setLevel(logging.INFO)

logger_summary = logging.getLogger(f"{__name__}.summary")
logger_summary.setLevel(logging.INFO)

debug_chains = True

class Priority(Enum):
    """Enumeration for the Priority, which is used in sorting the events in the simulation queue."""
    START_OF_DAY = 0
    FIRST_HALF_OF_DAY = 25
    LAST_HALF_OF_DAY = 75
    END_OF_DAY = 100

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class Event:
    """Base event class, from which all others inherit.

    Concrete subclasses should also inherit from one of the EventMixin classes
    defined below, and implement at least an `apply` method.
    """

    def __init__(self, module, *args, priority=Priority.FIRST_HALF_OF_DAY, **kwargs):
        """Create a new event.

        Note that just creating an event does not schedule it to happen; that
        must be done by calling Simulation.schedule_event.

        :param module: the module that created this event.
            All subclasses of Event take this as the first argument in their
            constructor, but may also take further keyword arguments.
        :param priority: a keyword-argument to set the priority (see Priority enum)
        """
        assert isinstance(priority, Priority), "priority argument should be a value from Priority enum"
        self.module = module
        self.sim = module.sim
        self.priority = priority
        self.target = None
        # This is needed so mixin constructors are called
        super().__init__(*args, **kwargs)

    def post_apply_hook(self):
        """Do any required processing after apply() completes."""

    def apply(self, target):
        """Apply this event to the given target.

        Must be implemented by subclasses.

        :param target: the target of the event
        """
        raise NotImplementedError
        
    def mni_values_differ(self, v1, v2):
    
        if isinstance(v1, list) and isinstance(v2, list):
            return v1 != v2  # simple element-wise comparison

        if pd.isna(v1) and pd.isna(v2):
            return False  # treat both NaT/NaN as equal
        return v1 != v2
        
    def compare_entire_mni_dicts(self,entire_mni_before, entire_mni_after):
        diffs = {}

        all_individuals = set(entire_mni_before.keys()) | set(entire_mni_after.keys())
            
        for person in all_individuals:
            if person not in entire_mni_before: # but is afterward
                for key in entire_mni_after[person]:
                    if self.mni_values_differ(entire_mni_after[person][key],self.sim.modules['PregnancySupervisor'].default_all_mni_values[key]):
                        if person not in diffs:
                            diffs[person] = {}
                        diffs[person][key] = entire_mni_after[person][key]
                    
            elif person not in entire_mni_after: # but is beforehand
                for key in entire_mni_before[person]:
                    if self.mni_values_differ(entire_mni_before[person][key],self.sim.modules['PregnancySupervisor'].default_all_mni_values[key]):
                        if person not in diffs:
                            diffs[person] = {}
                        diffs[person][key] = self.sim.modules['PregnancySupervisor'].default_all_mni_values[key]

            else: # person is in both
                # Compare properties
                for key in entire_mni_before[person]:
                    if self.mni_values_differ(entire_mni_before[person][key],entire_mni_after[person][key]):
                        if person not in diffs:
                            diffs[person] = {}
                        diffs[person][key] = entire_mni_after[person][key]

        return diffs
        
    def compare_population_dataframe_and_mni(self,df_before, df_after, entire_mni_before, entire_mni_after):
        """ This function compares the population dataframe and mni dictionary before/after a population-wide event has occurred.
        It allows us to identify the individuals for which this event led to a significant (i.e. property) change, and to store the properties which have changed as a result of it. """
        
        # Create a mask of where values are different
        diff_mask = (df_before != df_after) & ~(df_before.isna() & df_after.isna())
        diff_mni  = self.compare_entire_mni_dicts(entire_mni_before, entire_mni_after)
        
        # Create an empty list to store changes for each of the individuals
        chain_links = {}
        len_of_diff = len(diff_mask)

        # Loop through each row of the mask
        persons_changed = []
        
        for idx, row in diff_mask.iterrows():
            changed_cols = row.index[row].tolist()

            if changed_cols:  # Proceed only if there are changes in the row
                persons_changed.append(idx)
                # Create a dictionary for this person
                # First add event info
                link_info = {
                    'person_ID': idx,
                    'event': type(self).__name__,
                    'event_date': self.sim.date,
                }
                
                # Store the new values from df_after for the changed columns
                for col in changed_cols:
                    link_info[col] = df_after.at[idx, col]
 
                if idx in diff_mni:
                    # This person has also undergone changes in the mni dictionary, so add these here
                    for key in diff_mni[idx]:
                        link_info[col] = diff_mni[idx][key]
 
                # Append the event and changes to the individual key
                chain_links[idx] = str(link_info)
     
        # For individuals which only underwent changes in mni dictionary, save changes here
        if len(diff_mni)>0:
            for key in diff_mni:
                if key not in persons_changed:
                    # If individual hadn't been previously added due to changes in pop df, add it here
                    link_info = {
                        'person_ID': key,
                        'event': type(self).__name__,
                        'event_date': self.sim.date,
                    }
                    
                    for key_prop in diff_mni[key]:
                        link_info[key_prop] = diff_mni[key][key_prop]
                        
                    chain_links[key] = str(link_info)

        return chain_links
        
        
    def store_chains_to_do_before_event(self) -> tuple[bool, pd.Series, pd.DataFrame, dict, dict, bool]:
        """ This function checks whether this event should be logged as part of the event chains, and if so stored required information before the event has occurred. """

        # Initialise these variables
        print_chains = False
        df_before = []
        row_before = pd.Series()
        mni_instances_before = False
        mni_row_before = {}
        entire_mni_before = {}
        
        # Only print event if it belongs to modules of interest and if it is not in the list of events to ignore
        if all(sub not in str(self) for sub in self.sim.generate_event_chains_ignore_events):
        
        # Will eventually use this once I can actually GET THE NAME OF THE SELF
        #if not set(self.sim.generate_event_chains_ignore_events).intersection(str(self)):

            print_chains = True
            
            # Target is single individual
            if self.target != self.sim.population:
            
                # Save row for comparison after event has occurred
                row_before = self.sim.population.props.loc[abs(self.target)].copy().fillna(-99999)
                
                # Check if individual is already in mni dictionary, if so copy her original status
                mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
                if self.target in mni:
                    mni_instances_before = True
                    mni_row_before = mni[self.target].copy()
                
            else:

                # This will be a population-wide event. In order to find individuals for which this led to
                # a meaningful change, make a copy of the while pop dataframe/mni before the event has occurred.
                df_before = self.sim.population.props.copy()
                entire_mni_before = copy.deepcopy(self.sim.modules['PregnancySupervisor'].mother_and_newborn_info)
                
        return print_chains, row_before, df_before, mni_row_before, entire_mni_before, mni_instances_before
        
    def store_chains_to_do_after_event(self, row_before, df_before, mni_row_before, entire_mni_before, mni_instances_before) -> dict:
        """ If print_chains=True, this function logs the event and identifies and logs the any property changes that have occured to one or multiple individuals as a result of the event taking place. """
        
        chain_links = {}
    
        # Target is single individual
        if self.target != self.sim.population:
    
            # Copy full new status for individual
            row_after = self.sim.population.props.loc[abs(self.target)].fillna(-99999)
            
            # Check if individual is in mni after the event
            mni_instances_after = False
            mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
            if self.target in mni:
                mni_instances_after = True
            
            # Create and store event for this individual, regardless of whether any property change occurred
            link_info = {
                #'person_ID' : self.target,
                'person_ID' : self.target,
                'event' : type(self).__name__,
                'event_date' : self.sim.date,
            }
            
            # Store (if any) property changes as a result of the event for this individual
            for key in row_before.index:
                if row_before[key] != row_after[key]: # Note: used fillna previously, so this is safe
                    link_info[key] = row_after[key]
                    
            # Now check and store changes in the mni dictionary, accounting for following cases:
            # Individual is in mni dictionary before and after
            if mni_instances_before and mni_instances_after:
                for key in mni_row_before:
                    if self.mni_values_differ(mni_row_before[key], mni[self.target][key]):
                        link_info[key] = mni[self.target][key]
            # Individual is only in mni dictionary before event
            elif mni_instances_before and not mni_instances_after:
                default = self.sim.modules['PregnancySupervisor'].default_all_mni_values
                for key in mni_row_before:
                    if self.mni_values_differ(mni_row_before[key], default[key]):
                        link_info[key] = default[key]
            # Individual is only in mni dictionary after event
            elif mni_instances_after and not mni_instances_before:
                default = self.sim.modules['PregnancySupervisor'].default_all_mni_values
                for key in default:
                    if self.mni_values_differ(default[key], mni[self.target][key]):
                        link_info[key] = mni[self.target][key]
            # Else, no need to do anything
                    
            # Add individual to the chain links
            chain_links[self.target] = str(link_info)
            
        else:
            # Target is entire population. Identify individuals for which properties have changed
            # and store their changes.
            
            # Population frame after event
            df_after = self.sim.population.props
            entire_mni_after = copy.deepcopy(self.sim.modules['PregnancySupervisor'].mother_and_newborn_info)
            
            #  Create and store the event and dictionary of changes for affected individuals
            chain_links = self.compare_population_dataframe_and_mni(df_before, df_after, entire_mni_before, entire_mni_after)
                                        
        return chain_links


    def run(self):
        """Make the event happen."""
        
        # Collect relevant information before event takes place
        if self.sim.generate_event_chains:
            print_chains, row_before, df_before, mni_row_before, entire_mni_before, mni_instances_before = self.store_chains_to_do_before_event()
                
        self.apply(self.target)
        self.post_apply_hook()
        
        # Collect event info + meaningful property changes of individuals. Combined, these will constitute a 'link'
        # in the individual's event chain.
        if self.sim.generate_event_chains and print_chains:
            chain_links = self.store_chains_to_do_after_event(row_before, df_before, mni_row_before, entire_mni_before, mni_instances_before)
            
            # Create empty logger for entire pop
            pop_dict = {i: '' for i in range(FACTOR_POP_DICT)} # Always include all possible individuals
            pop_dict.update(chain_links)

            # Log chain_links here
            if len(chain_links)>0:
                print(chain_links)
                logger_chain.info(key='event_chains',
                                  data= pop_dict,
                                  description='Links forming chains of events for simulated individuals')


class RegularEvent(Event):
    """An event that automatically reschedules itself at a fixed frequency."""

    def __init__(self, module, *, frequency, end_date=None, **kwargs):
        """Create a new regular event.

        :param module: the module that created this event
        :param frequency: the interval from one occurrence to the next
            (must be supplied as a keyword argument)
        :type frequency: pandas.tseries.offsets.DateOffset
        """
        super().__init__(module, **kwargs)
        assert isinstance(frequency, DateOffset)
        self.frequency = frequency
        self.end_date = end_date

    def apply(self, target):
        """Apply this event to the given target.

        This is a no-op; subclasses should override this method.

        :param target: the target of the event
        """

    def post_apply_hook(self):
        """Schedule the next occurrence of this event."""
        next_apply_date = self.sim.date + self.frequency
        if not self.end_date or next_apply_date <= self.end_date:
            self.sim.schedule_event(self, next_apply_date)


class PopulationScopeEventMixin:
    """Makes an event operate on the entire population.

    This class is designed to be used via multiple inheritance along with one
    of the main event classes. It indicates that when an event happens, it is
    applied to the entire population, rather than a single individual.
    Contrast IndividualScopeEventMixin.

    Subclasses should implement `apply(self, population)` to contain their
    behaviour.
    """

    sim: Simulation

    def __init__(self, *args, **kwargs):
        """Create a new population-scoped event.

        This calls the base class constructor, passing any arguments through,
        and sets the event target as the whole population.
        """
        super().__init__(*args, **kwargs)
        self.target = self.sim.population

    def apply(self, population):
        """Apply this event to the population.

        Must be implemented by subclasses.

        :param population: the current population
        """
        raise NotImplementedError


class IndividualScopeEventMixin:
    """Makes an event operate on a single individual.

    This class is designed to be used via multiple inheritance along with one
    of the main event classes. It indicates that when an event happens, it is
    applied to a single individual, rather than the entire population.
    Contrast PopulationScopeEventMixin.

    Subclasses should implement `apply(self, person)` to contain their
    behaviour.
    """

    def __init__(self, *args, person_id, **kwargs):
        """Create a new individual-scoped event.

        This calls the base class constructor, passing any arguments through,
        and sets the event target as the provided person.

        :param person_id: the id of the person this event applies to
            (must be supplied as a keyword argument)
        """
        super().__init__(*args, **kwargs)
        self.target = person_id

    def apply(self, person_id):
        """Apply this event to the given person.

        Must be implemented by subclasses.

        :param person_id: the person the event happens to
        """
        raise NotImplementedError
