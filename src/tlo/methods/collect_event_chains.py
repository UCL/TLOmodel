from tlo.notify import notifier

from pathlib import Path
from typing import Optional
from tlo import Module, Parameter, Types, logging, population
from tlo.population import Population
import pandas as pd

from tlo.util import df_to_EAV, convert_chain_links_into_EAV, read_csv_files

import copy

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CollectEventChains(Module):

    def __init__(self, name=None):
        super().__init__(name)
    
        # This is how I am passing data from fnc taking place before event to the one after
        # It doesn't seem very elegant but not sure how else to go about it
        self.print_chains = False
        self.df_before = []
        self.row_before = pd.Series()
        self.mni_instances_before = False
        self.mni_row_before = {}
        self.entire_mni_before = {}
        
    PARAMETERS = {
        # Options within module
        "generate_event_chains": Parameter(
            Types.BOOL, "Whether or not we want to collect chains of events for individuals"
        ),
        "modules_of_interest": Parameter(
            Types.LIST, "Restrict the events collected to specific modules. If *, print for all modules"
        ),
        "events_to_ignore": Parameter(
            Types.LIST, "Events to be ignored when collecting chains"
        ),
        }
        
    def initialise_simulation(self, sim):
        notifier.add_listener("simulation.pop_has_been_initialised", self.on_notification_pop_has_been_initialised)
        notifier.add_listener("simulation.on_birth", self.on_notification_of_birth)
        notifier.add_listener("event.about_to_run", self.on_notification_event_about_to_run)
        notifier.add_listener("event.has_just_ran", self.on_notification_event_has_just_ran)
        
    def read_parameters(self, resourcefilepath: Optional[Path] = None):
        print("resource file path", resourcefilepath)
        self.load_parameters_from_dataframe(pd.read_csv(resourcefilepath/"ResourceFile_GenerateEventChains/parameter_values.csv"))

        # If modules of interest is '*', set by default to all modules included in the simulation
        if self.parameters["modules_of_interest"] == ['*']:
            self.parameters["modules_of_interest"] = list(self.sim.modules.keys())
        
    def initialise_population(self, population):
        pass

    def on_birth(self, mother, child):
        # Could the notification of birth simply take place here?
        pass
        
    def on_notification_pop_has_been_initialised(self, data):
        # When logging events for each individual to reconstruct chains, only the changes in individual properties will be logged.
        # At the start of the simulation + when a new individual is born, we therefore want to store all of their properties at the start.
        if self.parameters['generate_event_chains']:

            # EDNAV structure to capture status of individuals at the start of the simulation
            ednav = df_to_EAV(self.sim.population.props, self.sim.date, 'StartOfSimulation')

            logger.info(key='event_chains',
                               data = ednav.to_dict(),
                               description='Links forming chains of events for simulated individuals')
                               
                               
    def on_notification_of_birth(self, data):
                
        if self.parameters['generate_event_chains']:
            # When individual is born, store their initial properties to provide a starting point to the chain of property
            # changes that this individual will undergo as a result of events taking place.
            link_info = data['link_info']
            link_info.update(self.sim.population.props.loc[data['target']].to_dict())
            chain_links = {}
            chain_links[data['target']] = link_info

            ednav = convert_chain_links_into_EAV(chain_links)
            
            logger.info(key='event_chains',
                               data = ednav.to_dict(),
                               description='Links forming chains of events for simulated individuals')
                               
        
    def on_notification_event_about_to_run(self, data):
        """Do this when notified that an event is about to run. This function checks whether this event should be logged as part of the event chains, and if so stored required information before the event has occurred. """

        p = self.parameters

        # Only log event if
        # 1) generate_event_chains is set to True
        # 2) the event belongs to modules of interest and
        # 3) the event is not in the list of events to ignore
        if p['generate_event_chains'] and (data['module'] in p['modules_of_interest']) and (data['link_info']['EventName'] not in p['events_to_ignore']):
                      
            # Initialise these variables
            self.print_chains = False
            self.df_before = []
            self.row_before = pd.Series()
            self.mni_instances_before = False
            self.mni_row_before = {}
            self.entire_mni_before = {}
            
            self.print_chains = True
            
            # Target is single individual
            if not isinstance(data['target'], Population):

                # Save row for comparison after event has occurred
                self.row_before = self.sim.population.props.loc[abs(data['target'])].copy().fillna(-99999)
                
                # Check if individual is already in mni dictionary, if so copy her original status
                if 'PregnancySupervisor' in self.sim.modules:
                    mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
                    if data['target'] in mni:
                        self.mni_instances_before = True
                        self.mni_row_before = mni[data['target']].copy()
                else:
                    self.mni_row_before = None
                
            else:

                # This will be a population-wide event. In order to find individuals for which this led to
                # a meaningful change, make a copy of the while pop dataframe/mni before the event has occurred.
                self.df_before = self.sim.population.props.copy()
                if 'PregnancySupervisor' in self.sim.modules:
                    self.entire_mni_before = copy.deepcopy(self.sim.modules['PregnancySupervisor'].mother_and_newborn_info)
                else:
                    self.entire_mni_before = None

        return
        
    
    def on_notification_event_has_just_ran(self, data):
        """ If print_chains=True, this function logs the event and identifies and logs the any property changes that have occured to one or multiple individuals as a result of the event taking place. """

        p = self.parameters
        
        if p['generate_event_chains'] and self.print_chains:

            chain_links = {}
        
            # Target is single individual
            if not isinstance(data["target"], Population):
        
                # Copy full new status for individual
                row_after = self.sim.population.props.loc[abs(data['target'])].fillna(-99999)
                
                # Check if individual is in mni after the event
                mni_instances_after = False
                if 'PregnancySupervisor' in self.sim.modules:
                    mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
                    if data['target'] in mni:
                        mni_instances_after = True
                else:
                    mni_instances_after = None
                
                # Create and store event for this individual, regardless of whether any property change occurred
                link_info = data['link_info']
                
                # Store (if any) property changes as a result of the event for this individual
                for key in self.row_before.index:
                    if self.row_before[key] != row_after[key]: # Note: used fillna previously, so this is safe
                        link_info[key] = row_after[key]
                
                if 'PregnancySupervisor' in self.sim.modules:
                    # Now check and store changes in the mni dictionary, accounting for following cases:
                    # Individual is in mni dictionary before and after
                    if self.mni_instances_before and mni_instances_after:
                        for key in self.mni_row_before:
                            if self.mni_values_differ(self.mni_row_before[key], mni[data['target']][key]):
                                link_info[key] = mni[data['target']][key]
                    # Individual is only in mni dictionary before event
                    elif self.mni_instances_before and not mni_instances_after:
                        default = self.sim.modules['PregnancySupervisor'].default_all_mni_values
                        for key in self.mni_row_before:
                            if self.mni_values_differ(self.mni_row_before[key], default[key]):
                                link_info[key] = default[key]
                    # Individual is only in mni dictionary after event
                    elif mni_instances_after and not self.mni_instances_before:
                        default = self.sim.modules['PregnancySupervisor'].default_all_mni_values
                        for key in default:
                            if self.mni_values_differ(default[key], mni[data['target']][key]):
                                link_info[key] = mni[data['target']][key]
                    # Else, no need to do anything
                        
                # Add individual to the chain links
                chain_links[data['target']] = link_info
                
            else:
                # Target is entire population. Identify individuals for which properties have changed
                # and store their changes.
                
                # Population frame after event
                df_after = self.sim.population.props
                if 'PregnancySupervisor' in self.sim.modules:
                    entire_mni_after = copy.deepcopy(self.sim.modules['PregnancySupervisor'].mother_and_newborn_info)
                else:
                    entire_mni_after = None
                
                #  Create and store the event and dictionary of changes for affected individuals
                chain_links = self.compare_population_dataframe_and_mni(self.df_before, df_after, self.entire_mni_before, entire_mni_after)

                if chain_links:
                    # Convert chain_links into EAV
                    ednav = convert_chain_links_into_EAV(chain_links)

                    logger.info(key='event_chains',
                          data= ednav.to_dict(),
                          description='Links forming chains of events for simulated individuals')
                          
            # Reset variables
            self.print_chains = False
            self.df_before = []
            self.row_before = pd.Series()
            self.mni_instances_before = False
            self.mni_row_before = {}
            self.entire_mni_before = {}

        return
    
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
        if 'PregnancySupervisor' in self.sim.modules:
            diff_mni = self.compare_entire_mni_dicts(entire_mni_before, entire_mni_after)
        else:
            diff_mni = []
        
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
                    'EventName': type(self).__name__,
                }
                
                # Store the new values from df_after for the changed columns
                for col in changed_cols:
                    link_info[col] = df_after.at[idx, col]

                if idx in diff_mni:
                    # This person has also undergone changes in the mni dictionary, so add these here
                    for key in diff_mni[idx]:
                        link_info[col] = diff_mni[idx][key]

                # Append the event and changes to the individual key
                chain_links[idx] = link_info
     
        if 'PregnancySupervisor' in self.sim.modules:
            # For individuals which only underwent changes in mni dictionary, save changes here
            if len(diff_mni)>0:
                for key in diff_mni:
                    if key not in persons_changed:
                        # If individual hadn't been previously added due to changes in pop df, add it here
                        link_info = {
                            'EventName': type(self).__name__,
                        }
                        
                        for key_prop in diff_mni[key]:
                            link_info[key_prop] = diff_mni[key][key_prop]
                            
                        chain_links[key] = link_info

        return chain_links


        
