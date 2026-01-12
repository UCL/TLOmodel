import copy
from pathlib import Path
from typing import List, Optional

import pandas as pd

from tlo import Module, Parameter, Property, Types, logging
from tlo.notify import notifier
from tlo.population import Population

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class IndividualHistoryTracker(Module):

    def __init__(
        self,
        name: Optional[str] = None,
        modules_of_interest: Optional[List[str]] = None,
        events_to_ignore: Optional[List[str]] = None
    ):
        super().__init__(name)

        self.modules_of_interest = modules_of_interest
        self.events_to_ignore = events_to_ignore

        # This is how I am passing data from fnc taking place before event to the one after
        # It doesn't seem very elegant but not sure how else to go about it
        self.print_chains = False
        self.df_before = []
        self.row_before = pd.Series()
        self.mni_instances_before = False
        self.mni_row_before = {}
        self.entire_mni_before = {}
        self.consumable_access = {}
        self.cons_call_number_within_event = 0
        self.event_ID_in_sim = 1 # Initialise from 1 as the first event will be the start of the sim itself

    INIT_DEPENDENCIES = {"Demography"}

    PARAMETERS = {
        # Options within module
        "modules_of_interest": Parameter(
            Types.LIST, "Restrict the events collected to specific modules. If *, print for all modules"
        ),
        "events_to_ignore": Parameter(
            Types.LIST, "Events to be ignored when collecting chains"
        ),
    }
    
    PROPERTIES = {
        "iht_track_history": Property(Types.BOOL, "Whether the individual should be tracked by 
        "the individual history tracker or not")
    }

    def initialise_simulation(self, sim):
        notifier.add_listener("simulation.post-initialise", self.on_simulation_post_initialise)
        notifier.add_listener("simulation.post-do_birth", self.on_simulation_post_do_birth)
        notifier.add_listener("event.pre-run", self.on_event_pre_run)
        notifier.add_listener("event.post-run", self.on_event_post_run)
        notifier.add_listener("hsi_event.pre-run", self.on_event_pre_run)
        notifier.add_listener("hsi_event.post-run", self.on_event_post_run)
        notifier.add_listener("consumables.post-request_consumables", self.on_consumable_request)

    def read_parameters(self, resourcefilepath: Optional[Path] = None):
        self.load_parameters_from_dataframe(
            pd.read_csv(resourcefilepath/"ResourceFile_IndividualHistoryTracker/parameter_values.csv")
        )
        
    def df_to_eav(self, df, date, event_name):
        """Function to convert entire population dataframe into custom EAV"""
        eav = df.stack(dropna=False).reset_index()
        eav.columns = ['entity', 'attribute', 'value']
        eav['event_name'] = event_name
        eav['event_tag'] = 0 # First event
        eav = eav[["entity", "event_name", "event_tag", "attribute", "value"]]
        return eav


    def convert_chain_links_into_eav(self, chain_links):
        """Function to convert chain links into custom EAV"""
        rows = []

        for e, data in chain_links.items():
            event_name = data.get("event_name")
            event_tag  = self.event_ID_in_sim # access running counter
            
            for attr, val in data.items():
                if attr == "event_name" or attr == "event_tag":
                    continue
                
                rows.append({
                    "entity": e,
                    "event_name": event_name,
                    "event_tag": event_tag,
                    "attribute": attr,
                    "value": val
                })

        eav = pd.DataFrame(rows)
        
        return eav

    def initialise_population(self, population):
        # Use parameter file values by default, if not overwritten
        if self.modules_of_interest is None:
            self.modules_of_interest = self.parameters['modules_of_interest']

        if self.events_to_ignore is None:
            self.events_to_ignore = self.parameters['events_to_ignore']

        # If modules of interest is '*', set by default to all modules included in the simulation
        if self.modules_of_interest == ['*']:
            self.modules_of_interest = list(self.sim.modules.keys())
            
        # Initialise all individuals as being tracked by default
        pop = self.sim.population.props
        pop.loc[pop.is_alive, "iht_track_history"] = True

    def on_birth(self, mother, child):
        self.sim.population.props.at[child, "iht_track_history"] = True
        return
        
    def copy_of_pop_dataframe(self):
        df_copy = self.sim.population.props.copy()
        for col in df_copy.columns:
            df_copy[col] = df_copy[col].apply(
                lambda x: copy.deepcopy(x) if isinstance(x, (list, dict, pd.Series)) else x
            )
        return df_copy
    
    def copy_of_pop_dataframe_row(self, person_ID):
        copy_of_row = self.sim.population.props.loc[person_ID].copy()
        for col,val in copy_of_row.items():
            if isinstance(val, (list, dict, pd.Series)):
                copy_of_row[col] = copy.deepcopy(val)
        copy_of_row = copy_of_row.fillna(-99999)
        return copy_of_row
        
    def copy_of_mni(self):
        """Function to safely copy entire mni dictionary, ensuring that series attributes
        are safely copied too.
        """
        return copy.deepcopy(self.sim.modules['PregnancySupervisor'].mother_and_newborn_info)
    
    def copy_of_mni_row(self, person_ID):
        """Function to safely copy mni entry for single individual, ensuring that series attributes
        are safely copied too.
        """
        return copy.deepcopy(self.sim.modules['PregnancySupervisor'].mother_and_newborn_info[person_ID])
                    
    def log_eav_dataframe_to_individual_histories(self, df):
        for idx, row in df.iterrows():
            logger.info(key='individual_histories',
                               data = {
                                   "entity": row.entity,
                                   "attribute": row.attribute,
                                   "value": str(row.value),
                                   "event_name": row.event_name,
                                   "event_tag": row.event_tag
                               },
                               description='Links forming chains of events for simulated individuals')

    def on_simulation_post_initialise(self, data):
        # When logging events for each individual to reconstruct chains,
        # only the changes in individual properties will be logged.
        # At the start of the simulation + when a new individual is born,
        # we therefore want to store all of their properties
        # at the start.

        # EDNAV structure to capture status of individuals at the start of the simulation
        eav_plus_event = self.df_to_eav(self.sim.population.props, self.sim.date, 'StartOfSimulation')
        self.log_eav_dataframe_to_individual_histories(eav_plus_event)

    def on_simulation_post_do_birth(self, data):
        # When individual is born, store their initial properties to provide a starting point to the
        # chain of property changes that this individual will undergo
        # as a result of events taking place.
        link_info = {'event_name': 'Birth'}
        link_info.update(self.sim.population.props.loc[data['child_id']].to_dict())
        chain_links = {data['child_id']: link_info}

        eav_plus_event = self.convert_chain_links_into_eav(chain_links)
        self.log_eav_dataframe_to_individual_histories(eav_plus_event)

    def on_consumable_request(self,data):
        """Do this when notified that an individual has accessed consumables"""
        # Only log event if
        # 1) the event belongs to modules of interest and
        # 2) the event is not in the list of events to ignore
        if (data['module'] not in self.modules_of_interest) or (data['event_name'] in self.events_to_ignore):
            return
            
        # Copy this info for individual
        self.consumable_access[data['target']] = {
            ('ConsCall' + str(self.cons_call_number_within_event) + '_' + k): v
            for k, v in data.items() if k != 'target'}
            
        self.cons_call_number_within_event += 1
        return

    def on_event_pre_run(self, data):
        """Do this when notified that an event is about to run.
        This function checks whether this event should be logged as part of the event chains, a
        nd if so stored required information before the event has occurred.
        """

        # Create a unique identifier for this event to ensure events with the same name running on the same day
        # are not confused in the EAV log
        self.event_ID_in_sim += 1

        # Only log event if
        # 1) the event belongs to modules of interest and
        # 2) the event is not in the list of events to ignore
        if (data['module'] not in self.modules_of_interest) or (data['event_name'] in self.events_to_ignore):
            self.print_chains = False
            return

        # Initialise these variables
        self.df_before = []
        self.consumable_access = {}
        self.cons_call_number_within_event = 0
        self.row_before = pd.Series()
        self.mni_instances_before = False
        self.mni_row_before = {}
        self.entire_mni_before = {}
        self.print_chains = True

        # Target is single individual
        if not isinstance(data['target'], Population):
        
            # Save pop dataframe row for comparison after event has occurred
            self.row_before = self.copy_of_pop_dataframe_row(data['target'])
            pop = self.sim.population.props
            # Check if individual is already in mni dictionary, if so copy her original status
            if 'PregnancySupervisor' in self.sim.modules and (pop.loc[data['target'],'sex'] == 'F'):
                mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
                if data['target'] in mni:
                    self.mni_instances_before = True
                    self.mni_row_before = self.copy_of_mni_row(data['target'])
            else:
                self.mni_row_before = None
            
        else:

            # This will be a population-wide event. In order to find individuals for which this led to
            # a meaningful change, make a copy of the while pop dataframe/mni before the event has occurred.
            self.df_before = self.copy_of_pop_dataframe()
            if 'PregnancySupervisor' in self.sim.modules:
                self.entire_mni_before = self.copy_of_mni()
            else:
                self.entire_mni_before = None

    def on_event_post_run(self, data):
        """ If print_chains=True, this function logs the event and identifies and logs the any property
        changes that have occured to one or multiple individuals as a result of the event taking place.
        """

        if not self.print_chains:
            return

        chain_links = {}

        # Target is single individual
        if not isinstance(data['target'], Population):

            pop = self.sim.population.props

            # Copy full new status for individual
            row_after = self.copy_of_pop_dataframe_row(data['target'])

            # If individual qualified for the 'tracked' category either before OR
            # after the event occurred, the event will be logged:
            if self.row_before['iht_track_history'] or row_after['iht_track_history']:
            
                # Create and store event for this individual, regardless of whether any property change occurred
                link_info = {'event_name' : data['event_name']}
                if 'footprint' in data.keys():
                    HSI_specific_fields = {'footprint','level','treatment_ID','equipment','bed_days'}
                    for field in HSI_specific_fields:
                        link_info[field] = data[field]

                # Store (if any) property changes as a result of the event for this individual
                for key in self.row_before.index:
                    if self.row_before[key] != row_after[key]: # Note: used fillna previously, so this is safe
                        link_info[key] = row_after[key]

                # Check for any changes in mni dictionary
                if 'PregnancySupervisor' in self.sim.modules and pop.loc[data['target']].sex == 'F':
            
                    mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

                    # Check if individual is in mni after the event
                    mni_instances_after = False
                    if data['target'] in mni:
                        mni_instances_after = True
                
                    # Now check and store changes in the mni dictionary, accounting for following cases:
                    
                    # 1. Individual is not in mni neither before nor after event, can pass
                    if not self.mni_instances_before and not mni_instances_after:
                        pass
                    # 2. Individual is in mni dictionary before and after
                    if self.mni_instances_before and mni_instances_after:
                        for key in self.mni_row_before:
                            if self.mni_values_differ(self.mni_row_before[key], mni[data['target']][key]):
                                link_info[key] = mni[data['target']][key]
                    # 3. Individual is only in mni dictionary before event
                    elif self.mni_instances_before and not mni_instances_after:
                        default = self.sim.modules['PregnancySupervisor'].default_all_mni_values
                        for key in self.mni_row_before:
                            if self.mni_values_differ(self.mni_row_before[key], default[key]):
                                link_info[key] = default[key]
                    # 4. Individual is only in mni dictionary after event
                    elif mni_instances_after and not self.mni_instances_before:
                        default = self.sim.modules['PregnancySupervisor'].default_all_mni_values
                        for key in default:
                            if self.mni_values_differ(default[key], mni[data['target']][key]):
                                link_info[key] = mni[data['target']][key]

                # Add individual to the chain links
                chain_links[data['target']] = link_info
                
                # Update with consumable access info
                # Consumable access is only at individual level, so this should either be size 0 or 1
                assert len(self.consumable_access) == 0 or len(self.consumable_access) == 1
                if len(self.consumable_access) == 1:
                    chain_links[data['target']].update({k: v for k, v in
                                                        self.consumable_access[data['target']].items()
                                                        if k not in chain_links[data['target']]})
                    self.consumable_access = {}


        else:
            # Target is entire population. Identify individuals for which properties have changed
            # and store their changes.

            # Population dataframe after event
            df_after = self.copy_of_pop_dataframe()
            
            if 'PregnancySupervisor' in self.sim.modules:
                entire_mni_after = self.copy_of_mni()
            else:
                entire_mni_after = None

            #  Create and store the event and dictionary of changes for affected individuals
            chain_links = self.compare_population_dataframe_and_mni(self.df_before,
                                                                    df_after,
                                                                    self.entire_mni_before,
                                                                    entire_mni_after,
                                                                    data['event_name'])

        # Log chains
        if chain_links:
            # Convert chain_links into EAV-type dataframe
            eav_plus_event = self.convert_chain_links_into_eav(chain_links)
            # log it
            self.log_eav_dataframe_to_individual_histories(eav_plus_event)

        # Reset variables
        self.print_chains = False
        self.df_before = []
        self.row_before = pd.Series()
        self.mni_instances_before = False
        self.mni_row_before = {}
        self.entire_mni_before = {}
        self.consumable_access = {}
        self.cons_call_number_within_event = 0

    def mni_values_differ(self, v1, v2):

        if isinstance(v1, list) and isinstance(v2, list):
            return v1 != v2  # simple element-wise comparison

        if pd.isna(v1) and pd.isna(v2):
            return False  # treat both NaT/NaN as equal
        return v1 != v2

    def compare_entire_mni_dicts(self,entire_mni_before, entire_mni_after, set_of_tracked_individuals):
        diffs = {}

        all_individuals_in_mni = set(entire_mni_before.keys()) | set(entire_mni_after.keys())

        in_mni_and_tracked = all_individuals_in_mni.intersection(set_of_tracked_individuals)

        for person in in_mni_and_tracked:
            if person not in entire_mni_before: # but is afterward
                for key in entire_mni_after[person]:
                    if self.mni_values_differ(entire_mni_after[person][key],
                                              self.sim.modules['PregnancySupervisor'].default_all_mni_values[key]):
                        if person not in diffs:
                            diffs[person] = {}
                        diffs[person][key] = entire_mni_after[person][key]

            elif person not in entire_mni_after: # but is beforehand
                for key in entire_mni_before[person]:
                    if self.mni_values_differ(entire_mni_before[person][key],
                                              self.sim.modules['PregnancySupervisor'].default_all_mni_values[key]):
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

    def compare_population_dataframe_and_mni(self,df_before, df_after, entire_mni_before, entire_mni_after, event_name):
        """
        This function compares the population dataframe and mni dictionary before/after a population-wide e
        vent has occurred.
        It allows us to identify the individuals for which this event led to a significant (i.e. property) change,
        and to store the properties which have changed as a result of it.
        """
        # Create an empty dict to store changes for each of the individuals
        chain_links = {}

        # Individuals undergoing changes in the generap pop dataframe
        persons_changed = []

        # Find individuals which qualify as being tracked because they satisfied requirements either before OR after
        # the event occurred.
        assert df_before.index.equals(df_after.index), "Indices are not identical!"
        assert df_before.columns.equals(df_after.columns), "Columns of df_before and df_after do not match!"

        mask_of_tracked_individuals = df_before['iht_track_history'] | df_after['iht_track_history']
        set_of_tracked_individuals = set(mask_of_tracked_individuals.index[mask_of_tracked_individuals])
        
        # Only keep those individuals in dataframes
        df_before = df_before[mask_of_tracked_individuals]
        df_after = df_after[mask_of_tracked_individuals]

        # For those individuals, collect changes in the pop dataframe before/after the event
        same = df_before.eq(df_after) | (df_before.isna() & df_after.isna())
        diff_mask = ~same

        # Collect changes in the mni dictionary
        if 'PregnancySupervisor' in self.sim.modules:
            diff_mni = self.compare_entire_mni_dicts(entire_mni_before, entire_mni_after, set_of_tracked_individuals)
        else:
            diff_mni = []

        # Iterate over tracked individuals who experienced changes to properties as a result of the event
        for idx, row in diff_mask.iterrows():
        
            changed_cols = row.index[row].tolist()
            
            if changed_cols:  # Proceed only if there are changes in the row

                persons_changed.append(idx)
                # Create a dictionary for this person
                # First add event info
                link_info = {
                    'event_name': event_name,
                }

                # Store the new values from df_after for the changed columns
                for col in changed_cols:
                    link_info[col] = df_after.at[idx, col]

                # This person has also undergone changes in the mni dictionary, so add these here
                if idx in diff_mni:
                    for key in diff_mni[idx]:
                        link_info[col] = diff_mni[idx][key]

                # Append the event and changes to the individual key
                chain_links[idx] = link_info
        
        if 'PregnancySupervisor' in self.sim.modules:
            # For individuals which only underwent changes in mni dictionary, save changes here
            if len(diff_mni)>0:
                for key in diff_mni:
                    # If individual didn't also undergo changes in pop dataframe AND is tracked, add
                    if key not in persons_changed and key in set_of_tracked_individuals:
                        # If individual hadn't been previously added due to changes in pop df, add it here
                        link_info = {
                            'event_name': self.__class__.__name__,
                        }

                        for key_prop in diff_mni[key]:
                            link_info[key_prop] = diff_mni[key][key_prop]

                        chain_links[key] = link_info
        
        return chain_links




    
