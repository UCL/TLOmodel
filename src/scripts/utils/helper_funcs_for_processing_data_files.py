import pandas as pd
from pathlib import Path
from tlo.methods import demography


# Resource file path
rfp = Path("./resources")

def get_scaling_factor(parsed_output):
    """Find the factor that the model results should be multiplied by to be comparable to data"""
    # Get information about the real population size (Malawi Census in 2018)
    cens_tot = pd.read_csv(rfp / "ResourceFile_PopulationSize_2018Census.csv")['Count'].sum()
    cens_yr = 2018

    # Get information about the model population size in 2018 (and fail if no 2018)
    model_res = parsed_output['tlo.methods.demography']['population']
    model_yr = pd.to_datetime(model_res.date).dt.year

    if cens_yr in model_yr.values:
        model_tot = model_res.loc[model_yr == cens_yr, 'total'].values[0]
    else:
        print("WARNING: Model results do not contain the year of the census, so cannot scale accurately")
        model_tot = model_res.at[abs(model_yr - cens_yr).idxmin(), 'total']

    # Calculate ratio for scaling
    return cens_tot / model_tot

def get_list_of_gbd_causes():
    """Helper function to get the complete list of possible causes of deaths and disability (at their level 3).
    This uses the ResourceFile_Deaths_and_DALYS_GBD2019.csv"""

    raw = pd.read_csv(rfp / 'ResourceFile_Deaths_and_DALYS_GBD2019.csv')
    x = pd.unique(raw.cause_name)
    x.sort()
    return list(x)

def get_list_of_tlo_causes(output):
    """Helper function to get the complete list of possible causes of deaths and disability in the TLO model.
    todo - This should do a cold-read of the code, but, for now, it accepts an output data-structure from parse_log.
    """
    return pd.Series(list(set(pd.unique(output['tlo.methods.demography']['death']['cause'])))).sort_values()

def get_causes_mappers(output):
    """
    Make a dict that gives a mapping for each cause, from the GBD string and the strings put out from the TLO model.
    Accepts an argument of an output data-structure from parse_log

    :return: mapper_from_tlo_strings, mapper_from_gbd_strings

    todo - automate declaration and check that all tlo causes accounted for
    """

    gbd_causes = pd.Series(get_list_of_gbd_causes()).sort_values()
    tlo_causes = pd.Series(list(set(pd.unique(output['tlo.methods.demography']['death']['cause'])))).sort_values()

    # dalys = output['tlo.methods.healthburden']['dalys']
    # tlo_causes_disability = pd.Series(pd.unique(dalys.melt(id_vars=['date', 'sex', 'age_range', 'year'], var_name='cause')['cause'])).sort_values().str.split('_').apply(lambda x: x[1])

    # The TLO strings may be those used as cause of death or as a label for DALYS:

    causes = dict()
    causes['AIDS'] = {
        'gbd_strings': ['HIV/AIDS'],
        'tlo_strings': ['AIDS', 'Hiv']
    }
    causes['Malaria'] = {
        'gbd_strings': ['Malaria'],
        'tlo_strings': ['severe_malaria', 'Malaria']
    }
    causes['Childhood Diarrhoea'] = {
        'gbd_strings': ['Diarrheal diseases'],
        'tlo_strings': ['Diarrhoea_rotavirus',
                        'Diarrhoea_shigella',
                        'Diarrhoea_astrovirus',
                        'Diarrhoea_campylobacter',
                        'Diarrhoea_cryptosporidium',
                        'Diarrhoea_sapovirus',
                        'Diarrhoea_tEPEC',
                        'Diarrhoea_adenovirus',
                        'Diarrhoea_norovirus',
                        'Diarrhoea_ST-ETEC',
                        'Diarrhoea']
    }
    causes['Oesophageal Cancer'] = {
        'gbd_strings': ['Esophageal cancer'],
        'tlo_strings': ['OesophagealCancer']
    }
    causes['Epilepsy'] = {
        'gbd_strings': ['Other neurological disorders'],
        'tlo_strings': ['Epilepsy']
    }
    causes['Depression / Self-harm'] = {
        'gbd_strings': ['Self-harm'],
        'tlo_strings': ['Suicide', 'Depression']
    }
    causes['Complications in Labour'] = {
        'gbd_strings': ['Maternal disorders', 'Neonatal disorders', 'Congenital birth defects'],
        'tlo_strings': ['postpartum labour', 'labour', 'Labour']
    }

    # # Check that every gbd-string is included - looks like Epilepsy not:
    # for v in causes.values():
    #     for g in v['gbd_strings']:
    #         assert g in gbd_causes.values, f"{g} is not recognised in as a GBD cause of death"
    #     for t in v['tlo_strings']:
    #         assert (t in tlo_causes.values) or (t in tlo_causes_disability.values), f"{t} is not recognised in as a TLO cause of death"


    # Catch-all groups for Others:
    #  - map all the un-assigned gbd strings to Other
    all_gbd_strings_mapped = []
    for v in causes.values():
        all_gbd_strings_mapped.extend(v['gbd_strings'])

    gbd_strings_not_assigned = list(set(gbd_causes) - set(all_gbd_strings_mapped))

    causes['Other'] = {
        'gbd_strings': gbd_strings_not_assigned,
        'tlo_strings': ['Other']
    }

    # make the mappers:
    causes_df = pd.DataFrame.from_dict(causes, orient='index')

    #  - from tlo_strings (key=tlo_string, value=unified_name)
    mapper_from_tlo_strings = dict((v, k) for k, v in (
        causes_df.tlo_strings.apply(pd.Series).stack().reset_index(level=1, drop=True)
    ).iteritems())

    #  - from gbd_strings (key=gbd_string, value=unified_name)
    mapper_from_gbd_strings = dict((v, k) for k, v in (
        causes_df.gbd_strings.apply(pd.Series).stack().reset_index(level=1, drop=True)
    ).iteritems())

    # check that the mappers are exhaustive for all causes in both gbd and tlo
    # assert all([c in mapper_from_tlo_strings for c in tlo_causes])
    # assert all([c in mapper_from_gbd_strings for c in gbd_causes])

    return mapper_from_tlo_strings, mapper_from_gbd_strings

def age_cats(ages_in_years):
    """Accepts a pd.Series with age in single years and returns pd.Series in the age-categories compatible with GBD:
    (0-4, 5-9, ..., 90-94, 95+)"""

    # Get look-ups defined in the Demography module
    age_range_categories, age_range_lookup = get_age_range_categories()

    # Make an edit so that top-end group is 95+ to match with how GBD death are reported
    for age in range(95, 100):
        age_range_lookup[age] = '95+'

    if '95-99' in age_range_categories:
        age_range_categories.remove('95-99')
    if '100+' in age_range_categories:
        age_range_categories.remove('100+')
    if '95+' not in age_range_categories:
        age_range_categories.append('95+')

    age_cats = pd.Series(
        pd.Categorical(ages_in_years.map(age_range_lookup),
                       categories=age_range_categories, ordered=True)
    )
    return age_cats

def get_age_range_categories():
    """Get the age_range categories that are used in the TLO model"""
    dem = demography.Demography()
    age_range_categories = dem.AGE_RANGE_CATEGORIES.copy()
    age_range_lookup = dem.AGE_RANGE_LOOKUP.copy()

    return age_range_categories, age_range_lookup

def standardise_gbd_age_groups(ser):
    """Helper function to standardise the age-groups in the GBD data (0-4, 5-9, ..., 90-94, 95+)"""

    AGE_RANGE_CATEGORIES, _ = get_age_range_categories()
    ser = ser   .str.replace('to', '-')\
                .str.replace('95 plus', '95+')\
                .str.replace(' ', '')\
                .str.replace('1-4', '0-4')\
                .str.replace('<1year', '0-4')

    return pd.Categorical(ser, categories=AGE_RANGE_CATEGORIES, ordered=True)

def load_gbd_deaths_and_dalys_data(output):
    """Load the GBD data and format so that it is compatible with TLO model output"""

    gbd = pd.read_csv(rfp / "ResourceFile_Deaths_and_DALYS_GBD2019.csv")

    # map onto unified causes of death, standardised age-groups and collapse into age/cause count of death:
    _, mapper_from_gbd_strings = get_causes_mappers(output)

    gbd["unified_cause"] = gbd["cause_name"].map(mapper_from_gbd_strings)
    assert not gbd["unified_cause"].isna().any()

    # sort out labelling of sex:
    gbd['sex'] = gbd['sex_name'].map({'Male': 'M', 'Female': 'F'})

    # sort out age-groups:
    gbd['age_range'] = standardise_gbd_age_groups(gbd['age_name'])

    return gbd
