"""Unit tests for utility functions."""
import os
import pickle
import shutil
import string
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chisquare

import tlo.util
from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import demography
from tlo.util import DEFAULT_MOTHER_ID, convert_excel_files_to_csv, read_csv_files

path_to_files = Path(os.path.dirname(__file__))


@pytest.fixture
def rng(seed):
    return np.random.RandomState(seed % 2 ** 32)


def check_output_states_and_freq(
    rng, input_freq, prob_matrix, use_categorical_dtype=False, p_value_threshold=0.01
):
    expected_output_freq = prob_matrix.to_numpy() @ input_freq
    input_states = np.repeat(prob_matrix.columns.to_list(), input_freq).tolist()
    input_states = pd.Series(
        pd.Categorical(input_states) if use_categorical_dtype else input_states
    )
    output_states = tlo.util.transition_states(input_states, prob_matrix, rng=rng)
    output_freq = output_states.value_counts()
    # Ensure frequencies ordered as prob_matrix columns and zero counts included
    output_freq = np.array([output_freq.get(state, 0) for state in prob_matrix.columns])
    assert isinstance(output_states, pd.Series)
    assert output_states.dtype == input_states.dtype
    assert output_freq.sum() == len(input_states)
    # Perform Pearson's chi-squared test against null hypothesis of the frequencies of
    # the output states being distributed according to the expected multinomial
    # distribution. States with expected output frequency zero excluded to avoid
    # division by zero issues in evaluating test statistic
    assert chisquare(
        output_freq[expected_output_freq > 0],
        expected_output_freq[expected_output_freq > 0]
    ).pvalue > p_value_threshold
    return output_freq


@pytest.mark.parametrize('missing_state_index', [None, 0, 1, 2, 3])
def test_transition_states_uniform_input(rng, missing_state_index):
    states = list("abcd")
    prob_matrix = pd.DataFrame(columns=states, index=states)
    prob_matrix["a"] = [0.9, 0.1, 0.0, 0.0]
    prob_matrix["b"] = [0.1, 0.3, 0.6, 0.0]
    prob_matrix["c"] = [0.0, 0.2, 0.6, 0.2]
    prob_matrix["d"] = [0.0, 0.0, 0.3, 0.7]
    input_freq = np.full(len(states), 1000)
    if missing_state_index is not None:
        input_freq[missing_state_index] = 0
    check_output_states_and_freq(rng, input_freq, prob_matrix)


@pytest.mark.parametrize('use_categorical_dtype', [False, True])
def test_transition_states_fixed_state(rng, use_categorical_dtype):
    states = list("abcd")
    prob_matrix = pd.DataFrame(columns=states, index=states)
    prob_matrix["a"] = [0.9, 0.1, 0.0, 0.0]
    prob_matrix["b"] = [0.1, 0.3, 0.6, 0.0]
    prob_matrix["c"] = [0.0, 0.2, 0.8, 0.0]
    prob_matrix["d"] = [0.0, 0.0, 0.0, 1.0]
    input_freq = np.full(len(states), 1000)
    output_freq = check_output_states_and_freq(
        rng, input_freq, prob_matrix, use_categorical_dtype
    )
    # Transition of d state deterministic so should frequencies should match exactly
    assert output_freq[states.index("d")] == input_freq[states.index("d")]


def test_transition_states_removes_state(rng):
    states = list("abcd")
    prob_matrix = pd.DataFrame(columns=states, index=states)
    prob_matrix["a"] = [0.0, 1.0, 0.0, 0.0]
    prob_matrix["b"] = [0.0, 0.4, 0.6, 0.0]
    prob_matrix["c"] = [0.0, 0.2, 0.6, 0.2]
    prob_matrix["d"] = [0.0, 0.0, 0.3, 0.7]
    input_freq = np.full(len(states), 1000)
    output_freq = check_output_states_and_freq(rng, input_freq, prob_matrix)
    # No transitions to a state so should have zero frequency
    assert output_freq[states.index("a")] == 0


class TestCreateAgeRangeLookup:
    def test_default_range(self):
        """
        Given just a minimum and maximum age,
        should get a below age category, ranges of 5 and then an above maximum age category
        """
        ranges, lookup = tlo.util.create_age_range_lookup(10, 20)
        assert ranges == ["0-10", "10-14", "15-19", "20+"]
        for i in range(0, 10):
            assert lookup[i] == ranges[0]
        for i in range(10, 15):
            assert lookup[i] == ranges[1]
        for i in range(15, 20):
            assert lookup[i] == ranges[2]
        for i in range(20, 25):
            assert lookup[i] == ranges[3]

    def test_user_range(self):
        """
        Given just a minimum age, maximum age, and a range size of 10
        should get a below age category, ranges of 10 and then an above maximum age category
        """
        ranges, lookup = tlo.util.create_age_range_lookup(10, 30, 10)
        assert ranges == ["0-10", "10-19", "20-29", "30+"]
        for i in range(0, 10):
            assert lookup[i] == ranges[0]
        for i in range(10, 20):
            assert lookup[i] == ranges[1]
        for i in range(20, 30):
            assert lookup[i] == ranges[2]
        for i in range(30, 40):
            assert lookup[i] == ranges[3]

    def test_no_under_min_age(self):
        """
        Given just a minimum age of zero, and a maximum age
        should get a ranges of 5 and then an above maximum age category
        """
        ranges, lookup = tlo.util.create_age_range_lookup(0, 10)
        assert ranges == ["0-4", "5-9", "10+"]
        for i in range(0, 5):
            assert lookup[i] == ranges[0]
        for i in range(5, 10):
            assert lookup[i] == ranges[1]
        for i in range(10, 15):
            assert lookup[i] == ranges[2]


@pytest.mark.slow
def test_sample_outcome(tmpdir, seed):
    """Check that helper function `sample_outcome` works correctly."""

    # Create probability matrix for four individual (0-3) with four possible outcomes (A, B, C).
    probs = pd.DataFrame({
        'A': {0: 1.0, 1: 0.0, 2: 0.25, 3: 0.0},
        'B': {0: 0.0, 1: 1.0, 2: 0.25, 3: 0.0},
        'C': {0: 0.0, 1: 0.0, 2: 0.50, 3: 0.0},
    })
    rng = np.random.RandomState(seed=seed % 2 ** 32)

    list_of_results = list()
    n = 5000
    for i in range(n):
        list_of_results.append(tlo.util.sample_outcome(probs, rng))
    res = pd.DataFrame(list_of_results)

    assert (res[0] == 'A').all()
    assert (res[1] == 'B').all()
    assert (res[2].isin(['A', 'B', 'C'])).all()
    assert 3 not in res.columns

    for op in ['A', 'B', 'C']:
        prob = probs.loc[2, op]
        assert res[2].value_counts()[op] == pytest.approx(probs.loc[2, op] * n, abs=2 * np.sqrt(n * prob * (1 - prob)))


def test_logs_parsing(tmpdir):
    """test all functionalities of LogDict class inside utils.py.

        1.  ensure that logs are generated as expected
        2.  check expected keys are present within the logs
        3.  check that we're able to get metadata
        4.  check that output from method `items()` of LogsDict class return a generator
        5.  check that picked data can be properly generated

        """

    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    path_to_tmpdir = path_to_files / tmpdir

    start_date = Date(2010, 1, 1)
    end_date = Date(2012, 1, 1)
    popsize = 200

    # add file handler for the purpose of logging
    sim = Simulation(start_date=start_date, seed=0, log_config={
        'filename': 'logs_dict_class',
        'directory': tmpdir,
    })

    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath)
    )

    # Create a simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    file_path = sim.log_filepath
    outputs = parse_log_file(file_path)

    # check parse_log_file methods worked as expected - expected keys has to be in the LogsDict class
    assert 'tlo.methods.demography' in outputs

    # check that metadata is within the selected key
    assert '_metadata' in outputs['tlo.methods.demography'].keys()

    # check that method `items()` of LogsDict class returns a generator
    assert isinstance(outputs.items(), types.GeneratorType)

    # test we can create a .pickled file
    for key, output in outputs.items():
        if key.startswith("tlo."):
            with open(path_to_tmpdir / f"{key}.pickle", "wb") as f:
                pickle.dump(output, f)

        #   check a pickle file is created
        assert Path.is_file(path_to_tmpdir / f"{key}.pickle")

        # check the created pickle file is not empty
        assert os.path.getsize(path_to_tmpdir / f"{key}.pickle") != 0


def test_get_person_id_to_inherit_from(rng: np.random.RandomState):
    population_size = 1000
    num_test = 5
    for child_id, mother_id in rng.randint(0, population_size, size=(num_test, 2)):
        # population_dataframe and rng arguments should be unused if mother_id != -1
        assert mother_id == tlo.util.get_person_id_to_inherit_from(
            child_id, mother_id, population_dataframe=None, rng=None
        )

    # Test direct birth mothers, whose scope in person_id is [-population_size, -1]
    for child_id in rng.randint(0, population_size, size=num_test):
        for mother_id in rng.randint(-population_size, -1, size=num_test):
            assert abs(mother_id) == tlo.util.get_person_id_to_inherit_from(
                child_id, mother_id, population_dataframe=None, rng=None)

    population_dataframe = pd.DataFrame(
        {
            "is_alive": rng.choice((True, False), size=population_size),
            "sex": rng.choice(("F", "M"), size=population_size),
            "age_years": rng.randint(0, 100, size=population_size),
        }
    )

    # Test case of individuals initialised as adults at the start of the simulation
    mother_id = DEFAULT_MOTHER_ID
    for child_id in rng.choice(
        population_dataframe.index[population_dataframe.is_alive], size=num_test
    ):
        inherit_from_id = tlo.util.get_person_id_to_inherit_from(
            child_id, mother_id, population_dataframe, rng
        )
        assert inherit_from_id != mother_id
        assert inherit_from_id != child_id
        assert population_dataframe.loc[inherit_from_id].is_alive


def test_random_date_returns_date_sequential(rng):
    # start_date < end_date - sequential order
    num_iter = 20
    for year_init in rng.randint(1900, 2050, size=num_iter):
        year_fin = year_init + rng.randint(1, 100)
        start_date, end_date = Date(year_init, 1, 1), Date(year_fin, 1, 1)
        random_date = tlo.util.random_date(start_date, end_date, rng)
        assert isinstance(random_date, Date)
        assert start_date <= random_date < end_date


def test_random_date_returns_date_nonsequential(rng):
    # start_date >= end_date - nonsequential order
    num_iter = 20
    for year_init in rng.randint(1900, 2050, size=num_iter):
        year_fin = year_init - rng.randint(0, 100)
        start_date, end_date = Date(year_init, 1, 1), Date(year_fin, 1, 1)
        with pytest.raises(ValueError):
            tlo.util.random_date(start_date, end_date, rng)


def test_hash_dataframe(rng):
    """ Check that hash types:
                - are generated,
                - are equal for the same dataframes,
                - differ for different dataframes,
                - validate for lists.
        """

    def check_hash_is_valid(dfh):
        # assert hash_dataframe returns hash
        assert isinstance(dfh, str)
        # Try to interpret hash as a hexadecimal integer (should not raise exception)
        int(dfh, base=16)

    # generate dataframes of random strings
    dataframes = [
        pd.DataFrame(rng.choice(list(string.ascii_lowercase), size=(4, 3)))
        for _ in range(10)
    ]
    # account for lists
    for i in range(5):
        dataframes[i].at[1, 1] = [0, 1, 2]

    for i in range(len(dataframes)):
        # do checks on single dataframe dataframes[i]
        df_hash = tlo.util.hash_dataframe(dataframes[i])
        check_hash_is_valid(df_hash)
        # check hash returned remains constant over repeated calls
        assert df_hash == tlo.util.hash_dataframe(dataframes[i])
        for j in range(i):
            # do checks on dataframe pair (dataframes[i], dataframes[j])
            # check hash differs for different dataframes
            if not dataframes[i].equals(dataframes[j]):
                assert df_hash != tlo.util.hash_dataframe(dataframes[j])


def copy_files_to_temporal_directory_and_return_path(tmpdir):
    """ copy resource files in tests/resources to a temporal directory and return its path

    :param tmpdir: path to a temporal directory

    """
    resource_filepath = path_to_files / 'resources'
    tmpdir_resource_filepath = Path(tmpdir / 'resources')
    shutil.copytree(resource_filepath, tmpdir_resource_filepath)
    return tmpdir_resource_filepath


def test_pass_datatypes_to_read_csv_method(tmpdir):
    """ test passing column datatypes to read csv method. Final column datatype should change to what has been passed """
    # copy and get resource files path in the temporal directory
    path_to_tmpdir = Path(tmpdir)
    sample_data = pd.DataFrame(data={'numbers1': [5,6,8,4,9,6], 'numbers2': [19,27,53,49,75,56]}, dtype=int)
    sample_data.to_csv(tmpdir/'sample_data.csv', index=False)
    # read from the sample data file
    read_sample_data = read_csv_files(path_to_tmpdir, files=['sample_data'])
    # confirm column datatype is what was assigned
    assert read_sample_data.numbers1.dtype == 'int' and read_sample_data.numbers2.dtype == 'int'
    # define new datatypes
    datatype = {'numbers1': int, 'numbers2': float}
    # pass the new datatypes to read csv method and confirm datatype has changed to what has been declared now
    assign_dtype = read_csv_files(path_to_tmpdir, files=['sample_data'], dtype=datatype)
    assert assign_dtype.numbers1.dtype == 'int' and assign_dtype.numbers2.dtype == 'float'


def test_read_csv_file_method_passing_none_to_files_argument(tmpdir):
    """ test reading csv files with one file in the target resource file and setting to None the files argument

        Expectations
            1.  should return a dictionary
            2.  the dictionary key name should match file name
    """
    # copy and get resource files path in the temporal directory
    tmpdir_resource_filepath = copy_files_to_temporal_directory_and_return_path(tmpdir)
    #  choose an Excel file with one sheet in it and convert it to csv file
    convert_excel_files_to_csv(tmpdir_resource_filepath, files=['ResourceFile_load-parameters.xlsx'])
    # get the folder containing the newly converted csv file and check the expected behavior
    this_csv_resource_folder = tmpdir_resource_filepath/"ResourceFile_load-parameters"
    file_names = [csv_file_path.stem for csv_file_path in this_csv_resource_folder.rglob("*.csv")]
    one_csv_file_in_folder_dict = read_csv_files(this_csv_resource_folder, files=None)
    assert isinstance(one_csv_file_in_folder_dict, dict)
    assert set(one_csv_file_in_folder_dict.keys()) == set(file_names)


def test_read_csv_method_with_default_value_for_files_argument(tmpdir):
    """ read csv method when no file name(s) is supplied to the files argument
        i)  should return dictionary.
        ii) dictionary keys should match csv file names in resource folder
        iii)  all dictionary values should be dataframes

    :param tmpdir: path to a temporal directory

    """
    tmpdir_resource_filepath = copy_files_to_temporal_directory_and_return_path(tmpdir)
    file_names = [csv_file_path.stem for csv_file_path in tmpdir_resource_filepath.rglob("*.csv")]
    df_no_files = read_csv_files(tmpdir_resource_filepath)
    assert isinstance(df_no_files, dict)
    assert set(df_no_files.keys()) == set(file_names)
    assert all(isinstance(value, pd.DataFrame) for value in df_no_files.values())


def test_read_csv_method_with_one_file(tmpdir):
    """ test read csv method when one file name is supplied to files argument. should return a dataframe
    :param tmpdir: path to a temporal directory

    """
    tmpdir_resource_filepath = copy_files_to_temporal_directory_and_return_path(tmpdir)
    df = read_csv_files(tmpdir_resource_filepath, files='df_at_healthcareseeking')
    assert isinstance(df, pd.DataFrame)


def test_read_csv_method_with_multiple_files(tmpdir):
    """ read csv method when multiple file names are supplied.
             i) should return dictionary.
            ii) dictionary keys should match supplied file names
           iii)  all dictionary values should be dataframes

    :param tmpdir: path to a temporal directory

    """
    tmpdir_resource_filepath = copy_files_to_temporal_directory_and_return_path(tmpdir)
    file_names = ['df_at_healthcareseeking', 'df_at_init_of_lifestyle']
    df_dict = read_csv_files(tmpdir_resource_filepath, files=file_names)
    assert isinstance(df_dict, dict)
    assert set(df_dict.keys()) == set(file_names)
    for _key, dataframe in df_dict.items():
        assert isinstance(dataframe, pd.DataFrame)


def test_read_csv_method_output_matches_previously_used_read_excel(tmpdir):
    """ check read from csv method produces same output as the read Excel file
    :param tmpdir: path to a temporal directory

    """
    tmpdir_resource_filepath = copy_files_to_temporal_directory_and_return_path(tmpdir)
    excel_file_path = Path(tmpdir_resource_filepath
                           / 'ResourceFile_test_convert_to_csv/ResourceFile_test_convert_to_csv.xlsx')
    xls = pd.ExcelFile(excel_file_path)
    sheet_names = xls.sheet_names
    # convert the above Excel file into csv equivalent. we will use the newly converted files to determine if
    # loading parameters from Excel file will be equal to loading parameters from the converted csv files
    convert_excel_files_to_csv(folder=Path(tmpdir_resource_filepath / 'ResourceFile_test_convert_to_csv'),
                               files=[excel_file_path.name])

    # get excel sheet names
    df_excel = pd.read_excel(xls, sheet_name=sheet_names)

    # read newly converted csv files using read_csv_files method
    df_csv = read_csv_files(Path(str(excel_file_path).split('.')[0]),
                            files=sheet_names)

    # dictionary keys from both dataframe dictionaries should match
    assert isinstance(df_excel, dict) and isinstance(df_csv, dict)
    assert df_excel.keys() == df_csv.keys()
    for key in df_excel:
        assert df_excel[key].astype(str).equals(df_csv[key].astype(str))


def test_convert_excel_files_method(tmpdir):
    """ Test converting Excel files to csv equivalent is done as expected

        1) Excel file name should become the name of the folder containing the newly converted csv files
        2) Excel file sheet names should become csv file names
        3) if files are given, the function should only convert to excel only those given files in a folder
        4) if no files are given, all Excel files in the parent folder and subsequent folders within the parent folder
           should get converted to csv files

    """

    def check_logic_of_converting_excel_files_to_csv_files(folder: Path, files: list) -> None:
        """ check converting Excel files to csv files is done as expected
                1) check that a new directory to hold the newly created csv files has been created
                2) check that this new directory name matches the Excel file name it has been created from
                3) check csv files are created and that the csv names should match sheet names of an Excel file they
                have been created from
        """
        # check that the above function has created a folder named `ResourceFile_load-parameters`(name of the Excel
        # file) and a csv file named `parameter_values` (Excel file sheet name).
        excel_file_paths = [folder / file for file in files]

        for excel_file_path in excel_file_paths:
            xl = pd.ExcelFile(excel_file_path)
            path_to_new_directory = excel_file_path.with_suffix("")
            # new folder should be created
            assert path_to_new_directory.exists() and path_to_new_directory.is_dir()
            # the new folder name should be the same as the Excel file name
            assert excel_file_path.stem == path_to_new_directory.name
            for sheet_name in xl.sheet_names:
                path_to_new_file = Path(path_to_new_directory / f'{sheet_name}.csv')
                # new csv file(s) should be created with name(s) resembling sheet name(s) in excel file
                assert path_to_new_file.exists() and path_to_new_file.is_file()
                assert sheet_name == path_to_new_file.name.split('.')[0]


    # get resource file path
    resourcefilepath = path_to_files / 'resources'
    tmpdir_resourcefilepath = Path(tmpdir/'resources')
    shutil.copytree(resourcefilepath, tmpdir_resourcefilepath)

    # check convert to csv logic when a list of file name(s) is given
    excel_file = ['ResourceFile_load-parameters.xlsx']
    convert_excel_files_to_csv(tmpdir_resourcefilepath, files=excel_file)
    # check new folder containing csv file is created. The folder name and csv file name should resemble the supplied
    # Excel file name and sheet name respectively
    check_logic_of_converting_excel_files_to_csv_files(tmpdir_resourcefilepath, files=excel_file)

    # check convert to csv logic when no list of file name(s) is given
    excel_files = [file for file in tmpdir_resourcefilepath.rglob("*.xlsx")]
    if excel_files is None:
        excel_files = excel_file

    convert_excel_files_to_csv(tmpdir_resourcefilepath)
    # check behaviours are as expected. New folders containing csv files should be created with names resembling the
    # Excel file they were created from
    check_logic_of_converting_excel_files_to_csv_files(tmpdir_resourcefilepath, excel_files)
