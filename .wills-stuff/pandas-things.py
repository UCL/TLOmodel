import pandas as pd

my_dataframe = pd.DataFrame(
    data={"col1": [True, False, True, False], "col2": ["apples", "bananas", "pears", "oranges"], "person": [2, 5, 7, 1]},
)
my_dataframe.set_index("person", inplace=True)

print(my_dataframe)

holy_indices = [1, 7]

isin = my_dataframe.index.isin(holy_indices)

logical_opp = my_dataframe.col1 & isin
print(logical_opp)