import io
import numbers
import pandas as pd
import numpy as np

class Predictor(object):
    def __init__(self, property_name: str):
        """A Predictor variable for the regression model. The property_name is a property of the
         population dataframe e.g. age, sex, etc."""
        self.property_name = property_name
        self.conditions = list()
        self.else_condition_supplied = False
        print(f'{self.property_name}:')

    def when(self, condition, value):
        return self._coeff(condition, value)

    def otherwise(self, value):
        return self._coeff(value)

    def _coeff(self, *args):
        """Adds the coefficient for the Predictor. The arguments can be two:
                `coeff(condition, value)` where the condition evaluates the property value to true/false
                `coeff(value)` where the value is given to all unconditioned values of the property
        The second style (unconditioned value) only makes sense after one or more conditioned values
        """
        num_args = len(args)
        if num_args == 2:
            condition = args[0]
            coefficient = args[1]
        else:
            condition = None
            coefficient = args[0]

        if isinstance(condition, str):
            # Handle either a complex condition (begins with an operator) or implicit equality
            if condition[0] in ['=', '<', '>', '~', '(', '.']:
                parsed_condition = f'({self.property_name}{condition})'
            else:
                # numeric values don't need to be quoted
                if condition.isnumeric():
                    parsed_condition = f'({self.property_name} == {condition})'
                else:
                    parsed_condition = f'({self.property_name} == "{condition}")'
        elif isinstance(condition, bool):
            if condition:
                parsed_condition = f'({self.property_name} == True)'
            else:
                parsed_condition = f'({self.property_name} == False)'
        elif isinstance(condition, numbers.Number):
            parsed_condition = f'({self.property_name} == {condition}'
        elif condition is None:
            assert not self.else_condition_supplied, "You can only give one unconditioned value to predictor"
            self.else_condition_supplied = True
            parsed_condition = ''
        else:
            raise RuntimeError(f"Unhandled condition: {args}")

        print(f'\t{parsed_condition} -> {coefficient}')
        self.conditions.append((parsed_condition, coefficient))
        return self

    def predict(self, df: pd.DataFrame):
        """Will add the value(s) of this predictor to the output series, checking values in the supplied
        dataframe"""

        # We want to "short-circuit" values i.e. if an individual in a population matches a certain
        # condition, we don't want that individual to be matched in any subsequent conditions

        output = pd.Series(data = 0, index = df.index)
        touched = pd.Series(False, index=df.index)
        print("touched = pd.Series(False, index=output.index)")
        for condition, value in self.conditions:
            if condition:
                condition = condition + ' & (~@touched)'
            else:
                condition = '~@touched'
            mask = df.eval(condition)
            print(f"mask = df.eval({condition})")
            output[mask] += value
            print(f"output[mask] += {value}")
            touched = (touched | mask)
            print(f"touched = (touched | mask)")
        return output

class LinearModel(object):
    def __init__(self, type: str, intercept: float, *args: Predictor):
        """A logistic model has an intercept and one or more Predictor variables """

        assert type in ['linear', 'logistic', 'multiplicative'], 'Model type not recognised'
        self.type = type

        if self.type == 'linear':
            print("Prediction will be sum of each effect size.")
        elif self.type == 'logistic':
            print("Prediction will be transform to be 'probabilities. \
                                            Effect sizes assumed to be Odds Ratios.")
        elif self.type == 'multiplicative':
            print("Prediction will be multiplication of each effect size.")

        self.intercept = intercept
        self.predictors = list()
        for predictor in args:
            assert isinstance(predictor, Predictor)
            self.predictors.append(predictor)

    def predict(self, df: pd.DataFrame):
        """Will call each Predictor's `predict` methods passing the supplied dataframe"""

        # store the result of summing all the calculated values of Predictors
        res_by_predictor = pd.DataFrame(index=df.index)
        res_by_predictor['intercept'] = self.intercept

        for predictor in self.predictors:
            res_by_predictor[predictor] = predictor.predict(df)

        # Do appropriate transformation on output
        if self.type == 'linear':
            output = res_by_predictor.sum(axis=1)

        elif self.type == 'logistic':
            output = 1 / (1 + np.exp(-res_by_predictor.sum(axis=1)))

        elif self.type == 'multiplicative':
            output = res_by_predictor.prod(axis=1)

        return output


################################################################################
# EXAMPLE USAGE:
################################################################################

EXAMPLE_POP= """region_of_residence,li_urban,sex,age_years,sy_vomiting
Northern,True,M,12,False
Central,True,M,6,True
Northern,True,M,24,False
Southern,True,M,46,True
Central,True,M,91,True
Central,False,M,16,False
Southern,False,F,80,True
Northern,True,F,99,False
Western,True,F,63,False
Central,True,F,51,True
Central,True,M,57,False
Central,False,F,2,False
Central,True,F,93,False
Western,False,M,15,True
Western,False,M,5,False
Northern,True,M,29,True
Western,True,M,63,False
Southern,True,F,54,False
Western,False,M,94,False
Northern,False,F,91,True
Northern,True,M,29,False
"""

eq = LinearModel(
    'logistic',
    0.0,
    Predictor('region_of_residence').when('Northern', 0.1).when('Central', 0.2).when('Southern', 0.3),
    Predictor('li_urban').when(True, 0.01).otherwise(0.02),
    Predictor('sex').when('M', 0.001).when('F', 0.002),
    Predictor('age_years')
        .when('< 5', 0.0001)
        # alternative: .when('.between(0,5)', 0.0001)
        .when('< 15', 0.0002)
        .when('< 35', 0.0003)
        .when('< 60', 0.0004)
        .otherwise(0.0005),
    Predictor('sy_vomiting').when(True, 0.00001).otherwise(0.00002)
)

df = pd.read_csv(io.StringIO(EXAMPLE_POP))

predicted = eq.predict(df)
df['predicted'] = predicted

print(df.to_string())
