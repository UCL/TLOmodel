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

        # print(f'\t{parsed_condition} -> {coefficient}')
        self.conditions.append((parsed_condition, coefficient))
        return self

    def predict(self, df: pd.DataFrame):
        """Will add the value(s) of this predictor to the output series, checking values in the supplied
        dataframe"""

        # We want to "short-circuit" values i.e. if an individual in a population matches a certain
        # condition, we don't want that individual to be matched in any subsequent conditions

        output = pd.Series(data = np.nan, index = df.index)
        touched = pd.Series(False, index=df.index)
        print("touched = pd.Series(False, index=output.index)")
        for condition, value in self.conditions:
            if condition:
                condition = condition + ' & (~@touched)'
            else:
                condition = '~@touched'
            mask = df.eval(condition)
            print(f"mask = df.eval({condition})")
            output[mask] = value
            print(f"output[mask] += {value}")
            touched = (touched | mask)
            print(f"touched = (touched | mask)")
        return output

class LinearModel(object):
    def __init__(self, type: str, intercept: float, *args: Predictor):
        """
        A Linear model has an intercept and none or more Predictor variables.
        The type of model specifies how the results from the predictor are combined:
        'additive' -> adds the effect_sizes from the predictors
        'logisitc' -> multiples the effect_sizes from the predictors and applies the transform x/(1+x)
                [Thus, the intercept can be taken to be an Odds and effect_sizes Odds Ratios,
                and the prediction is a probability.]
        'multiplicative' -> multiplies the effect_sizes from the predictors
        """

        assert type in ['additive', 'logistic', 'multiplicative'], 'Model type not recognised'
        self.type = type

        self.intercept = intercept
        self.predictors = list()
        for predictor in args:
            assert isinstance(predictor, Predictor)
            self.predictors.append(predictor)

    def predict(self, df: pd.DataFrame):
        """Will call each Predictor's `predict` methods passing the supplied dataframe"""

        # Do some checks:
        assert self.intercept is not None, "Interceipt is not specified"
        assert all([pred.property_name in df.columns for pred in self.predictors]), "Predictor variables not in df"

        # Store the result of the calculated values of Predictors
        res_by_predictor = pd.DataFrame(index=df.index)
        res_by_predictor['intercept'] = self.intercept

        for predictor in self.predictors:
            res_by_predictor[predictor] = predictor.predict(df)

        # Do appropriate transformation on output
        if self.type == 'additive':
            # print("Linear Model: Prediction will be sum of each effect size.")
            output = res_by_predictor.sum(axis=1, skipna=True)

        elif self.type == 'logistic':
            # print("Logistic Regression Model: Prediction will be transform to probabilities. " \
            #       "Intercept assumed to be Odds and effect sizes assumed to be Odds Ratios.")
            odds = res_by_predictor.prod(axis=1, skipna=True)
            output = odds / (1 + odds)

        elif self.type == 'multiplicative':
            # print("Multiplicative Model: Prediction will be multiplication of each effect size.")
            output = res_by_predictor.prod(axis=1, skipna=True)

        return output
