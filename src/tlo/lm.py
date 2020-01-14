import logging
import numbers
from enum import Enum, auto

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Predictor(object):
    def __init__(self, property_name: str = None):
        """A Predictor variable for the regression model. The property_name is a property of the
         population dataframe e.g. age, sex, etc."""
        self.property_name = property_name
        self.conditions = list()
        self.else_condition_supplied = False

    def when(self, condition, value):
        return self._coeff(condition, value)

    def otherwise(self, value):
        assert self.property_name is not None, "Can't use `otherwise` condition on unnamed Predictor"
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

        # If there isn't a property name
        if self.property_name is None:
            # We use the supplied condition literally
            self.conditions.append((condition, coefficient))
            return self

        # Otherwise, the condition is applied on a specific property
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

        self.conditions.append((parsed_condition, coefficient))
        return self

    def predict(self, df: pd.DataFrame):
        """Will add the value(s) of this predictor to the output series, checking values in the supplied
        dataframe"""

        # We want to "short-circuit" values i.e. if an individual in a population matches a certain
        # condition, we don't want that individual to be matched in any subsequent conditions

        # result of assigning coefficients on this predictor
        output = pd.Series(data=np.nan, index=df.index)

        # keep a record of all rows matched by this predictor's conditions
        matched = pd.Series(False, index=df.index)

        for condition, value in self.conditions:
            # don't include rows that were matched by a previous condition
            if condition:
                unmatched_condition = f'{condition} & (~@matched)'
            else:
                unmatched_condition = '~@matched'

            # rows matching the current conditional
            mask = df.eval(unmatched_condition)

            # test if mask includes rows that were already matched by a previous condition
            assert not (matched & mask).any(), f'condition "{unmatched_condition}" matches rows already matched'

            # update elements in the output series with the corresponding value for the condition
            output[mask] = value

            logger.debug('predictor: %s; condition: %s; value: %s; matched rows: %d/%d',
                         self.property_name, condition, value, mask.sum(), len(mask))

            # add this condition's matching rows to the list of matched rows
            matched = (matched | mask)
        return output


class LinearModelType(Enum):
    """
    The type of model specifies how the results from the predictor are combined:
    'additive' -> adds the effect_sizes from the predictors
    'logisitc' -> multiples the effect_sizes from the predictors and applies the transform x/(1+x)
    [Thus, the intercept can be taken to be an Odds and effect_sizes Odds Ratios,
    and the prediction is a probability.]
    'multiplicative' -> multiplies the effect_sizes from the predictors
    """
    ADDITIVE = auto()
    LOGISTIC = auto()
    MULTIPLICATIVE = auto()


class LinearModel(object):
    def __init__(self, lm_type: LinearModelType, intercept: float, *args: Predictor):
        """
        A linear model has an intercept and zero or more Predictor variables.
        """
        assert lm_type in LinearModelType, 'Model should be one of the prescribed LinearModelTypes'
        self.lm_type = lm_type

        assert isinstance(intercept, float), "Intercept is not specified"
        self.intercept = intercept

        self.predictors = list()
        for predictor in args:
            assert isinstance(predictor, Predictor)
            self.predictors.append(predictor)

    def predict(self, df: pd.DataFrame):
        """Will call each Predictor's `predict` methods passing the supplied dataframe"""
        assert all([p.property_name in df.columns
                    for p in self.predictors
                    if p.property_name is not None]), "Predictor variables not in df"

        # Store the result of the calculated values of Predictors
        res_by_predictor = pd.DataFrame(index=df.index)
        res_by_predictor['__intercept__'] = self.intercept

        for predictor in self.predictors:
            res_by_predictor[predictor] = predictor.predict(df)

        # Do appropriate transformation on output
        if self.lm_type is LinearModelType.ADDITIVE:
            # print("Linear Model: Prediction will be sum of each effect size.")
            return res_by_predictor.sum(axis=1, skipna=True)

        elif self.lm_type is LinearModelType.LOGISTIC:
            # print("Logistic Regression Model: Prediction will be transform to probabilities. " \
            #       "Intercept assumed to be Odds and effect sizes assumed to be Odds Ratios.")
            odds = res_by_predictor.prod(axis=1, skipna=True)
            return odds / (1 + odds)

        elif self.lm_type is LinearModelType.MULTIPLICATIVE:
            # print("Multiplicative Model: Prediction will be multiplication of each effect size.")
            return res_by_predictor.prod(axis=1, skipna=True)

        return None
