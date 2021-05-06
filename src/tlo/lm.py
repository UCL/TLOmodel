import numbers
from enum import Enum, auto
from math import prod
from typing import Any, Callable, Union

import numpy as np
import pandas as pd

from tlo import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Predictor(object):

    def __init__(self, property_name: str = None, external: bool = False):
        """A Predictor variable for the regression model. The property_name is a property of the
         population dataframe e.g. age, sex, etc."""
        self.property_name = property_name

        # If this is a property that is not part of the population dataframe
        if external:
            assert property_name is not None, "Can't have an unnamed external predictor"
            self.property_name = f'__{self.property_name}__'

        self.conditions = list()
        self.callback = None
        self.has_otherwise = False

    def when(self, condition: Union[str, float, bool], value: float) -> 'Predictor':
        assert self.callback is None, "Can't use `when` on Predictor with function"
        return self._coeff(condition=condition, coefficient=value)

    def otherwise(self, value: float) -> 'Predictor':
        assert self.property_name is not None, "Can't use `otherwise` condition on unnamed Predictor"
        assert self.callback is None, "Can't use `otherwise` on Predictor with function"
        return self._coeff(coefficient=value)

    def apply(self, callback: Callable[[Any], float]) -> 'Predictor':
        assert self.property_name is not None, "Can't use `apply` on unnamed Predictor"
        assert len(self.conditions) == 0, "Can't specify `apply` on Predictor with when/otherwise conditions"
        assert self.callback is None, "Can't specify more than one callback for a Predictor"
        self.callback = callback
        return self

    def _coeff(self, *, coefficient, condition=None) -> 'Predictor':
        """Adds the coefficient for the Predictor. The arguments can be two:
                `coeff(condition, value)` where the condition evaluates the property value to true/false
                `coeff(value)` where the value is given to all unconditioned values of the property
        The second style (unconditioned value) only makes sense after one or more conditioned values
        """
        # If there isn't a property name
        if self.property_name is None:
            # We use the supplied condition literally
            self.conditions.append((condition, coefficient))
            return self

        # Otherwise, the condition is applied on a specific property
        if isinstance(condition, str):
            # Handle either a complex condition (begins with an operator) or implicit equality
            if condition[0] in ['!', '=', '<', '>', '~', '(', '.']:
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
            parsed_condition = f'({self.property_name} == {condition})'
        elif condition is None:
            assert not self.has_otherwise, "You can only give one unconditioned value to predictor"
            self.has_otherwise = True
            parsed_condition = None
        else:
            raise RuntimeError(f"Unhandled condition: {condition}")

        self.conditions.append((parsed_condition, coefficient))
        return self

    def __str__(self):
        if self.property_name and self.property_name.startswith('__'):
            name = f'{self.property_name.strip("__")} (external)'
        else:
            name = self.property_name
        if self.callback:
            return f"{name} -> callback({self.callback})"
        out = []
        previous_condition = None
        for condition, value in self.conditions:
            if condition is None:
                out.append(f'{" " * len(previous_condition)} -> {value} (otherwise)')
            else:
                out.append(f"{condition} -> {value}")
                previous_condition = condition
        return "\n  ".join(out)


class LinearModelType(Enum):
    """
    The type of model specifies how the results from the predictor are combined:
    'additive' -> adds the effect_sizes from the predictors
    'logisitic' -> multiples the effect_sizes from the predictors and applies the transform x/(1+x)
    [Thus, the intercept can be taken to be an Odds and effect_sizes Odds Ratios,
    and the prediction is a probability.]
    'multiplicative' -> multiplies the effect_sizes from the predictors
    """
    ADDITIVE = auto()
    LOGISTIC = auto()
    MULTIPLICATIVE = auto()
    # the 'custom' is used internally by the custom() method
    CUSTOM = auto()


class LinearModel(object):

    def __init__(self, lm_type: LinearModelType, intercept: float, *args: Predictor):
        """
        A linear model has an intercept and zero or more Predictor variables.
        """
        assert lm_type in LinearModelType, 'Model should be one of the prescribed LinearModelTypes'
        self.lm_type = lm_type

        assert isinstance(intercept, (float, int)), "Intercept is not specified or wrong type."
        self.intercept = intercept

        self.predictors = list()
        for predictor in args:
            assert isinstance(predictor, Predictor)
            self.predictors.append(predictor)

    @staticmethod
    def custom(predict_function, **kwargs):
        """Define a linear model using the supplied function

        The function acts as a drop-in replacement to the predict function and must implement the interface:

            (df: Union[pd.DataFrame, pd.Series], rng: np.random.RandomState = None, **kwargs)

        It is the responsibility of the caller of predict to ensure they pass either a dataframe or an
        individual record as expected by the custom function.

        See test_custom() in test_lm.py for a couple of examples.
        """
        # create an instance of a custom linear model
        custom_model = LinearModel(LinearModelType.CUSTOM, 0)
        # replace this instance's predict method
        # see https://stackoverflow.com/questions/28127874/monkey-patching-python-an-instance-method
        custom_model.predict = predict_function.__get__(custom_model, LinearModel)
        # save value to any keyword arguments inside of this linear model
        for k, v in kwargs.items():
            # check the name doesn't already exist
            assert not hasattr(custom_model, k), f"Cannot store argument '{k}' as name already exists; change name."
            setattr(custom_model, k, v)
        return custom_model

    def model_string_and_callback_predictors(self):
        null_op_value = 0 if self.lm_type == LinearModelType.ADDITIVE else 1
        predictor_strings = []
        callback_predictors = []
        for predictor in self.predictors:
            if predictor.callback is None:
                has_catch_all_condition = False
                for i, (condition, value) in enumerate(predictor.conditions):
                    if i == 0:
                        if condition is None:
                            predictor_str = f"{value}"
                            any_prev_conds = f"True"
                            has_catch_all_condition = True
                        else:
                            predictor_str = f"({condition}) * {value}"
                            any_prev_conds = f"{condition}"
                    else:
                        if condition is None:
                            predictor_str += f" + (~({any_prev_conds})) * {value}"
                            has_catch_all_condition = True
                        else:
                            predictor_str += f" + (~({any_prev_conds}) & {condition}) * {value}"
                            any_prev_conds += f" | {condition}"
                if not has_catch_all_condition:
                    predictor_str += f" + ~({any_prev_conds}) * {null_op_value}" 
                predictor_strings.append(f"({predictor_str})")
            else:
                callback_predictors.append(predictor)

        if len(predictor_strings) > 0:
    
            if (self.lm_type == LinearModelType.ADDITIVE and self.intercept != 0) or self.intercept != 1:
                predictor_strings.append(f"{self.intercept}")

        if self.lm_type == LinearModelType.ADDITIVE:
            model_string = " + ".join(predictor_strings)
        else:
            model_string = " * ".join(predictor_strings)

        return model_string, callback_predictors


    def predict(self, df: pd.DataFrame, rng: np.random.RandomState = None, **kwargs) -> pd.Series:

        if kwargs:
            new_columns = {}
            for column_name, value in kwargs.items():
                new_columns[f'__{column_name}__'] = kwargs[column_name]
            df = df.assign(**new_columns)
        
        converted_columns = {}
        for column_name, dtype in zip(df.columns, df.dtypes):
            if isinstance(dtype, pd.CategoricalDtype) and dtype.categories.dtype == "int":
                converted_columns[column_name] = df[column_name].astype("int")
        if len(converted_columns) > 0:
            df = df.assign(**converted_columns)

        model_string, callback_predictors = self.model_string_and_callback_predictors()

        if model_string != "":

            try:
                result = df.eval(model_string)
            except (TypeError, ValueError):
                result = df.eval(model_string, engine="python")

        else:
            result = pd.Series(data=self.intercept, index=df.index)

        if len(callback_predictors) > 0:
            callback_results = [df[p.property_name].apply(p.callback) for p in callback_predictors]
            if self.lm_type == LinearModelType.ADDITIVE:
                result += sum(callback_results)
            else:
                result *= prod(callback_results)

        if self.lm_type == LinearModelType.LOGISTIC:
            result = result / (1 + result)

        if rng:
            outcome = rng.random_sample(len(result)) < result
            # pop the boolean out of the series if we have a single row, otherwise return the series
            if len(outcome) == 1:
                return outcome.iloc[0]
            else:
                return outcome
        else:
            return result

    @staticmethod
    def multiplicative(*predictors: Predictor):
        """Returns a multplicative LinearModel with intercept=1.0

        :param predictors: One or more Predictor objects defining the model
        """
        return LinearModel(LinearModelType.MULTIPLICATIVE, 1.0, *predictors)

    def __str__(self):
        out = "LinearModel(\n"\
              f"  {self.lm_type},\n"\
              f"  intercept = {self.intercept},\n"
        for predictor in self.predictors:
            out += f'  {predictor}\n'
        out += ")"
        return out
