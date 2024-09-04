import ast
import builtins
import numbers
from enum import Enum, auto
from math import prod
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.core.computation.parsing import clean_column_name

from tlo import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Predictor(object):

    def __init__(
        self,
        property_name: str = None,
        external: bool = False,
        conditions_are_mutually_exclusive: Optional[bool] = None,
        conditions_are_exhaustive: Optional[bool] = False,
    ):
        """A Predictor variable for the regression model.

        :param property_name: A property of the population dataframe e.g. age, sex, etc.
            or if ``external=True`` the name of the external property that will be
            passed as a keyword argument to the ``LinearModel.predict`` method.
        :param external: Whether the named property is external (``True``) and so will
            be passed as a keyword argument to the ``LinearModel.predict`` method) or is
            a property of the population dataframe (``False``).
        :param conditions_are_mutually_exclusive: Whether the set of conditions that
            are declared for this predictor are all mutually exclusive, that is, for any
            pair of conditions, one condition evaluating to ``True`` implies the other
            must evaluate to ``False``. If this is declared to be the case a more
            efficient method of evaluation will be used in ``LinearModel.predict``. Note
            however that the validity of this declaration will not be checked so if this
            is set to ``True`` for predictors with non-mutually exclusive conditions,
            the model output will be erroneous.
        :param conditions_are_exhaustive: Whether the set of conditions that are
            declared for this predictor are all exhaustive, that is at least one
            condition will always be ``True`` irrespective of the value of the property.
            If this is declared to be the case, a more efficient method of evaluation
            maye be used in ``LinearModel.predict`, though if a catch-all ``otherwise``
            condition is included this flag will provide no benefit. Note that the
            validity of this declaration will not be checked so if this is set to
            ``True`` for predictors with non-exhaustive conditions, the model output
            will be erroneous.
        """
        self.property_name = property_name

        # If this is a property that is not part of the population dataframe
        if external:
            assert property_name is not None, "Can't have an unnamed external predictor"
            self.property_name = f'__{self.property_name}__'

        self.conditions = list()
        self.callback = None
        self.has_otherwise = False
        self.conditions_are_mutually_exclusive = conditions_are_mutually_exclusive
        self.conditions_are_exhaustive = conditions_are_exhaustive

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

    def __init__(
        self,
        lm_type: LinearModelType,
        intercept: Union[float, int],
        *predictors: Predictor
    ):
        """A linear model has an intercept and zero or more ``Predictor`` variables.

        :param lm_type: Model type to use.
        :param intercept: Intercept term for the model.
        :param *predictors: Any ``Predictor`` instances to use in computing output.
        """
        assert lm_type in LinearModelType, (
            "Model should be one of the prescribed LinearModelTypes"
        )
        self._lm_type = lm_type

        assert isinstance(intercept, (float, int)), (
            "Intercept is not specified or wrong type."
        )
        assert np.isfinite(intercept), "Intercept must not be NaN or infinite"
        self._intercept = intercept

        # Store predictors as tuple and expose via read-only property to prevent
        # updates after model initialisation
        self._predictors = tuple(predictors)
        non_predictors = [p for p in self._predictors if not isinstance(p, Predictor)]
        assert len(non_predictors) == 0, (
            f"One or more predictors are of invalid type: {non_predictors}"
        )

        self._parse_predictors()

    @property
    def lm_type(self) -> LinearModelType:
        """The model type."""
        return self._lm_type

    @property
    def intercept(self) -> Union[float, int]:
        """The intercept value for the model."""
        return self._intercept

    @property
    def predictors(self) -> Tuple[Predictor]:
        """The predictors used in calculating the model output."""
        return self._predictors

    @staticmethod
    def multiplicative(*predictors: Predictor):
        """Returns a multplicative LinearModel with intercept=1.0

        :param predictors: One or more Predictor objects defining the model
        """
        return LinearModel(LinearModelType.MULTIPLICATIVE, 1.0, *predictors)

    @staticmethod
    def custom(predict_function, **kwargs):
        """Define a linear model using the supplied function

        The function acts as a drop-in replacement to the predict function and must
        implement the interface:

            (
                self: LinearModel,
                df: Union[pd.DataFrame, pd.Series],
                rng: Optional[np.random.RandomState] = None,
                **kwargs
            ) -> pd.Series

        It is the responsibility of the caller of predict to ensure they pass either
        a dataframe or an individual record as expected by the custom function.

        See test_custom() in test_lm.py for a couple of examples.
        """
        # create an instance of a custom linear model
        custom_model = LinearModel(LinearModelType.CUSTOM, 0)
        # replace this instance's predict method
        # see https://stackoverflow.com/a/28127947
        custom_model.predict = predict_function.__get__(custom_model, LinearModel)
        # save value to any keyword arguments inside of this linear model
        for k, v in kwargs.items():
            # check the name doesn't already exist
            assert not hasattr(custom_model, k), (
                f"Cannot store argument '{k}' as name already exists; change name.")
            setattr(custom_model, k, v)
        return custom_model

    def _parse_predictors(self):
        """Set model string, callback predictors and predictor names from predictors.

        Sets `self._model_string` to an expression string (to be evaluated by
        ``pandas.DataFrame.eval``) corresponding to the evaluation of the model output
        for the subset of the predictors which do not define a custom callback function
        and the model intercept, or an empty string if no non-callback predictors are
        present.

        Additionally sets `self._callback_predictors` to a tuple of the omitted
        predictors with custom callback functions and `self._predictor_names` to a set
        of strings corresponding to names specified in the predictors.
        """
        # For additive models a zero coefficient corresponds to no effect while for
        # multiplicative and logistic models the relevant value is one
        null_coeff_value = 0 if self.lm_type == LinearModelType.ADDITIVE else 1
        predictor_strings = []
        callback_predictors = []
        self._predictor_names = set()
        for predictor in self.predictors:
            if predictor.callback is None:
                if predictor.property_name is not None:
                    self._predictor_names.add(predictor.property_name)
                else:
                    # If no property_name specified, predictor conditions will
                    # contain one or more column names therefore parse condition
                    # strings and filter for all name nodes. This will also
                    # add non-column names such as builtin functions so need to
                    # check if names are actually columns before using
                    for condition, _ in predictor.conditions:
                        self._predictor_names.update(
                            node.id for node in ast.walk(ast.parse(condition))
                            if isinstance(node, ast.Name)
                        )
                has_catch_all_condition = False
                for i, (condition, value) in enumerate(predictor.conditions):
                    if i == 0:
                        if condition is None:
                            # 'otherwise' fallback condition - always True. If used as
                            # first condition any other conditions will be ignored as
                            # this condition matches all
                            predictor_str = f"{value}"
                            any_prev_conds = "True"
                            has_catch_all_condition = True
                            break
                        else:
                            predictor_str = f"({condition}) * {value}"
                            any_prev_conds = f"{condition}"
                    else:
                        if condition is None:
                            # 'otherwise' fallback condition - matches all not
                            # so far matched therefore can ignore any remaining
                            # conditions
                            predictor_str += f" + (~({any_prev_conds})) * {value}"
                            has_catch_all_condition = True
                            break
                        elif predictor.conditions_are_mutually_exclusive:
                            # conditions have been declared to be mutually exclusive
                            # therefore we can just multiply conditions by coefficient
                            # values as condition == ~any_prev_conds & condition
                            predictor_str += f" + ({condition}) * {value}"
                            any_prev_conds += f" | {condition}"

                        else:
                            # conditions are potentially non-mutually exclusive and
                            # are applied sequentially in order specified on subset
                            # not matching any previous conditions
                            predictor_str += (
                                f" + (~({any_prev_conds}) & {condition}) * {value}")
                            any_prev_conds += f" | {condition}"
                # If the predictor neither declares that the conditions are exhaustive
                # (i.e. all cases are covered an any_prev_conds is guaranteed to be
                # True) nor an 'otherwise' catch-all condition has been used (in which
                # case any_prev_conds is also guaranteed to be True) then add term
                # corresponding to no effect when no previous conditions matched
                if not (predictor.conditions_are_exhaustive or has_catch_all_condition):
                    predictor_str += f" + ~({any_prev_conds}) * {null_coeff_value}"
                predictor_strings.append(f"({predictor_str})")
            else:
                self._predictor_names.add(predictor.property_name)
                callback_predictors.append(predictor)

        self._callback_predictors = tuple(callback_predictors)

        if len(predictor_strings) > 0:

            if self.intercept != null_coeff_value:
                # Only need to include intercept if its non-zero in additive models
                # or non-unity in multiplicative/logistic models
                predictor_strings.append(f"{self.intercept}")

        if self.lm_type == LinearModelType.ADDITIVE:
            self._model_string = " + ".join(predictor_strings)
        else:
            self._model_string = " * ".join(predictor_strings)

    def _get_column_resolvers(
        self,
        df: pd.DataFrame,
        **external_variables
    ) -> Dict[str, pd.Series]:
        """Construct mapping from predictor column names to column values.

        For use in ``resolvers`` argument to ``pandas.eval`` call.

        Compared to ``pandas.DataFrame._get_cleaned_column_resolvers()`` here only the
        column names present in the model predictors are included when constructing the
        returned dictionary. For dataframes with a large number of columns this is more
        performant than iterating over all columns, of which typically only a small
        subset are used in each linear model. Any external variables specified in
        predictors are also included with dunder-wrapped keys (e.g '__ext_var__').
        """
        column_resolvers = {}
        for name in self._predictor_names:
            # predictor_names may contain built-in names that are not columns
            # therefore we need to check if name is column in dataframe
            col = df.get(name)
            if col is not None:
                cleaned_name = clean_column_name(name)
                if (
                    isinstance(col.dtype, pd.CategoricalDtype)
                    and np.issubdtype(col.dtype.categories.dtype, np.integer)
                ):
                    # `pandas.eval` raises an error when using boolean operations
                    # on series with a categorical dtype with integer categories
                    # therefore if any such columns are present we convert to
                    # double-precision floats - this should be safe providing only
                    # integer categories which have exact floating point representations
                    # are used (which is likely to be the case)
                    column_resolvers[cleaned_name] = col.astype(np.float64)
                else:
                    column_resolvers[cleaned_name] = col
        for name, value in external_variables.items():
            column_resolvers[f"__{name}__"] = pd.Series(value, index=df.index)
        return column_resolvers

    def predict(
        self,
        df: pd.DataFrame,
        rng: Optional[np.random.RandomState] = None,
        squeeze_single_row_output=True,
        **kwargs
    ) -> pd.Series:
        """Evaluate linear model output for a given set of input data.

        :param df: The input ``DataFrame`` containing the input data to evaluate the
          model with.
        :param rng: If set to a NumPy ``RandomState`` instance, returned output will
          be boolean ``Series`` corresponding to Bernoulli random variables sampled
          according to probabilities specified by model output. Otherwise model
          output directly returned.
        :param squeeze_single_row_output: If ``rng`` argument is not ``None`` and this
          argument is set to ``True``, the output for a ``df`` input with a single-row
          will be a scalar boolean value rather than a boolean ``Series``.
        :param **kwargs: Values for any external variables included in model
          predictors.
        """
        # Check that all names specified in predictors are either a column name, an
        # external variable in kwargs (with __ prefix/suffix removed) or a built-in
        for name in self._predictor_names:
            assert (
                name in df
                or (
                    name.startswith("__")
                    and name.endswith("__")
                    and name.strip("__") in kwargs
                )
                or name in builtins.__dict__
            ), (
                f"Predictors include unknown name {name}"
            )

        column_resolvers = self._get_column_resolvers(df, **kwargs)

        if self._model_string != "":
            result = pd.eval(
                self._model_string,
                resolvers=(column_resolvers,),
                engine="python"
            )
        else:
            result = pd.Series(data=self.intercept, index=df.index)

        if len(self._callback_predictors) > 0:
            callback_results = [
                column_resolvers[p.property_name].apply(p.callback)
                for p in self._callback_predictors
            ]
            if self.lm_type == LinearModelType.ADDITIVE:
                result += sum(callback_results)
            else:
                result *= prod(callback_results)

        # Ensure result of floating point type even if all predictor coefficients
        # are integer but intercept is floating point
        if isinstance(self.intercept, float) and result.dtype == int:
            result = result.astype(float)
        # Result series sometimes picks up name from one of predictors - set to
        # None so comparisons with unnamed series in tests pass
        result.name = None

        if self.lm_type == LinearModelType.LOGISTIC:
            # Below is equivalent to result = result / (1 + result) but will give correct
            # output where any elements in result are inf (--> 1.0) or 0.0 (--> 0.0).
            result = (1 / (1 + 1 / result))

        # If the user supplied a random number generator then they want outcomes,
        # not probabilities
        if rng:
            outcome = rng.random_sample(len(result)) < result
            # pop the boolean out of the series if we have a single row,
            # otherwise return the series
            if len(outcome) == 1 and squeeze_single_row_output:
                return outcome.iloc[0]
            else:
                return outcome
        else:
            return result

    def __str__(self):
        out = "LinearModel(\n"\
              f"  {self.lm_type},\n"\
              f"  intercept = {self.intercept},\n"
        for predictor in self.predictors:
            out += f'  {predictor}\n'
        out += ")"
        return out
