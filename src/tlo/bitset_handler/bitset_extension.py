"""
Current limitations:

- We can't do {"a"} in Series yet (need to find appropriate method overwrite)
"""
from __future__ import annotations

import re
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterable,
    Optional,
    Set,
    Sequence,
    Tuple,
    Type,
    TypeAlias,
)

import numpy as np
from numpy.dtypes import BytesDType
from numpy.typing import NDArray
import pandas as pd
from pandas._typing import type_t, TakeIndexer
from pandas.core.arrays.base import ExtensionArray
from pandas.core.dtypes.base import ExtensionDtype

if TYPE_CHECKING:
    from pandas._typing import type_t

# Assume nodes are strings, else we can't construct from string when passed the name!
# We can likely get around this with some careful planning, but we'd have to figure out how
# to pass type-metadata for the elements from inside the output of self.name, so that casting
# was successful.
ALLOWABLE_ELEMENT_TYPES = (str,)
CastableForPandasOps: TypeAlias = (
    "NodeType"
    | Iterable["NodeType"]
    | NDArray[np.uint8]
    | NDArray[np.bytes_]
    | "BitsetArray"
)
NodeType: TypeAlias = str


class BitsetDtype(ExtensionDtype):
    """
    A Bitset is represented by a fixed-width string, whose characters are each a uint8.
    Elements of the set map 1:1 to these characters.

    If the elements set is indexed starting from 0, then:
    - The quotient of these indices (modulo 8) is the character within the string that contains the bit representing the element,
    - The remainder (modulo 8) is the index within said character that represents the element itself.

    The element map takes an element of the bitset as a key, and returns a tuple whose first element is the
    corresponding string-character index, and the latter the uint8 representation of the element within that
    string character.
    """
    _element_map: Dict[NodeType, Tuple[int, np.uint8]]
    _elements: Tuple[NodeType]
    _index_map: Dict[Tuple[int, np.uint8], NodeType]
    _metadata = ("_elements",)

    @classmethod
    def construct_array_type(cls) -> type_t[BitsetArray]:
        return BitsetArray

    @classmethod
    def construct_from_string(cls, string: str) -> BitsetDtype:
        """
        Construct an instance of this class by passing in a string of the form
        that str(<instance of this class>) produces.
        
        That is, given a string of the form
        bitset(#elements): e1, e2, e3, ...

        this method will return a BitsetDtype with elements e1, e2, e3, ... etc.
        """
        if not isinstance(string, str):
            raise TypeError(f"'construct_from_string' expects a string, got {type(string)}")

        string_has_bitset_prefix = re.match("bitset\((\d+)\):", string)
        if string_has_bitset_prefix:
            string = string.removeprefix(string_has_bitset_prefix.group(0))
        if "," not in string:
            raise TypeError(
                "Need at least 2 (comma-separated) elements in string to construct bitset."
            )
        return BitsetDtype(s.strip() for s in string.split(","))

    @property
    def elements(self) -> Tuple[NodeType]:
        return self._elements

    @property
    def fixed_width(self) -> int:
        """
        Fixed-length of the character string that represents this bitset.
        """
        return int(np.ceil(self.n_elements / 8.0))

    @property
    def n_elements(self) -> int:
        return len(self._elements)

    @property
    def na_value(self) -> np.bytes_:
        return self.type(self.fixed_width)

    @property
    def name(self) -> str:
        return self.__str__()

    @property
    def np_array_dtype(self) -> BytesDType:
        return BytesDType(self.fixed_width)

    @property
    def type(self) -> Type[np.bytes_]:
        return self.np_array_dtype.type

    def __init__(self, elements: Iterable[NodeType]) -> None:
        # Take only unique elements.
        # Sort elements alphabetically for consistency when constructing Bitsets that
        # represent the same items.
        # Cast all element types to strings so that construct_from_string does not need
        # metadata about the type of each element.
        provided_elements = sorted([e for e in elements])
        if not all(
            isinstance(e, ALLOWABLE_ELEMENT_TYPES) for e in provided_elements
        ):
            raise TypeError(f"BitSet elements must be one of type: {ALLOWABLE_ELEMENT_TYPES}")
        self._elements = tuple(
            sorted(set(provided_elements), key=lambda x: provided_elements.index(x))
        )

        if len(self._elements) <= 1:
            raise ValueError("Bitsets must have at least 2 possible elements (use bool for 1-element sets).")

        # Setup the element map and its inverse, one-time initialisation cost.
        self._element_map = {
            e: (index // 8, np.uint8(2 ** (index % 8)))
            for index, e in enumerate(self._elements)
        }
        self._index_map = {loc: element for element, loc in self._element_map.items()}

    def __repr__(self) -> str:
        return f"bitset({self.n_elements}): {', '.join(str(e) for e in self._elements)}"

    def __str__(self) -> str:
        return self.__repr__()

    def as_bytes(self, collection: Iterable[NodeType] | NodeType) -> np.bytes_:
        """
        Return the bytes representation of this set or single element.
        """
        return np.bytes_(self.as_uint8_array(collection))

    def as_set(self, binary_repr: np.bytes_) -> Set[NodeType]:
        """
        Return the set corresponding to the binary representation provided.
        """
        elements_in_set = set()
        for char_index, byte_value in enumerate(binary_repr):
            bin_rep = format(byte_value, "b")
            elements_in_set |= {
                self._index_map[(char_index, np.uint8(2**i))]
                for i, bit in enumerate(reversed(bin_rep))
                if bit == "1"
            }
        return elements_in_set

    def as_uint8_array(self, collection: Iterable[NodeType] | NodeType) -> NDArray[np.uint8]:
        """
        Return the collection of elements as a 1D array of ``self.fixed_width`` uint8s.
        Each uint8 corresponds to the bitwise representation of a single character
        in a character string.

        A single element will be broadcast to a (1,) numpy array.
        """
        if isinstance(collection, ALLOWABLE_ELEMENT_TYPES):
            collection = set(collection)
        
        output = np.zeros((self.fixed_width, 1), dtype=np.uint8)
        for element in collection:
            char, bin_repr = self._element_map[element]
            output[char] |= bin_repr
        return output.squeeze(axis=1)

    def element_loc(self, element: NodeType) -> Tuple[int, np.uint8]:
        """
        Location in of the bit corresponding to the element in this bitset.

        Each element in the bitset is mapped to a single bit via the _element_map, and
        can be located by specifying both:
        - The index of the character in the fixed-width string that represents the bitset.
        - The power of 2 within the uint8 representation of the the single character that corresponds to the element.

        For example, a bitset of 18 elements is stored as a fixed-width string of 3 characters,
        giving 24 bits to utilise. These are further subdivided into groups of 8, the first 8
        corresponding to the uint8 representation of the 0-indexed character, and so on. Each element within
        this bitset is assigned a power of two within one of the character representations.

        :param element: Element value to locate.
        :returns: The character index, and ``np.uint8`` representation of the element, unpacked in that order.
        """
        return self._element_map[element]


class BitsetArray(ExtensionArray):
    """
    Represents a series of Bitsets; each element in the series is a fixed-width bytestring,
    which represents some possible combination of elements of a bitset as defined by
    ``self.dtype``.

    When extracting a single entry via ``.loc`` or ``.at``, the value returned is a ``set``.
    This means that operations such as ``self.loc[0] |= {"1"}`` will behave as set operations
    from base Python. This is achieved by setting the behaviour of the ``__setitem__`` method
    to interpret ``set`` values as representations of the underlying bitset, thus causing them
    to be cast to their bytestring representation being being assigned.

    Supported Operations (slices)
    -----------------------------
    When operating on slices or masks of the series, we have to re-implement the desired operators
    so that users can continue to pass ``set``s as scalar arguments on the left.

    Currently implemented methods are:

    = :
        Directly assign the value on the right to the entry/entries on the left.
    +, | :
        Perform union of the values on the left with those on the right.
    +=, |= :
        In-place union; add values on the right to the sets on the left.
    & :
        Perform intersection of the values on the left with those on the right.
    &= :
        In-place intersection; retain only elements on the left that appear on the right.
    -, -= :
        Remove the values on the right from the sets on the left.
    """

    _data: NDArray[np.bytes_]
    _dtype: BitsetDtype

    @staticmethod
    def uint8s_to_byte_string(arr: np.ndarray[np.uint8]) -> NDArray[np.bytes_]:
        """
        Returns a view of an array of ``np.uint8``s of shape ``(M, N)``
        as an array of ``M`` fixed-width byte strings of size ``N``.
        """
        fixed_width = arr.shape[1]
        return arr.view(f"{fixed_width}S").squeeze()

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[BitsetArray]) -> BitsetArray:
        concat_data = np.concatenate(bsa._data for bsa in to_concat)
        return cls(concat_data, to_concat[0].dtype)

    @classmethod
    def _from_sequence(
        cls, scalars: Iterable[Set[NodeType]], *, dtype: BitsetDtype | None = None, copy: bool = False
    ) -> BitsetArray:
        """
        Construct a new BitSetArray from a sequence of scalars.

        Parameters
        ----------
        scalars : Sequence[Set[NodeType] | NodeType]
            Sequence of sets of elements (or single-values to be interpreted as single-element sets)
        dtype : dtype, optional
            Cast to this datatype, only BitsetDtype is supported if not None. If None, an attempt will be made to
            construct an appropriate BitsetDtype using the scalar values provided.
        copy : bool, default False
            If True, copy the underlying data.
        """
        # Check that we have only been passed sets as scalars. Implicitly convert single-items to sets.
        for s in scalars:
            if not isinstance(s, set):
                if isinstance(s, ALLOWABLE_ELEMENT_TYPES):
                    s = set(s)
                else:
                    raise ValueError(f"{s} cannot be cast to an element of a bitset.")
        assert all(isinstance(s, set) for s in scalars), "Not all scalars have been cast correctly."

        # If no dtype has been provided, attempt to construct an appropriate BitsetDtype.
        if dtype is None:
            # Determine the elements in the bitset by looking through the scalars
            all_elements = set()
            for s in scalars:
                # Take union of sets to form list of all possible Bitset elements
                all_elements |= s
            dtype = BitsetDtype(all_elements)
        elif not isinstance(dtype, BitsetDtype):
            raise TypeError(f"BitsetArray cannot be constructed with dtype {dtype}")

        # With an appropriate dtype, we can construct the data array to pass to the constructor.
        # We will need to convert each of our scalars to their binary representations before passing though.
        data = np.zeros((len(scalars),), dtype=dtype.np_array_dtype)
        view_format = f"{dtype.fixed_width}B" if dtype.fixed_width != 1 else "(1,1)B"
        data_view = data.view(view_format)
        for series_index, s in enumerate(scalars):
            for element in s:
                char, u8_repr = dtype.element_loc(element=element)
                data_view[series_index, char] |= u8_repr
        return cls(data, dtype, copy=copy)

    @classmethod
    def _from_factorized(cls, uniques: np.ndarray, original: BitsetArray) -> BitsetArray:
        return cls(uniques, original.dtype)

    @property
    def _uint8_view_format(self) -> str:
        """
        Format string to be applied to self._data, so that the output of

        self._data.view(<this function>)

        returns a numpy array of shape (len(self), self.dtype.fixed_width)
        and dtype uint8.
        """
        return f"({self.dtype.fixed_width},)B"

    @property
    def _uint8_view(self) -> NDArray[np.bytes_]:
        """
        Returns a view of the fixed-width byte strings stored in ``self._data``
        as an array of ``numpy.uint8``s, with shape

        ``(len(self._data), self.dtype.fixed_width)``.

        Each row ``i`` of this view corresponds to a bitset stored in this array.
        The value at index ``i, j`` in this view is the ``uint8`` that represents
        character ``j`` in ``self._data[i]``, which can have bitwise operations
        performed on it.  
        """
        return self._data.view(self._uint8_view_format)

    @property
    def dtype(self) -> BitsetDtype:
        return self._dtype

    @property
    def nbytes(self) -> int:
        return self._data.nbytes

    def __init__(
        self,
        data: Iterable[BytesDType] | np.ndarray[BytesDType],
        dtype: BitsetDtype,
        copy: bool = False,
    ) -> None:
        """ """
        if not isinstance(dtype, BitsetDtype):
            raise TypeError("BitsetArray must have BitsetDtype data.")

        self._data = np.array(data, copy=copy, dtype=dtype.type)
        self._dtype = dtype

    def __add__(
        self, other: CastableForPandasOps
    ) -> BitsetArray:
        """
        Element-wise union, delegates to ``__or__``.
        """
        return self.__or__(other)

    def __and__(self,other: CastableForPandasOps
    ) -> BitsetArray:
        """
        Entry-wise intersection with other.

        - If other is ``NodeType`` or ``Iterable[NodeType]``, perform entry-wise AND with the set
        representing the passed element values.
        - If other is ``BitsetArray`` of compatible shape, take entry-wise intersection.
        - If other is compatible ``np.ndarray``, take entry-wise intersection.

        Under the hood this is bitwise AND with other; self AND other.
        """
        return BitsetArray(
            self.__operate_bitwise(
                lambda A, B: A & B, other, return_as_bytestring=True
            ),
            dtype=self.dtype,
        )

    def __cast_to_uint8(self, other: CastableForPandasOps) -> None:
        """
        Casts the passed object to a ``np.uint8`` array that is compatible with bitwise operations
        on ``self._uint8_view``. See the docstring for behaviour in the various usage cases.

        Scalar elements:
            Cast to single-element sets, then treated as set.
        
        Sets:
            Are converted to the (array of) uint8s that represents the set.
        
        ``np.ndarray``s of ``np.uint8``
            Are returned if they have the same number of columns as ``self._uint8_view``.
        
        ``np.ndarray``s of ``np.dtype("Sx")``
            If ``x`` corresponds to the same fixed-width as ``self.dtype.np_array_dtype``, are cast
            to the corresponding ``np.uint8`` view, like ``self._uint8_view`` is from ``self._data``.
        
        BitsetArrays
            Return their ``_uint8_view`` attribute.
        """
        if isinstance(other, ALLOWABLE_ELEMENT_TYPES):
            # Treat single-elements as single-element sets
            other = set(other)
        if isinstance(other, BitsetArray):
            if self.dtype != other.dtype:
                raise TypeError("Cannot cast a different Bitset to this one!")
            else:
                cast = other._uint8_view
        elif isinstance(other, np.ndarray):
            if other.dtype == np.uint8 and other.shape[0] == self._uint8_view.shape[0]:
                # Compatible uint8s, possibly a view of another fixed-width bytestring array
                cast = other
            elif other.dtype == self.dtype.np_array_dtype:
                # An array of compatible fixed-width bytestrings
                cast = other.view(self._uint8_view_format)
        else:
            # Must be a collection of elements (or will error), so cast.
            cast = self.dtype.as_uint8_array(other)
        return cast

    def __operate_bitwise(
        self,
        op: Callable[[NDArray[np.uint8], NDArray[np.uint8]], NDArray[np.uint8]],
        r_value: CastableForPandasOps,
        l_value: Optional[CastableForPandasOps] = None,
        return_as_bytestring: bool = False,
    ) -> NDArray[np.bytes_] | NDArray[np.uint8]:
        """
        Perform a bitwise operation on two compatible ``np.ndarray``s of ``np.uint8``s.

        By default, the left value passed to the operator is assumed to be ``self._uint8_data``.

        Return value is the result of the bitwise operation, as an array of uint8s. If you wish
        to have this converted to the corresponding bytestring(s) before returning, use the
        return_as_bytestring argument.

        :param op: Bitwise operation to perform on input values.
        :param r_value: Right-value to pass to the operator.
        :param l_value: Left-value to pass to the operator.
        :param return_as_bytestring: Result will be returned as a fixed-width bytestring.
        """
        l_value = self._uint8_view if l_value is None else self.__cast_to_uint8(l_value)
        op_result = op(l_value, self.__cast_to_uint8(r_value))
        if return_as_bytestring:
            op_result = self.uint8s_to_byte_string(op_result)
        return op_result

    def __eq__(self, other) -> bool:
        if isinstance(other, (pd.Series, pd.DataFrame, pd.Index)):
            return NotImplemented
        elif isinstance(other, set):
            ans = self._data == self.dtype.as_bytes(other)
        else:
            ans = self._data = other
        return np.squeeze(ans)

    def __getitem__(self, item: int | slice | NDArray) -> BitsetArray:
        return (
            self.dtype.as_set(self._data[item])
            if isinstance(item, int)
            else BitsetArray(self._data[item], dtype=self.dtype)
        )

    def __len__(self) -> int:
        return self._data.shape[0]

    def __or__(
        self, other: CastableForPandasOps
    ) -> BitsetArray:
        """
        Entry-wise union with other.

        - If other is ``NodeType`` or ``Iterable[NodeType]``, perform entry-wise OR with the set
        representing the passed element values.
        - If other is ``BitsetArray`` of compatible shape, take entry-wise union.
        - If other is compatible ``np.ndarray``, take entry-wise union.

        Under the hood this is bitwise OR with other; self OR other.
        """
        return BitsetArray(
            self.__operate_bitwise(
                lambda A, B: A | B, other, return_as_bytestring=True
            ),
            dtype=self.dtype,
        )

    def __setitem__(
        self,
        key: int | slice | NDArray,
        value: (
            np.bytes_
            | NodeType
            | Set[NodeType]
            | Sequence[np.bytes_ | NodeType| Set[NodeType]]
        ),
    ) -> None:
        if isinstance(value, ALLOWABLE_ELEMENT_TYPES + (set,)):
            # Interpret this as a "scalar" set that we want to set all values to
            value = self.dtype.as_bytes(value)
        elif isinstance(value, np.bytes_):
            # Value is a scalar that we don't need to convert
            pass
        else:
            # Assume value is a sequence, and we will have to convert each value in turn
            value = [
                v if isinstance(v, np.bytes_) else self.dtype.as_bytes(v) for v in value
            ]
        self._data[key] = value

    def __sub__(
        self, other: CastableForPandasOps
    ) -> BitsetArray:
        """
        Remove elements from the Bitsets represented here.

        - If other is ``NodeType``, remove the single element from all series entries.
        - If other is ``Iterable[NodeType]``, remove all elements from all series entries.
        - If other is ``BitsetArray`` of compatible shape, take element-wise complements of series entries.
        - If other is compatible ``np.ndarray``, take element-wise complements of series entries.

        Under the hood this the bitwise operation self AND (NOT other).
        """
        return BitsetArray(
            self.__operate_bitwise(
                lambda A, B: A & (~B), other, return_as_bytestring=True
            ),
            dtype=self.dtype,
        )

    def _formatter(self, boxed: bool = False) -> Callable[[BytesDType], str | None]:
        if boxed: # If rendering an individual data value
            return lambda x: ",".join(x)
        return repr # Render the table itself

    def copy(self) -> BitsetArray:
        return BitsetArray(self._data, self.dtype, copy=True)

    def isna(self) -> NDArray:
        """
        TODO: This isn't a great way to express missing data, but equally a bitset doesn't really ever contain missing data...
        """
        return np.isnan(self._data)

    def take(
        self,
        indices: TakeIndexer,
        *,
        allow_fill: bool = False,
        fill_value: Optional[BytesDType | Set[NodeType]] = None,
    ) -> BitsetArray:
        if allow_fill:
            if isinstance(fill_value, set):
                fill_value = self.dtype.as_bytes(fill_value)
            elif fill_value is None:
                fill_value = self.dtype.na_value
            elif not isinstance(fill_value, self.dtype.type):
                raise TypeError(
                    f"Fill value must be of type {self.dtype.type} (got {type(fill_value).__name__})"
                )
            scalars = np.empty((len(indices), ), dtype=self.dtype.type)
            scalars[indices[indices >= 0]] = self._data[indices[indices >= 0]]
            scalars[indices[indices < 0]] = fill_value
        else:
            scalars = np.take(self._data, indices)
        return self._from_sequence(scalars)


if __name__ == "__main__":

    normal = pd.Series([1, 2, 3, 4, 5])

    bdt = BitsetDtype.construct_from_string("a, b, c, d")
    big_bdt = BitsetDtype.construct_from_string("0, 1, 2, 3, 4, 5, 6, 7, 8")

    small_instance = bdt.as_bytes({"a", "d"})
    small_set_from_instance = bdt.as_set(small_instance)

    big_instance = big_bdt.as_bytes({"4", "8"})
    set_from_big_instance = big_bdt.as_set(big_instance)

    small_s = pd.Series([{"a"}, {"b", "c"}, {"d", "a"}], dtype=bdt)
    big_s = pd.Series([{"1"}, {"4", "8"}], dtype=big_bdt)

    small_s == {"a"}
    big_s == {"4", "8"}

    f = small_s + {"a"}
    g = big_s + {"1"}

    print(g)
    g += {"1"}
    print(g)
    g -= {"1"}
    print(g)

    pass
