from __future__ import annotations

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
import pandas as pd
from pandas._typing import type_t, TakeIndexer
from pandas.core.arrays.base import ExtensionArray
from pandas.core.dtypes.base import ExtensionDtype

if TYPE_CHECKING:
    from pandas._typing import type_t


ALLOWABLE_ELEMENT_TYPES = (str, int, float)
NodeType: TypeAlias = str | int | float

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
    _elements: Tuple[NodeType] # NB: we never assume elements are strings, could bind this to a type
    _index_map: Dict[Tuple[int, np.uint8], NodeType]
    _metadata = ("n_elements", "elements")

    @classmethod
    def construct_array_type(cls) -> type_t[BitsetArray]:
        return BitsetArray

    @classmethod
    def construct_from_string(cls, string: str, delimiter: str = ",") -> BitsetDtype:
        """
        Construct an instance of this class by passing in a string of set values, separated by the delimiter.
        Whitespace will be trimmed from the set elements automatically.
        """
        return BitsetDtype(s.strip() for s in string.split(delimiter))

    @property
    def elements(self) -> Set[NodeType]:
        return set(self._elements)

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

    @property
    def bin_format(self) -> str:
        return f"{self.fixed_width}B" if self.fixed_width != 1 else "B"

    @property
    def str_format(self) -> str:
        return f"{self.fixed_width}S" if self.fixed_width != 1 else "S"

    def __init__(self, elements: Iterable[NodeType]) -> None:
        # Take only unique elements, and preserve order of the input iterable for consistency
        # reasons.
        provided_elements = [e for e in elements]
        self._elements = tuple(
            sorted(set(provided_elements), key=lambda x: provided_elements.index(x))
        )
        if len(self._elements) == 0:
            raise ValueError("Bitsets must have at least 1 possible element.")

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

    def as_bytes(self, collection: Iterable[NodeType]) -> str:
        """
        Return the bytes representation of this set.
        """
        output: np.ndarray[np.uint8] = self.type(self.fixed_width).view(self.bin_format)
        if self.fixed_width == 1:
            for element in collection:
                _, bin_repr = self._element_map[element]
                output |= bin_repr
        else:
            for element in collection:
                char, bin_repr = self._element_map[element]
                output[char] |= bin_repr
        return output.tobytes()

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

    _data: np.ndarray[BytesDType]
    _dtype: BitsetDtype

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
        data = (
            np.zeros((len(scalars), ), dtype=dtype.np_array_dtype)
            if dtype.fixed_width != 1
            else np.zeros((len(scalars), 1), dtype=dtype.np_array_dtype)
        )
        data_view = data.view(dtype.bin_format)
        for series_index, s in enumerate(scalars):
            for element in s:
                char, u8_repr = dtype.element_loc(element=element)
                data_view[series_index, char] |= u8_repr
        return cls(data, dtype, copy=copy)

    @classmethod
    def _from_factorized(cls, uniques: np.ndarray, original: BitsetArray) -> BitsetArray:
        return cls(uniques, original.dtype)

    @property
    def _binary_data(self) -> np.ndarray[BytesDType]:
        return self._data.view(self.dtype.bin_format)

    @property
    def dtype(self) -> BitsetDtype:
        return self._dtype

    @property
    def nbytes(self) -> int:
        return self._data.nbytes

    def __init__(
        self,
        data: Iterable[BytesDType],
        dtype: BitsetDtype,
        copy: bool = False,
    ) -> None:
        """ """
        if not isinstance(dtype, BitsetDtype):
            raise TypeError("BitsetArray must have BitsetDtype data.")

        self._data = np.array(data, copy=copy, dtype=dtype.type)
        self._dtype = dtype

    def __eq__(self, other) -> bool:
        if isinstance(other, (pd.Series, pd.DataFrame, pd.Index)):
            return NotImplemented
        elif isinstance(other, set):
            ans = self._data == self.dtype.as_bytes(other)
        else:
            ans = self._data = other
        return np.squeeze(ans)

    def __getitem__(self, item: int | slice | np.ndarray) -> BitsetArray:
        return (
            self._data[item]
            if isinstance(item, int)
            else BitsetArray(self._data[item], dtype=self.dtype)
        )

    def __len__(self) -> int:
        return self._data.shape[0]

    def _formatter(self, boxed: bool = False) -> Callable[[BytesDType], str | None]:
        if boxed: # If rendering an individual data value
            return lambda x: ",".join(self.dtype.as_set(x))
        return repr # Render the table itself

    def copy(self) -> BitsetArray:
        return BitsetArray(self._data, self.dtype, copy=True)

    def isna(self) -> np.ndarray:
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
    pass
