from typing import Any, List

def sort_preserving_order(to_sort: List[Any], relative_to: List[Any]) -> List:
    """
    """
    def sort_key(item):
        try:
            return relative_to.index(item)
        except ValueError:
            return len(relative_to)
    return sorted(to_sort, key=sort_key)

my_list = ["a", "b", "c", "d"]
custom_order = ["c", "a"]

print(sort_preserving_order(my_list, custom_order))

if "emergency" in "do_emergency":
    print("hi")