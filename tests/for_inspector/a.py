import inspect

class Person:

    def __init__(self, name):
        self._name = name

    def set_name(self, name):
        self._name = name

    def get_name(self):
        return self._name


class Employee(Person):
    def __init__(self, name):
        super().__init__(name)
        self._salary = 0

    def pay(self, salary):
        self._salary = salary

    def get_details(self):
        info = f"Name: {self._name}, Salary: {self._salary}"
        return info


if __name__ == '__main__':
    p = Person("john")
    #import pdb;pdb.set_trace()
    print(f"The name of this person is {p.get_name()}")
    e = Employee("peter")
    e.pay(30000)
    print(f"Employee details: {e.get_details()}")


