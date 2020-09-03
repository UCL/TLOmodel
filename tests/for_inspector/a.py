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


class Father:
    def __init__(self, age):
        self._age = age

    def get_age(self):
        return self._age

class Mother:
    def __init__(self, job):
        self._job = job

    def get_job(self):
        return self._job

class Offspring (Father, Mother):
    def __init__(self, age, job, name):
        #super(Father, self).__init__(age)
        #super(Mother, self).__init__(job)
        self._age = age
        self._job = job
        self._name = name

    def get_name(self):
        return self._name

    def get_info(self):
        str = f"Name: {self.get_name()}, Age: {self.get_age()}, Job: {self.get_job()}"
        return str


if __name__ == '__main__':
    p = Person("john")
    #import pdb;pdb.set_trace()
    print(f"The name of this person is {p.get_name()}")
    e = Employee("peter")
    e.pay(30000)
    print(f"Employee details: {e.get_details()}")

    o = Offspring(111, "Thief", "Bilbo")
    print(o.get_info())
