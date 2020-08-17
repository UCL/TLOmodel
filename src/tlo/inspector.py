# Best to run this from within PyCharm configuration
# with working directory set to (e.g.)
# /Users/matthewgillman/PycharmProjects/TLOmodel
# and Script path
# /Users/matthewgillman/PycharmProjects/TLOmodel/src/tlo/inspector.py
import inspect
from tlo.methods import mockitis

if __name__ == '__main__':

    # This will print out all classes in mockitis.py
    # including those imported by mockitis.py
    stuff = inspect.getmembers(mockitis)
    for name, obj in stuff:
        if inspect.isclass(obj):
            print(name)
            print (obj)

    # Just obtain mockitis.py's classes:
    # Is this robust enough?
    print ("Classes defined in mockitis.py only:")
    leader = "tlo.methods.mockitis"
    for name, obj in stuff:
        if leader in str(obj) and inspect.isclass(obj):
            print (obj)


