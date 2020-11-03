#-
from time import time
from beacon import Beacon

#- Main
if __name__ == "__main__":

    #- Load Text Example
    with open("examples/example_1.txt", "r") as f:
        text = f.read()

    #- Instantiate Beacon
    beacon = Beacon()

    #- Run beacon on text
    t = time()
    result = beacon(text)
    elapsed = time() - t

    #- Print result
    print(result)
    print("Computed in (sec): {}".format(elapsed))
