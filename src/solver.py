import math
from tabulate import tabulate


class Solver:

    def demo(self, a, b, c):
        return a * b * c

if __name__ == '__main__':
    solver = Solver()

while True:
    a = int(input("a: "))
    b = int(input("b: "))
    c = int(input("c: "))
    result = solver.demo(a, b, c)
    table = [("a", a), ("b", b), ("c", c), ("result", result)]
    print(tabulate(table))