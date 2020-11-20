import math
from tabulate import tabulate


def Fibonacci(n):
    if n<=0:
        print("Incorrect input")
    # First Fibonacci number is 0
    elif n==1:
        return 0
    # Second Fibonacci number is 1
    elif n==2:
        return 1
    else:
        return Fibonacci(n-1)+Fibonacci(n-2)

if __name__ == '__main__':
    n = 1
    while True:
        result = Fibonacci(n)
        table = [("n", n), ("fibonacci", result)]
        print(tabulate(table))
        print()
        n += 1
        if n == 10:
            print("Bye!")
            break