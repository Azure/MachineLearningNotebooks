# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse

print("*********************************************************")
print("Hello Azure ML!")

parser = argparse.ArgumentParser()
parser.add_argument('--numbers-in-sequence', type=int, dest='num_in_sequence', default=10,
                    help='number of fibonacci numbers in sequence')
args = parser.parse_args()
num = args.num_in_sequence


def fibo(n):
    if n < 2:
        return n
    else:
        return fibo(n - 1) + fibo(n - 2)


try:
    from azureml.core import Run
    run = Run.get_context()
    print("Log Fibonacci numbers.")
    for i in range(0, num - 1):
        run.log('Fibonacci numbers', fibo(i))
    run.complete()
except:
    print("Warning: you need to install Azure ML SDK in order to log metrics.")

print("*********************************************************")
