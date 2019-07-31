# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse

print("*********************************************************")
print("Hello Azure ML!")

parser = argparse.ArgumentParser()
parser.add_argument('--numbers-in-sequence', type=int, dest='num_in_sequence', default=10,
                    help='number of fibonacci numbers in sequence')

# This is how you can use a bool argument in Python. If you want the 'my_bool_var' to be True, just pass it
# in Estimator's script_param as script+params:{'my_bool_var': ''}.
# And, if you want to use it as False, then do not pass it in the Estimator's script_params.
# You can reverse the behavior by setting action='store_false' in the next line.
parser.add_argument("--my_bool_var", action='store_true')

args = parser.parse_args()
num = args.num_in_sequence
my_bool_var = args.my_bool_var


def fibo(n):
    if n < 2:
        return n
    else:
        return fibo(n - 1) + fibo(n - 2)


try:
    from azureml.core import Run
    run = Run.get_context()
    print("The value of boolean parameter 'my_bool_var' is {}".format(my_bool_var))
    print("Log Fibonacci numbers.")
    for i in range(0, num - 1):
        run.log('Fibonacci numbers', fibo(i))
    run.complete()
except:
    print("Warning: you need to install Azure ML SDK in order to log metrics.")

print("*********************************************************")
