# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

print("*********************************************************")
print("Hello Azure ML!")

try:
    from azureml.core import Run
    run = Run.get_context()
    print("Log Fibonacci numbers.")
    run.log_list('Fibonacci numbers', [0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
    run.complete()
except:
    print("Warning: you need to install Azure ML SDK in order to log metrics.")

print("*********************************************************")
