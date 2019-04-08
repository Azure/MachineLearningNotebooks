# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
import os

print("*********************************************************")
print("Hello Azure ML!")

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, help="data directory")
parser.add_argument('--output', type=str, help="output")
args = parser.parse_args()

print("Argument 1: %s" % args.datadir)
print("Argument 2: %s" % args.output)

if not (args.output is None):
    os.makedirs(args.output, exist_ok=True)
    print("%s created" % args.output)

try:
    from azureml.core import Run
    run = Run.get_context()
    print("Log Fibonacci numbers.")
    run.log_list('Fibonacci numbers', [0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
    run.complete()
except:
    print("Warning: you need to install Azure ML SDK in order to log metrics.")

print("*********************************************************")
