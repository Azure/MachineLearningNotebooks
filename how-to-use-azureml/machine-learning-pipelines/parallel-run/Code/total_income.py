# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import os


def init():
    print("Init")


def run(mini_batch):
    print(f'run method start: {__file__}, run({mini_batch})')
    total_income = mini_batch["INCOME"].sum()
    print(f'total_income = {total_income}')
    mini_batch["total_income"] = total_income

    return mini_batch
