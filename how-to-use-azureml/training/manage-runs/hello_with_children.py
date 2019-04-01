# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

from azureml.core import Run

run = Run.get_context()

child_runs = run.create_children(count=5)
for c, child in enumerate(child_runs):
    child.log(name="Hello from child run ", value=c)
    child.complete()
