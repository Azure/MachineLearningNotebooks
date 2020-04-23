# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

from azureml.core import Run

submitted_run = Run.get_context()
submitted_run.log(name="message", value="Hello from run!")
