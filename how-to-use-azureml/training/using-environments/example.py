# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

# Very simple script to demonstrate run in environment
# Print message passed in as environment variable
import os

print(os.environ.get("MESSAGE"))
