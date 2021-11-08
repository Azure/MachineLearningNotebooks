# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import os


def init():
    print("Init")


# For partition per folder/column jobs, ParallelRunStep pass an optional positional parameter `mini_batch_context`
# to the `run` function in user's entry script, which contains information of the mini_batch.
def run(mini_batch, mini_batch_context):
    print(f"run method start: {__file__}, run({mini_batch}, {mini_batch_context})")
    # `partition_key_value` is a dict that corresponds to the mini_batch, the keys of the dict are those specified
    # in `partition_keys` in ParallelRunConfig.
    print(f"partition_key_value = {mini_batch_context.partition_key_value}")
    # `dataset` is the dataset object that corresponds to the mini_batch, which is a subset of the input dataset
    # filtered by condition specified in `partition_key_value`.
    print(f"dataset = {mini_batch_context.dataset}")

    print(f"file_count_of_mini_batch = {len(mini_batch)}")
    file_name_list = []
    file_size_list = []
    total_file_size_of_mini_batch = 0
    for file_path in mini_batch:
        file_name_list.append(os.path.basename(file_path))
        file_size = os.path.getsize(file_path)
        file_size_list.append(file_size)
        total_file_size_of_mini_batch += file_size
    print(f"total_file_size_of_mini_batch = {total_file_size_of_mini_batch}")
    file_size_ratio_list = [file_size * 1.0 / total_file_size_of_mini_batch for file_size in file_size_list]

    # If `output_action` is set to `append_row` in ParallelRunConfig for FileDataset input(as is in this sample
    # notebook), the return value of `run` method is expected to be a list/tuple of same length with the
    # input parameter `mini_batch`, and each element in the list/tuple would form a row in the result file by
    # calling the Python builtin `str` function.
    # If you want to specify the output format, please format and return str value as in this example.
    return [
        ",".join([str(x) for x in fields])
        for fields in zip(
            file_name_list,
            file_size_list,
            file_size_ratio_list,
            [mini_batch_context.partition_key_value["user"]] * len(mini_batch),
            [mini_batch_context.partition_key_value["genres"]] * len(mini_batch),
            [total_file_size_of_mini_batch] * len(mini_batch),
        )
    ]
