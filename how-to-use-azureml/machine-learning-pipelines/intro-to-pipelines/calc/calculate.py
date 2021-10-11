# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import argparse
import os
import codecs

print("In calculate.py")
parser = argparse.ArgumentParser("calculate")
parser.add_argument("--arg_num1", type=int, help="First number as parameter")
parser.add_argument("--arg_num2", type=int, help="Second number as parameter")
parser.add_argument("--file_num1", type=str, help="First number, read from file")
parser.add_argument("--file_num2", type=str, help="Second number, read from file")
parser.add_argument("--output_sum", type=str, help="output_sum directory")
parser.add_argument("--output_product", type=str, help="output_product directory")

args = parser.parse_args()

print("Argument 1: %s" % args.arg_num1)
print("Argument 2: %s" % args.arg_num2)
print("Argument 3: %s" % args.file_num1)
print("Argument 4: %s" % args.file_num2)
print("Argument 5: %s" % args.output_sum)
print("Argument 6: %s" % args.output_product)


def get_number_from_file(file_path):
    with codecs.open(file_path, "r", encoding="utf-8-sig") as f:
        val = int(f.read())
        f.close()
        return val


def get_num(arg_num, file_num):
    if arg_num is None and not file_num:
        return 0
    else:
        num = arg_num if arg_num is not None else get_number_from_file(file_num)
        return num


def write_num_to_file(num, file_path):
    if file_path is not None and file_path != '':
        output_dir = file_path
    else:
        output_dir = '.'
    filename = output_dir

    if output_dir != '.' and not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    fo = open(filename, 'w+')
    fo.write(str(num))
    fo.close()


num1 = get_num(args.arg_num1, args.file_num1)
num2 = get_num(args.arg_num2, args.file_num2)
res_sum = num1 + num2
res_product = num1 * num2
print("results: sum:", res_sum, ", product:", res_product)
write_num_to_file(res_sum, args.output_sum)
write_num_to_file(res_product, args.output_product)
