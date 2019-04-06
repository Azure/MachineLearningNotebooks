import argparse
import os
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', dest="input_file", default = "C://temp/attrition_pipe/output")
parser.add_argument('--output_dir', dest="output_dir", default = "C://temp/attrition_pipe/output")

args = parser.parse_args()
print("all args: ", args)

cwd = os.getcwd()
print("cwd:", cwd)
print("dir of cwd", os.listdir(cwd))
parent = os.path.dirname(args.input_file)
print("input_dir_parent:", parent)
print("dir of input_dir_parent:", os.listdir( parent))

print("file path:", args.input_file)

with open(os.path.join(args.input_file)) as f:
    metrics = json.load(f)

pprint(metrics, width = 1)