import argparse
import os

#azureml-sdk
# from azureml.core import Run

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', dest="input_dir", default = "C://temp/attrition_pipe/output")
parser.add_argument('--output_dir', dest="output_dir", default = "C://temp/attrition_pipe/output")


args = parser.parse_args()
print("all args: ", args)


cwd = os.getcwd()
print("cwd:", cwd)
print("dir of cwd", os.listdir(cwd))
parent = os.path.dirname(args.input_dir)
print("input_dir_parent:", parent)
print("dir of input_dir_parent:", os.listdir( parent))


with open(os.path.join(parent, 'metrics')) as f:
    metrics = json.load(f)

print(metrics)
# print("input_dir:", args.input_dir)
# print("dir of input_dir", os.listdir( args.input_dir))