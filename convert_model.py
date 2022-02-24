import argparse

import torch

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, required=True, help='path to load model')
parser.add_argument('--output_path', type=str, required=True, help='path to save model')

args = parser.parse_args()

model = torch.load(args.model_path)
model.eval()

for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

torch.save(model.state_dict(), args.output_path)