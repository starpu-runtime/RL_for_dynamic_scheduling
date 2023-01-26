import argparse

from env.utils import *
from model import ModelHeterogene

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, required=True, help='path to load model')
parser.add_argument('--output_path', type=str, required=True, help='path to save model')
parser.add_argument('--input_dim', type=int, default=16, help='input dim')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dim')
parser.add_argument('--ngcn', type=int, default=0, help='number of gcn')
parser.add_argument('--nmlp', type=int, default=1, help='number of mlp to compute probs')

args = parser.parse_args()

print(f"Loading model from path {args.model_path}")

model = ModelHeterogene(input_dim=args.input_dim, hidden_dim=args.hidden_dim, ngcn=args.ngcn, nmlp=args.nmlp, jittable=True)

model.load_state_dict(torch.load(args.model_path), strict=False)
model.eval()

ts_model = torch.jit.script(model)

print(f"Saving model to path {args.output_path}")

ts_model.save(args.output_path)
