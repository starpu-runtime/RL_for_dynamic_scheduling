import argparse

from env.utils import *
from model import ModelHeterogene

# torch.nn.Module.dump_patches = True

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, required=True, help='path to load model')

args = parser.parse_args()
config_enhanced = vars(args)

model = ModelHeterogene(input_dim=16, ngcn=0, nmlp=1)

# print("Expected model:\n")
# print(model)

data_model = torch.load(args.model_path)

# print("\nLoaded model: ")
# for d in data_model:
#     print(d)

model.load_state_dict(torch.load(args.model_path), strict=False)
model.eval()

task_data = ggen_cholesky(n_vertex=4, noise=0)

nGPU = 2
p = len(np.array([1] * nGPU + [0] * (4 - nGPU)))
running = -1 * np.ones(p)  # array of task number
ready_tasks = [0]
window = 0

visible_graph, node_num = compute_sub_graph(task_data, torch.tensor(
    np.concatenate((running[running > -1], ready_tasks)), dtype=torch.long), window)
ready = isin(node_num, torch.tensor(ready_tasks)).float()

n_succ = torch.sum((node_num == task_data.edge_index[0]).float(), dim=1).unsqueeze(-1)
n_pred = torch.sum((node_num == task_data.edge_index[1]).float(), dim=1).unsqueeze(-1)

task_num = task_data.task_list[node_num.squeeze(-1)]
if isinstance(task_num, Task):
    task_type = torch.tensor([[4]])
else:
    task_type = torch.tensor([task.type for task in task_num]).unsqueeze(-1)

num_classes = 4
one_hot_type = (task_type == torch.arange(num_classes).reshape(1, num_classes)).float()

remaining_time = torch.zeros(node_num.shape[0])
remaining_time = remaining_time.unsqueeze(-1)

descendant_features_norm = (task_data.add_features_descendant()[0] / torch.sum(task_data.x, dim=0))[node_num].squeeze(1)
node_type = torch.ones(node_num.shape[0])
node_type = node_type.unsqueeze(-1)

min_ready_gpu = torch.FloatTensor([1]).repeat(node_num.shape[0]).unsqueeze((-1))
min_ready_cpu = torch.FloatTensor([1]).repeat(node_num.shape[0]).unsqueeze((-1))

running_1 = isin(node_num, torch.tensor(running[running > -1])).squeeze(-1)
running_1 = running_1.unsqueeze(-1).float()

visible_graph.x = torch.cat((n_succ, n_pred, one_hot_type, ready, running_1, remaining_time,
                             descendant_features_norm, node_type, min_ready_gpu, min_ready_cpu), dim=1)

# print(f"n_succ: {n_succ}")
# print(f"n_pred: {n_pred}")
# print(f"one_hot_type: {one_hot_type}")
# print(f"ready: {ready}")

data = {
    "graph": visible_graph,
    "node_num": node_num,
    "ready": ready
}

out = model(visible_graph.x, visible_graph.edge_index, ready)

print(out)

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

torch.save(model.state_dict(), "model_116.0_statedict_1.pth")
