import argparse
from pprint import pprint
from queue import Queue

import gym
import torch
from torch.utils.tensorboard import SummaryWriter

from a2c import A2C
from env.utils import TaskGraph
from model import ModelHeterogene

# import numpy as np
# import torch

action_queue = Queue()
data_queue = Queue()

parser = argparse.ArgumentParser()

# Training settings
parser.add_argument('--model_path', type=str, default='none', help='path to load model')
parser.add_argument('--output_model_path', type=str, default='none', help='path to save model')
parser.add_argument('--num_env_steps', type=int, default=10 ** 4, help='num env steps')
parser.add_argument('--num_processes', type=int, default=1, help='num proc')
parser.add_argument('--lr', type=float, default=10 ** -2, help='learning rate')
parser.add_argument('--eps', type=float, default=10 ** -1, help='Random seed.')
parser.add_argument('--optimizer', type=str, default='rms', help='sgd or adam or rms')
parser.add_argument('--scheduler', type=str, default='lambda', help='lambda or cyclic')
parser.add_argument('--step_up', type=float, default=100, help='step_size_up for cyclic scheduler')
parser.add_argument('--sched_ratio', type=float, default=10, help='lr ratio for cyclic scheduler')
parser.add_argument('--entropy_coef', type=float, default=0.002, help='entropy loss weight')
parser.add_argument('--gamma', type=float, default=1, help='inflation')
parser.add_argument('--loss_ratio', type=float, default=0.5, help='value loss weight')
parser.add_argument('--trajectory_length', type=int, default=40, help='batch size')
parser.add_argument('--log_interval', type=int, default=10, help='evaluate every log_interval steps')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--agent', type=str, default='A2C', help='A2C')
parser.add_argument("--result_name", type=str, default="results.csv", help="filename where results are stored")

# model settings
parser.add_argument('--input_dim', type=int, default=13, help='input dim')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dim')
parser.add_argument('--ngcn', type=int, default=0, help='number of gcn')
parser.add_argument('--nmlp', type=int, default=1, help='number of mlp to compute probs')
parser.add_argument('--nmlp_value', type=int, default=1, help='number of mlp to compute v')
parser.add_argument('--res', action='store_true', default=False, help='with residual connexion')
parser.add_argument('--withbn', action='store_true', default=False, help='with batch norm')

# env settings
parser.add_argument('--n', type=int, default=4, help='number of tiles')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs')
parser.add_argument('--nCPU', type=int, default=3, help='number of cores')
parser.add_argument('--window', type=int, default=0, help='window')
parser.add_argument('--noise', type=float, default=0, help='noise')
parser.add_argument('--env_type', type=str, default='QR', help='chol or LU or QR')
parser.add_argument('--seed_env', type=int, default=42, help='Random seed env ')


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class StarPUEnv(gym.Env):
    def __init__(self):
        self.num_steps = 0
        self.time = 0
        self.has_just_started = True
        self.task_count = 0
        self.tasks_left = 0

    def read_scheduler_data(self, queue):
        data = read_queue(queue)
        number_tasks = data['number_tasks']
        self.tasks_left = data['tasks_left']
        self.time = data['time']
        tasks_types = []

        for task_type in data['tasks_types']:
            task_numbers = torch.arange(4).reshape(1, 4)
            tasks_types.append(task_numbers.eq(task_type).long())

        x = torch.cat((
            torch.tensor(data['number_successors']).reshape(number_tasks, 1),
            torch.tensor(data['number_predecessors']).reshape(number_tasks, 1),
            torch.vstack(tasks_types),
            torch.tensor(data['tasks_ready']).reshape(number_tasks, 1),
            torch.tensor(data['tasks_running']).reshape(number_tasks, 1),
            torch.tensor(data['remaining_time']).reshape(number_tasks, 1),
            torch.tensor(data['normalized_path_lengths']).reshape(number_tasks, 1),
            torch.tensor(data['node_type']).repeat(number_tasks, 1),
            torch.tensor(data['min_ready_gpu']).repeat(number_tasks, 1),
            torch.tensor(data['min_ready_cpu']).repeat(number_tasks, 1)), dim=1)

        graph_data = TaskGraph(x,
                               torch.tensor(data['edge_index_vector']).reshape(2, len(data['edge_index_vector']) // 2),
                               None)

        return data, graph_data, number_tasks

    def step(self, action):
        self.num_steps += 1

        if action != -1:
            self.task_count += 1

        # Tell the scheduler which action to take right now
        # (schedule the task associated to the ID returned, or skip if action > nb_tasks)
        print(f"{bcolors.WARNING}[training_script] Sending action {action} to scheduler{bcolors.ENDC}")
        append_queue(action_queue, int(action))

        # 'Ask' the scheduler for data (processors, tasks ready, etc.)
        data, graph_data, number_tasks = self.read_scheduler_data(data_queue)
        print(f"{bcolors.WARNING}[training_script] Step graph data: {graph_data}{bcolors.ENDC}")

        # always false until there are no more tasks to schedule
        done = self.tasks_left == 0

        # self.time -> time since start of execution
        reward = - self.time if done else 0

        print(f"{bcolors.WARNING}[training_script] Time: {self.time}, Reward: {reward}{bcolors.ENDC}")

        info = {'episode': {'r': reward, 'length': self.num_steps, 'time': self.time}, 'bad_transition': False}

        return {'graph': graph_data, 'node_num': torch.arange(number_tasks).reshape(number_tasks, 1),
                'ready': torch.tensor(data['tasks_ready']).reshape(number_tasks, 1)}, reward, done, info

    def reset(self):
        self.time = 0
        self.num_steps = 0
        self.task_count = 0

        # 'Ask' the scheduler for data (processors, tasks ready, etc.)
        if not self.has_just_started:
            print(f"{bcolors.WARNING}[training_script] Telling scheduler to reset{bcolors.ENDC}")
            append_queue(action_queue, -2)

        print(f"{bcolors.WARNING}[training_script] Waiting for scheduler to send initial data{bcolors.ENDC}")
        data, graph_data, number_tasks = self.read_scheduler_data(data_queue)
        print(f"{bcolors.WARNING}[training_script] Reset graph data: {graph_data}{bcolors.ENDC}")

        self.has_just_started = False

        return {'graph': graph_data, 'node_num': torch.arange(number_tasks).reshape(number_tasks, 1),
                'ready': torch.tensor(data['tasks_ready']).reshape(number_tasks, 1)}

    def render(self, mode="human"):
        pass


def read_queue(queue):
    return queue.get(block=True)


def append_queue(queue, data):
    queue.put(data)


def train(argv=None):
    if argv is None:
        argv = {}
    else:
        print(f"{bcolors.WARNING}[training_script] Received arguments from StarPU:")
        pprint(argv)

    args = parser.parse_args(argv)
    config_enhanced = vars(args)
    writer = SummaryWriter('runs')

    print(f"{bcolors.WARNING}[training_script] Current config_enhanced is:")
    pprint(config_enhanced)

    env = StarPUEnv()

    model = ModelHeterogene(input_dim=args.input_dim,
                            hidden_dim=args.hidden_dim,
                            ngcn=args.ngcn,
                            nmlp=args.nmlp,
                            nmlp_value=args.nmlp_value,
                            res=args.res,
                            withbn=args.withbn)

    agent = A2C(config_enhanced, env, model=model, writer=writer)

    best_perf, _ = agent.training_batch()
