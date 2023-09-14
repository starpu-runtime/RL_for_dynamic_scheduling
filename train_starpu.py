import argparse
import logging
from pprint import pformat
from queue import Queue

import gym
import torch
from torch.utils.tensorboard import SummaryWriter

from a2c import A2C
from common_logging import training_logger
from convergence_functions import ConvergenceFunctions
from env.utils import TaskGraph
from model import ModelHeterogene

# import numpy as np
# import torch

action_queue = Queue()
data_queue = Queue()
reward_data_queue = Queue()
end_queue = Queue()

convergence_status_queue = Queue()
convergence_ack_queue = Queue()


def function_name(name):
    return name


parser = argparse.ArgumentParser()

# Training settings
parser.add_argument("--model_path", type=str, default="none", help="path to load model")
parser.add_argument("--output_model_path", type=str, default="none", help="path to save model")
parser.add_argument("--torchscript_output_model_path", type=str, default="none", help="path to save torchscript model")
parser.add_argument("--num_env_steps", type=int, default=10**4, help="num env steps")
parser.add_argument("--num_episodes", type=int, default=5, help="num episodes")
parser.add_argument("--num_processes", type=int, default=1, help="num proc")
parser.add_argument("--lr", type=float, default=10**-2, help="learning rate")
parser.add_argument("--eps", type=float, default=10**-1, help="Random seed.")
parser.add_argument("--optimizer", type=str, default="rms", help="sgd or adam or rms")
parser.add_argument("--scheduler", type=str, default="lambda", help="lambda or cyclic")
parser.add_argument("--step_up", type=float, default=100, help="step_size_up for cyclic scheduler")
parser.add_argument("--sched_ratio", type=float, default=10, help="lr ratio for cyclic scheduler")
parser.add_argument("--entropy_coef", type=float, default=0.002, help="entropy loss weight")
parser.add_argument("--gamma", type=float, default=1, help="inflation")
parser.add_argument("--loss_ratio", type=float, default=0.5, help="value loss weight")
parser.add_argument("--trajectory_length", type=int, default=40, help="batch size")
parser.add_argument("--log_interval", type=int, default=10, help="evaluate every log_interval steps")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--agent", type=str, default="A2C", help="A2C")
parser.add_argument("--result_name", type=str, default="results.csv", help="filename where results are stored")

parser.add_argument("--convergence_threshold", type=float, default=50, help="")
parser.add_argument("--required_buffer_elements", type=int, default=10, help="")
parser.add_argument("--convergence_function", type=function_name, default="average", help="")

# model settings
parser.add_argument("--input_dim", type=int, default=13, help="input dim")
parser.add_argument("--hidden_dim", type=int, default=128, help="hidden dim")
parser.add_argument("--ngcn", type=int, default=0, help="number of gcn")
parser.add_argument("--nmlp", type=int, default=1, help="number of mlp to compute probs")
parser.add_argument("--nmlp_value", type=int, default=1, help="number of mlp to compute v")
parser.add_argument("--res", action="store_true", default=False, help="with residual connexion")
parser.add_argument("--withbn", action="store_true", default=False, help="with batch norm")

# env settings
parser.add_argument("--n", type=int, default=4, help="number of tiles")
parser.add_argument("--nGPU", type=int, default=1, help="number of GPUs")
parser.add_argument("--nCPU", type=int, default=3, help="number of cores")
parser.add_argument("--window", type=int, default=0, help="window")
parser.add_argument("--noise", type=float, default=0, help="noise")
parser.add_argument("--env_type", type=str, default="QR", help="chol or LU or QR")
parser.add_argument("--seed_env", type=int, default=42, help="Random seed env ")

parser.add_argument(
    "--logging_level",
    type=str,
    default="info",
    choices=["info", "debug", "error", "critical", "warn"],
    help="logging level (default: %(default)s)",
)


class StarPUEnv(gym.Env):
    def __init__(self, convergence_instance, convergence_method):
        self.num_steps = 0
        self.time = 0
        self.task_count = 0
        self.tasks_left = 0
        self.node_num = None
        self.ready_tasks = None
        self.number_tasks = 0
        self.reward = 0
        self.convergence_instance = convergence_instance
        self.convergence_method = convergence_method
        self.performance = 0
        self.converged = False
        self.training_ended = False

    def has_converged(self):
        self.converged = self.convergence_method(self.performance)

        return self.converged

    def generate_empty_tensor(self):
        self.tasks_left = 0
        self.number_tasks = 0
        self.ready_tasks = torch.tensor([]).reshape(0, 1)

        x = torch.tensor([]).reshape(0, 13)
        edge_index = torch.tensor([]).reshape(2, 0)
        graph_data = TaskGraph(x, edge_index, None)

        return graph_data

    def get_reward(self, reset=False):
        done = read_queue(end_queue)

        if reset or done:
            # If get_reward was called by self.reset() it means we have finished an execution, either prematurely or
            # because we've reached the end of the execution as signaled by the scheduler
            # In that case, we signal the scheduler to continue execution if training hasn't finished and stop if it has
            append_queue(convergence_status_queue, True if self.training_ended else False)
            training_logger.info(f"Training ended: {self.training_ended}")

        if reset or not done:
            # If get_reward was called by self.reset() or the execution wasn't done (signaled by the scheduler), then
            # the reward is set to 0
            self.reward = 0
        else:
            # If not, that means get_reward was called by self.step() and the scheduler has signaled the end of the
            # execution and a reward should be given

            gflops, max_gflops = read_queue(reward_data_queue)
            self.performance = gflops
            self.convergence_instance.append_buffer(self.performance)

            # The goal is to maximize the reward, trending towards a positive value
            self.reward = -(max_gflops - gflops) / 500

        return self.reward

    def read_scheduler_data(self, queue):
        data = read_queue(queue)

        self.number_tasks = number_tasks = data["number_tasks"]
        self.tasks_left = data["tasks_left"]
        self.ready_tasks = torch.tensor(data["tasks_ready"]).reshape(number_tasks, 1)
        self.time = data["time"]
        tasks_types = []

        for task_type in data["tasks_types"]:
            task_numbers = torch.arange(4).reshape(1, 4)
            tasks_types.append(task_numbers.eq(task_type).long())

        x = torch.cat(
            (
                torch.tensor(data["number_successors"]).reshape(number_tasks, 1),
                torch.tensor(data["number_predecessors"]).reshape(number_tasks, 1),
                torch.vstack(tasks_types),
                torch.tensor(data["tasks_ready"]).reshape(number_tasks, 1),
                torch.tensor(data["tasks_running"]).reshape(number_tasks, 1),
                torch.tensor(data["remaining_time"]).reshape(number_tasks, 1) / 100,
                torch.tensor(data["normalized_path_lengths"]).reshape(number_tasks, 1),
                torch.tensor(data["node_type"]).repeat(number_tasks, 1),
                torch.tensor(data["min_ready_gpu"]).repeat(number_tasks, 1) / 100,
                torch.tensor(data["min_ready_cpu"]).repeat(number_tasks, 1) / 100,
            ),
            dim=1,
        )
        edge_index = torch.tensor(data["edge_index_vector"]).reshape(2, len(data["edge_index_vector"]) // 2)
        graph_data = TaskGraph(x, edge_index, None)

        self.node_num = torch.arange(self.number_tasks).reshape(self.number_tasks, 1)

        return graph_data

    def step(self, action):
        self.num_steps += 1

        if action != -1:
            self.task_count += 1

        # Tell the scheduler which action to take right now
        # (schedule the task associated to the ID returned, or skip if action is -1)
        training_logger.info(f"Sending action {action} to scheduler")
        append_queue(action_queue, int(action))

        # Ask for reward, either 0 if execution hasn't finished yet or a negative value if it has
        reward = self.get_reward()

        # Read data pushed by the scheduler if the execution hasn't finished (reward is 0), or generate an
        # empty tensor otherwise to continue execution
        graph_data = self.read_scheduler_data(data_queue) if reward == 0 else self.generate_empty_tensor()

        training_logger.info(f"Graph data: {graph_data}")

        # always false until there are no more tasks to schedule
        done = self.tasks_left == 0

        training_logger.info(f"Tasks left: {self.tasks_left}")
        training_logger.info(f"Reward: {reward}")

        info = {"episode": {"r": reward, "length": self.num_steps, "time": self.time}, "bad_transition": False}

        return {"graph": graph_data, "node_num": self.node_num, "ready": self.ready_tasks}, reward, done, info

    def reset(self, init=False):
        self.time = 0
        self.num_steps = 0
        self.task_count = 0
        self.reward = 0

        if not init:
            # We only need to tell the scheduler to reset if it hasn't reset by itself (completed its execution)
            training_logger.info(f"Telling scheduler to reset")
            append_queue(action_queue, -2)

            self.reward = self.get_reward(reset=True)

        training_logger.info("Waiting for scheduler to send initial data")
        graph_data = self.read_scheduler_data(data_queue)
        training_logger.info(f"Reset graph data: {graph_data}")

        return {"graph": graph_data, "node_num": self.node_num, "ready": self.ready_tasks}

    def render(self, mode="human"):
        pass


def read_queue(queue):
    data = queue.get(block=True)
    training_logger.warn(f"Number of elements left in queue after get: {queue.qsize()}")

    return data


def append_queue(queue, data):
    training_logger.warn(f"Number of elements left in queue before append: {queue.qsize()}")

    queue.put(data)


def convert_model(args, model_path):
    model = torch.load(model_path)

    model_state_dict = ModelHeterogene(
        input_dim=args.input_dim, hidden_dim=args.hidden_dim, ngcn=args.ngcn, nmlp=args.nmlp, jittable=True
    )
    model_state_dict.load_state_dict(model.state_dict(), strict=False)
    model_torchscript = torch.jit.script(model_state_dict)

    training_logger.info(f"Saving model to path {args.torchscript_output_model_path}")

    model_torchscript.save(args.torchscript_output_model_path)


def train(argv=None):
    args = parser.parse_args({} if argv is None else argv)

    training_logger.setLevel(logging.getLevelName(args.logging_level.upper()))

    training_logger.info(f"Received arguments from StarPU:\n{pformat(argv)}")

    convergence_instance = ConvergenceFunctions(
        required_buffer_elements=args.required_buffer_elements, convergence_threshold=args.convergence_threshold
    )
    convergence_method = getattr(convergence_instance, args.convergence_function, None)

    config_enhanced = vars(args)
    writer = SummaryWriter("runs")

    training_logger.info(f"Current config_enhanced is:\n{pformat(config_enhanced)}")

    env = StarPUEnv(convergence_instance=convergence_instance, convergence_method=convergence_method)

    model = ModelHeterogene(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        ngcn=args.ngcn,
        nmlp=args.nmlp,
        nmlp_value=args.nmlp_value,
        res=args.res,
        withbn=args.withbn,
    )

    agent = A2C(config_enhanced, env, model=model, writer=writer)

    best_perf, _ = agent.training()

    convert_model(args, args.output_model_path)

    return 0
