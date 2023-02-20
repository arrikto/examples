import os
import sys
import argparse
import functools
import torch
import torch.profiler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    # BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    # enable_wrap,
    # wrap,
)
from tqdm import tqdm


RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))


def setup() -> None:
    """Initialize torch distributed."""
    torch.manual_seed(42)
    dist.init_process_group(backend="nccl")


def cleanup() -> None:
    """Cleanup torch distributed."""
    dist.destroy_process_group()


class Net(nn.Module):
    """Define a simple CNN for MNIST."""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, epoch, batch, optimizer, sampler=None):
    """Train the model for one epoch."""
    model.train()
    
    if sampler:
        sampler.set_epoch(epoch)
    
    data, target = batch[0].to(device), batch[1].to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target, reduction="sum")
    loss.backward()
    optimizer.step()


def test(model, device, test_loader):
    """Test the model."""
    model.eval()
    ddp_loss = torch.zeros(3).to(device)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            ddp_loss[0] += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            ddp_loss[2] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if RANK == 0:
        test_loss = ddp_loss[0] / ddp_loss[2]
        print("Test set: Avg loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss, int(ddp_loss[1]), int(ddp_loss[2]),
            100. * ddp_loss[1] / ddp_loss[2]))


def main(args):
    # setup distributed training
    setup()

    # create the train and test datasets
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(".data", download=True, transform=transform)
    test_dataset = datasets.MNIST(".data", train=False, transform=transform)

    train_sampler = DistributedSampler(train_dataset, rank=RANK,
                                       num_replicas=WORLD_SIZE, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, rank=RANK,
                                      num_replicas=WORLD_SIZE)

    # create the train and test dataloaders
    train_kwargs = {"batch_size": args.batch_size, "sampler": train_sampler}
    test_kwargs = {"batch_size": args.test_batch_size, "sampler": test_sampler}
    cuda_kwargs = {"num_workers": 2, "pin_memory": True, "shuffle": False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    # define the auto wrap policy for FSDP
    wrap_policy = functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=args.min_num_params)

    # set the device to use
    # for multi-processing scenarios each process gets a LOCAL_RANK
    # environment variable which will be the index of the device as well
    # if a GPU device is available
    device = torch.device(LOCAL_RANK if torch.cuda.is_available() else "cpu")

    model = Net().to(device)

    # whether to offload the model parameters and gradients to CPU
    # when not in use
    cpu_offload = CPUOffload(offload_params=args.cpu_offload)
    
    # wrap the model with FSDP
    model = FSDP(model, cpu_offload=cpu_offload, auto_wrap_policy=wrap_policy)

    # define the optimizer and the learning rate scheduler
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=1,  # during this phase profiler is not active
                warmup=1,  # during this phase profiler starts tracing, but the results are discarded
                active=3,  # during this phase profiler traces and records data
                repeat=2),  # specifies an upper bound on the number of cycles
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                f"/logs/fsdp/{args.version}"),
            profile_memory=True,
            record_shapes=True,
            with_stack=True
        ) as profiler:
            # train the model
            for batch in tqdm(train_loader, desc=f"Training - Epoch {epoch}"):
                train(model, device, epoch, batch, optimizer, train_sampler)
                profiler.step()
        
        # test the model
        test(model, device, test_loader)
        # update the learning rate
        scheduler.step()

    if args.save_model:
        # use a barrier to make sure training is done on all ranks
        dist.barrier()
        
        # save the model
        if RANK == 0:
            model_state = model.state_dict()
            torch.save(model_state, "mnist_cnn.pt")

    # tear down distributed training
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="PyTorch MNIST Fully Sharded Data Parallel Example")

    parser.add_argument("--batch-size", type=int, default=64, metavar="B",
                        help="input batch size for training (default: 64)")
    parser.add_argument("--test-batch-size", type=int, default=1000,
                        metavar="TB",
                        help="input batch size for testing (default: 1000)")
    parser.add_argument("--epochs", type=int, default=5, metavar="E",
                        help="number of epochs to train (default: 5)")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR",
                       help="learning rate (default: 0.01)")
    parser.add_argument("--gamma", type=float, default=0.9, metavar="G",
                        help="learning rate step gamma (default: 0.9)")
    parser.add_argument("--seed", type=int, default=1, metavar="S",
                        help="random seed (default: 1)")
    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="disables CUDA training (default: False)")
    parser.add_argument("--min-num-params", type=int, default=int(1e8),
                        metavar="P",
                        help="minimum number of parameters for an FSDP unit"
                             " (default: 1e8)")
    parser.add_argument("--cpu_offload", action="store_true", default=False,
                        help="offload the parameters and gradients of the"
                             " model to CPU when not in use (default: False)")
    parser.add_argument("--version", type=str, default="version_0",
                        metavar="V",
                        help="name of the experiment for logging purposes")
    parser.add_argument("--save-model", action="store_true", default=False,
                        help="For Saving the current Model")
    parser.add_argument("--save-dir", type=str, default=".", metavar="D",
                        help="directory to save the model")

    args = parser.parse_args()

    sys.exit(main(args))

