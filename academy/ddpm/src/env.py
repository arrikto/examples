# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

from lightning_fabric.plugins.environments.cluster_environment import ClusterEnvironment

log = logging.getLogger(__name__)


class CustomKubeflowEnvironment(ClusterEnvironment):
    """Distributed training using the `PyTorchJob`_ operator on `Kubeflow`_.

    .. _PyTorchJob: https://www.kubeflow.org/docs/components/training/pytorch/
    .. _Kubeflow: https://www.kubeflow.org
    """

    @property
    def creates_processes_externally(self) -> bool:
        return "LOCAL_RANK" in os.environ

    @property
    def main_address(self) -> str:
        return os.environ["MASTER_ADDR"]

    @property
    def main_port(self) -> int:
        return int(os.environ["MASTER_PORT"])

    @staticmethod
    def detect() -> bool:
        """Return ``True`` if the process was launched using PyTorchJob."""
        required_env_vars = {"KUBERNETES_PORT", "MASTER_ADDR", "MASTER_PORT",
                             "WORLD_SIZE", "RANK"}
        # torchelastic sets these. Make sure we're not in torchelastic
        excluded_env_vars = {"GROUP_RANK", "LOCAL_WORLD_SIZE"}
        env_vars = os.environ.keys()
        return (required_env_vars.issubset(env_vars)
                and excluded_env_vars.isdisjoint(env_vars))

    def world_size(self) -> int:
        return int(os.environ["WORLD_SIZE"])

    def set_world_size(self, size: int) -> None:
        self._world_size = size

    def global_rank(self) -> int:
        return self._global_rank

    def set_global_rank(self, rank: int) -> None:
        self._global_rank = rank

    def local_rank(self) -> int:
        return int(os.environ.get("LOCAL_RANK", 0))

    def node_rank(self) -> int:
        return int(os.environ["RANK"])
