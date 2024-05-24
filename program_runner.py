# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility class to run operations on a program."""

import contextlib
import dataclasses
import os
import shutil
import tarfile
import tempfile

from absl import flags
from absl import logging
import docker
import requests


_DOCKER_RUNTIME = flags.DEFINE_string(
    "docker_runtime", "runc", "The docker runtime."
)


@dataclasses.dataclass(frozen=True)
class RunInfo:
  stdout_log: str
  stderr_log: str
  exit_info: str
  output_path: str


class ProgramRunner:
  """Utility class to run a program."""

  def __init__(self, code_tar_path: str, docker_image: str):
    self._code_tar_path = code_tar_path
    self._working_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    self._docker_client = docker.DockerClient.from_env()
    self._docker_image = docker_image
    self.reset_code_dir()

  def reset_code_dir(self) -> None:
    """Ensures that the working dir contains the original state of the repo."""

    # Delete all contents
    for filename in os.listdir(self._working_dir.name):
      file_path = os.path.join(self._working_dir.name, filename)
      if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)
      elif os.path.isdir(file_path):
        shutil.rmtree(file_path)

    # Re-extract tar.gz
    with tarfile.open(os.path.join(self._code_tar_path), "r:gz") as tar:
      tar.extractall(self._working_dir.name)

  @property
  def code_dir(self) -> str:
    return self._working_dir.name

  def _cleanup_temp_files(self) -> None:
    self._docker_client.containers.run(
        image=self._docker_image,
        command="chmod -R 777 /code/",
        volumes={
            self._working_dir.name: {"bind": "/code/", "mode": "rw"},
        },
        network_disabled=True,
        runtime=_DOCKER_RUNTIME.value,
        detach=False,
        remove=True,
    )

  @contextlib.contextmanager
  def run(self, script: str, timeout_sec: int = 600, mem_limit: str = "10g"):
    """Run an operation on the program."""
    with (
        tempfile.TemporaryDirectory() as outputs_tmpdir,
        tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as op_tmpdir,
    ):
      with open(os.path.join(op_tmpdir, "run.sh"), "w") as f:
        f.write(script)

      container = self._docker_client.containers.run(
          image=self._docker_image,
          command="sh /op/run.sh",
          volumes={
              self._working_dir.name: {"bind": "/code/", "mode": "rw"},
              outputs_tmpdir: {"bind": "/outputs/", "mode": "rw"},
              op_tmpdir: {"bind": "/op/", "mode": "ro"},
          },
          network_disabled=True,
          stderr=True,
          stdout=True,
          runtime=_DOCKER_RUNTIME.value,
          detach=True,
          mem_limit=mem_limit,
          nano_cpus=1_000_000_000 * 2,  # 2 CPUs per run.
      )
      try:
        exit_info = str(container.wait(timeout=timeout_sec))
      except requests.exceptions.ConnectionError:
        exit_info = "Timeout: Container stopped."
        container.pause()

      stdout_log = container.logs(stderr=False, stdout=True).decode()
      stderr_log = container.logs(stderr=True, stdout=False).decode()

      try:
        container.stop()
        container.remove(force=True)
      except docker.errors.APIError:
        logging.exception("Could not stop or remove container.")

      yield RunInfo(
          stdout_log=stdout_log,
          stderr_log=stderr_log,
          exit_info=exit_info,
          output_path=outputs_tmpdir,
      )
      try:
        self._cleanup_temp_files()
      except PermissionError:
        logging.exception("Could not cleanup temp files.")
