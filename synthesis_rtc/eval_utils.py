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

"""A generic wrapper around evaluation."""

from collections.abc import Callable
import concurrent.futures
import itertools
import os
import tempfile

from absl import logging
import docker
import requests

from roundtrip_correctness import rtc_data as rtcd
from roundtrip_correctness.synthesis_rtc import task as d2s


def run_and_check_exit_code(
    code: str, image_name: str, docker_runtime: str
) -> bool:
  """Run the given code and check if a non 0 exit code has been returned."""
  docker_client = docker.from_env()
  with tempfile.TemporaryDirectory() as code_tmpdir:
    with open(os.path.join(code_tmpdir, "program.py"), "w") as f:
      f.write(code)

    container = docker_client.containers.run(
        image=image_name,
        command="python /code/program.py",
        volumes={
            code_tmpdir: {"bind": "/code/", "mode": "ro"},
        },
        network_disabled=True,
        runtime=docker_runtime,
        ulimits=[
            docker.types.containers.Ulimit(name="nproc", soft=100, hard=10000),
            docker.types.containers.Ulimit(
                name="cpu", soft=60, hard=120
            ),  # Seconds
            docker.types.containers.Ulimit(
                name="memlock", soft=10 * 1024, hard=1024 * 1024
            ),  # in KB
        ],
        detach=True,
    )
    try:
      exit_info = str(container.wait(timeout=90))
      logging.info("Task exited with %s", exit_info)
      container.stop()
      container.remove(force=True)
    except requests.exceptions.ConnectionError:
      logging.exception("Timeout for task: Container stopped.")
      exit_info = '"Timeout for task %s: Container stopped."'
      container.pause()

    return exit_info == "{'Error': None, 'StatusCode': 0}"


def evaluate_generated_output(
    generation_samples: rtcd.GenerationSamplesForDatapoint[
        d2s.SynthesisRtcExample
    ],
    check_pass: Callable[[str], concurrent.futures.Future[bool]],
) -> rtcd.EvaluatedGenerationSamplesForDatapoint[d2s.SynthesisRtcExample]:
  """Concurrently evaluate the generated output."""
  gen_samples_pass_futures = []
  for fw_sample in generation_samples.generation_samples:
    fw_sample_pass_futures = []
    gen_samples_pass_futures.append(fw_sample_pass_futures)
    for bw_sample in fw_sample.backward_samples:
      fw_sample_pass_futures.append(check_pass(bw_sample.text))

  baseline_samples_pass_futures = []
  if generation_samples.baseline_samples:
    for baseline_sample in generation_samples.baseline_samples:
      baseline_samples_pass_futures.append(check_pass(baseline_sample.text))

  all_futures = list(
      itertools.chain(baseline_samples_pass_futures, *gen_samples_pass_futures)
  )
  concurrent.futures.wait(all_futures)

  evaluated_problem_samples = rtcd.EvaluatedGenerationSamplesForDatapoint[
      d2s.SynthesisRtcExample
  ](
      samples=generation_samples,
      generation_samples_consistencies={
          "pass": [
              [1.0 if f.result() else 0.0 for f in fw_sample_pass_futures]
              for fw_sample_pass_futures in gen_samples_pass_futures
          ]
      },
      baseline_samples_consistencies={
          "pass": [
              1.0 if f.result() else 0.0 for f in baseline_samples_pass_futures
          ]
      },
  )
  evaluated_problem_samples.validate_sizes()
  return evaluated_problem_samples
