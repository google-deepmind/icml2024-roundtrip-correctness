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

"""Run RTC execution evaluation on HumanEval."""

from collections.abc import Sequence
import concurrent.futures
import gzip
import io
import textwrap
from typing import Any

from absl import app
from absl import flags
import tensorflow_datasets as tfds

from roundtrip_correctness import rtc_data as med
from roundtrip_correctness.synthesis_rtc import eval_utils
from roundtrip_correctness.synthesis_rtc import task as srtc


_SAMPLES_PATH = flags.DEFINE_string(
    "samples_path",
    None,
    "Path to the directory storing the output samples.",
    required=True,
)

_NUM_CONCURRENT_CONTAINERS = flags.DEFINE_integer(
    "num_concurrent_containers", 10, "The number of concurrent containers."
)

_OUTPUT_FILE = flags.DEFINE_string(
    "output_file",
    None,
    "The output json file containing the scores.",
    required=True,
)

_IMAGE_NAME = flags.DEFINE_string(
    "image_name", "python:3.11-alpine", "The image name."
)
_DOCKER_RUNTIME = flags.DEFINE_string(
    "docker_runtime", "runc", "The Docker runtime."
)


def _humaneval_by_id() -> dict[int, Any]:
  """Load HumanEval dataset by their task id."""
  ds = tfds.load("huggingface:openai_humaneval/openai_humaneval")
  samples = {}
  for sample in ds["test"]:
    samples[
        int(
            sample["task_id"]
            .numpy()
            .decode()
            .removeprefix("HumanEval/")
            .removesuffix(".py")
        )
    ] = sample
  return samples


def _samples_by_id() -> (
    dict[int, med.GenerationSamplesForDatapoint[srtc.SynthesisRtcExample]]
):
  """Load samples by their MBPP problem id."""
  samples_per_problem = {}

  with gzip.open(open(_SAMPLES_PATH.value, "rb")) as f:
    for line in f:
      sample = med.GenerationSamplesForDatapoint.from_json(line.decode())
      sample.datapoint = srtc.SynthesisRtcExample.from_dict(sample.datapoint)
      problem_id = int(
          sample.datapoint.filename[len("HumanEval/") : -len(".py")]
      )
      assert problem_id not in samples_per_problem
      samples_per_problem[problem_id] = sample
  return samples_per_problem


def _construct_runnable_test(
    humaneval_problem_def: dict[str, Any],
    datapoint: srtc.SynthesisRtcExample,
    sample: str,
) -> str:
  """Creates the string of a test in HumanEval."""
  with io.StringIO() as sb:
    bw_sample = textwrap.indent(
        textwrap.dedent(sample),
        datapoint.indentation,
    )
    # Force-fix first line indentation
    # bw_sample = datapoint.indentation + bw_sample.lstrip()
    sb.write(
        f"{datapoint.code_before_hole}\n{bw_sample}\n{datapoint.code_after_hole}\n"
    )

    sb.write(humaneval_problem_def["test"].numpy().decode())
    sb.write("\n")
    sb.write(
        f"check({humaneval_problem_def['entry_point'].numpy().decode()})\n"
    )
    return sb.getvalue()


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  humaneval_dataset = _humaneval_by_id()
  eval_samples_by_id = _samples_by_id()
  with (
      gzip.open(open(_OUTPUT_FILE.value, "wb"), "wb") as f,
      concurrent.futures.ThreadPoolExecutor(
          max_workers=_NUM_CONCURRENT_CONTAINERS.value
      ) as pool,
  ):
    for problem_id, problem_samples in eval_samples_by_id.items():
      # This is fine since check_pass is used per-iteration,
      # pylint: disable=cell-var-from-loop
      def check_pass(predicted_code: str) -> concurrent.futures.Future[bool]:
        code = _construct_runnable_test(
            humaneval_dataset[problem_id],
            problem_samples.datapoint,
            predicted_code,
        )
        return pool.submit(
            eval_utils.run_and_check_exit_code,
            code,
            _IMAGE_NAME.value,
            _DOCKER_RUNTIME.value,
        )

      # pylint: enable=cell-var-from-loop
      evaluated_problem_samples = eval_utils.evaluate_generated_output(
          problem_samples, check_pass
      )

      f.write(evaluated_problem_samples.to_json().encode())
      f.write(b"\n")


if __name__ == "__main__":
  app.run(main)
