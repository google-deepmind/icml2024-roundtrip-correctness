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

"""Execute unit tests to check pass rate."""

from collections.abc import Iterator, Sequence
import concurrent
import gzip
import json
import os
import textwrap

from absl import app
from absl import flags
from absl import logging

from roundtrip_correctness import program_runner
from roundtrip_correctness import rtc_data as rtcd
from roundtrip_correctness.synthesis_rtc import eval_utils
from roundtrip_correctness.synthesis_rtc import task as synthesis_rtc


_INPUT_FILE = flags.DEFINE_string(
    "input_file", None, "The input jsonl.gz file to evaluate.", required=True
)
_TARGET_PROGRAM_TAR = flags.DEFINE_string(
    "target_program_tar",
    None,
    "The tar.gz file containing the code of the target program.",
    required=True,
)
_IMAGE_NAME = flags.DEFINE_string(
    "image_name",
    None,
    "The name of the Docker image to use.",
    required=True,
)
_OUTPUT_FILE = flags.DEFINE_string(
    "output_file",
    None,
    "The output jsonl file containing the scores.",
    required=True,
)

_RUN_COMMAND = flags.DEFINE_string(
    "run_command",
    "pytest --json-report --json-report-file=/outputs/pytest-report.json"
    " /code/",
    "The command to run to test the program.",
)


def _get_pytest_summary(runner: program_runner.ProgramRunner) -> dict[str, int]:
  with runner.run(_RUN_COMMAND.value) as result:
    report_path = os.path.join(result.output_path, "pytest-report.json")
    if not os.path.exists(report_path):
      return {}
    with open(report_path) as f:
      return json.load(f)["summary"]


def prefix_and_suffix_for_example(
    input_example: synthesis_rtc.SynthesisRtcExample,
    runner: program_runner.ProgramRunner,
) -> tuple[str, str, str]:
  """Returns prefix and suffix for a given example."""
  filepath = input_example.filename.split("::")[-1]
  with open(os.path.join(runner.code_dir, filepath)) as f:
    lines = f.read().splitlines(keepends=True)
  start_point = input_example.start_point
  end_point = input_example.end_point

  prefix = "".join(lines[: start_point[0]])
  suffix = "".join(lines[end_point[0] + 1 :])
  actual_code_in_hole = "".join(lines[start_point[0] : end_point[0] + 1])

  if actual_code_in_hole.removeprefix("\n").removesuffix(
      "\n"
  ) != input_example.code_in_hole.removeprefix("\n").removesuffix("\n"):
    raise ValueError("Code in file and in sample didn't match.")
  return filepath, prefix, suffix


def _compute_sample_pass(
    filepath: str,
    prefix: str,
    suffix: str,
    indentation: str,
    baseline_pytest_report: dict[str, int],
    bw_sample: str,
) -> bool:
  """Computes if unit tests still pass."""
  runner = program_runner.ProgramRunner(
      _TARGET_PROGRAM_TAR.value, _IMAGE_NAME.value
  )
  bw_sample_text = textwrap.indent(
      textwrap.dedent(bw_sample),
      indentation,
  )
  # Force-fix first line indentation
  bw_sample_text = indentation + bw_sample_text.lstrip()

  with open(os.path.join(runner.code_dir, filepath), "w") as f:
    f.write(f"{prefix}\n{bw_sample_text}\n{suffix}")
  pytest_report = _get_pytest_summary(runner)

  # Note that this may be approximate. It is possible that this fixes
  # a test from the baseline causing another one to fail.
  passes_tests = pytest_report == baseline_pytest_report
  logging.info(
      "Sample for %s (%s)",
      pytest_report,
      "pass" if passes_tests else "fail",
  )
  return passes_tests


def _eval_sample(
    input_example: rtcd.GenerationSamplesForDatapoint[
        synthesis_rtc.SynthesisRtcExample
    ],
    baseline_pytest_report: dict[str, int],
) -> (
    rtcd.EvaluatedGenerationSamplesForDatapoint[
        synthesis_rtc.SynthesisRtcExample
    ]
    | None
):
  """Evaluates a single sample."""
  runner = program_runner.ProgramRunner(
      _TARGET_PROGRAM_TAR.value, _IMAGE_NAME.value
  )
  try:
    filepath, prefix, suffix = prefix_and_suffix_for_example(
        input_example.datapoint, runner
    )
  except ValueError:
    logging.exception(
        "Code in file and in sample didn't match. Skipping example."
    )
    return None

  with concurrent.futures.ThreadPoolExecutor(5) as executor:

    def _check_sample_pass(
        synthesized_code: str,
    ) -> concurrent.futures.Future[bool]:
      return executor.submit(
          _compute_sample_pass,
          filepath,
          prefix,
          suffix,
          input_example.datapoint.indentation,
          baseline_pytest_report,
          synthesized_code,
      )

    evaluated_samples = eval_utils.evaluate_generated_output(
        input_example, _check_sample_pass
    )

    return evaluated_samples


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  def load_samples() -> (
      Iterator[
          rtcd.GenerationSamplesForDatapoint[synthesis_rtc.SynthesisRtcExample]
      ]
  ):
    with gzip.open(open(_INPUT_FILE.value, "rb")) as f:
      for line in f:
        sample = rtcd.GenerationSamplesForDatapoint.from_json(line.decode())
        sample.datapoint = synthesis_rtc.SynthesisRtcExample.from_dict(
            sample.datapoint
        )
        yield sample

  with gzip.open(open(_OUTPUT_FILE.value, "ab"), "ab") as f:
    runner = program_runner.ProgramRunner(
        _TARGET_PROGRAM_TAR.value, _IMAGE_NAME.value
    )
    # Get baseline
    baseline_pytest_report = _get_pytest_summary(runner)
    logging.info("Baseline pytest report: %s", baseline_pytest_report)

    for input_example in load_samples():
      sample = _eval_sample(input_example, baseline_pytest_report)
      if sample is None:
        continue
      f.write(sample.to_json().encode())
      f.write(b"\n")


if __name__ == "__main__":
  app.run(main)
