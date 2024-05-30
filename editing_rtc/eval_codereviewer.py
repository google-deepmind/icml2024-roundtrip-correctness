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

import collections
from collections.abc import Sequence
import gzip

from absl import app
from absl import flags

from roundtrip_correctness import rtc_data as med
from roundtrip_correctness.editing_rtc import task as ertc


_SAMPLES_PATH = flags.DEFINE_string(
    "samples_path",
    None,
    "Path to the directory storing the output samples.",
    required=True,
)

_OUTPUT_FILE = flags.DEFINE_string(
    "output_file",
    None,
    "The output json file containing the scores.",
    required=True,
)


def _compute_metrics(original: str, prediction: str) -> dict[str, float]:
  return {"exact_match": float(prediction == original)}


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  evaluated_samples = []
  with gzip.open(open(_SAMPLES_PATH.value, "rb")) as f:
    for line in f:
      sample = med.GenerationSamplesForDatapoint.from_json(line.decode())
      sample.datapoint = ertc.EditingRtcExample.from_dict(sample.datapoint)
      original = sample.datapoint.code_after_edit.strip()
      baseline_scores = collections.defaultdict(list)
      all_backward_scores = collections.defaultdict(list)
      for generation_sample in sample.generation_samples:
        backward_scores = collections.defaultdict(list)
        for backward_sample in generation_sample.backward_samples:
          prediction = backward_sample.text.split("[old]")[0].strip()
          pred_metrics = _compute_metrics(original, prediction)
          for metric, value in pred_metrics.items():
            backward_scores[metric].append(value)

        for m, l in backward_scores.items():
          all_backward_scores[m].append(l)

      if sample.baseline_samples:
        for baseline_sample in sample.baseline_samples:
          prediction = baseline_sample.text.split("[old]")[0].strip()
          pred_metrics = _compute_metrics(original, prediction)
          for metric, value in pred_metrics.items():
            baseline_scores[metric].append(value)

      evaluated_sample = med.EvaluatedGenerationSamplesForDatapoint[
          ertc.EditingRtcExample
      ](
          samples=sample,
          generation_samples_consistencies=all_backward_scores,
          baseline_samples_consistencies=baseline_scores,
      )
      evaluated_samples.append(evaluated_sample)

  with gzip.open(open(_OUTPUT_FILE.value, "wb"), "wb") as f:
    for evaluated_sample in evaluated_samples:
      f.write(evaluated_sample.to_json().encode())
      f.write(b"\n")


if __name__ == "__main__":
  app.run(main)
