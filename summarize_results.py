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

r"""Summarize the RTC results."""

import collections
from collections.abc import Sequence
import gzip
import json
import os
from typing import Any

from absl import app
from absl import flags
import numpy as np
import rich.console
import rich.table

from roundtrip_correctness import rtc_data as rtcd

_INPUT_DATA_PATH = flags.DEFINE_string(
    "input_data_path", None, "Input data", required=True
)
_OUTPUT_PER_EXAMPLE_STATS = flags.DEFINE_string(
    "output_per_example_stats", None, "Output per-example stats to a json file."
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  def example_to_group_id(example) -> str:
    return example["filename"].split("::")[0]

  generation_path = _INPUT_DATA_PATH.value

  per_group_stats = collections.defaultdict(
      lambda: collections.defaultdict(list)
  )

  if not os.path.exists(generation_path):
    raise FileNotFoundError(f"{generation_path} does not exist.")

  table = rich.table.Table(title=f"Stats for {_INPUT_DATA_PATH.value}")
  table.add_column("Name")
  table.add_column("Avg", justify="right")

  generation_scores = collections.defaultdict(list)
  with gzip.open(open(generation_path, "rb")) as f:
    try:
      for line in f:
        sample: rtcd.EvaluatedGenerationSamplesForDatapoint[Any] = (
            rtcd.EvaluatedGenerationSamplesForDatapoint.from_json(line.decode())
        )
        sample.samples = rtcd.GenerationSamplesForDatapoint.from_dict(
            sample.samples
        )
        sample_scores = sample.compute_scores()
        for metric_name, metric_value in sample_scores.items():
          generation_scores[metric_name].append(metric_value)
          per_group_stats[example_to_group_id(sample.samples.datapoint)][
              metric_name
          ].append(metric_value)
    except EOFError:
      print("Could not load entire file.")

  for metric_name, metric_values in generation_scores.items():
    table.add_row(
        metric_name,
        f"{np.mean(metric_values):.3f}",
    )

  if _OUTPUT_PER_EXAMPLE_STATS.value:
    with open(_OUTPUT_PER_EXAMPLE_STATS.value, "w") as f:
      json.dump(per_group_stats, f)

  console = rich.console.Console()
  console.print(table)


if __name__ == "__main__":
  app.run(main)
