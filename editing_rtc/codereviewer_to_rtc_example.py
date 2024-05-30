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

"""Convert CodeReviewer to an EditingRtcExample dataset."""

from collections.abc import Iterator, Sequence
import gzip
import json

from absl import app
from absl import flags

from roundtrip_correctness.editing_rtc import task as editing_rtc

_INPUT_PATH = flags.DEFINE_string(
    "input_path",
    None,
    "The path to the downloaded CodeReviewer test jsonl file.",
    required=True,
)


_OUT_PATH = flags.DEFINE_string(
    "out_path", None, "The path to write the output to.", required=True
)


def _remove_annotation_symbol(block: str, symbol: str) -> str:
  """Removes the diff-related annotation symbol from the block."""
  return "\n".join(l.removeprefix(symbol) for l in block.splitlines())


def _iter_codereviewer() -> Iterator[editing_rtc.EditingRtcExample]:
  """Iterates over CodeReviewer examples and generates EditingRtc examples ."""
  with open(_INPUT_PATH.value, "r") as f:
    for line in f:
      ex = json.loads(line)
      yield editing_rtc.EditingRtcExample(
          filename="code_reviewer_refinement_test_{}".format(ex["ghid"]),
          code_before_edit=_remove_annotation_symbol(ex["old"], "-"),
          code_after_edit=_remove_annotation_symbol(ex["new"], "+"),
          ground_truth_edit_description=ex["comment"],
      )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  with gzip.open(open(_OUT_PATH.value, "wb"), "wb") as f:
    # NOTE: The CodeReviewer test set is very large, and so we use only
    # a random subset of it for our experiments.
    for e in _iter_codereviewer():
      f.write(e.to_json().encode())
      f.write(b"\n")


if __name__ == "__main__":
  app.run(main)
