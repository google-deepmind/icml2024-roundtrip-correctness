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

"""A CLI to extract samples from a given file."""

from collections.abc import Sequence
import glob
import gzip
import logging
import os

from absl import app
from absl import flags

from roundtrip_correctness.synthesis_rtc import example_gen


_INPUT_FOLDER = flags.DEFINE_string(
    "input_folder", None, "Input folder.", required=True
)

_OUT_PATH = flags.DEFINE_string("out_path", None, "Output path.", required=True)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  with gzip.open(open(_OUT_PATH.value, "wb"), "wb") as f:
    for filename in glob.iglob(
        os.path.join(_INPUT_FOLDER.value, "*"), recursive=True
    ):
      if "test" in filename.lower():
        logging.info("Skipping %s because it looks like a test.", filename)
        continue
      for example in example_gen.extract_samples_from_file(filename):
        example.filename = os.path.relpath(filename, _INPUT_FOLDER.value)
        f.write(example.to_json().encode())
        f.write(b"\n")


if __name__ == "__main__":
  app.run(main)
