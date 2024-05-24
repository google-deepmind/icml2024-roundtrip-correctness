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

"""Convert MBPP to a Desc2Synth dataset."""

from collections.abc import Iterator, Sequence
import gzip
import re

from absl import app
from absl import flags
import libcst as cst
import tensorflow_datasets as tfds

from roundtrip_correctness.synthesis_rtc import task as synth_rtc

_OUT_PATH = flags.DEFINE_string(
    "out_path", None, "The path to write the output to.", required=True
)


class _RemoveDocstring(cst.CSTTransformer):
  """Removes docstrings from code."""

  def __init__(self):
    super().__init__()
    self.removed_text = []

  def leave_FunctionDef(
      self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
  ) -> cst.FunctionDef:
    try:
      if isinstance(updated_node.body.body[0].body[0].value, cst.SimpleString):
        docstring_text: str = updated_node.body.body[0].body[0].value.value
        docstring_text = (
            docstring_text.removeprefix("'''")
            .removesuffix("'''")
            .removeprefix('"""')
            .removesuffix('"""')
        )
        docstring_text = re.sub(r"\s+", " ", docstring_text)
        self.removed_text.append(docstring_text.strip())
        return updated_node.with_changes(
            body=updated_node.body.with_changes(body=updated_node.body.body[1:])
        )
    except:  # pylint: disable=bare-except
      pass
    return updated_node


def _iter_humaneval() -> Iterator[synth_rtc.SynthesisRtcExample]:
  """Iterates over all HumanEval problems and generates the code-to-be-traced."""
  ds = tfds.load("huggingface:openai_humaneval/openai_humaneval")

  for d in ds["test"]:
    prompt = d["prompt"].numpy().decode()
    canonical_solution = d["canonical_solution"].numpy().decode()
    if not prompt.endswith("\n"):
      prompt = prompt + "\n"

    i = 0
    while canonical_solution[i : i + 1].isspace():
      i += 1
    indent = canonical_solution[:i]
    module = cst.parse_module(f"{prompt}{indent}...\n")
    docstring_remover = _RemoveDocstring()
    code_no_docstrings = module.visit(docstring_remover).code

    code_no_docstrings = code_no_docstrings.splitlines(keepends=True)
    num_lines = len(code_no_docstrings)

    task_id = d["task_id"].numpy().decode()

    yield synth_rtc.SynthesisRtcExample(
        filename=f"{task_id}.py",
        start_point=(num_lines, 0),
        end_point=(num_lines + len(canonical_solution.splitlines()) + 1, 0),
        # 1-based line numbering
        code_before_hole="".join(code_no_docstrings[:-1]),  # Remove "..."
        code_after_hole="",
        code_in_hole=canonical_solution,
        line_comment_prefix="#",
        ground_truth_description=docstring_remover.removed_text[-1]
        if docstring_remover.removed_text
        else None,
    )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  with gzip.open(open(_OUT_PATH.value, "wb"), "wb") as f:
    for e in _iter_humaneval():
      f.write(e.to_json().encode())
      f.write(b"\n")


if __name__ == "__main__":
  app.run(main)
