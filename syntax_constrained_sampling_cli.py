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

"""A CLI interface for visualizing AST Sampling output."""
from collections.abc import Sequence

from absl import app
from absl import flags
import colorama
import py_tree_sitter as ts

from roundtrip_correctness import syntax_constrained_sampling
from roundtrip_correctness import text_utils
from roundtrip_correctness import treesitter_utils

_INPUT_FILE = flags.DEFINE_string(
    "input_file", None, "Input file.", required=True
)
NUM_SAMPLES = flags.DEFINE_integer(
    "num_samples", 100, "The number of samples to generate."
)
_TRUNCATE_CONTEXT_NUM_CHARS = flags.DEFINE_integer(
    "truncate_context_num_chars",
    180,
    "The number of characters to truncate the context when showing the"
    " results.",
)


def _is_python_sampling_context(node: ts.Node) -> bool:
  return node.type == "function_definition"


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  suffix = _INPUT_FILE.value.split(".")[-1]
  tree = treesitter_utils.parse(
      _INPUT_FILE.value, treesitter_utils.SUFFIX_TO_TS_LANGUAGE[suffix]
  )

  samples = syntax_constrained_sampling.sample(
      tree,
      _is_python_sampling_context,
      exclude_node_predicate=lambda n: False,
      num_samples=NUM_SAMPLES.value,
  )

  for sampled_span in samples:
    span_with_context = text_utils.get_span_with_context(
        tree.text,
        sampled_span.start_pos,
        sampled_span.end_pos,
        _TRUNCATE_CONTEXT_NUM_CHARS.value,
    )
    print(
        f"{span_with_context.left_context}"
        f"{colorama.Fore.RED}{colorama.Style.BRIGHT}{span_with_context.span}"
        f"{colorama.Style.RESET_ALL}{span_with_context.right_context}"
    )
    print(
        "-----------------------------------------------------------------------"
    )


if __name__ == "__main__":
  app.run(main)
