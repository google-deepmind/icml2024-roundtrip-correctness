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

"""Example generation for Synthesis RTC."""

from collections.abc import Iterator

from absl import flags
from absl import logging
import colorama
import py_tree_sitter as ts

from roundtrip_correctness import syntax_constrained_sampling
from roundtrip_correctness import text_utils
from roundtrip_correctness import treesitter_utils
from roundtrip_correctness.synthesis_rtc import task as synth_rtc


_PRINT_SAMPLES = flags.DEFINE_bool("print_samples", False, "Print samples.")
_CONTEXT_SIZE = flags.DEFINE_integer(
    "context_size", 1024, "Context size in chars."
)
_MIN_SPAN_BYTE_LENGTH = flags.DEFINE_integer(
    "min_span_byte_length", 64, "Minimum code region span byte length."
)
_MAX_SPAN_BYTE_LENGTH = flags.DEFINE_integer(
    "max_span_byte_length", 512, "Maximum code region span byte length."
)


def _is_python_sampled_context(node: ts.Node) -> bool:
  """Samples nodes only inside functions."""
  return (
      node.type == "block"
      and node.parent
      and node.parent.type == "function_definition"
  )


def _is_java_sampled_context(node: ts.Node) -> bool:
  """Samples nodes only inside methods."""
  return (
      node.type == "block"
      and node.parent
      and node.parent.type == "method_declaration"
  )


_IS_SAMPLED_CONTEXT = {
    "py": _is_python_sampled_context,
    "java": _is_java_sampled_context,
}


def _visit_children_python(node: ts.Node) -> Iterator[ts.Node]:
  """Vist simple statements only."""
  if node.type in (
      "future_import_statement",
      "import_statement",
      "import_from_statement",
      "print_statement",
      "assert_statement",
      "expression_statement",
      "return_statement",
      "delete_statement",
      "raise_statement",
      "pass_statement",
      "break_statement",
      "continue_statement",
      "global_statement",
      "nonlocal_statement",
      "exec_statement",
  ):
    # Do not visit "fine-grained statements"
    return
  yield from node.children


def _visit_children_java(node: ts.Node) -> Iterator[ts.Node]:
  match node.type:
    case (
        "expression_statement"
        | "local_variable_declaration"
        | "return_statement"
        | "throw_statement"
    ):
      # Stop traversing
      return
    case "if_statement":
      yield from node.children_by_field_name("consequence")
      yield from node.children_by_field_name("alternative")
    case "for_statement" | "enhanced_for_statement" | "while_statement":
      yield from node.children_by_field_name("body")
    case "try_with_resources_statement" | "catch_clause" | "finally_clause":
      yield from node.children_by_field_name("body")
    case _:
      yield from node.children


_VISIT_CHILDREN = {"py": _visit_children_python, "java": _visit_children_java}


def _python_exclude_nodes(node: ts.Node) -> bool:
  """Keep only single/multi-line statements, except from docstrings."""
  return not node.type.endswith("_statement") or (  # Docstrings
      node.type == "expression_statement"
      and len(node.children) == 1
      and node.children[0].type == "string"
  )


def _java_exclude_nodes(node: ts.Node) -> bool:
  return node.type in (
      "line_comment",
      "block_comment",
      "block",
      "switch_block",
      "{",
      "}",
      ":",
      "catch_clause",
      "finally_clause",
  ) or (
      # Nested if-elses
      node.type == "if_statement"
      and node.parent
      and node.parent.type == "if_statement"
  )


_EXCLUDE_NODES = {"py": _python_exclude_nodes, "java": _java_exclude_nodes}


_LINE_COMMENT_PREFIX = {"py": "#", "java": "//", "js": "//"}


class _NotFullLineError(Exception):
  """An exception to indicate that the sampled node is only part of a line.

  This can happen in some cases, e.g., in Java
    if (foo) bar()

  While filtering this out might be possible, we ignore such relatively rare
    cases.
  """


def _adjust_start_and_end_pos(
    code_bytes, start_pos, end_pos
) -> tuple[int, int]:
  """Adjusts start and end pos to whole lines."""
  # Ensure that start_pos is just after \n
  while (
      start_pos - 1 >= 0
      and code_bytes[start_pos - 1 : start_pos].isspace()
      and code_bytes[start_pos : start_pos + 1] != b"\n"
  ):
    start_pos -= 1
  if code_bytes[start_pos : start_pos + 1] != b"\n":
    raise _NotFullLineError()

  start_pos += 1

  # Ensure that end_pos ends in \n
  # This consumes any subsequent comment. In rare cases, this may consume
  #  part of the subsequent statement (e.g. when there are multiple
  # statements in a single line). This is indented behavior for now.
  while (
      end_pos < len(code_bytes) and code_bytes[end_pos : end_pos + 1] != b"\n"
  ):
    end_pos += 1
  end_pos += 1

  return start_pos, end_pos


def extract_samples_from_file(
    filepath: str,
    eligible_lines: set[int] | None = None,
    max_num_samples: int = 10,
    max_samples_per_kb: int = 5,
    sample_with_replacement: bool = False,
) -> Iterator[synth_rtc.SynthesisRtcExample]:
  """Extracts samples from the input file."""
  suffix = filepath.split(".")[-1]
  if suffix not in treesitter_utils.SUFFIX_TO_TS_LANGUAGE:
    return
  tree = treesitter_utils.parse(
      filepath, treesitter_utils.SUFFIX_TO_TS_LANGUAGE[suffix]
  )
  code_bytes: bytes = tree.text
  if tree.root_node.has_error:
    logging.warning("Error parsing %s", filepath)
    return

  if eligible_lines:

    def is_eligible_subtree_predicate(node: ts.Node) -> bool:
      if not _IS_SAMPLED_CONTEXT[suffix](node):
        return False

      return any(
          l in eligible_lines
          # Tree sitter has 0-based lines. Range must be inclusive.
          for l in range(node.start_point[0] + 1, node.end_point[0] + 2)
      )

  else:
    is_eligible_subtree_predicate = _IS_SAMPLED_CONTEXT[suffix]

  samples = syntax_constrained_sampling.sample(
      tree,
      is_eligible_subtree_predicate=is_eligible_subtree_predicate,
      exclude_node_predicate=_EXCLUDE_NODES[suffix],
      visit_children=_VISIT_CHILDREN[suffix],
      min_bytes_length=_MIN_SPAN_BYTE_LENGTH.value,
      max_bytes_length=_MAX_SPAN_BYTE_LENGTH.value,
      # Hard-coded heuristic
      num_samples=min(
          max_num_samples, max_samples_per_kb * len(code_bytes) // 1024
      ),  # 2 samples per KB of code
      sample_with_replacement=sample_with_replacement,
  )

  if not samples:
    logging.warning("No elements found in %s", filepath)

  for sampled_span in samples:
    try:
      start_pos, end_pos = _adjust_start_and_end_pos(
          code_bytes, sampled_span.start_pos, sampled_span.end_pos
      )
    except _NotFullLineError:
      logging.warning("A non-full line node was sampled and ignored.")
      continue

    span_with_context = text_utils.get_span_with_context(
        code_bytes, start_pos, end_pos, _CONTEXT_SIZE.value
    )

    yield synth_rtc.SynthesisRtcExample(
        filename=filepath,
        start_point=sampled_span.start_point,
        end_point=sampled_span.end_point,
        code_before_hole=span_with_context.left_context,
        code_after_hole=span_with_context.right_context,
        code_in_hole=span_with_context.span,
        line_comment_prefix=_LINE_COMMENT_PREFIX[suffix],
    )

    if _PRINT_SAMPLES.value:
      print(
          f"{colorama.Back.LIGHTRED_EX}{colorama.Fore.BLACK}{filepath}  "
          f" {sampled_span.start_point}-->{sampled_span.end_point}"
          f"{colorama.Style.RESET_ALL}\n{span_with_context.left_context}"
          f"{colorama.Fore.RED}{colorama.Style.BRIGHT}{span_with_context.span}"
          f"{colorama.Style.RESET_ALL}{span_with_context.right_context}"
      )

      print(
          "--------------------------------------------------------------------"
      )
