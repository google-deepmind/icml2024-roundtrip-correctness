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

"""Creates line-level code spans with additional context."""
import dataclasses


@dataclasses.dataclass(frozen=True)
class SpanWithContext:
  """A span with context."""

  left_context: str
  span: str
  right_context: str


def get_span_with_context(
    file_bytes: bytes,
    start_pos: int,
    end_pos: int,
    context_chars: int,
    truncation_delimiter: str = "...",
) -> SpanWithContext:
  """Gets the span and its truncated context.

  The `context_chars` are the minimum number of characters (*not* bytes) that
  should be included in the context. This function ensures that entire lines are
  always included. The `context_chars` is equally distributed on the left and
  right of the span. If the left contexts is smaller than the additional space
  is provided for the rest of the right context and vice-versa.

  This utility is useful for trucating inputs for large language models.

  Args:
    file_bytes: The bytes of a the input text.
    start_pos: The stating position in the byte stream of the target span.
    end_pos: The end position in the byte stream of the target span.
    context_chars: The minimum number of context characters on the left and
      right of the span to include. This will be expanded to include the entire
      lines.
    truncation_delimiter: The delimiter to use at the beginning and end of the
      trucated context, when truncation happens.

  Returns:
    The span with the relevant context.
  Raises:
    ValueError if start_pos >= end_pos, i.e., the span is empty.
  """
  if start_pos >= end_pos:
    raise ValueError(
        f"Starting position ({end_pos}) must be after starting position"
        f" ({start_pos})."
    )

  hole_prefix = file_bytes[:start_pos].decode()
  hole = file_bytes[start_pos:end_pos].decode()
  hole_suffix = file_bytes[end_pos:].decode()

  # Use the same number of left and right context, unless one of them is
  # shorter.
  right_context_budget = context_chars // 2
  left_context_budget = context_chars - right_context_budget

  if len(hole_prefix) < left_context_budget:
    right_context_budget += left_context_budget - len(hole_prefix)
  elif len(hole_suffix) < right_context_budget:
    left_context_budget += right_context_budget - len(hole_suffix)

  # Truncate to include the entire lines.
  left_idx = hole_prefix.rfind(
      "\n", 0, max(0, len(hole_prefix) - left_context_budget)
  )
  if left_idx != -1 and left_idx > 0:
    hole_prefix = truncation_delimiter + hole_prefix[left_idx:]

  right_idx = hole_suffix.find("\n", right_context_budget)
  if right_idx != -1 and right_idx < len(hole_suffix) - 1:
    hole_suffix = hole_suffix[:right_idx] + "\n" + truncation_delimiter

  return SpanWithContext(hole_prefix, hole, hole_suffix)
