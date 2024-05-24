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

"""Tests for text_utils."""
from typing import Final

from absl.testing import absltest
from absl.testing import parameterized

from roundtrip_correctness import text_utils

_EXAMPLE_INPUT: Final[bytes] = b"""\
this is a test text.

It contains empty lines
and
random text used to test

the text_utils
"""


class TextUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="middle_span",
          start_pos=46,
          end_pos=46 + len("and\nrandom text used to test\n"),
          context_chars=10,
          expected_out=text_utils.SpanWithContext(
              "...\nIt contains empty lines\n",
              "and\nrandom text used to test\n",
              "\nthe text_utils\n",
          ),
      ),
      dict(
          testcase_name="single_line",
          start_pos=25,
          end_pos=25 + len("contains"),
          context_chars=1,
          expected_out=text_utils.SpanWithContext(
              "...\nIt ", "contains", " empty lines\n..."
          ),
      ),
      dict(
          testcase_name="no_truncate_left",
          start_pos=0,
          end_pos=0 + len("this is a test text.\n"),
          context_chars=8,
          expected_out=text_utils.SpanWithContext(
              "", "this is a test text.\n", "\nIt contains empty lines\n..."
          ),
      ),
      dict(
          testcase_name="no_truncate_right",
          start_pos=76,
          end_pos=76 + len("the text_utils\n"),
          context_chars=5,
          expected_out=text_utils.SpanWithContext(
              "...\nrandom text used to test\n\n", "the text_utils\n", ""
          ),
      ),
  )
  def test_span_with_context_with_context_chars(
      self, start_pos: int, end_pos: int, context_chars: int, expected_out: str
  ):
    output = text_utils.get_span_with_context(
        _EXAMPLE_INPUT, start_pos, end_pos, context_chars
    )
    self.assertEqual(output, expected_out)


if __name__ == "__main__":
  absltest.main()
