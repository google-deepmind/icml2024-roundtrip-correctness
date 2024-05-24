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

"""Tests for syntax_constrained_sampling."""

from typing import Final

from absl.testing import absltest
import intervaltree

from roundtrip_correctness import syntax_constrained_sampling
from roundtrip_correctness import treesitter_utils

_SAMPLE_CODE: Final[bytes] = """\
import foo

def bar(x: float, y: float) -> float:
  return x ** 2 + y ** 2

def baz(text: str) -> int:
  if foo.test(text):
    return foo.baz(0)
  return -1
""".encode()


_TRUE_PREDICATE = lambda x: True
_FALSE_PREDICATE = lambda x: False


class SyntaxConstrainedSamplingTest(absltest.TestCase):

  def test_all_nodes_covered(self):
    tree = treesitter_utils.parse_code(
        _SAMPLE_CODE, treesitter_utils.SUFFIX_TO_TS_LANGUAGE["py"]
    )
    collected_nodes = syntax_constrained_sampling._collect_eligible_nodes(
        tree,
        is_eligible_subtree_predicate=_TRUE_PREDICATE,
        exclude_node_predicate=_FALSE_PREDICATE,
        visit_children=syntax_constrained_sampling.visit_all_children,
        min_bytes_length=1,
        max_bytes_length=1000,
    )

    interval_tree = intervaltree.IntervalTree()
    for node in collected_nodes:
      interval_tree.addi(node.start_byte, node.end_byte)

    interval_tree.split_overlaps()
    interval_tree.merge_neighbors()

    # The entire code is covered (at least once)
    self.assertLen(interval_tree, 1)
    single_interval = next(iter(interval_tree))
    self.assertEqual(single_interval.begin, 0)
    self.assertLen(_SAMPLE_CODE, single_interval.end)

    # And the sum of all node weights is equal to the number of non-whitespace
    #  characters
    _, weights = syntax_constrained_sampling._weight_candidates(
        collected_nodes, _SAMPLE_CODE
    )
    self.assertAlmostEqual(weights.sum(), 118)

  def test_only_fns_context(self):
    tree = treesitter_utils.parse_code(
        _SAMPLE_CODE, treesitter_utils.SUFFIX_TO_TS_LANGUAGE["py"]
    )
    collected_nodes = syntax_constrained_sampling._collect_eligible_nodes(
        tree,
        is_eligible_subtree_predicate=lambda n: n.type == "function_definition",
        exclude_node_predicate=lambda n: n.type == "unary_operator",
        visit_children=syntax_constrained_sampling.visit_all_children,
        min_bytes_length=0,
        max_bytes_length=1000,
    )

    interval_tree = intervaltree.IntervalTree()
    all_spans = set()
    for node in collected_nodes:
      interval_tree.addi(node.start_byte, node.end_byte)
      all_spans.add(tree.text[node.start_byte : node.end_byte])

    # Are the whole functions covered only?
    interval_tree.split_overlaps()
    interval_tree.merge_neighbors()
    covered_spans = set()
    for interval in interval_tree:
      covered_spans.add(_SAMPLE_CODE[interval.begin : interval.end])

    self.assertSetEqual(
        # The text of the two functions defined `bar` and `baz`.
        covered_spans,
        {_SAMPLE_CODE[12:74], _SAMPLE_CODE[76:-1]},
    )

    # Are some subexpressions appearing within the functions in all_spans?
    self.assertContainsSubset(
        {
            b"x ** 2",
            b"y ** 2",
            b"x ** 2 + y ** 2",
            b"x",
            b"y",
            b"**",
            b"return -1",
            b"text",
            b"foo.test",
        },
        all_spans,
    )

    # The excluded child nodes are _not_ present
    self.assertNotIn(b"-1", all_spans)


if __name__ == "__main__":
  absltest.main()
