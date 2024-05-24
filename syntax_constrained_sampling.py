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

"""Syntax-constrained code span sampling for infilling-style tasks."""
import collections
from collections.abc import Callable, Iterator, Sequence
import dataclasses
import re
from typing import Final

from absl import logging
import intervaltree
import numpy as np
import py_tree_sitter as ts


@dataclasses.dataclass(frozen=True)
class SampledSpan:
  # The start-end range in bytes.
  start_pos: int
  end_pos: int

  # The (line, column) range.
  start_point: tuple[int, int]
  end_point: tuple[int, int]

  # The sampled Tree-Sitter node, for reference
  node: ts.Node


def visit_all_children(node: ts.Node) -> Iterator[ts.Node]:
  """Visits all children of a node."""
  yield from node.children


def sample(
    tree: ts.Tree,
    is_eligible_subtree_predicate: Callable[[ts.Node], bool],
    exclude_node_predicate: Callable[[ts.Node], bool],
    visit_children: Callable[[ts.Node], Iterator[ts.Node]] = visit_all_children,
    min_bytes_length: int = 8,
    max_bytes_length: int = 1024,
    num_samples: int = 1,
    sample_with_replacement: bool = False,
    temperature: float = 0.8,
) -> Sequence[SampledSpan]:
  """Syntax-constrained code span sampling.

  Samples (=syntax nodes or sequences of syntax nodes) are sampled such that
  each character has equal probability of appearing within a sample among all
  included characters. A bias towards longer sequences can be induced by
  decreasing the sampling temperature.

  If randomness needs to be controlled, the np.seed must be set prior to
  invoking this method.

  Args:
    tree: The tree-sitter parsed tree
    is_eligible_subtree_predicate: Returns true when visiting a node whose
      children can be eligible samples.
    exclude_node_predicate: Returns true if a particular node is to be excluded
      from sampling.
    visit_children: A function that given a node, yields the children to be
      visited for consideration.
    min_bytes_length: The minimum byte length of a sample.
    max_bytes_length: The maximum byte length of a sample.
    num_samples: How many holes to sample.
    sample_with_replacement: How to sample.
    temperature: Sampling temperature. A non-negative number. As this tends to 0
      the longest permitted node will be returned. As this tends to +inf, the
      sample nodes will be selected uniformly at random independently of their
      length.

  Returns:
    A list of the sampled nodes.
  """
  if temperature < 0:
    raise ValueError("Temperature must be non-negative.")

  eligible_nodes = _collect_eligible_nodes(
      tree,
      is_eligible_subtree_predicate,
      exclude_node_predicate,
      visit_children=visit_children,
      min_bytes_length=min_bytes_length,
      max_bytes_length=max_bytes_length,
  )
  logging.info(
      "Found %s eligible nodes",
      len(eligible_nodes),
  )
  if not eligible_nodes:
    return []

  eligible_nodes, weights = _weight_candidates(eligible_nodes, tree.text)

  weights = weights ** (1 / temperature)
  selected_nodes = np.random.choice(
      eligible_nodes,
      p=weights / weights.sum(),
      size=min(num_samples, len(eligible_nodes)),
      replace=sample_with_replacement,
  )

  return [
      SampledSpan(n.start_byte, n.end_byte, n.start_point, n.end_point, n)
      for n in selected_nodes
  ]


@dataclasses.dataclass(frozen=True, slots=True)
class _NodeGroup:
  """A group of consecutive syntax nodes, such as sequential statements.

  Assumes input nodes are sorted and consecutive.
  """

  nodes: Sequence[ts.Node]

  def __post_init__(self):
    if len(self.nodes) <= 1:
      raise ValueError("Node group should have more than 1 underlying node")

  @property
  def start_byte(self) -> int:
    return self.nodes[0].start_byte

  @property
  def end_byte(self) -> int:
    return self.nodes[-1].end_byte

  @property
  def id(self) -> int:
    return self.nodes[-1].id - self.nodes[0].id

  @property
  def type(self) -> str:
    return "_NodeGroup"

  @property
  def start_point(self) -> tuple[int, int]:
    return self.nodes[0].start_point

  @property
  def end_point(self) -> tuple[int, int]:
    return self.nodes[-1].end_point


_CandidateNode = ts.Node | _NodeGroup


@dataclasses.dataclass(frozen=True)
class _HashableNodeWrapper:
  """A wrapper to make a tree-sitter node hashable via object identity.

  This is required to store tree-sitter nodes to a dictionary.
  """

  node: _CandidateNode

  def __hash__(self) -> int:
    return self.node.id

  def __eq__(self, other):
    if not isinstance(other, _HashableNodeWrapper):
      return False
    return self.node is other.node


_IS_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")


def _weight_candidates(
    candidate_nodes: Sequence[_CandidateNode], code_bytes: bytes
) -> tuple[Sequence[_CandidateNode], np.ndarray]:
  """Computes candidate node weights.

  We want each candidate to be selected proportionally to its char length.
  However, some candidates overlap with each other and should be downweighted.

  For example, in the expression `(a + b) + c` the `a + b` can appear both
  as standalone candidate and within the original expression. We want to
  downweight these two candidates so that (parts of) the expression have
  equal probability of being selected.

  Note that whitespace is excluded from the length of each node.

  Args:
    candidate_nodes: An interval tree with the text span of CandidateNode to be
      considered for sampling.
    code_bytes: The bytes of the code file.

  Returns:
    A dictionary of each _CandidateNode with its length weight.
  """
  nodes_interval_tree = intervaltree.IntervalTree()
  for candidate_node in candidate_nodes:
    nodes_interval_tree.addi(
        candidate_node.start_byte, candidate_node.end_byte, candidate_node
    )

  def _accu(
      accumulator: list[_CandidateNode], node: _CandidateNode
  ) -> list[_CandidateNode]:
    accumulator.append(node)
    return accumulator

  nodes_interval_tree.split_overlaps()
  nodes_interval_tree.merge_overlaps(
      data_reducer=_accu, data_initializer=list()
  )

  candidate_weights = collections.defaultdict(float)
  for interval in nodes_interval_tree:
    interval: intervaltree.Interval

    num_whitespace_chars = sum(
        len(ws_span)
        for ws_span in _IS_WHITESPACE_RE.findall(
            code_bytes[interval.begin : interval.end].decode()
        )
    )

    candidates_in_interval: list[_CandidateNode] = interval.data
    weighted_interval_length = (
        interval.end - interval.begin - num_whitespace_chars
    ) / len(candidates_in_interval)

    for candidate in candidates_in_interval:
      candidate_weights[
          _HashableNodeWrapper(candidate)
      ] += weighted_interval_length

  nodes_interval_tree, weights = zip(*candidate_weights.items())
  return [c.node for c in nodes_interval_tree], np.array(weights)


def _collect_eligible_nodes(
    tree: ts.Tree,
    is_eligible_subtree_predicate: Callable[[ts.Node], bool],
    exclude_node_predicate: Callable[[ts.Node], bool],
    visit_children: Callable[[ts.Node], Iterator[ts.Node]],
    min_bytes_length: int,
    max_bytes_length: int,
) -> Sequence[_CandidateNode]:
  """Visits the tree and retrieves all eligible nodes."""
  to_visit: list[tuple[ts.Node, bool]] = [(
      tree.root_node,
      is_eligible_subtree_predicate(tree.root_node),
  )]
  eligible_nodes = []
  while to_visit:
    current_node, is_eligible_subtree = to_visit.pop()

    node_char_len = current_node.end_byte - current_node.start_byte
    if node_char_len <= min_bytes_length:
      # Neither this node nor its children will be longer than min_char_length
      continue

    # If node is of appropriate size and is not excluded, add to interval tree.
    if (
        min_bytes_length < node_char_len <= max_bytes_length
        and is_eligible_subtree
        and not exclude_node_predicate(current_node)
    ):
      eligible_nodes.append(current_node)

    children = tuple(visit_children(current_node))
    to_visit.extend(
        (c, is_eligible_subtree | is_eligible_subtree_predicate(c))
        for c in children
    )

    if not is_eligible_subtree:
      # Do not sample groups of children of unsamplable nodes.
      continue

    # Add sequences of nodes (groups) from the children (e.g., sequential
    # statements).
    for i, child_i in enumerate(children):
      if exclude_node_predicate(child_i):
        continue

      for j in range(i + 2, len(children)):
        node_group = _NodeGroup(children[i:j])
        node_group_len = node_group.end_byte - node_group.start_byte
        if (
            min_bytes_length < node_group_len <= max_bytes_length
            and not exclude_node_predicate(node_group)
        ):
          eligible_nodes.append(node_group)
        elif node_group_len > max_bytes_length:
          break  # Do not expand group any further, it will be longer.

  return eligible_nodes
