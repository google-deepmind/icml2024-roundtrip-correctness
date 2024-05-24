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

"""The interface of an interactive prompt, such as an LLM."""

from __future__ import annotations

import abc
from collections.abc import Callable, Collection
import io
from typing import NamedTuple, Optional, TypeVar


class PromptOutput(NamedTuple):
  text: str
  logp: float


_T = TypeVar('_T', bound='InteractivePromptInterface')


class InteractivePromptInterface(abc.ABC):
  """Interface for consecutively interacting with an LLM.

  The interface allows multiple consecutive interactions where either the
  LLM or this interface appends text to the model "prompt". Given the
  state of the prompt, the model can be asked to generate a single
  response, or to sample multiple responses. The interaction can then
  continue by appending more text through the interface or asking the
  model to generate additional text.
  """

  def __init__(self, starting_state: Optional[str] = None):
    self._state = io.StringIO()
    if starting_state:
      self._state.write(starting_state)

  @property
  def state(self) -> str:
    """The state of the prompt."""
    return self._state.getvalue()

  def extend(self, text: str) -> None:
    """Append the given text to the prompt changing its state."""
    self._state.write(text)

  @abc.abstractmethod
  async def sample(
      self,
      *,
      prefix: str,
      stop_token: str | Collection[str],
      num_samples: int = 1,
      max_length: int = 64,
  ) -> list[PromptOutput]:
    """Samples generations from the prompt.

    This does not affect the state of the prompt.

    Arguments:
      prefix: the input prefix in the beginning of the generation.
      stop_token: a string representation of the token or tokens that completes
        the generation.
      num_samples: the minimum number of samples to generate.
      max_length: the maximum length of the generation in
        implementation-specific tokens (e.g., BPE). This is used to avoid
        excessively long generations.

    Returns:
      A list of `PromptOutput` tuples containing the text and the log prob
      of each generation.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def fork(self: _T) -> _T:
    """Forks the prompt and its state."""
    raise NotImplementedError


PromptFactory = Callable[[], InteractivePromptInterface]


def truncate_to_stop_token(
    text: str, stop_tokens: str | Collection[str]
) -> str:
  """Truncates the text to the given stop tokens."""
  if isinstance(stop_tokens, str):
    stop_tokens = (stop_tokens,)

  for tok in stop_tokens:
    tok_idx = text.find(tok)
    if tok_idx != -1:
      text = text[: tok_idx + len(tok)]
  return text
