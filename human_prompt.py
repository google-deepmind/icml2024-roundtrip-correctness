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

"""A dummy prompt that defers input to a human in the console, rather than am LLM."""

from __future__ import annotations

from collections.abc import Callable, Collection

import colorama

from roundtrip_correctness import llm_interface as llm


def _print_to_out(x) -> None:
  print(x, end="")


class HumanPrompt(llm.InteractivePromptInterface):
  """A prompt that defers input to a human in the console instead of an LLM.

  This class is to be used mainly for debugging purposes since it does not
  require a connection to an LLM.

  This class is _not_ thread safe.
  """

  def _reset_prompt_display(self) -> None:
    """Resets the tty and prints out the state."""
    _print_to_out(chr(27) + "[2J")  # Reset terminal
    _print_to_out(self._state.getvalue())

  def extend(self, text: str) -> None:
    super().extend(text)
    _print_to_out(text)

  async def sample(
      self,
      *,
      prefix: str,
      stop_token: str | Collection[str],
      num_samples: int = 1,
      max_length: int = 64,
  ) -> list[llm.PromptOutput]:
    samples = []
    for i in range(num_samples):
      self._state.write(prefix)
      _print_to_out(prefix)

      _print_to_out(str(colorama.Fore.YELLOW))
      try:
        _print_to_out(f"Type sample {i+1}: ")
        generated = input()
      finally:
        _print_to_out(colorama.Fore.RESET)

      self._state.write(generated)
      self._state.write("\n")  # Implied by `input()` returning.
      samples.append(llm.PromptOutput(generated, 0))
    return samples

  def fork(self) -> HumanPrompt:
    return HumanPrompt(self._state.getvalue())


def get_human_prompt_factory() -> Callable[[], llm.InteractivePromptInterface]:
  """Returns the factory for RTC prompts."""
  return lambda: HumanPrompt("")
