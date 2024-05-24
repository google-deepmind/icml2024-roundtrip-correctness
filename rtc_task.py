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

"""The interface for self-consistency evaluations."""

from __future__ import annotations

import abc
import asyncio
from collections.abc import Collection, Iterator
import io
import typing
from typing import AsyncIterator, Final, Generic, Type, TypeVar, final

from absl import flags
import dataclasses_json

from roundtrip_correctness import llm_interface as llm
from roundtrip_correctness import rtc_data as rtcd


_LOG_PROMPTS_AND_RESPONSES = flags.DEFINE_string(
    'log_prompts_and_responses',
    None,
    'Optional location to log prompt and responses for debugging.',
)

_TInput = TypeVar('_TInput', bound=dataclasses_json.DataClassJsonMixin)


class RoundTripCorrecntessTask(Generic[_TInput], abc.ABC):
  """An abstract class for roundtrip correctness evaluation tasks.

  Implementors of this class need to be able to:
  * Construct forward prompts.
  * Extract the relevant info from the output of the forward step.
  * Construct backward prompts.

  Attributes:
    forward_prompt_factory: A lambda that returns a prompt interface to use for
      the forwards step.
    backward_prompt_factory: A lambda that returns a prompt interface to use for
      the backwards step.
    num_forward_examples: The number of forward samples to draw.
    num_backward_examples: The number of backward examples to draw _per forward_
      sample.
    max_forward_generation_len: The maximum length for the decoder.
    max_backward_generation_len: The maximum length for the decoder.
  """

  def __init__(
      self,
      forward_prompt_factory: llm.PromptFactory,
      backward_prompt_factory: llm.PromptFactory,
      forward_prompt_instruction: str,
      backward_prompt_instruction: str,
      forward_few_shot_examples: Collection[_TInput],
      backward_few_shot_examples: Collection[_TInput],
      n_forward_samples: int,
      n_backward_samples: int,
      max_forward_generation_len: int,
      max_backward_generation_len: int,
      example_separator_token: str,
      stopping_tokens: Collection[str] = (),
      max_concurrent_examples: int = 25,
  ):
    self._fw_prompt_factory = forward_prompt_factory
    self._bw_prompt_factory = backward_prompt_factory
    self._n_forward_examples = n_forward_samples
    self._n_backward_examples = n_backward_samples

    self._max_forward_generation_len = max_forward_generation_len
    self._max_backward_generation_len = max_backward_generation_len

    self._example_separator: Final[str] = example_separator_token
    self._stopping_tokens: Final[set[str]] = set(stopping_tokens) | {
        example_separator_token
    }

    self._forward_prompt_prefix: Final[str] = (
        self._render_forward_prompt_prefix(
            forward_prompt_instruction, forward_few_shot_examples
        )
    )
    self._backward_prompt_prefix: Final[str] = (
        self._render_backwards_prompt_prefix(
            backward_prompt_instruction, backward_few_shot_examples
        )
    )

    # This controls the maximum number of examples (backward and forward)
    # that will be computed concurrently. The reason for having this is
    # that due to asyncio's concurrent nature all examples would be
    # computed concurrently, meaning that it might take a very
    # long time until the first full end-to-end sample is fully
    # generated.
    self._max_concurrent_examples = asyncio.Semaphore(max_concurrent_examples)

  input_type: Type[_TInput]

  def __init_subclass__(cls) -> None:
    # Store generic type to allow to be retrieved during run-time.
    cls.input_type = typing.get_args(cls.__orig_bases__[0])[0]  # type: ignore

  @abc.abstractmethod
  def _format_datapoint_for_forward(
      self, datapoint: _TInput
  ) -> tuple[str, str]:
    """Return the input and target (when available) for the given datapoint."""
    raise NotImplementedError

  def _render_forward_prompt_prefix(
      self,
      forward_prompt_instruction: str,
      forward_few_shot_examples: Collection[_TInput],
  ) -> str:
    """Renders the string for the few-shot prefix of the prompt."""
    with io.StringIO() as sb:
      if forward_prompt_instruction:
        sb.write(forward_prompt_instruction)
        sb.write('\n')
      for example in forward_few_shot_examples:
        context, target = self._format_datapoint_for_forward(example)
        sb.write(f'{context} {target}{self._example_separator}')

      return sb.getvalue()

  @final
  def _format_forward_prompt(self, datapoint: _TInput) -> rtcd.PromptAndTarget:
    """Creates the prompt and expected target for the forward pass.

    Args:
      datapoint: The datapoint used to create the RTC data.

    Returns:
      The forward prompt, target (if known), and stopping token.
    """
    context, target = self._format_datapoint_for_forward(datapoint)
    return rtcd.PromptAndTarget(
        context=f'{self._forward_prompt_prefix}{context}',
        target=target,
        stopping_tokens=self._stopping_tokens,
    )

  @property
  @abc.abstractmethod
  def _baseline_forward_prediction(self) -> str | None:
    """Returns the constant non-informative forward "sample", if supported."""
    raise NotImplementedError

  @abc.abstractmethod
  def _format_datapoint_for_backward(
      self,
      datapoint: _TInput,
      backward_prompt_instruction: str | None = None,
  ) -> tuple[str, str]:
    """Formats the data point for the backwards prompt."""
    raise NotImplementedError

  @final
  def _render_backwards_prompt_prefix(
      self,
      backward_prompt_instruction: str,
      backward_few_shot_examples: Collection[_TInput],
  ) -> str:
    """Renders the string for the few-shot prefix of the prompt."""
    with io.StringIO() as sb:
      if backward_prompt_instruction:
        sb.write(backward_prompt_instruction)
        sb.write('\n')
      for example in backward_few_shot_examples:
        context, target = self._format_datapoint_for_backward(example)
        sb.write(f'{context}{target}{self._example_separator}')

      return sb.getvalue()

  @final
  def _construct_backward_prompt(
      self, datapoint: _TInput, forward_sampled_target: str
  ) -> rtcd.PromptAndTarget:
    """Creates the prompt and expected target for the backward pass.

    Args:
      datapoint: The datapoint used to create the RTC data.
      forward_sampled_target: The output sample of the LLM in the forward mode.

    Returns:
      The backward prompt, target (if known), and stopping token.
    """

    eod_idx = forward_sampled_target.find(self._example_separator)
    if eod_idx != -1:
      forward_sampled_target = forward_sampled_target[:eod_idx]

    context, target = self._format_datapoint_for_backward(
        datapoint, forward_sampled_target
    )
    return rtcd.PromptAndTarget(
        self._backward_prompt_prefix + context,
        target,
        stopping_tokens=self._stopping_tokens,
    )

  @final
  async def _sample_lm(
      self,
      prompt_factory: llm.PromptFactory,
      context: str,
      num_samples: int,
      stopping_tokens: str | Collection[str],
      max_out_len: int,
  ) -> list[rtcd.Sample]:
    """Samples an LLM."""
    prompt_outputs: list[llm.PromptOutput] = await prompt_factory().sample(
        prefix=context,
        stop_token=stopping_tokens,
        num_samples=num_samples,
        max_length=max_out_len,
    )

    if _LOG_PROMPTS_AND_RESPONSES.value:
      with open(_LOG_PROMPTS_AND_RESPONSES.value, 'a') as f:
        for prompt_output in prompt_outputs:
          f.write('\n\n###>>>##### Prompt #######<<<<####\n')
          f.write(context)
          f.write('\n\n###>>>##### Response #######<<<<####\n')
          f.write(prompt_output.text)

    return [rtcd.Sample(p.text, p.logp) for p in prompt_outputs]

  @final
  async def _generate_backwards_for(
      self, datapoint: _TInput, fw_sample: rtcd.Sample
  ) -> rtcd.GenerationSample:
    """Generate backwards samples for a single forward sample."""
    bw_prompt = self._construct_backward_prompt(datapoint, fw_sample.text)
    bw_samples = []
    for bw_sample in await self._sample_lm(
        self._bw_prompt_factory,
        bw_prompt.context,
        num_samples=self._n_backward_examples,
        stopping_tokens=bw_prompt.stopping_tokens,
        max_out_len=self._max_backward_generation_len,
    ):
      bw_sample: rtcd.Sample
      bw_samples.append(bw_sample)

    return rtcd.GenerationSample(
        forward_sample=fw_sample, backward_samples=bw_samples
    )

  @final
  async def _get_samples_for(
      self,
      datapoint: _TInput,
  ):
    """Queries an LLM to get RTC samples for scoring.

    The output samples are to be used for computing the RTC score.

    Args:
      datapoint: The input data point.

    Returns:
      The scoring samples for the input data point.
    """

    async with self._max_concurrent_examples:
      forward_prompt = self._format_forward_prompt(datapoint)
      forward_samples: list[rtcd.Sample] = await self._sample_lm(
          self._fw_prompt_factory,
          forward_prompt.context,
          num_samples=self._n_forward_examples,
          stopping_tokens=forward_prompt.stopping_tokens,
          max_out_len=self._max_forward_generation_len,
      )

      fw_and_bw_samples = await asyncio.gather(*(
          self._generate_backwards_for(datapoint, fw_sample)
          for fw_sample in forward_samples
      ))

      if self._baseline_forward_prediction:
        baseline_samples = (
            await self._generate_backwards_for(
                datapoint,
                rtcd.Sample(self._baseline_forward_prediction, 0.0),
            )
        ).backward_samples
      else:
        baseline_samples = None

      generation_samples_for_datapoint = rtcd.GenerationSamplesForDatapoint(
          datapoint=datapoint,
          generation_samples=fw_and_bw_samples,
          baseline_samples=baseline_samples,
      )

      return generation_samples_for_datapoint

  @final
  async def generate_rtc_samples(
      self, input_data: Iterator[_TInput]
  ) -> AsyncIterator[rtcd.GenerationSamplesForDatapoint[_TInput]]:
    """Samples concurrently an LLM for RTC samples.

    Args:
      input_data: The input data iterator.

    Yields:
      The RTC samples.
    """
    all_tasks = [self._get_samples_for(datapoint) for datapoint in input_data]
    for samples in asyncio.as_completed(all_tasks):
      yield await samples

  @classmethod
  @abc.abstractmethod
  def samples_to_html(
      cls,
      samples: rtcd.GenerationSamplesForDatapoint[_TInput],
  ) -> str:
    """Creates and HTML string visualzing the samples."""
    raise NotImplementedError
