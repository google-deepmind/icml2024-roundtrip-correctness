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

"""An OpenAI-compatible prompt (e.g., used with vLLM)."""

from __future__ import annotations

import asyncio
from collections.abc import Collection
import functools
import time
from typing import Callable, Optional

from absl import logging
import aiohttp
import gin

from roundtrip_correctness import llm_interface as llm


@gin.configurable(
    denylist=[
        'concurrent_connections_semaphores',
        'starting_state',
    ]
)
class OpenAICompatiblePrompt(llm.InteractivePromptInterface):
  """An HTTP-based LLM prompt."""

  def __init__(
      self,
      model_path: str,
      concurrent_connections_semaphores: asyncio.Semaphore,
      starting_state: Optional[str] = None,
      temperature: float = 0.8,
  ):
    super().__init__(starting_state)
    self._model_path = model_path
    self._url, self._model = model_path.split('@', maxsplit=1)
    self._temperature = temperature
    self._concurrent_connections_semaphores = concurrent_connections_semaphores
    logging.info('Using model %s at %s', self._model, self._url)

  async def _call_llm(
      self,
      prompt: str,
      max_length: Optional[int] = None,
      num_samples: int = 1,
  ) -> list[str]:
    async with aiohttp.ClientSession() as session:
      rq = {
          'max_tokens': max_length or 2048,
          'temperature': self._temperature,
          'model': self._model,
          'prompt': prompt,
          'n': num_samples,
      }
      async with session.post(self._url, json=rq) as response:
        response = await response.json()
        return [r['text'] for r in response['choices']][:num_samples]

  async def generate(
      self,
      *,
      prefix: str,
      stop_token: str | Collection[str],
      max_length: int = 64,
  ) -> llm.PromptOutput:
    """See base class."""
    self._state.write(prefix)
    async with self._concurrent_connections_semaphores:
      logging.info('Initiating request...')
      start_time = time.time()
      predictions = await self._call_llm(self._state.getvalue(), max_length)
      logging.info('Wait time to get response %s', time.time() - start_time)

    predicted_text = llm.truncate_to_stop_token(predictions[0], stop_token)

    self._state.write(' ')  # The call seems to consume or imply a space (?)
    self._state.write(predicted_text)
    return llm.PromptOutput(predicted_text, 0)

  async def sample(
      self,
      *,
      prefix: str,
      stop_token: str | Collection[str],
      num_samples: int = 1,
      max_length: int = 64,
  ) -> list[llm.PromptOutput]:
    """See base class."""
    if self._temperature == 0 and num_samples > 1:
      raise ValueError(
          'Sampling with temperature=0 does not make sense. Increase the'
          ' temperature.'
      )
    if num_samples <= 0:
      raise ValueError('num_samples must be > 0')

    num_samples_collected = 0
    samples = []
    async with self._concurrent_connections_semaphores:
      while True:
        logging.info('Initiating Sample() request...')
        start_time = time.time()
        generated = await self._call_llm(
            self._state.getvalue() + prefix,
            max_length,
            max(1, num_samples - len(samples)),
        )
        logging.info(
            'Wait time to get Sample() response %s', time.time() - start_time
        )

        num_samples_collected += len(generated)
        samples.extend(
            llm.PromptOutput(
                llm.truncate_to_stop_token(text, stop_token),
                0,
            )
            for text in generated
        )

        if num_samples_collected >= num_samples:
          break

    return samples

  def fork(self) -> OpenAICompatiblePrompt:
    """See base class."""
    return OpenAICompatiblePrompt(
        self._model_path,
        self._concurrent_connections_semaphores,
        self._state.getvalue(),
    )


@gin.configurable
class OpenAICompatiblePromptFactory:
  """A factory of OpenAI-compatible prompt factories, e.g., for vLLM.

  This class is to be used with dependency injection to provide a single
  point of defining the path. Prompt factories constructed through this
  class are guaranteed to not exceed the maximum number of concurrent requests
  to the model.
  """

  def __init__(
      self,
      model_path: str = gin.REQUIRED,
      max_concurrent_requests: int = gin.REQUIRED,
      temperature: float = 0.8,
  ):
    """Constructs the object.

    Args:
      model_path: The path to the model server.
      max_concurrent_requests: The maximum number of concurrent requests to make
        to the server.
      temperature: The sampling temperature to use.
    """
    self._connections_semaphore = asyncio.Semaphore(max_concurrent_requests)
    self._model_path = model_path
    self._temperature = temperature

  def get_prompt_factory(
      self, prompt_txt: str
  ) -> Callable[[], OpenAICompatiblePrompt]:
    """Returns a function yielding `OpenAICompatiblePrompt`s.

    Args:
      prompt_txt: The prompt prefix to be used.

    Returns: A callable that creates `OpenAICompatiblePrompt`s for the given
      server and with `prompt_txt` starting state.
    """
    return functools.partial(
        OpenAICompatiblePrompt,
        model_path=self._model_path,
        concurrent_connections_semaphores=self._connections_semaphore,
        starting_state=prompt_txt,
        temperature=self._temperature,
    )


@gin.configurable
def get_oai_compatible_prompt_factory(
    model_url: str = gin.REQUIRED,
    max_concurrent_requests: int = gin.REQUIRED,
    temperature: float = gin.REQUIRED,
) -> Callable[[], llm.InteractivePromptInterface]:
  """Returns the factory for RTC prompts."""
  return OpenAICompatiblePromptFactory(
      model_path=model_url,
      max_concurrent_requests=max_concurrent_requests,
      temperature=temperature,
  ).get_prompt_factory('')
