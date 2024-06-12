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

"""Loader for running RTC Evaluation."""

from collections.abc import Iterator
import contextlib
import gzip
import hashlib
import os
from typing import AsyncIterator, Type, TypeVar

from absl import logging
import dataclasses_json
import gin
import tqdm.asyncio as tqdm

from roundtrip_correctness import rtc_data as rtcd
from roundtrip_correctness import rtc_task as rtct


_T = TypeVar("_T")


async def _to_tuple_iter(
    data: AsyncIterator[_T], as_first_element: bool
) -> AsyncIterator[tuple[_T, None] | tuple[None, _T]]:
  async for s in data:
    if as_first_element:
      yield (s, None)
    else:
      yield (None, s)


def _compute_hash(json_str) -> str:
  return hashlib.sha256(json_str.encode()).hexdigest()


@gin.configurable
async def run(
    task: rtct.RoundTripCorrectnessTask = gin.REQUIRED,
    input_data_path: str = gin.REQUIRED,
    output_data_path: str = gin.REQUIRED,
    html_output_file: str | None = gin.REQUIRED,
) -> None:
  """Runs roundtrip-correctness sampling and generation.

  Args:
    task: The concrete RTC task.
    input_data_path: The location of the input data.
    output_data_path: The target output location for the samples.
    html_output_file: An optional location where an HTML visualization will be
      output, only is generation mode.

  Raises:
    ValueError: when html_output_file is set in scoring mode.
  """
  input_data_class: Type[dataclasses_json.DataClassJsonMixin] = task.input_type
  generation_output_filepath = output_data_path + "-generation.jsonl.gz"

  seen_samples: set[str] = set()

  def _add_datapoint(d) -> None:
    seen_samples.add(_compute_hash(input_data_class.from_dict(d).to_json()))

  if os.path.exists(generation_output_filepath):
    with gzip.open(open(generation_output_filepath, "rb")) as f:
      for line in f:
        _add_datapoint(
            rtcd.GenerationSamplesForDatapoint.from_json(
                line.decode()
            ).datapoint
        )
    logging.info(
        "Resuming: Found %d existing samples in %s",
        len(seen_samples),
        generation_output_filepath,
    )

  def _dataset_iter() -> Iterator[dataclasses_json.DataClassJsonMixin]:
    with gzip.open(open(input_data_path, "rb")) as f:
      for line in f:
        datapoint = input_data_class.from_json(line.decode())
        if _compute_hash(datapoint.to_json()) not in seen_samples:
          yield datapoint

  if html_output_file:
    html_ctx_manager = open(html_output_file, "a")
  else:
    html_ctx_manager = contextlib.nullcontext()

  with (
      gzip.open(open(generation_output_filepath, "ab"), "a") as f_gen,
      html_ctx_manager as f_html,
  ):
    async for generation_sample in tqdm.tqdm(
        task.generate_rtc_samples(_dataset_iter()),
        initial=len(seen_samples),
    ):

      if f_gen:
        f_gen.write(generation_sample.to_json().encode())
        f_gen.write(b"\n")
        f_gen.flush()
        if f_html:
          f_html.write(task.samples_to_html(generation_sample))
          f_html.write("\n")
          f_html.flush()
