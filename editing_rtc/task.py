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

"""Definition of the Code Edit <-> NL Description RTC task."""

from __future__ import annotations

import dataclasses
import io
from typing import Collection

import dataclasses_json
import gin

from roundtrip_correctness import llm_interface as llm
from roundtrip_correctness import rtc_data as rtcd
from roundtrip_correctness import rtc_task as rtct


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class EditingRtcExample:
  """A code edit <-> natural language description RTC example."""

  filename: str
  code_before_edit: str
  code_after_edit: str
  ground_truth_edit_description: str | None = None

  def __post_init__(self):
    """Validates the data and computes dependent fields."""
    if self.code_before_edit == self.code_after_edit:
      raise ValueError(
          '`code_before_edit` and `code_after_edit` should not be the same.'
      )


@gin.configurable
class EditingRtc(rtct.RoundTripCorrecntessTask[EditingRtcExample]):
  """A code edit <-> natural language description RTC task."""

  def __init__(
      self,
      forward_prompt_factory: llm.PromptFactory,
      backward_prompt_factory: llm.PromptFactory,
      forward_prompt_instruction: str,
      backward_prompt_instruction: str,
      forward_few_shot_examples: Collection[EditingRtcExample],
      backward_few_shot_examples: Collection[EditingRtcExample],
      max_forward_generation_len: int = 128,
      max_backward_generation_len: int = 512,
      n_forward_samples: int = 3,
      n_backward_samples: int = 1,
      example_separator_token: str = '\n\n',
      stopping_tokens: Collection[str] = (),
      max_concurrent_examples: int = 25,
  ):
    super().__init__(
        forward_prompt_factory,
        backward_prompt_factory,
        forward_prompt_instruction,
        backward_prompt_instruction,
        forward_few_shot_examples,
        backward_few_shot_examples,
        n_forward_samples,
        n_backward_samples,
        max_forward_generation_len,
        max_backward_generation_len,
        example_separator_token,
        stopping_tokens,
        max_concurrent_examples,
    )

  def _format_datapoint_for_forward(
      self, datapoint: EditingRtcExample
  ) -> tuple[str, str]:
    return (
        (
            f'[old]\n{datapoint.code_before_edit}\n'
            f'[new]\n{datapoint.code_after_edit}\n[edit description]'
        ),
        datapoint.ground_truth_edit_description,
    )

  @property
  def _baseline_forward_prediction(self) -> str:
    return 'Edit.'

  def _format_datapoint_for_backward(
      self,
      datapoint: EditingRtcExample,
      backward_prompt_instruction: str | None = None,
  ) -> tuple[str, str]:
    if (
        backward_prompt_instruction is None
        and datapoint.ground_truth_edit_description is None
    ):
      raise ValueError(
          'Either `backward_prompt_instruction` or'
          ' `datapoint.ground_truth_edit_description` must be provided.'
      )

    edit_description = (
        backward_prompt_instruction or datapoint.ground_truth_edit_description
    )
    return (
        (
            f'[old]\n{datapoint.code_before_edit}\n[edit description] '
            f'{edit_description}\n[new]\n'
        ),
        datapoint.code_after_edit,
    )

  @classmethod
  def samples_to_html(
      cls,
      samples: rtcd.GenerationSamplesForDatapoint[EditingRtcExample],
  ) -> str:
    """See base class."""
    with io.StringIO() as sb:
      input_sample = samples.datapoint
      sb.write(f'<h2>{input_sample.filename}</h2>\n')
      sb.write('<pre>\n')
      sb.write('<span style="color:darkred; font-weight:bold;">')
      sb.write(input_sample.code_before_edit)
      sb.write('</span>\n')
      sb.write('<span style="color:darkgreen; font-weight:bold;">')
      sb.write(input_sample.code_after_edit)
      sb.write('</span>')
      sb.write('</pre>\n<h4>Forward Samples</h4>\n')
      sb.write('<ol>')
      for sample in samples.generation_samples:
        sb.write(
            f'<li>{sample.forward_sample.text} '
            f'({sample.forward_sample.logprob:.2f})</li>\n'
        )
      sb.write('</ol>\n')

      sb.write('<h4>Backwards Samples</h4>\n<table>')
      for i, sample in enumerate(samples.generation_samples, start=1):
        sb.write(f'<tr><td>{i}</td>\n')
        for bw_sample in sample.backward_samples:
          sb.write(
              f'<td><pre> {bw_sample.text}</pre>'
              f' <br/>({bw_sample.logprob:.2f})</td>'
          )
        sb.write('</tr>\n')

      sb.write('</table>\n')
      return sb.getvalue()
