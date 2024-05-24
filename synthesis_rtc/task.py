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

"""Definition of the Code <-> NL Description MachineEval task."""

from __future__ import annotations

import dataclasses
import io
import textwrap
from typing import Collection

import dataclasses_json
import gin

from roundtrip_correctness import llm_interface as llm
from roundtrip_correctness import rtc_data as rtcd
from roundtrip_correctness import rtc_task as rtct


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class SynthesisRtcExample:
  """A code span <-> natural language description RTC example."""

  filename: str
  start_point: tuple[int, int]
  end_point: tuple[int, int]

  # The string of the code before the hole. This must end with a new line.
  code_before_hole: str

  # The string of the code after the hole.
  code_after_hole: str

  # The expected hole in the code. This must end with a new line.
  code_in_hole: str

  # Different programming languages need different comment prefixes.
  line_comment_prefix: str
  indentation: str = dataclasses.field(init=False)

  ground_truth_description: str | None = None

  def __post_init__(self):
    """Validates the data and computes dependent fields."""
    if self.code_before_hole and self.code_before_hole[-1] != '\n':
      raise ValueError('`code_before_hole` should end with a new line.')
    if not self.code_in_hole or not self.code_in_hole.strip():
      raise ValueError('`code_in_hole` should not be empty.')
    if self.code_in_hole[-1] != '\n':
      raise ValueError('`code_in_hole` should end with a new line.')

    indent_idx = 0
    while indent_idx < len(self.code_in_hole):
      if not self.code_in_hole[indent_idx].isspace():
        break
      indent_idx += 1
    self.indentation = self.code_in_hole[:indent_idx]

  def code_with_annotated_code_region(
      self, start_region_text: str, end_region_text: str
  ) -> str:
    return (
        f'{self.code_before_hole}'
        f'{self.indentation}{self.line_comment_prefix}{start_region_text}\n'
        f'{self.code_in_hole}'
        f'{self.indentation}{self.line_comment_prefix}{end_region_text}\n'
        f'{self.code_after_hole}'
    )

  def code_with_todo_at_hole(self, todo_comment: str) -> str:
    todo_comment = f'TODO(LLM): {todo_comment}'
    todo_comment = textwrap.indent(
        todo_comment, f'{self.indentation}{self.line_comment_prefix} '
    )
    return f'{self.code_before_hole}{todo_comment}\n{self.code_after_hole}'


@gin.configurable
class SynthesisRtc(rtct.RoundTripCorrecntessTask[SynthesisRtcExample]):
  """A code span <-> natural language description RTC task."""

  def __init__(
      self,
      forward_prompt_factory: llm.PromptFactory,
      backward_prompt_factory: llm.PromptFactory,
      forward_prompt_instruction: str,
      backward_prompt_instruction: str,
      forward_few_shot_examples: Collection[SynthesisRtcExample],
      backward_few_shot_examples: Collection[SynthesisRtcExample],
      max_forward_generation_len: int = 128,
      max_backward_generation_len: int = 512,
      n_forward_samples: int = 3,
      n_backward_samples: int = 3,
      region_start_text: str = '<<<region start>>>',
      region_end_text: str = '<<<region end>>>',
      example_separator_token: str = '\n\n',
      stopping_tokens: Collection[str] = (),
      max_concurrent_examples: int = 25,
  ):
    # These need to come first for the few-shot prompts to be rendered.
    self.region_start_text = region_start_text
    self.region_end_text = region_end_text
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
      self, datapoint: SynthesisRtcExample
  ) -> tuple[str, str]:
    return (
        datapoint.code_with_annotated_code_region(
            self.region_start_text, self.region_end_text
        )
        + '\nDescription:',
        datapoint.ground_truth_description,
    )

  @property
  def _baseline_forward_prediction(self) -> str:
    return 'Implement.'

  def _format_datapoint_for_backward(
      self,
      datapoint: SynthesisRtcExample,
      backward_prompt_instruction: str | None = None,
  ) -> tuple[str, str]:
    if (
        backward_prompt_instruction is None
        and datapoint.ground_truth_description is None
    ):
      raise ValueError(
          'Either `backward_prompt_instruction` or'
          ' `datapoint.ground_truth_description` must be provided.'
      )

    return (
        datapoint.code_with_todo_at_hole(
            backward_prompt_instruction or datapoint.ground_truth_description
        )
        + '\nCode:\n',
        textwrap.dedent(datapoint.code_in_hole),
    )

  @classmethod
  def samples_to_html(
      cls,
      samples: rtcd.GenerationSamplesForDatapoint[SynthesisRtcExample],
  ) -> str:
    """See base class."""
    with io.StringIO() as sb:
      input_sample = samples.datapoint
      sb.write(
          f'<h2>{input_sample.filename} '
          f'({input_sample.start_point}-{input_sample.end_point})</h2>\n'
      )
      sb.write('<pre>\n')
      sb.write(input_sample.code_before_hole)
      sb.write('<span style="color:darkred; font-weight:bold;">')
      sb.write(input_sample.code_in_hole)
      sb.write('</span>')
      sb.write(input_sample.code_after_hole)
      sb.write('</pre>\n<h4>Forward Samples</h4>\n')
      sb.write('<ol>')
      for generation_sample in samples.generation_samples:
        fw_sample = generation_sample.forward_sample
        sb.write(f'<li>{fw_sample.text} ({fw_sample.logprob:.2f})</li>\n')
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
