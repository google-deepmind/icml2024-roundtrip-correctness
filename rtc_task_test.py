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

import unittest
from unittest import mock
from absl.testing import absltest
from roundtrip_correctness import llm_interface as llm
from roundtrip_correctness import rtc_data as rtcd
from roundtrip_correctness import rtc_task


class _RTCTaskStub(rtc_task.RoundTripCorrectnessTask[str]):

  def _format_datapoint_for_forward(self, datapoint: str) -> tuple[str, str]:
    return (
        'fw_input:' + datapoint,
        'fw_target:' + datapoint,
    )

  @property
  def _baseline_forward_prediction(self) -> str:
    return 'mock_baseline'

  def _format_datapoint_for_backward(
      self,
      datapoint: str,
      backward_prompt_instruction: str | None = None,
  ) -> tuple[str, str]:
    return (
        'bw_input:' + datapoint + ':' + backward_prompt_instruction,
        'bw_target:' + datapoint,
    )

  @classmethod
  def samples_to_html(
      cls,
      samples: rtcd.GenerationSamplesForDatapoint[str],
  ) -> str:
    raise NotImplementedError


@mock.patch.multiple(llm.InteractivePromptInterface, __abstractmethods__=set())
def _create_llm_mock():
  fw_prompt = llm.InteractivePromptInterface()
  fw_prompt.sample = mock.AsyncMock(
      return_value=[
          llm.PromptOutput('FW_sample1', -1),
          llm.PromptOutput('FW_sample2', -2),
          llm.PromptOutput('FW_sample3', -3),
      ]
  )
  fw_prompt.query = mock.AsyncMock(
      return_value={'FW_sample1': -1, 'FW_sample2': -2, 'FW_sample3': -3}
  )

  bw_prompt = llm.InteractivePromptInterface()
  bw_prompt.sample = mock.AsyncMock(
      return_value=[
          llm.PromptOutput('BW_sample1', -1),
          llm.PromptOutput('BW_sample2', -2),
      ]
  )

  bw_prompt.query = mock.AsyncMock(
      return_value={'mock_baseline': -10, 'bw_target:D1': -1}
  )
  return fw_prompt, bw_prompt


class RTCTaskTest(absltest.TestCase, unittest.IsolatedAsyncioTestCase):

  async def test_generation(self):
    fw_prompt_mock, bw_prompt_mock = _create_llm_mock()

    task = _RTCTaskStub(
        lambda: fw_prompt_mock,
        lambda: bw_prompt_mock,
        n_forward_samples=3,
        n_backward_samples=2,
        max_forward_generation_len=100,
        max_backward_generation_len=101,
        forward_prompt_instruction='',
        backward_prompt_instruction='',
        forward_few_shot_examples=(),
        backward_few_shot_examples=(),
        example_separator_token='\n',
    )
    samples = [s async for s in task.generate_rtc_samples(['D1'])]
    fw_prompt_mock.sample.assert_called_once_with(
        prefix='fw_input:D1', stop_token={'\n'}, num_samples=3, max_length=100
    )

    fixed_args = dict(
        stop_token={'\n'},
        num_samples=2,
        max_length=101,
    )
    bw_prompt_mock.sample.assert_has_calls([
        mock.call(prefix=f'bw_input:D1:FW_sample{i}', **fixed_args)
        for i in range(1, 4)
    ])
    bw_prompt_mock.query.assert_not_called()

    self.assertLen(samples, 1)
    sample = samples[0]
    self.assertEqual(sample.datapoint, 'D1')
    self.assertLen(sample.generation_samples, 3)
    self.assertEqual(
        sample.generation_samples[0].forward_sample,
        rtcd.Sample('FW_sample1', logprob=-1),
    )
    self.assertEqual(
        sample.generation_samples[1].forward_sample,
        rtcd.Sample('FW_sample2', logprob=-2),
    )
    self.assertEqual(
        sample.generation_samples[2].forward_sample,
        rtcd.Sample('FW_sample3', logprob=-3),
    )
    for i in range(3):
      self.assertSequenceEqual(
          sample.generation_samples[i].backward_samples,
          [
              rtcd.Sample('BW_sample1', logprob=-1),
              rtcd.Sample('BW_sample2', logprob=-2),
          ],
      )


if __name__ == '__main__':
  absltest.main()
