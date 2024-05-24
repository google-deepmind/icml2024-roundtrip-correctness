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

"""The data structures for RTC."""

from collections.abc import Collection
import dataclasses
from typing import Generic, TypeVar

import dataclasses_json

_TInput = TypeVar('_TInput')


@dataclasses.dataclass(frozen=True)
class PromptAndTarget:
  context: str
  target: str | None
  stopping_tokens: str | Collection[str]


@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True)
class Sample:
  """A sample from an LM."""

  text: str
  logprob: float

  def __post_init__(self):
    if self.logprob > 0:
      raise ValueError('Log-probabilities must be non-positive.')


@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True)
class GenerationSample:
  """An RTC sample."""

  forward_sample: Sample
  backward_samples: list[Sample]


@dataclasses_json.dataclass_json
@dataclasses.dataclass()
class GenerationSamplesForDatapoint(Generic[_TInput]):
  """RTC samples for a datapoint."""

  datapoint: _TInput
  generation_samples: list[GenerationSample]
  baseline_samples: list[Sample] | None


@dataclasses_json.dataclass_json
@dataclasses.dataclass()
class EvaluatedGenerationSamplesForDatapoint(Generic[_TInput]):
  """Generation samples for a datapoint with per-backward sample metrics."""

  samples: GenerationSamplesForDatapoint[_TInput]

  # Dictionaries from metric name -> per-sample-metric (e.g. pass unit tests,
  # BLEU, exact match)
  generation_samples_consistencies: dict[str, list[list[float]]]
  baseline_samples_consistencies: dict[str, list[float]] | None

  def validate_sizes(self):
    """Shallow Validation of the data struct."""
    for metric_for_samples in self.generation_samples_consistencies.values():
      for evals, gen_sample in zip(
          metric_for_samples, self.samples.generation_samples, strict=True
      ):
        if len(evals) != len(gen_sample.backward_samples):
          raise ValueError(
              'The number of samples in the generation sample eval must match'
              ' the number of metrics computed.'
          )

    if self.baseline_samples_consistencies:
      missing_metrics = set(self.generation_samples_consistencies) ^ set(
          self.baseline_samples_consistencies
      )
      if missing_metrics:
        raise ValueError(
            'The following metrics are missing from either the baseline or'
            f' generation sample evals: `{missing_metrics}`.'
        )

      if not all(
          len(s) == len(self.samples.baseline_samples)
          for s in self.baseline_samples_consistencies.values()
      ):
        raise ValueError(
            'The number of samples in the baseline samples eval must match the'
            ' number of metrics computed.'
        )

  def _compute_baseline_scores(
      self, metric_name: str, computed_metrics: dict[str, float]
  ) -> None:
    """Computes the baseline-based consistency scores."""
    baseline_eval_for_metric = self.baseline_samples_consistencies[metric_name]

    avg_baseline_metric = sum(baseline_eval_for_metric) / len(
        baseline_eval_for_metric
    )
    computed_metrics[f'{metric_name}-baseline-avg'] = avg_baseline_metric
    computed_metrics[f'{metric_name}-lift'] = (
        computed_metrics[f'rtc-avg-{metric_name}'] - avg_baseline_metric
    )

  def _compute_rtc_scores(
      self, metric_name: str, computed_metrics: dict[str, float]
  ) -> None:
    """Computes the RTC scores."""
    generation_scores_for_metric = self.generation_samples_consistencies[
        metric_name
    ]
    rtc_avg, rtc_pass_at_k = 0.0, 0.0

    for _, sample_eval in zip(
        self.samples.generation_samples,
        generation_scores_for_metric,
        strict=True,
    ):
      rtc_avg += sum(sample_eval) / len(sample_eval)
      rtc_pass_at_k += 1 if sum(sample_eval) > 0 else 0

    computed_metrics[f'rtc-avg-{metric_name}'] = rtc_avg / len(
        self.samples.generation_samples
    )

    computed_metrics[f'rtc-{metric_name}-at-k'] = 1 if rtc_pass_at_k > 0 else 0
    if self.baseline_samples_consistencies:
      self._compute_baseline_scores(metric_name, computed_metrics)

  def compute_scores(self) -> dict[str, float]:
    """Computes the RTC metrics."""
    computed_metrics = {}
    for metric_name in self.generation_samples_consistencies:
      self._compute_rtc_scores(metric_name, computed_metrics)

    return computed_metrics
