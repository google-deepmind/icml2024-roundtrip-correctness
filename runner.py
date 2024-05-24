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

"""A CLI to invoke Roundtrip Correctness Sampling."""

import asyncio
from collections.abc import Sequence

from absl import app
from absl import flags
import gin

from roundtrip_correctness import rtc

_GIN_FILE = flags.DEFINE_multi_string(
    'gin_file', None, 'List of paths to the config files.'
)

_GIN_PARAM = flags.DEFINE_multi_string(
    'gin_param',
    [],
    'Gin parameter bindings. Use the flag for each parameter you want to bind.',
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  gin.parse_config_files_and_bindings(
      _GIN_FILE.value, _GIN_PARAM.value, print_includes_and_imports=True
  )
  asyncio.run(rtc.run())


if __name__ == '__main__':
  app.run(main)
