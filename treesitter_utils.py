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

"""Utilities for the Tree-Sitter library."""

import immutabledict
import py_tree_sitter as ts

SUFFIX_TO_TS_LANGUAGE: immutabledict.immutabledict[str, ts.Language] = (
    immutabledict.immutabledict({
        "py": ts.Language("", "python"),
        "java": ts.Language("", "java"),
    })
)


def parse(filepath: str, language: ts.Language) -> ts.Tree:
  """Parse a file with TreeSitter.

  Args:
    filepath: The path of the file.
    language: The TreeSitter language object. To be retrieved from the constant
      above.

  Returns:
    The parsed TreeSitter Tree.
  """
  with open(filepath, "rb") as f:
    code_bytes = f.read()
  return parse_code(code_bytes, language)


def parse_code(code_bytes: bytes, language: ts.Language) -> ts.Tree:
  """Parse code with TreeSitter.

  Args:
    code_bytes: The code to parse.
    language: The TreeSitter language object. To be retrieved from the constant
      above.

  Returns:
    The parsed TreeSitter Tree.
  """
  parser = ts.Parser()
  parser.set_language(language)

  return parser.parse(code_bytes)
