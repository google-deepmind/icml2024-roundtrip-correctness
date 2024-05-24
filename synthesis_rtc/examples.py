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

"""Hand-written examples to be used for few shot prompting."""

from roundtrip_correctness.synthesis_rtc import task as synth_rtc

# pylint: disable=g-long-lambda
# Constants as lambdas to be used in Gin later.
FWD_TASK_DESCRIPTION = lambda: (
    "Describe briefly and concisely with imperative natural language the code"
    " region defined by <<region start>> and <<region end>> in the code"
    " excerpts below.\n"
)

BWD_TASK_DESCRIPTION = lambda: (
    "Synthesize the snippet of code that resolves the TODO(LLM) comment.\n"
)

EXAMPLE1 = synth_rtc.SynthesisRtcExample(
    filename="",
    start_point=(0, 0),
    end_point=(0, 0),
    line_comment_prefix="#",
    code_before_hole="""\
...
PARSER = argparse.ArgumentParser(
    description="Helper for check file hashes in Makefile instead of bare timestamps"
)
PARSER.add_argument("dst", metavar="DST", type=pathlib.Path)
PARSER.add_argument("-d", "--debug", action="store_true", default=False)


def main(argv):
    args = PARSER.parse_args(argv)
    dst = args.dst
    assert dst.suffix == ".hash"
    dirname = dst.parent
    if dirname.name != ".hash":
        if args.debug:
            print(f"Invalid name {dst} -> dirname {dirname}", file=sys.stderr)
        return 0
    dirname.mkdir(exist_ok=True)
    src_dir = dirname.parent
    src_name = dst.stem  # drop .hash
    full_src = src_dir / src_name
    hasher = hashlib.sha256()
""",
    code_in_hole="""\
    try:
        hasher.update(full_src.read_bytes())
    except OSError:
        if args.debug:
            print(f"Cannot open {full_src}", file=sys.stderr)
        return 0
    src_hash = hasher.hexdigest()
""",
    code_after_hole="""\
    if dst.exists():
        dst_hash = dst.read_text()
    else:
        dst_hash = ""
    if src_hash != dst_hash:
        dst.write_text(src_hash)
        print(f"re-hash {src_hash}")
    else:
        if args.debug:
            print(f"Skip {src_hash} checksum, up-to-date")
    return 0
""",
    ground_truth_description=(
        "Compute hash digest of file content in `src_hash`, if an error"
        " happens, exit with error code 0."
    ),
)

EXAMPLE2 = synth_rtc.SynthesisRtcExample(
    filename="",
    start_point=(0, 0),
    end_point=(0, 0),
    line_comment_prefix="#",
    code_before_hole="""\
...
def get_language_keywords(language: str) -> FrozenSet[str]:
    \"\"\"
    Returns the keywords of a programming language.
    There are some inconsistencies across languages wrt to
    what is considered a keyword. For example, the true/false
    literals are considered keywords in many languages. However,
    we exclude them here for consistency. We also exclude special
    functions-like keywords, such as `die()` in PHP.
    \"\"\"
    language = language.lower()
    if language == 'python':
        return frozenset(k for k in keyword.kwlist if k != 'True' and k != 'False')
""",
    code_in_hole="""\
    elif language in _LANGUAGE_TO_FILENAME:
        name = _LANGUAGE_TO_FILENAME[language]
        with open(os.path.join(os.path.dirname(__file__), name)) as f:
            return frozenset(l.strip() for l in f if len(l.strip()) > 0)
""",
    code_after_hole="""\
    else:
        raise Exception('Language keywords `%s` not supported yet. Consider contributing it to dpu-utils.' % language)
...
""",
    ground_truth_description=(
        "If `language` appears in the _LANGUAGE_TO_FILENAME dictionary, get the"
        " filename and return a frozen set containing each non-empty line in"
        " that file, without any leading or preceding whitespace."
    ),
)

EXAMPLE3 = synth_rtc.SynthesisRtcExample(
    filename="",
    start_point=(0, 0),
    end_point=(0, 0),
    line_comment_prefix="#",
    code_before_hole="""\
...
def get_commit(oid):
    parents = []

    commit = data.get_object(oid, 'commit').decode()
    lines = iter(commit.splitlines())
    for line in itertools.takewhile(operator.truth, lines):
""",
    code_in_hole="""\
        key, value = line.split(' ', 1)
""",
    code_after_hole="""\
        if key == 'tree':
            tree = value
        elif key == 'parent':
            parents.append(value)
        else:
            assert False, f'Unknown field {key}'

    message = '\n'.join(lines)
    return Commit(tree=tree, parents=parents, message=message)
""",
    ground_truth_description=(
        "Split the line on the first space into two chunks: the key and the"
        " value."
    ),
)

EXAMPLES = lambda: (EXAMPLE1, EXAMPLE2, EXAMPLE3)
