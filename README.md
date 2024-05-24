# Round-trip Correctness

This is the code that accompanies ["Unsupervised Evaluation of Code LLMs with Round-Trip Correctness"](https://arxiv.org/abs/2402.08699). ICML 2024.

## Installation

To run this repository please clone this repository and install the dependencies
in `requirements.txt`.

## Usage

### Computing SynthesisRTC on HumanEval

First, generate the input examples

```bash
python roundtrip_correctness/synthesis_rtc/humaneval_to_rtc_example.py \
      --out_path /path/to/humaneval-inputs.jsonl.gz
```

Then, run the RTC sampling loop.

```bash
python roundtrip_correctness/runner.py \
      --gin_file path/to/config.gin \
      --gin_param="OUTPUT_PATH='/path/to/humaneval-rtc-samples'" \
      --gin_param="HTML_PATH='/path/to/he-samples-visualization.html'" \
      --gin_param="MODEL_PATH='http://localhost:8001/v1/completions@/models/bigcode-starcoder2-15b'" \
      --gin_param="INPUT_PATH='/path/to/humaneval-inputs.jsonl.gz'"
```
This assumes a [vLLM](https://docs.vllm.ai/en/latest/) server serving at `http://localhost:8001/v1/completions`
an LLM with the path of `/models/bigcode-starcoder2-15b`.
The parameters of RTC sampling can be directly tweaked in the Gin configuration file.
A sample configuration (with the SynthesisRTC defaults) is found at
[`sample_config.gin`](./sample_config.gin).


Next compute the semantic equivalence of the backward samples by using unit
test as a similarity function.

```bash
python roundtrip_correctness/synthesis_rtc/eval_humaneval.py \
      --samples_path /path/to/humaneval-rtc-samples-generation.jsonl.gz \
      --output_file /path/to/evaluated-samples.jsonl.gz
```
Note that the `--samples_path` is the path previously using in the `OUTPUT_PATH`
gin parameter.

Finally, compute the summary RTC statistics

```bash
python roundtrip_correctness/summarize_results.py --input_data /path/to/evaluated-samples.jsonl.gz
```

### Computing SynthesisRTC on other code
Previously, we showed an example of how to compute RTC for HumanEval. Next,
we focus on the more general process for computing SynthesisRTC in arbitrary
code.

First, extract the input examples by pointing to a folder `/path/to/input/code/`
containing the relevant code files (e.g., the root folder of a git repository).

```
python roundtrip_correctness/synthesis_rtc/example_gen_cli.py \
      --input_folder "/path/to/input/code/" \
      --out_path /path/to/output/program-samples.jsonl.gz
```
The `--context_size`, `--min_span_byte_length`, and `--max_span_byte_length`
allow to further parameterize the example generation. Currently Python and Java
are supported; introducing new languages requires using a
[TreeSitter](https://tree-sitter.github.io/tree-sitter/) grammar and defining
the syntactic constructs that are allowed. See
[`synthesis_rtc/example_gen.py`](synthesis_rtc/example_gen.py) as a starting
point.

> [!IMPORTANT]
> The examples are extracted via syntactic analysis of the code.
> It is recommended that the samples are filtered by removing those that are
> not adequately captured by unit tests, as discussed in the paper.

Next, create the LLM samples. Assuming a vLLM server
at `http://localhost:8001/v1/completions` with a model path of `/models/bigcode-starcoder2-15b`, run

```bash
python roundtrip_correctness/runner.py \
      --gin_file path/to/config.gin \
      --gin_param="OUTPUT_PATH='/path/to/mycode-rtc-samples.jsonl.gz'" \
      --gin_param="HTML_PATH='/path/to/samples-visualization.html'" \
      --gin_param="MODEL_PATH='http://localhost:8001/v1/completions@/models/bigcode-starcoder2-15b'" \
      --gin_param="INPUT_PATH='/path/to/output/program-samples.jsonl.gz'"
```
where `INPUT_PATH` is the file generated in the previous step and `OUTPUT_PATH`
is the location where the samples will be stored.

Next, compute the correctness of the roundtrip. This step cannot
be generalized to arbitrary code and the code provided will need to be adapted
for different languages and similarity functions.
Here we include an example that assumes
a Docker installation along with an image `rtc-image` that contains all the relevant dependencies for
testing a Python program with pytest and generating a JSON report.
For example, the `Dockerfile` of `rtc-image` may look like

```Dockerfile
FROM python:3.11
RUN pip install absl-py pytest pytest-json-report
```

We also require a `code.tar.gz` with all the code in `/path/to/input/code/`
that was previously used to create the evaluation examples in `/path/to/output/program-samples.jsonl.gz`
with the RTC samples generated in the previous step at `/path/to/mycode-rtc-samples.jsonl.gz`.
Next, run the evalution of the roundtrip samples

```bash
python synthesis_rtc/eval_for_program.py \
      --input_file /path/to/mycode-rtc-samples.jsonl.gz \
      --target_program_tar /path/to/code.tar.gz \
      --image_name "rtc-image" \
      --output_file /path/to/evaluated-samples.jsonl.gz
```

Finally, obtain the RTC statistics

```bash
python roundtrip_correctness/summarize_results.py --input_data /path/to/evaluated-samples.jsonl.gz
```


### RTC data structures
The core RTC data structures are defined in [`rtc_data.py`](./rtc_data.py).

* `GenerationSamplesForDatapoint` contains a set of LLM generation samples for
      a single datapoint.
* `EvaluatedGenerationSamplesForDatapoint` contains the generated samples
      evaluated for semantic similarity ($sim(\cdot)$ in the paper).

## Citing this work

Please use

```latex
@inproceedings{allamanis2024unsupervised,
      title={Unsupervised Evaluation of Code {LLM}s with Round-Trip Correctness},
      author={Allamanis, Miltiadis and Panthaplackel, Sheena and Yin, Pengcheng},
      booktitle={International Conference on Machine Learning},
      year={2024},
}
```

## License and disclaimer

Copyright 2024 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
