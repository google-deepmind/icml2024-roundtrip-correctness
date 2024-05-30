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

"""CodeReviewer validation examples to be used for few shot prompting."""

from roundtrip_correctness.editing_rtc import task as editing_rtc

# pylint: disable=g-long-lambda
# Constants as lambdas to be used in Gin later.

FWD_TASK_DESCRIPTION = lambda: (
    "Describe concisely and accurately with natural language the"
    " edit shown in the diff below.\n"
    "The diff displays an added line with + and a deleted line with -."
)

BWD_TASK_DESCRIPTION = lambda: (
    "Edit the code shown to achieve the provided edit description.\n"
    "Display a diff with + for an added line and - for a deleted line.\n"
)

FWD_EDITS_AS_NEW_CODE_TASK_DESCRIPTION = lambda: (
    "Describe concisely and accurately with natural language the"
    " differences between the old and new code shown below."
)

BWD_EDITS_AS_NEW_CODE_TASK_DESCRIPTION = lambda: (
    "Write new code which applies the change described in the edit description"
    " to the old code.\n"
)


EXAMPLE1 = editing_rtc.EditingRtcExample(
    filename="code_reviewer_refinement_valid_242",
    code_before_edit="""\
      c.DevAuthMethodId = c.flagDevAuthMethodId
    }
    if c.flagDevUsername != "" {
      if len(c.flagDevUsername) < 5 {
        c.UI.Error("Invalid dev username, must be longer than 5 characters")
        return 1
      }
      c.DevUsername = c.flagDevUsername
    }
    if c.flagDevPassword != "" {
      if len(c.flagDevPassword) < 7 {
        c.UI.Error("Invalid dev username, must be longer than 7 characters")
        return 1
      }
      c.DevPassword = c.flagDevPassword
    }
""",
    code_after_edit="""\
      c.DevAuthMethodId = c.flagDevAuthMethodId
    }
    if c.flagDevUsername != "" {
      c.DevUsername = c.flagDevUsername
    }
    if c.flagDevPassword != "" {
      c.DevPassword = c.flagDevPassword
    }
""",
    ground_truth_edit_description=(
        "Please remove the length check here and for the password below. It's"
        " dev mode, let them set it to whatever they want."
    ),
)


EXAMPLE2 = editing_rtc.EditingRtcExample(
    filename="code_reviewer_refinement_valid_2444",
    code_before_edit="""\
 }
 func (s *Syncer) applySvc(skey svcKey, sinfo k8sp.ServicePort, eps []k8sp.Endpoint,
  cleanupDerived func(uint32) error) (error, bool) {
  var (
    err        error
""",
    code_after_edit="""\
 }
 func (s *Syncer) applySvc(skey svcKey, sinfo k8sp.ServicePort, eps []k8sp.Endpoint,
  cleanupDerived func(uint32) error) (bool, error) {
  var (
    err        error
""",
    ground_truth_edit_description=(
        "Please make the error argument the last one; that's the go convention."
    ),
)

EXAMPLE3 = editing_rtc.EditingRtcExample(
    filename="code_reviewer_refinement_valid_1909",
    code_before_edit="""\
           A template URL is optional. If no URL is specified it will use the default template provided by CocoaPods.
         DESC
        self.arguments = '[NAME] [TEMPLATE_URL]'
         def initialize(argv)
           @name = argv.shift_argument
""",
    code_after_edit="""\
           A template URL is optional. If no URL is specified it will use the default template provided by CocoaPods.
         DESC
        self.arguments = 'NAME [TEMPLATE_URL]'
         def initialize(argv)
           @name = argv.shift_argument
""",
    ground_truth_edit_description=(
        "Name should not be indicated in square braces as it is a required"
        " parameter."
    ),
)

EXAMPLES = lambda: (EXAMPLE1, EXAMPLE2, EXAMPLE3)
