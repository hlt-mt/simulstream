# Copyright 2025 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import logging
import torch
import numpy as np

from types import SimpleNamespace
from typing import List

from simulstream.metrics.detokenizers import get_detokenizer
from simulstream.server.speech_processors import IncrementalOutput, SpeechProcessor, class_load
from simulstream.server.speech_processors import SAMPLE_RATE


logger = logging.getLogger(__name__)


try:
    from simuleval.agents.agent import SEGMENT_TYPE_DICT
    from simuleval.agents.actions import Action
    from simuleval.data.segments import SpeechSegment
except Exception as e:
    logger.error(
        "Not able to import SimulEval. Please install SimulEval or add it to your PYTHONPATH.")
    # In case we are running unit tests, avoid failures when importing the Wrapper
    import os
    is_testing = os.getenv("IS_TESTING")
    if not (is_testing is not None and bool(is_testing)):
        raise e
    import builtins
    # Temporarily inject types to satisfy simuleval_wrapper imports
    # These need to remain in builtins for type annotations in the class definition
    builtins.Action = type("Action", (), {})
    builtins.SpeechSegment = type("SpeechSegment", (), {})


class SimulEvalWrapper(SpeechProcessor):
    """
    Wrapper processor that calls the configured `simuleval_agent` implemented on SimulEval>=1.1.0.
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        agent_class_name = getattr(config, "simuleval_agent")
        agent_class = class_load(agent_class_name)
        config.source_segment_size = config.speech_chunk_size * 1000
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.simuleval_agent = agent_class(config)
        self.latency_unit = config.latency_unit
        self.segment_type = self._segment_type_class()
        self.emission_started = False
        self.detokenizer = get_detokenizer(config)

    def _segment_type_class(self):
        return SEGMENT_TYPE_DICT[self.simuleval_agent.target_type]

    @classmethod
    def load_model(cls, config: SimpleNamespace):
        """
        In SimulEval, the model is loaded in the init of the `simuleval_agent` and,
        therefore, a copy of the model for each client is created.
        """
        pass

    def set_target_language(self, language: str) -> None:
        if hasattr(self.simuleval_agent, "tgt_lang"):
            self.simuleval_agent.tgt_lang = language
        else:
            logger.warning("Unable to set the target language for SimulEval agent.")

    def set_source_language(self, language: str) -> None:
        pass

    def tokens_to_string(self, tokens: List[str]) -> str:
        return self.detokenizer(tokens)

    def _process_action(self, action: Action) -> str:
        """
        Processes a SimulEval action and updates the target state accordingly.

        If the action is of type READ, no output is produced and an empty prediction string is
        returned. Otherwise (WRITE action), the method extracts the generated content from the
        action, wraps it in the appropriate target segment type, and updates the SimulEval agent's
        target state with this new segment.

        Args:
            action (Action): The current SimulEval action to process. It can either
                request reading more input (READ) or writing an output (WRITE).

        Returns:
            str: The predicted output text if the action is WRITE, or an empty string
                if the action is READ.
        """
        if action.is_read():
            return ""

        prediction = action.content
        segment = self.segment_type(index=0, content=prediction, finished=action.finished)
        self.simuleval_agent.states.update_target(segment)
        return prediction

    def _build_incremental_outputs(self, generated_text: str) -> IncrementalOutput:
        """
        Transform the prediction string from `Action.content` of SimulEval into the required
        IncrementalOutput format. The token conversion follows the original SimulEval Instance
        handling (https://github.com/facebookresearch/SimulEval/blob/
        536de8253b82d805c9845440169a5010ff507357/simuleval/evaluator/instance.py#L209) with the
        sole exception of not removing generated spaces in the character-level languages.
        Since SimulEval only supports incremental outputs, no tokens are deleted.
        """
        if self.latency_unit in ["word", "spm"]:
            generated_tokens = generated_text.strip().split()
        elif self.latency_unit == "char":
            generated_tokens = list(generated_text.strip())
        else:
            raise NotImplementedError

        if self.emission_started and self.latency_unit == "word":
            generated_text = " " + generated_text

        self.emission_started = True

        return IncrementalOutput(
            new_tokens=generated_tokens,
            new_string=generated_text,
            deleted_tokens=[],
            deleted_string="",
        )

    def process_chunk(self, waveform: np.float32) -> IncrementalOutput:
        source_segment = SpeechSegment(
            index=0,
            content=waveform.tolist(),
            sample_rate=SAMPLE_RATE,
            finished=False,
            tgt_lang=self.simuleval_agent.tgt_lang,
        )
        self.simuleval_agent.states.update_source(source_segment)
        action = self.simuleval_agent.policy(self.simuleval_agent.states)
        prediction = self._process_action(action)
        new_output = self._build_incremental_outputs(prediction)
        return new_output

    def end_of_stream(self) -> IncrementalOutput:
        self.simuleval_agent.states.source_finished = True
        action = self.simuleval_agent.policy(self.simuleval_agent.states)
        prediction = self._process_action(action)
        new_output = self._build_incremental_outputs(prediction)
        return new_output

    def clear(self) -> None:
        """ In SimulEval, the agent is reset inside `simuleval_agent` """
        self.simuleval_agent.reset()
        self.emission_started = False
