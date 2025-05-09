# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Union

import torch
from megatron.core.transformer.transformer_config import TransformerConfig

from nemo.collections.llm.gpt.model.gemma3 import Gemma3Config, Gemma3Config4B, Gemma3Config12B, Gemma3Config27B
from nemo.collections.vlm.gemma3vl.model.base import Gemma3VLConfig, Gemma3VLModel
from nemo.collections.vlm.gemma3vl.model.vision import Gemma3VLVisionConfig, Gemma3VLMultimodalProjectorConfig
from nemo.lightning import io, teardown

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer


@dataclass
class Gemma3VLConfig4B(Gemma3VLConfig):
    """Gemma3 VL config 4B"""

    language_transformer_config: Gemma3Config = field(default_factory=lambda: Gemma3Config4B())
    vision_transformer_config: Gemma3VLVisionConfig = field(default_factory=lambda: Gemma3VLVisionConfig())
    vision_projection_config: Gemma3VLMultimodalProjectorConfig = field(
        default_factory=lambda: Gemma3VLMultimodalProjectorConfig(input_size=1152, hidden_size=2560)
    )


@dataclass
class Gemma3VLConfig12B(Gemma3VLConfig):
    """Gemma3 VL config 12B"""

    language_transformer_config: Gemma3Config = field(default_factory=lambda: Gemma3Config12B())
    vision_transformer_config: Gemma3VLVisionConfig = field(default_factory=lambda: Gemma3VLVisionConfig())
    vision_projection_config: Gemma3VLMultimodalProjectorConfig = field(
        default_factory=lambda: Gemma3VLMultimodalProjectorConfig(input_size=1152, hidden_size=3840)
    )


@dataclass
class Gemma3VLConfig27B(Gemma3VLConfig):
    """Gemma3 VL config 27B"""

    language_transformer_config: Gemma3Config = field(default_factory=lambda: Gemma3Config27B())
    vision_transformer_config: Gemma3VLVisionConfig = field(default_factory=lambda: Gemma3VLVisionConfig())
    vision_projection_config: Gemma3VLMultimodalProjectorConfig = field(
        default_factory=lambda: Gemma3VLMultimodalProjectorConfig(input_size=1152, hidden_size=5376)
    )


@io.model_importer(Gemma3VLModel, "hf")
class Gemma3VLImporter(io.ModelConnector["Gemma3ForConditionalGeneration", Gemma3VLModel]):
    """Gemma3 VL model HF importer"""

    def init(self) -> Gemma3VLModel:
        # pylint: disable=C0115,C0116
        return Gemma3VLModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        # pylint: disable=C0115,C0116
        from transformers import Gemma3ForConditionalGeneration

        source = Gemma3ForConditionalGeneration.from_pretrained(str(self))
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted HF Gemma3VL model to NeMo, saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        # TODO:
        return

    @property
    def tokenizer(self) -> "AutoTokenizer":
        # pylint: disable=C0115,C0116
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(str(self))

    @property
    def config(self) -> Gemma3VLConfig:
        # pylint: disable=C0115,C0116
        from transformers import Gemma3Config as HFGemma3Config

        name = str(self)
        source = HFGemma3Config.from_pretrained(name)
        source_text = source.text_config
        source_vision = source.vision_config

        if source_text.num_hidden_layers == 34:
            language_transformer_config = Gemma3Config4B()
        elif source_text.num_hidden_layers == 48:
            language_transformer_config = Gemma3Config12B()
        elif source_text.num_hidden_layers == 62:
            language_transformer_config = Gemma3Config27B()
        else:
            raise ValueError(f"Unrecognized import model: {name}")
        vision_transformer_config = Gemma3VLVisionConfig()
        vision_projection_config = Gemma3VLMultimodalProjectorConfig(
            input_size=source_vision.hidden_size,
            hidden_size=source_text.hidden_size,
        )

        output = Gemma3VLConfig(
            language_transformer_config=language_transformer_config,
            vision_transformer_config=vision_transformer_config,
            vision_projection_config=vision_projection_config,
        )
        return output
