# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from nemo.collections.llm.gpt.model.base import GPTConfig, GPTModel
from nemo.lightning import io, teardown
from nemo.lightning.io.state import TransformFns
from nemo.lightning.pytorch.optim import OptimizerModule

if TYPE_CHECKING:
    from transformers import MixtralForCausalLM

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


@dataclass
class MixtralConfig(GPTConfig):
    """
    Base config for Mixtral models.
    """

    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    gated_linear_unit: bool = True

    num_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_query_groups: int = 8
    ffn_hidden_size: int = 14336
    max_position_embeddings: int = 4096
    seq_length: int = 4096
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = False

    # MoE
    num_moe_experts: int = 8
    moe_aux_loss_coeff: float = 0.01
    moe_router_topk: int = 2
    moe_router_pre_softmax: bool = True
    moe_token_dispatcher_type: str = "alltoall"
    moe_router_load_balancing_type: str = 'aux_loss'

    init_method_std: float = 0.02
    layernorm_epsilon: float = 1e-5
    # rotary
    rotary_percent: float = 1.0
    rotary_base: float = 1000000.0
    bf16: bool = True
    params_dtype: torch.dtype = torch.bfloat16

    # fusions
    apply_rope_fusion: bool = True
    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = True
    masked_softmax_fusion: bool = False


@dataclass
class MixtralConfig8x3B(MixtralConfig):
    """
    NeMo's Mixtral-8x3B model variant
    https://github.com/NVIDIA/NeMo-Framework-Launcher/blob/main/launcher_scripts/conf/training/mixtral/mixtral_8x3b.yaml
    """

    num_layers: int = 32
    hidden_size: int = 2560
    num_attention_heads: int = 32
    ffn_hidden_size: int = 8960
    max_position_embeddings: int = 4096
    seq_length: int = 4096


@dataclass
class MixtralConfig8x7B(MixtralConfig):
    """
    Config for Mixtral-8x7B model
    Official announcement: https://mistral.ai/news/mixtral-of-experts/
    """

    num_layers: int = 32
    hidden_size: int = 4096
    ffn_hidden_size: int = 14336
    max_position_embeddings: int = 4096
    seq_length: int = 4096


@dataclass
class MixtralConfig8x22B(MixtralConfig):
    """
    Config for Mixtral-8x22B model
    Official announcement: https://mistral.ai/news/mixtral-8x22b/
    """

    num_layers: int = 56
    hidden_size: int = 6144
    num_attention_heads: int = 48
    ffn_hidden_size: int = 16384
    max_position_embeddings: int = 4096
    seq_length: int = 4096


class MixtralModel(GPTModel):
    """Mcore-based MixtralModel"""

    def __init__(
        self,
        config: Optional[Union[MixtralConfig8x7B, MixtralConfig8x22B]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        """Mcore-based MixtralModel ctor"""
        super().__init__(
            config or MixtralConfig8x7B(), optim=optim, tokenizer=tokenizer, model_transform=model_transform
        )


@io.model_importer(MixtralModel, ext="hf")
class HFMixtralImporter(io.ModelConnector["MixtralForCausalLM", MixtralModel]):
    """HF to NeMo importer"""

    def init(self) -> MixtralModel:
        """init"""
        return MixtralModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        """Import model from HF"""
        from transformers import MixtralForCausalLM

        source = MixtralForCausalLM.from_pretrained(str(self), torch_dtype='auto', use_safetensors=True)
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        """State-dict converter"""
        mapping = {
            "model.layers.*.self_attn.o_proj.weight": "decoder.layers.*.self_attention.linear_proj.weight",
            "model.layers.*.input_layernorm.weight": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "model.layers.*.post_attention_layernorm.weight": "decoder.layers.*.pre_mlp_layernorm.weight",
            # MoE
            "model.layers.*.block_sparse_moe.experts.*.w2.weight": "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",  # pylint: disable=line-too-long
            "model.layers.*.block_sparse_moe.gate.weight": "decoder.layers.*.mlp.router.weight",
            # lm-head
            "model.norm.weight": "decoder.final_layernorm.weight",
        }

        transforms = [
            io.state_transform(
                source_key=(
                    "model.layers.*.self_attn.q_proj.weight",
                    "model.layers.*.self_attn.k_proj.weight",
                    "model.layers.*.self_attn.v_proj.weight",
                ),
                target_key="decoder.layers.*.self_attention.linear_qkv.weight",
                fn=TransformFns.merge_qkv,
            ),
            io.state_transform(
                source_key=(
                    "model.layers.*.block_sparse_moe.experts.*.w1.weight",
                    "model.layers.*.block_sparse_moe.experts.*.w3.weight",
                ),
                target_key="decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
                fn=TransformFns.merge_fc1,
            ),
            _import_embedding,
            _import_lm_head,
        ]
        return io.apply_transforms(source, target, mapping=mapping, transforms=transforms)

    @property
    def tokenizer(self) -> "AutoTokenizer":
        """Configures tokenizer"""
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(self.save_hf_tokenizer_assets(str(self)))

    @property
    def config(self) -> MixtralConfig8x7B | MixtralConfig8x22B:
        """Returns Mcore config from HF"""
        from transformers import GenerationConfig
        from transformers import MixtralConfig as HfMixtralConfig

        config = HfMixtralConfig.from_pretrained(str(self))
        generation_config = GenerationConfig.from_pretrained(str(self))
        config_cls = MixtralConfig8x7B
        if '8x22b' in str(self).lower():
            config_cls = MixtralConfig8x22B
        return config_cls(
            bf16=getattr(config, "torch_dtype", None) == torch.bfloat16,
            activation_func=F.silu,
            # network
            num_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            kv_channels=getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads),
            max_position_embeddings=config.max_position_embeddings,  # TODO
            seq_length=config.max_position_embeddings,
            # RoPE
            position_embedding_type='rope',
            rotary_base=config.rope_theta,
            # Transformer config
            num_attention_heads=config.num_attention_heads,
            num_query_groups=config.num_key_value_heads,
            num_moe_experts=config.num_local_experts,
            moe_router_topk=config.num_experts_per_tok,
            moe_router_pre_softmax=True,
            # norm
            normalization='RMSNorm',
            layernorm_epsilon=config.rms_norm_eps,
            # Init
            init_method_std=config.initializer_range,
            gated_linear_unit=True,
            # Vocab
            make_vocab_size_divisible_by=128,
            # CPU init
            use_cpu_initialization=True,
            perform_initialization=False,
            params_dtype=getattr(config, "torch_dtype", torch.bfloat16),
            generation_config=generation_config,
        )


@io.state_transform(
    source_key="model.embed_tokens.weight",
    target_key="embedding.word_embeddings.weight",
)
def _import_embedding(ctx: io.TransformCTX, embedding):
    """_import_embedding"""
    embedding_weight = ctx.source.model.embed_tokens.weight
    vocab_size = embedding_weight.shape[0]
    ctx.target_state['embedding.word_embeddings.weight'][:vocab_size, :].copy_(embedding_weight)
    return ctx.target_state['embedding.word_embeddings.weight']


@io.state_transform(
    source_key="lm_head.weight",
    target_key="output_layer.weight",
)
def _import_lm_head(ctx: io.TransformCTX, embedding):
    """import head"""
    lm_head_weight = ctx.source.lm_head.weight
    vocab_size = lm_head_weight.shape[0]
    ctx.target_state['output_layer.weight'][:vocab_size, :].copy_(lm_head_weight)
    return ctx.target_state['output_layer.weight']


@io.model_exporter(MixtralModel, "hf")
class HFMixtralExporter(io.ModelConnector[MixtralModel, "MixtralForCausalLM"]):
    """NeMo to HF exporter"""

    def init(self) -> "MixtralForCausalLM":
        """HFMixtralExporter initialization"""
        from transformers import AutoModelForCausalLM
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            return AutoModelForCausalLM.from_config(self.config)

    def apply(self, output_path: Path) -> Path:
        """export to hf format"""
        # TODO: Make it work with lazy init
        # with torch.device("meta"):
        #     target = self.init()
        target = self.init()
        source, _ = self.nemo_load(str(self))
        target = self.convert_state(source, target)

        # TODO: Make sure we don't need to do this
        target = target.cpu()
        target.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        return output_path

    def convert_state(self, source, target):
        """convert state"""
        mapping = {
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            # MoE
            "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight": "model.layers.*.block_sparse_moe.experts.*.w2.weight",  # pylint: disable=line-too-long
            "decoder.layers.*.mlp.router.weight": "model.layers.*.block_sparse_moe.gate.weight",
            # lm-head
            "decoder.final_layernorm.weight": "model.norm.weight",
        }

        transforms = [
            io.state_transform(
                source_key="embedding.word_embeddings.weight",
                target_key="model.embed_tokens.weight",
                fn=TransformFns.prune_padding,
            ),
            io.state_transform(
                source_key="output_layer.weight",
                target_key="lm_head.weight",
                fn=TransformFns.prune_padding,
            ),
            io.state_transform(
                source_key="decoder.layers.*.self_attention.linear_qkv.weight",
                target_key=(
                    "model.layers.*.self_attn.q_proj.weight",
                    "model.layers.*.self_attn.k_proj.weight",
                    "model.layers.*.self_attn.v_proj.weight",
                ),
                fn=TransformFns.split_qkv,
            ),
            io.state_transform(
                source_key="decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
                target_key=(
                    "model.layers.*.block_sparse_moe.experts.*.w1.weight",
                    "model.layers.*.block_sparse_moe.experts.*.w3.weight",
                ),
                fn=TransformFns.split_fc1,
            ),
        ]
        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=transforms,
        )

    @property
    def tokenizer(self):
        """return tokenizer"""
        return io.load_context(str(self), subpath="model").tokenizer

    @property
    def config(self) -> "MixtralConfig":
        """return hf-config from mcore"""
        # Either MixtralConfig8x7B or MixtralConfig8x22B
        source: MixtralConfig8x7B = io.load_context(str(self), subpath="model.config")

        from transformers import MixtralConfig as HfMixtralConfig

        return HfMixtralConfig(
            architectures=["MixtralForCausalLM"],
            num_hidden_layers=source.num_layers,
            hidden_size=source.hidden_size,
            intermediate_size=source.ffn_hidden_size,
            max_position_embeddings=source.max_position_embeddings,
            seq_length=source.max_position_embeddings,
            # RoPe
            rope_theta=source.rotary_base,
            # transformer config
            num_attention_heads=source.num_attention_heads,
            num_key_value_heads=source.num_query_groups,
            num_local_experts=source.num_moe_experts,
            num_experts_per_tok=source.moe_router_topk,
            # norm
            rms_norm_eps=source.layernorm_epsilon,
            # init
            initializer_range=source.init_method_std,
            # vocab
            vocab_size=self.tokenizer.vocab_size,
            head_dim=source.kv_channels,
        )


__all__ = [
    "MixtralConfig",
    "MixtralConfig8x3B",
    "MixtralConfig8x7B",
    "MixtralConfig8x22B",
    "MixtralModel",
]
