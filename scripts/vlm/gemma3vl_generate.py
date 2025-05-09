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

"""
Example:
    python scripts/vlm/gemma3_generate.py --load_from_hf
"""

import argparse

import requests
import torch
from PIL import Image

from transformers import AutoProcessor, Gemma3ForConditionalGeneration

import nemo.lightning as nl
from nemo.collections.vlm import Gemma3VLConfig4B, Gemma3VLModel
from nemo.utils import logging

HF_GEMMA3 = "google/gemma-3-4b-it"


def main(args) -> None:
    # pylint: disable=C0115,C0116
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        ckpt_include_optimizer=False,
    )
    trainer = nl.Trainer(
        devices=args.tp_size * args.pp_size,
        max_steps=10,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        val_check_interval=10,
        limit_val_batches=0,
    )

    processor = AutoProcessor.from_pretrained(HF_GEMMA3)
    hf_tokenizer = processor.tokenizer

    fabric = trainer.to_fabric()
    # Decide whether to import or load the model based on the input arguments
    args.load_from_hf = True
    if args.load_from_hf:
        model = fabric.import_model(f"hf://{HF_GEMMA3}", Gemma3VLModel)
    else:
        model = Gemma3VLModel(Gemma3VLConfig4B(), tokenizer=hf_tokenizer)
        model = fabric.load_model(args.local_model_path, model)
    model = model.module.cuda()
    model.eval()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": args.image_url,
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    print(inputs)
    return

    with torch.no_grad():
        input_ids = inputs['input_ids'].clone().to("cuda")
        pixel_values = inputs['pixel_values'].clone().to("cuda")

        # Greedy generation loop
        generated_ids = input_ids
        for _ in range(args.osl):
            output = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
            )

            next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

            input_ids = generated_ids
            # If the generated token is the end of sequence token, stop generating
            if next_token_ids.item() == hf_tokenizer.eos_token_id:
                break

        generated_ids[generated_ids < 0] = 0
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        generated_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        logging.info("======== GENERATED TEXT OUTPUT ========")
        logging.info(f"{args.image_url}, \t\t{generated_texts}")
        logging.info("=======================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma3 Multimodal Inference")
    parser.add_argument(
        "--load_from_hf",
        action="store_true",
        help="Flag to indicate whether to load the model from Hugging Face hub.",
    )
    parser.add_argument(
        "--local_model_path",
        type=str,
        default=None,
        help="Local path to the model if not loading from Hugging Face.",
    )
    parser.add_argument(
        "--image_url",
        type=str,
        default="https://github.com/google/gemma_pytorch/blob/main/scripts/images/test_image.jpg?raw=true",
        help="URL of the image to use for inference.",
    )
    parser.add_argument('--osl', type=int, default=30, help='output seq length')
    parser.add_argument('--tp_size', type=int, default=1, help='tensor parallel size')
    parser.add_argument('--pp_size', type=int, default=1, help='pipeline parallel size')
    args = parser.parse_args()

    main(args)
