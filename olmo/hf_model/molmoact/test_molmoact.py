import argparse
from PIL import Image
import requests

import torch

from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForImageTextToText, GenerationConfig



def main():

    checkpoint_dir = "/weka/oe-training-default/jiafeid/MolmoAct/checkpoints-molmoact/MolmoAct-7B-D-Pretrain-0812"

    processor = AutoProcessor.from_pretrained(
        checkpoint_dir,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
        padding_side="left",
    )

    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint_dir,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    ).to("cuda")

    instruction = "pick up the black bowl on the stove and place it on the plate"
    prompt = (
        f"The task is {instruction}. "
        "What is the action that the robot should take. "
        f"To figure out the action that the robot should take to {instruction}, let's think through it step by step. "
        "First, what is the depth map for this image? "
        "Second, what is the trajectory of the end effector? "
        "Based on the depth map of the image and the trajectory of the end effector, what is the action that the robot should take?"
    )
    messages = [
        {
            "role": "user",
            "content": [dict(type="text", text=prompt)]
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    img1 = Image.open("/weka/oe-training-default/jiafeid/MolmoAct/data/libero_modified_proprio/libero_spatial_no_noops/primary/0000000/0000.png")
    img2 = Image.open("/weka/oe-training-default/jiafeid/MolmoAct/data/libero_modified_proprio/libero_spatial_no_noops/wrist/0000000/0000.png")
    img = [img1, img2]

    inputs = processor(
        images=[img],
        text=text,
        padding=True,
        return_tensors="pt",
    ).to("cuda")


    def cast_float_dtype(t: torch.Tensor):
        if torch.is_floating_point(t):
            t = t.to(torch.bfloat16)
        return t

    inputs = {k: cast_float_dtype(v.to(model.device)) for k, v in inputs.items()}



    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            generated_ids = model.generate(
                **inputs, 
                generation_config=GenerationConfig(max_new_tokens=448, stop_strings="<|endoftext|>"),
                tokenizer=processor.tokenizer
            )

    generated_tokens = generated_ids[:, inputs['input_ids'].size(1):]
    generated_text = processor.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(generated_text)

    depth = model.parse_depth(generated_text)
    print(depth)

    trace = model.parse_trace(generated_text)
    print(trace)

    action = model.parse_action(generated_text, unnorm_key="fractal20220817_data")
    print(action)



if __name__ == "__main__":
    main()


## pip install transformers==4.52.3