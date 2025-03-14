# CWDE615 3/13/25
# Script that runs the model on the data in the provided directories.
import os
import argparse

# TODO: In each model function, set up the model using the script provided by its maintainers and call it likewise on the data.


def llava1_6(data):
	# NOTE: Different models have different imports, so we place them inside the function.
	from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
	import torch
	import requests

	# Get llava1.6 from HF. See llava-hf/llava-v1.6-mistral-7b-hf on HF.
	processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

	model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
	model.to("cuda:0")

	# With transformers>=4.48, we no longer have to load a PIL image.
	messages = [
	    {
	        "role": "user",
	        "content": [
	            {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"}
	            {"type": "text", "text": "What is shown in this image?"},
	        ],
	    },
	]

	inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
	output = model.generate(**inputs, max_new_tokens=50)

	return output

def qwen2_5(data):
	from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
	from qwen_vl_utils import process_vision_info


	# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
	model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
	    "Qwen/Qwen2.5-VL-7B-Instruct",
	    torch_dtype=torch.bfloat16,
	    attn_implementation="flash_attention_2",
	    device_map="auto",
	)

	# default processer
	# The default range for the number of visual tokens per image in the model is 4-16384.
	processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

	messages = [
	    {
	        "role": "user",
	        "content": [
	            {
	                "type": "image",
	                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
	            },
	            {"type": "text", "text": "Describe this image."},
	        ],
	    }
	]

	# Preparation for inference
	text = processor.apply_chat_template(
	    messages, tokenize=False, add_generation_prompt=True
	)
	image_inputs, video_inputs = process_vision_info(messages)
	inputs = processor(
	    text=[text],
	    images=image_inputs,
	    videos=video_inputs,
	    padding=True,
	    return_tensors="pt",
	)
	# inputs = inputs.to("cuda") # TODO: Uncomment with environment active.

	# Inference: Generation of the output
	generated_ids = model.generate(**inputs, max_new_tokens=128)
	generated_ids_trimmed = [
	    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
	]
	output_text = processor.batch_decode(
	    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
	)

	return output_text



def model_router(arg = None, data = None):
	if arg == "llava-hf/llava-v1.6-mistral-7b-hf":
		return llava1_6(data)
	elif arg = "Qwen/Qwen2.5-VL-7B-Instruct":
		return qwen2_5(data)
	elif arg = None:
		print("No arg provided to router.")
		return None
	else:
		print("Invalid arg provided to router.")
		return None

if __name__ == "__main__":
	# TODO: Use args to call the right model_router function, which in turn, calls the various other model functions on the same data.
	argparse.ArgumentParser(
		prog = "RUN MODEL",
		description = "This script runs a model whose HF tag is in the 'm' flag using the prompt stored in PROMPT using the comics in the 'comics' folder as input and writing output to 'output'."
	)

	parser.add_argument('-m','--model',required=True,help='the model to be run')
	args = parser.parse_args()

	text = model_router(args.model)

	assert text is not None
	print(text)
