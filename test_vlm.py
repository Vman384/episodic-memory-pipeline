from vllm import LLM, SamplingParams
from PIL import Image

# Initialize the VLM with V100-safe parameters
llm = LLM(
    model="Qwen/Qwen2-VL-7B-Instruct", 
    enforce_eager=True,
    dtype="half",                  # CRITICAL: Forces standard float16 for Volta GPUs
    max_model_len=4096,            # CRITICAL: Restricts context size to prevent OOM
    gpu_memory_utilization=0.9     # Leaves a little buffer for PyTorch overhead
)

prompt = "<|im_start|>user\n<|image_pad|>\nWhat is in this image?<|im_end|>\n<|im_start|>assistant\n"

image = Image.open("test.jpg")

sampling_params = SamplingParams(temperature=0.2, max_tokens=100)

outputs = llm.generate(
    {"prompt": prompt, "multi_modal_data": {"image": image}},
    sampling_params=sampling_params
)

print(outputs[0].outputs[0].text)