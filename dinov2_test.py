from transformers import AutoModel, AutoImageProcessor
from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map
import torch
from PIL import Image
import requests

# Step 1: Load model config and processor
model_name = "/home/lshen/data/huggingface/dinov2-large"
processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=True)

# Step 2: Load model with empty weights and infer device map
from transformers import AutoConfig
config = AutoConfig.from_pretrained(model_name)

with init_empty_weights():
    model = AutoModel.from_config(config)

device_map = infer_auto_device_map(model, max_memory={i: "16GiB" for i in range(torch.cuda.device_count())})

# Dispatch model across multiple GPUs
model = AutoModel.from_pretrained(model_name, device_map=device_map)

# Step 3: Prepare input image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
inputs = processor(images=image, return_tensors="pt")

# Step 4: Move inputs to the **first device** in the device_map
first_device = list(device_map.values())[0]
device = torch.device(first_device)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Step 5: Forward pass
with torch.no_grad():
    outputs = model(**inputs)
    print(outputs.last_hidden_state.shape)
