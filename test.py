from model import create_model
from PIL import Image
import torch

# Path to the downloaded checkpoint
checkpoint_path = "/home/lshen/data/huggingface/"

# Create the model, change the text_model to `Alibaba-NLP/gte-large-en-v1.5` if use sail_dinov2_gte
model = create_model(
    text_model_name=checkpoint_path + "NV-Embed-v2",
    vision_model_name=checkpoint_path + "dinov2-large",
    head_weights_path=checkpoint_path + "sail/sail_dinov2l_nv2.pt",
    target_dimension=1024,
)
model.eval()  # Set model to evaluation mode

# Prepare images and texts
image_processor = model.image_processor
texts = ["a dog", "a cat"]
dog_image = Image.open("asset/dog.jpg").convert("RGB")
cat_image = Image.open("asset/cat.jpg").convert("RGB")
images = image_processor(images=[dog_image, cat_image], return_tensors="pt")

# Generate features and probabilities
with torch.no_grad():
    image_features = model.encode_image(images, normalize=True)
    text_features = model.encode_text(texts, text_list=texts, normalize=True)
text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# Print the label probabilities
print("Label probs:", text_probs)