from transformers import AutoImageProcessor, AutoModel, CLIPVisionModel
import torch
import torch.nn as nn
from PIL import Image, ImageFile    
import os
from typing import List, Tuple, Dict, Any, Union, Optional
from torchvision import transforms as pth_transforms
from transformers import ImageProcessingMixin, AutoConfig
try:
    from transformers.image_processing_base import BaseBatchFeature
except:
    pass
from packaging import version
import transformers
from .dinoforseg import Dinov2EncoderForSegmentation
from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map
from .mae import get_mae_vit
from .ibot import get_ibot_vit
from .ijepa import get_ijepa_vit
from concurrent.futures import ThreadPoolExecutor, as_completed

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def modify_vit(type_: str) -> None:
    from transformers.models.dinov2 import modeling_dinov2
    if type_ == 'seg':
        print("Using Dinov2EncoderForSegmentation")
        modeling_dinov2.Dinov2Encoder = Dinov2EncoderForSegmentation
    else:
        pass

class CustomImageProcessor(ImageProcessingMixin):
    def __init__(self, size=224):
        super().__init__()
        # 定义图像预处理流水线
        self.image_processor = pth_transforms.Compose([
            pth_transforms.Resize(256, interpolation=pth_transforms.InterpolationMode.BICUBIC),
            pth_transforms.CenterCrop(size),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, images, return_tensors=None):
        if not isinstance(images, list):
            images = [images]
        # 处理单个图像或批量图像
    
        processed_images = [self.image_processor(image) for image in images]
        # 将图像堆叠为一个 tensor
        processed_images = torch.stack(processed_images)
   
        
        # 返回 tensor，如果指定了 `return_tensors='pt'`
        if return_tensors == 'pt':
            return processed_images
        else:
            return processed_images


class ImageEmbedding(nn.Module):
    def __init__(self, model_name="facebook/dinov2-base", device=None, seg: bool = False, agg_mode='concat'):
        super(ImageEmbedding, self).__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')        
        self.seg = seg
        self.agg_mode = agg_mode
        self.model_name = model_name
        
        config = AutoConfig.from_pretrained(model_name)
        with init_empty_weights():
            model = AutoModel.from_config(config)
        device_map = infer_auto_device_map(
            model, 
            max_memory={i: "16GiB" for i in range(torch.cuda.device_count())}
            #no_split_module_classes=["DinoVisionTransformerLayer"]
            )

        if any(x in model_name for x in ['ibot', 'mae', 'dinov1', 'ml-aim', 'ijepa']):
            # load from local
            if 'ibot' in model_name:
                self.model = get_ibot_vit(model_name)
            elif 'mae' in model_name:
                self.model = get_mae_vit(model_name)
            elif 'ijepa' in model_name:
                self.model = get_ijepa_vit(model_name)
            # load from torch hub
            elif 'dinov1' in model_name:
                if 'vitb16' in model_name:
                    self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
                    self.model.embed_dim = 768
                elif 'resnet' in model_name:
                    self.model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
                    self.model.embed_dim = 2048
            elif 'ml-aim' in model_name:
                if '1B' in model_name:
                    self.model = torch.hub.load("apple/ml-aim", "aim_1B")
                    self.model.embed_dim = 2048
                elif '600M' in model_name:
                    self.model = torch.hub.load("apple/ml-aim", "aim_600M")
                    self.model.embed_dim = 1536
                else:
                    raise ValueError(f"Invalid model name: {model_name}")
            if torch.cuda.device_count()>1:
                print("Using DataParallel for custom models")
                self.model = nn.DataParallel(self.model)
                self.model = self.model.to(self.device).half()
            else:
                self.model = self.model.to(self.device).half()
            self.model.dtype = torch.float16
            self.image_processor = CustomImageProcessor()
        
        # load from huggingface
        elif not seg and any(x in model_name.lower() for x in ['dinov2']) and version.parse(transformers.__version__) >= version.parse("4.45.0"):
            # check if huggingface version is greater than 4.38.0
            print("Using model with SDPA")
            self.SDPA = True
            self.model = AutoModel.from_pretrained(model_name, attn_implementation="sdpa", torch_dtype=torch.float16, device_map=device_map)
            self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        elif any(x in model_name.lower() for x in ['clip']):
            self.model = CLIPVisionModel.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device_map)
            self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        elif any(x in model_name.lower() for x in ['aimv2']):
            self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True, device_map=device_map)
            self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        else:
            if seg:
                modify_vit('seg')
            self.SDPA = False
            self.model = AutoModel.from_pretrained(model_name, attn_implementation="eager", torch_dtype=torch.float16, device_map=device_map)
            self.image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

    def load_images_from_directory(self, images_path: List[str]) -> List[Image.Image]:

        def load_single_image(args):
            idx, image_path = args
            with Image.open(image_path) as img:
                return idx, img.convert("RGB")
        
        images = [None] * len(images_path)
        with ThreadPoolExecutor() as executor:
            # Submit tasks with indices to maintain order
            futures = {executor.submit(load_single_image, (i, path)): i 
                      for i, path in enumerate(images_path)}
            
            # Collect results in order
            for future in as_completed(futures):
                idx, img = future.result()
                images[idx] = img
                
        return images

    def get_visual_embeddings_from_directory(self, images_path: List[str]):
        images = self.load_images_from_directory(images_path)
        inputs = self.image_processor(images, return_tensors="pt").to(self.device, dtype=self.model.dtype)
        with torch.autocast(device_type=self.device, dtype=self.model.dtype):
            with torch.no_grad():
                return self.forward(inputs)

    def _get_model_device(self):
        """Get the appropriate device for model inputs in multi-GPU setup"""
        if torch.cuda.device_count()>1:
            if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
                # Get the device of the embedding layer (usually the first layer)
                device_map = self.model.hf_device_map
                
                # Look for embedding layer device
                for layer_name, device in device_map.items():
                    if 'embed' in layer_name.lower() or 'patch' in layer_name.lower():
                        return torch.device(device)
                
                # Fallback to first device in map
                return torch.device(next(iter(device_map.values())))
            elif hasattr(self.model, 'module'):
                # DataParallel case
                return next(self.model.module.parameters()).device
            else:
                # Fallback to first parameter's device
                return next(self.model.parameters()).device
        else:
            return torch.device(self.device)
        
    def forward(self, inputs, patch_mode=False, attetion_type='qk', ignore_residual=False):
        """
        Get embeddings from the vision encoder.

        Args:
            nocls: if True, return mean of patch tokens, else concat CLS token and mean of patch tokens.
            patch_mode: if True, return all patch tokens.
            ignore_residual: if True, skip the residual connection in the last layer.
        """
        input_device = self._get_model_device()
        
        if self.seg:
            self.model.encoder.attetion_type = attetion_type
            self.model.encoder.ignore_residual = ignore_residual
        
                # Move inputs to correct device
        if isinstance(inputs, dict):
            inputs = {k: v.to(input_device) for k, v in inputs.items()}
        else:
            inputs = inputs.to(input_device, non_blocking=True)
                 
        if any(x in self.model_name.lower() for x in ['mae']):
            if isinstance(inputs, torch.Tensor):
                outputs = self.model.forward_features(inputs)
            else:
                outputs = self.model.forward_features(inputs['pixel_values'])
        else:
            if isinstance(inputs, torch.Tensor):
                outputs = self.model(inputs)
            elif isinstance(inputs, dict) or isinstance(inputs, BaseBatchFeature):
                if any(x in self.model_name.lower() for x in ['ibot', 'dinov1', 'ml-aim', 'ijepa']):
                    outputs = self.model(inputs['pixel_values'])
                else:
                    # huggingface transformer vision model
                    outputs = self.model(**inputs)
            else:
                raise ValueError(f"Unsupported input type: {type(inputs)}")
            
        # extract the embeddings
        if any(x in self.model_name.lower() for x in ['ijepa']):
            embedding = outputs.mean(dim=1)
        elif any(x in self.model_name.lower() for x in ['clip']):
            embedding = outputs.pooler_output
        elif any(x in self.model_name.lower() for x in ['aimv2']):
            embedding = torch.mean(outputs.last_hidden_state, dim=1)
        else:
            sequence_output = outputs[0]  # batch_size, sequence_length, hidden_size

            if patch_mode:
                patch_tokens = sequence_output[:, 1:]
                cls_token = sequence_output[:, 0].unsqueeze(1).repeat(1, patch_tokens.shape[1], 1)
                embedding = torch.cat([cls_token, patch_tokens], dim=-1)
            else:
                if any(x in self.model_name.lower() for x in ['ibot', 'mae', 'dinov1']):
                    embedding = outputs
                elif any(x in self.model_name.lower() for x in ['aim']):
                    embedding = outputs[1]
                else:
                    cls_token = sequence_output[:, 0]
                    patch_tokens = sequence_output[:, 1:]
                    if self.agg_mode == 'patch': 
                        embedding = patch_tokens.mean(dim=1)
                    elif self.agg_mode == 'cls':
                        embedding = cls_token
                    elif self.agg_mode == 'concat':
                        embedding = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
                    else:
                        raise ValueError(f"Invalid agg_mode: {self.agg_mode}")

        return embedding


if __name__ == "__main__":
    processor = ImageEmbedding()
    breakpoint()
    image_dirs = ["path/to/dir1", "path/to/dir2"]
    hidden_states = processor.get_visual_embeddings_from_directory(image_dirs)
    for states in hidden_states:
        print(list(states.shape))
