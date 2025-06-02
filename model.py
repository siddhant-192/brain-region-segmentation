########## model.py ##########

import torch
import torch.nn as nn
import os
import numpy as np
import cv2
from torch.utils.data import Dataset
from pathlib import Path
import torch.nn.functional as F
from monai.losses import DiceLoss
from kornia.losses import BinaryFocalLossWithLogits
import kornia.filters as KF
import segmentation_models_pytorch as smp
import traceback
from typing import Dict, List, Tuple, Union

class BrainSegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None, img_size=4096):
        """
        Dataset for brain segmentation that loads whole images.

        Args:
            data_dir (str): Directory where the data is stored
            transform (callable, optional): Transforms to apply to the images and the labels
            img_size (int): Size to resize the image to (default: 4096)
        """
        self.data_dir  = data_dir
        self.transform = transform
        self.img_size = img_size

        self.images_dir = os.path.join(data_dir, "images")
        self.labels_dir = os.path.join(data_dir, "labels")

        self.image_filenames = sorted([f for f in os.listdir(self.images_dir) if f.endswith(('.jpg'))])

        self.class_to_idx = {
            "Thalamus": 0,
            "Caudate nucleus": 1,
            "Putamen": 2,
            "Globus pallidus": 3,
            "Nucleus accumbens": 4,
            "Internal capsule": 5,
            "Substantia innominata": 6,
            "Fornix": 7,
            "Anterior commissure": 8,
            "Ganglionic eminence": 9,
            "Hypothalamus": 10,
            "Amygdala": 11,
            "Hippocampus": 12,
            "Choroid plexus": 13,
            "Lateral ventricle": 14,
            "Olfactory tubercle": 15,
            "Pretectum": 16,
            "Inferior colliculus": 17,
            "Superior colliculus": 18,
            "Tegmentum": 19,
            "Pons": 20,
            "Medulla": 21,
            "Cerebellum": 22,
            "Corpus callosum": 23,
            "Cerebral cortex": 24
        }

        self.idx_to_class = {v:k for k,v in self.class_to_idx.items()}

        self.colors = {
            0:  "#6ff151",  # Thalamus
            1:  "#65ce0c",  # Caudate nucleus
            2:  "#d6cb4d",  # Putamen
            3:  "#e5feba",  # Globus pallidus
            4:  "#8e6f03",  # Nucleus accumbens
            5:  "#8e4527",  # Internal capsule
            6:  "#4f9b02",  # Substantia innominata
            7:  "#ac561e",  # Fornix
            8:  "#b7cc25",  # Anterior commissure
            9:  "#876f47",  # Ganglionic eminence
            10: "#3fe936",  # Hypothalamus
            11: "#74af15",  # Amygdala
            12: "#bd885c",  # Hippocampus
            13: "#b5e60a",  # Choroid plexus
            14: "#88b151",  # Lateral ventricle
            15: "#ecad5e",  # Olfactory tubercle
            16: "#707166",  # Pretectum
            17: "#a1830f",  # Inferior colliculus
            18: "#ff9b3d",  # Superior colliculus
            19: "#eaeea1",  # Tegmentum
            20: "#cc7e39",  # Pons
            21: "#fcae1b",  # Medulla
            22: "#4e4137",  # Cerebellum
            23: "#de998d",  # Corpus callosum
            24: "#50fa0d",  # Cerebral cortex
        }

    def __len__(self):
        """
        Get the number of images (datapoints) present in the dataset
        """
        return len(self.image_filenames)
    
    def __getitem__(self, index):
        """
        Get a single whole image and its label.
        """
        img_filename = self.image_filenames[index]
        img_path = os.path.join(self.images_dir, img_filename)

        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        original_h, original_w = img.shape[:2]

        # Resize to target size while preserving aspect ratio
        if (original_h, original_w) != (self.img_size, self.img_size):
            img, scale, x_offset, y_offset = self._resize_preserve_aspect(img, self.img_size)
        else:
            scale = 1.0
            x_offset = 0
            y_offset = 0
        
        # Parse YOLO format labels
        label_filename = img_filename.replace(".jpg", ".txt")
        label_path = os.path.join(self.labels_dir, label_filename)

        segments, classes = self._parse_yolo_labels(label_path, original_w, original_h)

        # Apply any transformation
        if self.transform:
            img, segments = self.transform(img, segments)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Create target dict
        target = {
            'segments': segments,
            'cls': torch.tensor(classes, dtype=torch.int64),
            'original_size': (original_h, original_w)
        }

        return img_tensor, target

    def _resize_preserve_aspect(self, img, target_size):
        """
        Resize the image preserving ratio with padding.
        Returns: resized_image, scale, x_offset, y_offset
        """
        h, w = img.shape[:2]
        scale = min(target_size / w, target_size / h)
        
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        # Create square image with padding
        square = np.zeros((target_size, target_size, 3), dtype=np.uint8)

        # Center the resized image
        x_offset = (target_size - new_w) // 2
        y_offset = (target_size - new_h) // 2

        square[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        return square, scale, x_offset, y_offset

    def _parse_yolo_labels(self, label_path, orig_w, orig_h):
        """
        Parse YOLO format labels and apply the same transformations as the image
        """
        segments, classes = [], []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.read().strip().splitlines():
                    parts = line.split()
                    cls = int(parts[0])
                    coords = list(map(float, parts[1:]))

                    # Convert normalized coordinates to absolute coordinates in original image space
                    abs_coords = []
                    for i in range(0, len(coords), 2):
                        x_abs = coords[i] * orig_w
                        y_abs = coords[i+1] * orig_h
                        
                        abs_coords.append(x_abs)
                        abs_coords.append(y_abs)
                    
                    segments.append(abs_coords)
                    classes.append(cls)
        
        return segments, classes

class LightweightDecoder(nn.Module):
    """
    Lightweight decoder with skip connections.
    Converts 32 feature channels to 25 class channels with spatial upsampling.
    """
    def __init__(self, in_channels=32, num_classes=25):
        super().__init__()
        self.num_classes = num_classes

        # Simple but effective decoder
        self.decoder_blocks = nn.ModuleList([
            # Block 1: 32 -> 16 channels, 2x upsampling
            nn.Sequential(
                nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Upsample(
                    scale_factor=2, 
                    mode='bilinear', 
                    align_corners=False
                    )
            ),
            # Block 2: 16 -> 8 channels, 2x upsampling
            nn.Sequential(
                nn.Conv2d(16, 8, 3, padding=1, bias=False),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                nn.Upsample(
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=False
                )
            )
        ])

        # Final classification layer
        self.final_conv = nn.Conv2d(8, num_classes, kernel_size=1, bias=True)

        # Skip connection projections for input features
        self.input_skip_proj = nn.Conv2d(3, 8, kernel_size=1, bias=False)
        nn.init.xavier_normal_(self.input_skip_proj.weight, gain=0.1)
    
    def forward(self, features, original_input=None):
        """
        Args:
            features: (B, 32, H, W) - feature maps from fusion
            original_input: (B, 3, H, W) - original RGB input for skip connection
        """
        x = features

        # Apply decoding blocks
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x) # Progressive upsample and feature refinement

        # Skip connection from original input
        if original_input is not None:
            # Resize original input to match current spatial size
            skip_input = F.interpolate(
                original_input,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            # Project to matching channels and add
            skip_features = self.input_skip_proj(skip_input)
            x = x + 0.2 * skip_features # SKIP CONNECTION
        
        # Final classification
        output = self.final_conv(x)
        
        return output

class PatchFusionYOLO(nn.Module):
    def __init__(self, base_model, num_classes=25, learn_fusion_weights=True, freeze_base=True):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.freeze_base = freeze_base

        # CRITICAL FIX: Freeze the base model weights
        if self.freeze_base:
            self._freeze_base_model()

        self.stride = int(max(base_model.stride)) if hasattr(base_model, "stride") else 32

        # Image size parameters
        self.imgsz_4096 = 4096
        self.imgsz_2048 = 2048

        # Patch parameters
        self.overlap_percent = 10
        self.central_patch_overlap = 5

        # Expected output channels from YOLO backbone
        self.expected_channels = 32

        self.decoder = LightweightDecoder(
            in_channels=self.expected_channels, #32
            num_classes=self.num_classes        #25
        )

        # Initialize learnable fusion weights with better defaults and constraints
        self.learn_fusion_weights = learn_fusion_weights
        if learn_fusion_weights:
            # Initialize with more weight on the central patch (3x the others)
            initial_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 3.0])
            initial_weights = initial_weights / initial_weights.sum()  # Normalize
            self.patch_weights_logits = nn.Parameter(torch.log(initial_weights / (1 - initial_weights)))
            
            # Initialize resolution weight with slight preference for high-res (4096)
            self.resolution_weight_logit = nn.Parameter(torch.tensor(0.2))  # Sigmoid(0.2) ≈ 0.55
            
            # Initialize boundary weight
            self.boundary_weight = nn.Parameter(torch.tensor(0.2))
        else:
            # Fixed weights
            self.register_buffer('patch_weights', torch.tensor([1.0, 1.0, 1.0, 1.0, 3.0]) / 7.0)
            self.register_buffer('resolution_weight', torch.tensor(0.55))
            self.register_buffer('boundary_weight', torch.tensor(0.2))

        # Channel projection layers (lazily initialized when needed)
        self.channel_projections = nn.ModuleDict()

        # Feature channel attention (operates on 32 feature channels)
        self.feature_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.expected_channels, self.expected_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.expected_channels // 4, self.expected_channels, 1),
            nn.Sigmoid()
        )
    
    def _freeze_base_model(self):
        """
        Freeze all parameters in the base model to prevent them from being updated during training.
        This ensures we only train the additional layers on top.
        """
        print("Freezing base YOLOv9 model parameters...")
        frozen_params = 0
        for name, param in self.base_model.named_parameters():
            param.requires_grad = False
            frozen_params += param.numel()
        
        print(f"Frozen {frozen_params:,} parameters in base model")
        
        # Set base model to eval mode to ensure consistent behavior
        self.base_model.eval()

    def train(self, mode=True):
        """
        Override train method to ensure base model stays in eval mode when frozen.
        """
        super().train(mode)
        if self.freeze_base:
            self.base_model.eval()
        return self
    
    def get_trainable_parameters(self):
        """
        Return only the trainable parameters (not from base model).
        """
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append((name, param))
        return trainable_params
    
    def print_trainable_parameters(self):
        """
        Print summary of trainable vs frozen parameters.
        """
        trainable_params = 0
        frozen_params = 0
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"Trainable: {name} - {param.numel():,} params")
            else:
                frozen_params += param.numel()
        
        total_params = trainable_params + frozen_params
        print(f"\nParameter Summary:")
        print(f"Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        print(f"Frozen: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        print(f"Total: {total_params:,}")
        
    def forward(self, x):
        """
        Processes image using multi-scale patch fusion strategy
        """
        batch_size, channels, input_height, input_width = x.shape

        # Store original input for skip connections
        original_input = x

        # If batch size is >1 then process each image seperately
        if batch_size>1:
            outputs = []
            for i in range(batch_size):
                outputs.append(self._process_single_image(x[i:i+1]))
            result = torch.cat(outputs, dim=0)
        else:
            result =  self._process_single_image(x)
        
        # Resize the output back to the input dimentions
        if result.shape[2:] != (input_height, input_width):
            result = F.interpolate(
                result,
                size=(input_height, input_width),
                mode='bilinear',
                align_corners=False
            )
        
        # Ensure proper channel count (32 feature channels)
        result = self._ensure_channel_count(result)

        # Use decoder to get the final output
        final_output = self.decoder(result, original_input)

        return final_output
    
    @staticmethod
    def _pad_to_stride(x, stride):
        """
        When the dimentions are not multiple of the stride (should be a multiple) 
        then we pad the feature map appropriately so that the new padded feature 
        map becomes a multiple of the stride.
        
        We zero-pad only in the right bottom of the feature map.

        H and W are multiples of the stride after processing through the function.
        """
        h, w = x.shape[2:]
        pad_h = (stride - h % stride) % stride
        pad_w = (stride - w % stride) % stride
        if pad_h or pad_w:
            #(left, right, top, bottom) -> pad on right and bottom only
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, pad_h, pad_w
    
    def _spatial_tensor(self, outputs):
        """
        Return the segmentation prototype features from YOLOv9-seg outputs with improved robustness.
        YOLOv9-seg outputs a tuple where prototype features are at outputs[1][2]
        with shape (B, 32, H, W)
        """
        # Initialize tracking variable if not already present
        if not hasattr(self, '_feature_extracted'):
            self._feature_extracted = False
        
        # Direct path for YOLOv9-seg structure with better error handling
        try:
            if (isinstance(outputs, tuple) and len(outputs) == 2 and 
                isinstance(outputs[1], tuple) and len(outputs[1]) == 3 and
                torch.is_tensor(outputs[1][2]) and outputs[1][2].ndim == 4):
                
                # This is the expected structure
                features = outputs[1][2]
                
                # Log info on first successful extraction
                if not self._feature_extracted:
                    print(f"Successfully extracted YOLOv9-seg prototype features: shape={features.shape}")
                    self._feature_extracted = True
                
                # Validate expected channels
                if hasattr(self, 'expected_channels') and features.shape[1] != self.expected_channels:
                    print(f"Warning: Expected {self.expected_channels} channels but got {features.shape[1]}")
                
                return features
        except (IndexError, AttributeError) as e:
            print(f"Primary feature extraction path failed: {str(e)}")
        
        # Fallback logic for other structures with better error messages
        if torch.is_tensor(outputs):
            # Single tensor case
            features = outputs
            if not self._feature_extracted:
                print(f"Using direct tensor output: shape={features.shape}")
                self._feature_extracted = True
            return features
        
        if isinstance(outputs, (list, tuple)):
            # First, look for 4D tensors with expected channels
            expected_channels = getattr(self, 'expected_channels', 32)
            
            for i, o in enumerate(outputs):
                if torch.is_tensor(o) and o.ndim == 4 and o.shape[1] == expected_channels:
                    if not self._feature_extracted:
                        print(f"Found tensor with expected channels at outputs[{i}]: shape={o.shape}")
                        self._feature_extracted = True
                    return o
            
            # Then look for any 4D tensor
            for i, o in enumerate(outputs):
                if torch.is_tensor(o) and o.ndim == 4:
                    if not self._feature_extracted:
                        print(f"Found 4D tensor at outputs[{i}]: shape={o.shape}")
                        self._feature_extracted = True
                    return o
            
            # Recursive search in nested containers with position tracking
            for i, o in enumerate(outputs):
                if isinstance(o, (list, tuple)):
                    for j, nested_o in enumerate(o):
                        if torch.is_tensor(nested_o) and nested_o.ndim == 4:
                            if not self._feature_extracted:
                                print(f"Found nested tensor at outputs[{i}][{j}]: shape={nested_o.shape}")
                                self._feature_extracted = True
                            return nested_o
        
        # If we got here, we didn't find a usable tensor
        structure_info = self._describe_structure(outputs)
        raise RuntimeError(f"No usable tensor found in outputs. Structure: {structure_info}")

    def _describe_structure(self, obj, max_depth=3, current_depth=0):
        """Helper to describe the structure of complex nested outputs for debugging"""
        if current_depth >= max_depth:
            return "..."
        
        if torch.is_tensor(obj):
            return f"Tensor(shape={list(obj.shape)})"
        
        if isinstance(obj, (list, tuple)):
            if len(obj) == 0:
                return "Empty sequence"
            
            items = []
            for i, item in enumerate(obj):
                if i >= 3:  # Limit to first 3 items
                    items.append("...")
                    break
                items.append(self._describe_structure(item, max_depth, current_depth + 1))
            
            container_type = "list" if isinstance(obj, list) else "tuple"
            return f"{container_type}[{len(obj)}]({', '.join(items)})"
        
        return str(type(obj).__name__)

    def _ensure_channel_count(self, tensor, target_channels=None):
        """
        Properly map input channels to target channel count using learned projection
        """
        if target_channels is None:
            target_channels = self.expected_channels
            
        in_channels = tensor.shape[1]
        
        # If channels already match, return as is
        if in_channels == target_channels:
            return tensor
            
        # Create a unique key for this channel transformation
        key = f"in{in_channels}_out{target_channels}"
        
        # Create projection layer if it doesn't exist
        if key not in self.channel_projections:
            self.channel_projections[key] = nn.Conv2d(
                in_channels, 
                target_channels,
                kernel_size=1,
                bias=False
            )
            # Initialize with Kaiming initialization
            nn.init.kaiming_normal_(self.channel_projections[key].weight)
            
            # Move to same device
            self.channel_projections[key] = self.channel_projections[key].to(tensor.device)
        
        # Apply projection
        return self.channel_projections[key](tensor)
    
    def _process_single_image(self, x):
        """
        Process a single image through multi-scale fusion
        """
        # Process at both resolutions
        fusion_map_2048 = self._process_at_2048(x)
        fusion_map_4096 = self._process_at_4096(x)

        # Fuse multi-resolution results
        final_fusion_map = self._fuse_multi_resolution(fusion_map_2048, fusion_map_4096)

        # Ensure that final_fusion_map has the expected number of channels (32)
        final_fusion_map = self._ensure_channel_count(final_fusion_map)
        
        torch.cuda.empty_cache() 

        return final_fusion_map
    
    def _process_at_4096(self ,x):
        """
        Process image at 4096 resolution using sliding windows
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Resize to 4096 id needed
        if x.shape[2:] != (self.imgsz_4096, self.imgsz_4096):
            x = F.interpolate(
                x,
                size=(self.imgsz_4096, self.imgsz_4096),
                mode='bilinear',
                align_corners=False
            )
        
        batch_size, channels, height, width = x.shape

        # Define sliding windows
        windows = [
            (0, 0, width//2, height//2),
            (width//4, 0, 3*width//4, height//2),
            (width//2, 0, width, height//2),
            (0, height//4, width//2, 3*height//4),
            (width//4, height//4, 3*width//4, 3*height//4),
            (width//2, height//4, width, 3*height//4),
            (0, height//2, width//2, height),
            (width//4, height//2, 3*width//4, height),
            (width//2, height//2, width, height)
        ]

        # Sample input the get the output dimentions
        # Using small tensor to save memory
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, 128, 128, device=x.device)
            sample_outputs = self.base_model(dummy_input)
            sample_output = self._spatial_tensor(sample_outputs)
                
            output_channels = sample_output.shape[1]
        
        # Initialize fusion map
        fusion_map_global = torch.zeros(batch_size, output_channels, height, width, device=x.device)

        # Instead of processing all windows at once, process them in smaller batches
        # and free memory after each batch
        windows_batches = [windows[i:i+3] for i in range(0, len(windows), 3)]
        
        for batch_idx, window_batch in enumerate(windows_batches):
            # Process each window in the batch
            for window_idx, (x1, y1, x2, y2) in enumerate(window_batch):
                # Extract window
                window = x[:, :, y1:y2, x1:x2]
                
                # Process window
                processed_window = self._process_at_2048(window)
                processed_window = self._spatial_tensor(processed_window)
                
                # Add to global fusion map
                fusion_map_global[:, :, y1:y2, x1:x2] += processed_window
                
                # Clear memory immediately
                del window, processed_window
                torch.cuda.empty_cache()
            
            # Additional explicit memory cleanup after each batch
            torch.cuda.empty_cache()
        
        return fusion_map_global
    
    def _process_at_2048(self, x):
        """
        Process input at 2048 resolution using the 5 patch strategy
        """
        # Resize to 2048 if needed
        if x.shape[2:] != (self.imgsz_2048, self.imgsz_2048):
            x = F.interpolate(
                x,
                size=(self.imgsz_2048, self.imgsz_2048),
                mode='bilinear',
                align_corners=False
            )
        
        batch_size, channels, height, width = x.shape

        # Generate patches (4 quadrants + 1 central)
        patches_with_type = self._generate_patch_coords(height, width)

        # Sample output for output dimensions
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, 128, 128, device=x.device)
            sample_outputs = self.base_model(dummy_input)
            sample_output = self._spatial_tensor(sample_outputs)

            output_channels = sample_output.shape[1]
        
        # Initialize fusion map
        fusion_map = torch.zeros(batch_size, output_channels, height, width, device=x.device)

        # Normalize patch weights using softmax for stability
        if self.learn_fusion_weights:
            patch_weights_norm = F.softmax(self.patch_weights_logits, dim=0)
        else:
            patch_weights_norm = self.patch_weights

        for patch_idx, (patch_type, (x1, y1, x2, y2)) in enumerate(patches_with_type):
            # Extract patch
            patch = x[:, :, y1:y2, x1:x2]

            patch, ph, pw = self._pad_to_stride(patch, self.stride)

            # Process with base model
            processed_patch = self.base_model(patch)
            processed_patch = self._spatial_tensor(processed_patch)
            
            # FIX: Add safety check for tensor dimensions
            if processed_patch.dim() < 4:
                # Handle case where output doesn't have expected dimensions
                print(f"Warning: Expected 4D tensor but got {processed_patch.dim()}D tensor. Reshaping.")
                if processed_patch.dim() == 3:
                    # Add batch dimension if missing
                    processed_patch = processed_patch.unsqueeze(0)
                elif processed_patch.dim() < 3:
                    # Skip this patch if dimensions are too low
                    print(f"Error: Cannot process tensor with {processed_patch.dim()} dimensions")
                    continue
            
            # FIX: Proper dimension safety check before slicing
            if ph > 0 or pw > 0:
                if processed_patch.dim() >= 4 and processed_patch.size(2) > ph and processed_patch.size(3) > pw:
                    processed_patch = processed_patch[:, :, :processed_patch.size(2) - ph, :processed_patch.size(3) - pw]
                else:
                    # Log the shape for debugging
                    print(f"Warning: Cannot remove padding. Tensor shape: {processed_patch.shape}, ph={ph}, pw={pw}")

            # Resize if needed
            if processed_patch.shape[2:] != (y2-y1, x2-x1):
                processed_patch = F.interpolate(
                    processed_patch,
                    size=(y2-y1, x2-x1),
                    mode='bilinear',
                    align_corners=False
                )
            
            # Get appropriate weight based on the patch
            weight = patch_weights_norm[4] if patch_type == 'central' else patch_weights_norm[patch_idx]

            # Normalize feature maps before fusion
            processed_patch = F.normalize(processed_patch, p=2, dim=1)

            # Apply weight and add to fusion map
            fusion_map[:, :, y1:y2, x1:x2] += processed_patch * weight

        return fusion_map

    def _fuse_multi_resolution(self, fusion_map_2048, fusion_map_4096):
        """
        Enhanced multi-resolution fusion with skip connections for FEATURE channels.
        """
        # Resize the 2048 map to match 2096
        resized_fusion_map_2048 = F.interpolate(
            fusion_map_2048,
            size=fusion_map_4096.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        # SKIP CONNECTION 1: Global feature-level residual connection
        # This preserces information across all feature channels
        global_residual = fusion_map_4096 + 0.3 * resized_fusion_map_2048

        # Initialize the fusion map
        fused_map = torch.zeros_like(fusion_map_4096)

        # Apply learnable weights
        if self.learn_fusion_weights:
            resolution_weight = torch.sigmoid(self.resolution_weight_logit)
        else:
            resolution_weight = self.resolution_weight
        
        # Process each FEATURE channel (not class channel)
        for feature_idx in range(fusion_map_4096.shape[1]):
            # Extract feature maps
            map_4096 = fusion_map_4096[:, feature_idx:feature_idx+1]
            map_2048 = resized_fusion_map_2048[:, feature_idx:feature_idx+1]

            # Get corresponding global residuals
            residual_features = global_residual[:, feature_idx:feature_idx+1]

            # Create soft boundary mask based on the 2048 map
            mask_2048 = torch.sigmoid((map_2048 - self.boundary_weight) * 10.0)

            # Weighted combination using learnable parameter
            base_fusion = (
                map_4096 * resolution_weight +
                map_2048 * (1 - resolution_weight)
            ) * mask_2048

            # SKIP CONNECTION 2: Add feature-level residual
            fused_map[:, feature_idx:feature_idx+1] = base_fusion + 0.2 * residual_features
        
        if hasattr(self, 'feature_attention'):
            attention_weights = self.feature_attention(fused_map)
            fused_map = fused_map * attention_weights
        
        return fused_map
    
    def _generate_patch_coords(self, height, width):
        """
        Generate the coordinates for the 5 patches
        """
        # Calculate overlap in pixels
        overlap_w = int(width * self.overlap_percent / 100)
        overlap_h = int(height * self.overlap_percent / 100)

        # Calculate quadrant patch dimentions
        quad_patch_w = (width // 2) + overlap_w
        quad_patch_h = (height // 2) + overlap_h

        # Coordinates for quadrant patches
        top_left_coords = (0, 0, quad_patch_w, quad_patch_h)
        top_right_coords = (width // 2 - overlap_w, 0, width, quad_patch_h)
        bottom_left_coords = (0, height // 2 - overlap_h, quad_patch_w, height)
        bottom_right_coords = (width // 2 - overlap_w, height // 2 - overlap_h, width, height)

        # Calculate central patch dimentions and coordinates
        central_patch_w = (width // 2) + int(width * (self.central_patch_overlap *2 /10))
        central_patch_h = (height // 2) + int(height * (self.central_patch_overlap *2 /10))
        center_x, center_y = width // 2, height // 2

        central_coords = (
            max(0, center_x - central_patch_w // 2),
            max(0, center_y - central_patch_h // 2),
            min(width, center_x + central_patch_w // 2),
            min(height, center_y + central_patch_h // 2)
        )

        return [
            ('quadrant', top_left_coords),
            ('quadrant', top_right_coords),
            ('quadrant', bottom_left_coords),
            ('quadrant', bottom_right_coords),
            ('central', central_coords)
        ]

class BrainSegmentationLoss(nn.Module):
    def __init__(self, num_classes=25, dice_weight=1.0, focal_weight=0.5, boundary_weight=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight

        # Combination of losses
        self.dice_loss = DiceLoss(
            include_background=False,
            to_onehot_y=False,
            sigmoid=True,
            squared_pred=True,
            smooth_nr=1.0
        )

        self.focal_loss = BinaryFocalLossWithLogits(
            alpha=0.75,
            gamma=2.0,
            reduction='mean'
        )
    
    def forward(self, pred, target):
        """
        Calculate combined loss with improved shape handling
        
        Args:
            pred: Model predictions (B, C, H, W)
            target: Dictionary with segmentation information or tensor
        """
        # Debug shape information
        # print(f"Loss input - pred shape: {pred.shape}")
        
        # Handle different input types
        if isinstance(target, list) and len(target) > 0 and isinstance(target[0], dict):
            # Target is a list of dictionaries with segments info (new batch format)
            B = pred.size(0)  # true batch size from the model
            
            # Collect segments, classes, and sizes from batch
            target_segments = [t['segments'] for t in target]
            target_classes = [t['cls'] for t in target]
            target_sizes = [t['original_size'] for t in target]

            # Convert polygons → dense masks
            gt_masks = self._segments_to_masks(
                target_segments,
                target_classes,
                pred.shape[2:],  # (H, W)
                target_sizes,
            ).to(pred.device)
        elif isinstance(target, dict):
            # Legacy case: Target is a dictionary with segments info
            B = pred.size(0)  # true batch size from the model
            
            if len(target['segments']) != B:
                # We received polygons for a single image (or wrong batching) → wrap
                target_segments = [target['segments']]
                target_classes = [target['cls']]
                target_sizes = [target['original_size']]
            else:
                target_segments = target['segments']
                target_classes = target['cls']
                target_sizes = target['original_size']

            # Convert polygons → dense masks
            gt_masks = self._segments_to_masks(
                target_segments,
                target_classes,
                pred.shape[2:],  # (H, W)
                target_sizes,
            ).to(pred.device)
        else:
            # Target is already a dense mask
            gt_masks = target.to(pred.device, non_blocking=True)
        
        # Debug shape information
        # print(f"Loss calculation - pred shape: {pred.shape}, gt_masks shape: {gt_masks.shape}")
        
        # Ensure batch dimensions match
        if gt_masks.shape[0] != pred.shape[0]:
            if gt_masks.shape[0] > pred.shape[0]:
                gt_masks = gt_masks[:pred.shape[0]]  # Take first batch_size items
            else:
                # Repeat gt_masks to match batch_size
                repeats = [1] * len(gt_masks.shape)
                repeats[0] = pred.shape[0] // gt_masks.shape[0] + 1
                gt_masks = gt_masks.repeat(*repeats)[:pred.shape[0]]
        
        # Ensure channel dimensions match if needed
        if pred.shape[1] != gt_masks.shape[1]:
            if pred.shape[1] < gt_masks.shape[1]:
                pred = F.pad(pred, (0, 0, 0, 0, 0, gt_masks.shape[1] - pred.shape[1]))
            else:
                gt_masks = F.pad(gt_masks, (0, 0, 0, 0, 0, pred.shape[1] - gt_masks.shape[1]))
        
        # Calculate dice and focal losses
        dice = self.dice_loss(pred, gt_masks)
        focal = self.focal_loss(pred, gt_masks)

        # Calculate boundary loss using Sobel edge detection
        pred_sigmoid = torch.sigmoid(pred)
        pred_edges = KF.sobel(pred_sigmoid)
        target_edges = KF.sobel(gt_masks)
        boundary = F.mse_loss(pred_edges, target_edges)

        # Combine losses with weighting
        total_loss = dice * self.dice_weight + focal * self.focal_weight + boundary * self.boundary_weight
        # print("BrainSegmentation loss:", dice, focal, boundary)

        return total_loss
    
    def _segments_to_masks(self, segments, classes, output_size, original_size):
        # simply forward to our new helper:
        return segments_to_masks(
            segments, classes,
            output_size, original_size,
            num_classes=self.num_classes
        )
    
def rasterize_polygon(
    polygon: Union[List[float], torch.Tensor],
    output_size: Tuple[int,int],
    original_size: Tuple[int,int],
) -> torch.Tensor:
    """
    Rasterizes a polygon (given as a list or tensor of coordinates) onto a 2D torch 
    tensor mask of specified output size, preserving the aspect ratio relative to 
    the original image size.
    """
    output_h, output_w = output_size
    orig_h, orig_w   = original_size

    # early exit on empty
    if isinstance(polygon, torch.Tensor):
        if polygon.numel() == 0:
            return torch.zeros((output_h, output_w), dtype=torch.float32)
    elif not polygon:
        return torch.zeros((output_h, output_w), dtype=torch.float32)

    # always do CPU→numpy
    device = 'cpu'
    if isinstance(polygon, torch.Tensor):
        polygon_np = polygon.detach().cpu().numpy()
    else:
        polygon_np = np.array(polygon)

    # must be an even number of coords, at least 3 points
    if len(polygon_np) % 2 == 0 and len(polygon_np) >= 6:
        pts = polygon_np.reshape(-1, 2)

        # reproduce your _resize_preserve_aspect scaling
        scale = min(output_w / orig_w, output_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        x_off = (output_w - new_w) // 2
        y_off = (output_h - new_h) // 2

        scaled = pts.copy()
        scaled[:, 0] *= scale
        scaled[:, 1] *= scale
        scaled[:, 0] += x_off
        scaled[:, 1] += y_off

        pts_i = scaled.astype(np.int32)
        mask = np.zeros((output_h, output_w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts_i], 1)

        mask_t = torch.from_numpy(mask).float().to(device)
        return mask_t

    # fallback for malformed
    return torch.zeros((output_h, output_w), dtype=torch.float32, device=device)

def segments_to_masks(
    segments: List[List[List[float]]],
    classes:  List[List[int]],
    output_size:     Tuple[int,int],
    original_size:   List[Tuple[int,int]],
    num_classes:     int
) -> torch.Tensor:
    """
    Batch‐level mask builder, including
    hole‐punching and mutual‐exclusivity.
    Returns: Tensor of shape [B, num_classes, H, W].
    """
    batch_size = len(segments)
    H, W = output_size
    masks = torch.zeros((batch_size, num_classes, H, W), dtype=torch.float32)

    for b in range(batch_size):
        segs   = segments[b]
        clses  = classes[b]
        orig   = original_size[b]

        # 1) group polygons by class (same as before)
        by_cls: Dict[int, List[List[float]]] = {}
        for seg, cls in zip(segs, clses):
            if cls < num_classes:
                by_cls.setdefault(cls, []).append(seg)

        # 2) rasterize + hole-punch per class (same as before)
        sample_mask = torch.zeros((num_classes, H, W), dtype=torch.float32)
        for cls, polys in by_cls.items():
            pmasks = [rasterize_polygon(p, output_size, orig) for p in polys]
            if len(pmasks) == 1:
                cm = pmasks[0]
            else:
                areas = [m.sum().item() for m in pmasks]
                order = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)
                cm = pmasks[order[0]].clone()
                for i in order[1:]:
                    hole = pmasks[i].bool()
                    cm[hole] = 0.0
            sample_mask[cls] = cm
            for m in pmasks: del m

        # 3) FIXED: Softer mutual exclusivity for brain segmentation
        total = sample_mask.sum(dim=0)  # [H,W]
        overlap_mask = total > 1.5  # Only resolve significant overlaps
        
        if overlap_mask.any():
            # For overlapping pixels, use weighted combination instead of elimination
            for h in range(H):
                for w in range(W):
                    if overlap_mask[h, w]:
                        # Find overlapping classes at this pixel
                        overlapping_classes = []
                        for c in range(num_classes):
                            if sample_mask[c, h, w] > 0:
                                overlapping_classes.append(c)
                        
                        if len(overlapping_classes) > 1:
                            # Keep the class with highest priority (lowest index)
                            # but reduce intensity instead of eliminating
                            keep_class = min(overlapping_classes)
                            for c in overlapping_classes:
                                if c != keep_class:
                                    sample_mask[c, h, w] *= 0.3  # Reduce instead of eliminate

        masks[b] = sample_mask

    return masks