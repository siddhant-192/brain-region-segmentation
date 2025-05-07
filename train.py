########## train.py ##########

import os
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from ultralytics import YOLO
import cv2
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import datetime
import sys
import traceback

# Import the custom model classes
from model import BrainSegmentationDataset, BrainSegmentationLoss, PatchFusionYOLO

def ddp_setup():
    """
    Initialize distributed process group with improved settings
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        # Better environment variables for NCCL stability
        os.environ['NCCL_BLOCKING_WAIT'] = '1'  
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
        os.environ['NCCL_DEBUG'] = 'INFO'  # More detailed logs for debugging
        os.environ['NCCL_IB_TIMEOUT'] = '30'  # Higher timeout in seconds
        os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
        
        # Use a longer timeout for initialization
        timeout = datetime.timedelta(minutes=10)  # 10 mins
        
        # Try multiple initialization attempts
        max_attempts = 1
        for attempt in range(max_attempts):
            try:
                dist.init_process_group(
                    backend="nccl", 
                    rank=rank, 
                    world_size=world_size,
                    timeout=timeout
                )
                break
            except Exception as e:
                if attempt < max_attempts - 1:
                    print(f"Init attempt {attempt+1} failed: {e}. Retrying...")
                    time.sleep(10)  # Wait before retry
                else:
                    raise

        # Set device
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        print(f"Process {rank}/{world_size} using local GPU {local_rank}")
        
        return rank, world_size
    else:
        # Fallback to using single GPU
        return 0, 1

# Setup data augmentation
def get_transforms(is_train=True):
    """
    Define image transformations for training and validation
    """
    def transform(img, segments):
        # Apply augmentation only during training
        if is_train:
            # Random horizontal flip
            if np.random.random() > 0.5:
                img = cv2.flip(img, 1) # 1 for horizontal flip

                # Also flip segments
                for i in range(len(segments)):
                    points = np.array(segments[i]).reshape(-1, 2)
                    # Flip x coordinates
                    points[:, 0] = img.shape[1] - points[:, 0]
                    segments[i] = points.flatten().tolist()
            
            # Random vertical flip
            if np.random.random() > 0.5:
                img = cv2.flip(img, 0) # 0 for vertical flip
                
                # Flip segments
                for i in range(len(segments)):
                    points = np.array(segments[i]).reshape(-1,2)
                    # Flip y coordinates
                    points[:, 1] = img.shape[0] - points[:, 1]
                    segments[i] = points.flatten().tolist()
            
            # Random brightness/contrast
            if np.random.random() > 0.5:
                alpha = 1.0 + np.random.uniform(-0.2, 0.2) # contrast
                beta = np.random.uniform(-20, 20) # brightness
                img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            
            # Random gaussian blur
            if np.random.random() > 0.8:
                img = cv2.GaussianBlur(img, (5, 5), 0)
        
        return img, segments

    return transform

# Fixed visualize_prediction function with correct indexing

def visualize_prediction(image, pred_mask, gt_mask, class_colors, idx_to_class, image_idx, epoch, output_dir):
    """
    Visualize and save prediction results
    
    Args:
        image: Input image tensor (C, H, W)
        pred_mask: Predicted mask tensor (C, H, W)
        gt_mask: Ground truth mask tensor (C, H, W)
        class_colors: Dict of class colors
        idx_to_class: Dict mapping class indices to names
        image_idx: Index of the image in the batch
        epoch: Current epoch number
        output_dir: Directory to save the outputs
    """
    # Create visualization directory if it doesn't exist
    viz_dir = os.path.join(output_dir, f"viz_epoch_{epoch}")
    os.makedirs(viz_dir, exist_ok=True)

    # Convert tensors to numpy arrays
    image_np = image.permute(1, 2, 0).cpu().numpy()
    pred_mask_np = torch.sigmoid(pred_mask).cpu().numpy() > 0.5
    gt_mask_np = gt_mask.cpu().numpy() > 0.5

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(36, 12))

    # Plot the original image
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Plot ground truth mask overlay
    overlay = image_np.copy()
    for class_idx in range(gt_mask_np.shape[0]):
        if np.any(gt_mask_np[class_idx]):
            color = np.array([int(class_colors[class_idx][i:i+2], 16) / 255 for i in (1, 3, 5)])
            mask = gt_mask_np[class_idx]
            overlay = np.where(
                np.expand_dims(mask, axis=2),
                0.7 * color.reshape(1, 1, 3) + 0.3 * overlay,
                overlay
            )
    
    axes[1].imshow(overlay)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # Plot prediction mask overlay
    overlay = image_np.copy()
    for class_idx in range(pred_mask_np.shape[0]):
        if np.any(pred_mask_np[class_idx]):
            color = np.array([int(class_colors[class_idx][i:i+2], 16) / 255 for i in (1, 3, 5)])
            mask = pred_mask_np[class_idx]
            overlay = np.where(
                np.expand_dims(mask, axis=2),
                0.7 * color.reshape(1, 1, 3) + 0.3 * overlay,
                overlay
            )
    
    # FIX: Correct indexing for the third subplot
    axes[2].imshow(overlay)
    axes[2].set_title("Prediction")  # Corrected from axes[3]
    axes[2].axis("off")             # Corrected from axes[4]

    # Add legend
    classes_present = set()
    for c in range(gt_mask_np.shape[0]):
        if np.any(gt_mask_np[c]) or np.any(pred_mask_np[c]):
            classes_present.add(c)
    
    legend_elements = []
    for c in classes_present:
        color = np.array([int(class_colors[c][i:i+2], 16) / 255 for i in (1, 3, 5)])
        patch = plt.Rectangle((0, 0), 1, 1, fc=color)
        legend_elements.append((patch, f"{c}: {idx_to_class[c]}"))
    
    if legend_elements:
        patches, labels = zip(*legend_elements)
        fig.legend(patches, labels, loc='lower center', ncol=min(5, len(legend_elements)))
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"img_{image_idx}.png"))
    plt.close(fig)

def calculate_metrics(pred_masks, gt_masks, threshold=0.5):
    """
    Calculate the segmentation metrics aligned with MONAI DiceLoss implementation

    This metic calculation has the same implementation as the BrainSgmentationLoss class for uniformity and consitency.
    """
    # Ensure inputs are on the same device
    device = pred_masks.device
    
    # Print shapes for debugging
    # print(f"Debug - pred_masks shape: {pred_masks.shape}, gt_masks shape: {gt_masks.shape}")
    
    # Ensure batch dimensions match
    if pred_masks.shape[0] != gt_masks.shape[0]:
        if gt_masks.shape[0] > pred_masks.shape[0]:
            gt_masks = gt_masks[:pred_masks.shape[0]]
        else:
            # Repeat gt_masks to match batch_size (rarely needed)
            repeats = [1] * len(gt_masks.shape)
            repeats[0] = pred_masks.shape[0] // gt_masks.shape[0] + 1
            gt_masks = gt_masks.repeat(*repeats)[:pred_masks.shape[0]]
    
    try:
        # Convert predictions to probabilities using sigmoid (soft dice)
        pred_prob = torch.sigmoid(pred_masks)
        
        # Square predictions as in MONAI DiceLoss(squared_pred=True)
        pred_prob_squared = pred_prob ** 2
        
        # Calculate soft dice coefficient with squared predictions
        # Following MONAI implementation pattern
        intersection = torch.sum(pred_prob_squared * gt_masks, dim=(2,3))
        pred_sum = torch.sum(pred_prob_squared, dim=(2,3))
        gt_sum = torch.sum(gt_masks, dim=(2,3))
        
        # Add smooth_nr=1.0 in numerator as in MONAI implementation
        dice = (2.0 * intersection + 1.0) / (pred_sum + gt_sum + 1e-6)
        
        # Calculate IoU (Jaccard) using the same soft approach
        iou = intersection / (pred_sum + gt_sum - intersection + 1e-6)
        
        # Average over batch and classes
        mean_dice = dice.mean().detach().item()  # Add .detach() before .item() # code changes
        mean_iou = iou.mean().detach().item()    # Add .detach() before .item() # code changes
        
        # Calculate per class metrics by averaging across the batch dimension
        per_class_dice = dice.mean(dim=0)  # Shape: [num_classes]
        class_dice = per_class_dice.detach().cpu().numpy()  # Add .detach() before .cpu().numpy() # code changes
        
    except Exception as e:
        print(f"Error in metric calculation: {str(e)}")
        print(f"pred_masks shape: {pred_masks.shape}, gt_masks shape: {gt_masks.shape}")
        # Return default values on error
        return {
            'mean_dice': 0.0,
            'mean_iou': 0.0,
            'class_dice': np.array([0.0] * pred_masks.shape[1])
        }
    
    return {
        'mean_dice': mean_dice,  # Soft dice aligned with loss
        'mean_iou': mean_iou,    # Soft IoU
        'class_dice': class_dice  # Per-class soft dice
    }

def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch,
                    scaler=None, log_interval=10, grad_clip=1.0, mixed_precision=True,
                    gradient_accumulation_steps=1):
    """
    Train for one epoch with proper gradient accumulation and mixed precision.
    """
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    batch_count = 0
    
    # Clear gradients at the beginning
    optimizer.zero_grad()
    
    start_time = time.time()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} [Train]")

    dataloader_timeout = 300  # 5 minutes

    for batch_idx, (images, targets) in pbar:
        try:
            # Move data to device
            images = images.to(device)
            
            for i in range(len(targets)):
                for k in targets[i]:
                    if isinstance(targets[i][k], torch.Tensor):
                        targets[i][k] = targets[i][k].to(device)
            
            # Forward pass with mixed precision
            if mixed_precision and scaler is not None:
                with autocast():
                    outputs = model(images)
                    # Normalize loss by gradient accumulation steps
                    loss = criterion(outputs, targets) / gradient_accumulation_steps
                    
                    # Calculate metrics (doesn't need normalization)
                    target_segments = [t['segments'] for t in targets]
                    target_classes = [t['cls'] for t in targets]
                    target_sizes = [t['original_size'] for t in targets]
                    
                    gt_masks = criterion._segments_to_masks(
                        target_segments,
                        target_classes,
                        outputs.shape[2:],
                        target_sizes
                    ).to(device)
                    
                    metrics = calculate_metrics(outputs, gt_masks)
                    total_dice += metrics['mean_dice']
                    total_iou += metrics['mean_iou']
                
                # Backward pass with scaling (accumulate gradients)
                scaler.scale(loss).backward()
                
                # Only update weights after accumulating gradients for specified steps
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    # Now it's time to update - only unscale once right before the clip and step
                    if grad_clip > 0:
                        # Unscale gradients once before clipping
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    
                    # Update weights
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Zero gradients after optimizer step
                    optimizer.zero_grad()
            else:
                # Standard precision training
                outputs = model(images)
                loss = criterion(outputs, targets) / gradient_accumulation_steps
                
                # Calculate metrics
                target_segments = [t['segments'] for t in targets]
                target_classes = [t['cls'] for t in targets]
                target_sizes = [t['original_size'] for t in targets]
                
                gt_masks = criterion._segments_to_masks(
                    target_segments,
                    target_classes,
                    outputs.shape[2:],
                    target_sizes
                ).to(device)
                
                metrics = calculate_metrics(outputs, gt_masks)
                total_dice += metrics['mean_dice']
                total_iou += metrics['mean_iou']
                
                # Backward pass (accumulate gradients)
                loss.backward()
                
                # Only update weights after accumulating gradients
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    # Clip gradients if needed
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    
                    # Update weights
                    optimizer.step()
                    
                    # Zero gradients after optimizer step
                    optimizer.zero_grad()
            
            # Calculate full loss (not normalized) for reporting
            full_loss = loss.item() * gradient_accumulation_steps
            total_loss += full_loss
            batch_count += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{full_loss:.4f}",
                'dice': f"{metrics['mean_dice']:.4f}",
                'iou': f"{metrics['mean_iou']:.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        
    # Calculate average metrics
    avg_loss = total_loss / batch_count
    avg_dice = total_dice / batch_count
    avg_iou = total_iou / batch_count
    
    return {
        'loss': avg_loss,
        'dice': avg_dice,
        'iou': avg_iou,
        'time': time.time() - start_time
    }

def validate(model, val_loader, criterion, device, epoch, output_dir, idx_to_class, class_colors,
             visualize_every=10, max_visualizations=5, mixed_precision=False):
    """
    Run validation with improved memory management
    
    Args:
        model: The neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run validation
        epoch: Current epoch number
        output_dir: Directory to save visualizations
        idx_to_class: Dictionary mapping class indices to class names
        class_colors: Dictionary mapping class indices to colors
        visualize_every: How often to save visualizations
        max_visualizations: Maximum number of visualizations to save
        mixed_precision: Whether to use mixed precision
    
    Returns:
        dict: Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0  # Added IoU tracking
    batch_count = 0

    # Dictionary to store per class dice scores
    class_dice_scores = {}

    # Wrap with tqdm for progress bar
    pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch} [Val]")

    # Initialize mixed precision scaler if needed
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    
    with torch.no_grad():
        for batch_idx, (images, targets) in pbar:
            try:
                # Clear cache before processing each batch
                torch.cuda.empty_cache()
                
                # Move data to device
                images = images.to(device)
                # Updated code to handle targets as a list of dictionaries
                for i in range(len(targets)):
                    for k in targets[i]:
                        if isinstance(targets[i][k], torch.Tensor):
                            targets[i][k] = targets[i][k].to(device)
                
                # Forward pass with optional mixed precision
                if mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                else:
                    outputs = model(images)
                
                # print(f"Model output shape: {outputs.shape}")
                
                # Collect segments data from the list of targets
                target_segments = [t['segments'] for t in targets]
                target_classes = [t['cls'] for t in targets]
                target_sizes = [t['original_size'] for t in targets]

                # Generate ground truth masks for loss and metrics ON GPU
                gt_masks = criterion._segments_to_masks(
                    target_segments,
                    target_classes,
                    outputs.shape[2:],
                    target_sizes
                ).to(device)  # Move gt_masks to GPU if not already
                
                # print(f"Ground truth mask shape: {gt_masks.shape}")
                
                # Ensure batch dimensions match before calculating metrics
                if gt_masks.shape[0] != outputs.shape[0]:
                    print(f"Fixing batch dimension mismatch: gt_masks={gt_masks.shape[0]}, outputs={outputs.shape[0]}")
                    
                    # Take the first batch_size items from gt_masks
                    if gt_masks.shape[0] > outputs.shape[0]:
                        gt_masks = gt_masks[:outputs.shape[0]]
                    
                    # Or repeat gt_masks to match batch_size
                    elif gt_masks.shape[0] < outputs.shape[0]:
                        repeats = [1] * len(gt_masks.shape)
                        repeats[0] = outputs.shape[0] // gt_masks.shape[0] + 1
                        gt_masks = gt_masks.repeat(*repeats)[:outputs.shape[0]]
                
                # Calculate loss on GPU
                if mixed_precision:
                    with torch.cuda.amp.autocast():
                        loss = criterion(outputs, gt_masks)
                else:
                    loss = criterion(outputs, gt_masks)

                # Calculate metrics on GPU (no CPU transfer)
                metrics = calculate_metrics(outputs, gt_masks)
                
                # Update running statistics
                total_loss += loss.item()
                total_dice += metrics['mean_dice']
                total_iou += metrics['mean_iou']  # Track IoU
                batch_count += 1
                
                # Update per class dice scores
                for c in range(len(metrics['class_dice'])):
                    if c not in class_dice_scores:
                        class_dice_scores[c] = []
                    if not np.isnan(metrics['class_dice'][c]):
                        class_dice_scores[c].append(metrics['class_dice'][c])
                
                # Update progress bar with IoU included
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'dice': f"{metrics['mean_dice']:.4f}",
                    'iou': f"{metrics['mean_iou']:.4f}"  # Added IoU to progress bar
                })

                # Save visualization for select batches
                if batch_idx % visualize_every == 0 and batch_idx < max_visualizations * visualize_every:
                    try:
                        for i in range(min(images.size(0), 2)):  # Visualize max 2 images per batch
                            # Ensure index is valid for gt_masks
                            gt_idx = min(i, gt_masks.shape[0] - 1)
                            
                            # Move tensors to CPU only for visualization
                            img_cpu = images[i].cpu()
                            outputs_cpu = outputs[i].cpu()
                            gt_masks_cpu = gt_masks[gt_idx].cpu()
                            
                            # Do visualization on CPU
                            visualize_prediction(
                                img_cpu,
                                outputs_cpu,
                                gt_masks_cpu,
                                class_colors,
                                idx_to_class,
                                f"{batch_idx}_{i}",
                                epoch,
                                output_dir
                            )
                            
                            # Clean up CPU tensors
                            del img_cpu, outputs_cpu, gt_masks_cpu
                    except Exception as vis_error:
                        print(f"Error visualizing prediction: {str(vis_error)}")
                        traceback.print_exc()
                        continue
                
                # Clean up to save memory
                if batch_idx < len(val_loader) - 1:  # Keep last batch outputs for potential debugging
                    del outputs, gt_masks
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing batch {batch_idx} during validation: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Clear CUDA cache on error
                torch.cuda.empty_cache()
                continue
    
    # Final cache clear
    torch.cuda.empty_cache()
    
    # Handle case where no batches were successfully processed
    if batch_count == 0:
        print("Warning: No validation batches were successfully processed")
        return {
            'loss': float('inf'),
            'dice': 0.0,
            'iou': 0.0,
            'class_dice': {}
        }
            
    # Calculate average metrics
    avg_loss = total_loss / batch_count
    avg_dice = total_dice / batch_count
    avg_iou = total_iou / batch_count

    # Calculate average per class dice
    avg_class_dice = {c: np.mean(scores) for c, scores in class_dice_scores.items() if scores}

    # Log validation results
    # if wandb.run is not None:
    #     log_dict = {
    #         'val/loss': avg_loss,
    #         'val/dice': avg_dice,
    #         'val/iou': avg_iou,
    #         'val/epoch': epoch
    #     }

    #     # Log per class dice
    #     for c, dice in avg_class_dice.items():
    #         class_name = idx_to_class.get(c, f"class_{c}")
    #         log_dict[f'val/dice_{class_name}'] = dice
        
    #     wandb.log(log_dict)
    
    return {
        'loss': avg_loss,
        'dice': avg_dice,
        'iou': avg_iou,
        'class_dice': avg_class_dice
    }

def brain_segmentation_collate_fn(batch):
    """
    Custom collate function for brain segmentation data with variable-sized segments.
    
    Args:
        batch: A list of tuples (image, target) from the dataset
    
    Returns:
        A tuple of (batched_images, list_of_targets)
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    # Stack images into a batch tensor
    images = torch.stack(images, dim=0)
    
    # Keep targets as a list of dictionaries
    # Do not attempt to collate the segments
    
    return images, targets

def main():
    parser = argparse.ArgumentParser(description="Train brain segmentation model with custom training loop")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--yolo_weights', type=str, required=True, help='Path to YOLOv9 weights')
    parser.add_argument('--img_size', type=int, default=4096, help='Image size for training')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='brain-segmentation', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity name')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--skip_validation', action='store_true', help='Skip validation for debugging')
    parser.add_argument('--use_ddp', action='store_true', help='Force using DDP even on single GPU')
    parser.add_argument('--val_img_size', type=int, default=4096, help='Validation image size (default: same as training)')
    parser.add_argument('--val_batch_size', type=int, default=1, help='Validation batch size (default: 1)')
    parser.add_argument('--memory_efficient', action='store_true', help='Use memory efficient operations')
    parser.add_argument('--track_memory', action='store_true', help='Track GPU memory usage during training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of updates steps to accumulate before performing a backward/update pass')
    parser.add_argument('--find_unused_parameters', action='store_true', help='Enable find_unused_parameters in DDP')
    parser.add_argument('--sync_bn', action='store_true', help='Use synchronized batch normalization')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ', device)

    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"PatchFusionYOLO-{time.strftime('%Y%m%d_%H%M%S')}",
        )
    
    # Only run ddp_setup if using torchrun or explicitly requested
    use_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    
    if use_distributed or args.use_ddp:
        rank, world_size = ddp_setup()
        is_main = rank == 0  # only rank-0 prints/saves/logs
        device = torch.device(f'cuda:{rank}')
    else:
        # Single GPU mode
        rank = 0
        world_size = 1
        is_main = True
    
    # Create dataset:
    train_dataset = BrainSegmentationDataset(
        data_dir=os.path.join(args.data_dir, "train"),
        transform=get_transforms(is_train=True),
        img_size=args.img_size
    )

    val_img_size = args.val_img_size if args.val_img_size else args.img_size

    val_dataset = BrainSegmentationDataset(
        data_dir=os.path.join(args.data_dir, 'val'),
        transform=get_transforms(is_train=False),
        img_size=val_img_size 
    )

    # Create data loaders
    if use_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )

        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
    else:
        # Use regular samplers for non-distributed training
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size // world_size or 1,
        sampler=train_sampler,
        num_workers=0, #args.num_workers,
        pin_memory=True,
        collate_fn=brain_segmentation_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=brain_segmentation_collate_fn,
    )

    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

    # Initialize mixed precision scaler if requested
    scaler = GradScaler() if args.mixed_precision else None

    # Load base YOLO model
    base_model = YOLO(args.yolo_weights)
    print(f"Loaded base YOLO model: {args.yolo_weights}")

    # Create model
    model = PatchFusionYOLO(
        base_model=base_model.model,
        num_classes=25,
        learn_fusion_weights=True
    ).to(device)

    # Sync batch norm for better stability in distributed training
    if args.sync_bn and use_distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if use_distributed:
        model = DDP(
            model, 
            device_ids=[rank], 
            output_device=rank, 
            find_unused_parameters=args.find_unused_parameters,
            broadcast_buffers=True,  # Ensure buffers are synced
            bucket_cap_mb=25         # Smaller bucket size for more stable comms
        )

    # Criterion loss function
    criterion = BrainSegmentationLoss(
        num_classes=25,
        dice_weight=1.0,
        focal_weight=0.5,
        boundary_weight=0.3
    )

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Create learning rate scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # Exponential moving average for model parameters
    ema = None
    if torch.cuda.is_available():
        try:
            from torch_ema import ExponentialMovingAverage
            ema = ExponentialMovingAverage(model.parameters(), decay=0.9999) # High decay value is a common choice in medical imaging applications where stability and reliability are more important than rapid adaptation to new data.
            print("Using EMA for model parameters.")
        except ImportError:
            print("torch-ema not found, skipping EMA")
    
    # Resume the checkpoint if specified
    start_epoch = 0
    best_dice = 0.0
    best_iou = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint.get('epoch', 0)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'ema' in checkpoint and ema is not None:
                ema.load_state_dict(checkpoint['ema'])
            best_dice = checkpoint.get('best_dice', 0.0)
            best_iou = checkpoint.get('best_iou', 0.0)
            print(f"Loaded checkpoint from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {args.resume}")
    
    # Wrap everything in try-finally to ensure proper cleanup
    try:
        # Training loop
        print(f"Starting training from epoch {start_epoch + 1} to {args.num_epochs}")
        for epoch in range(start_epoch, args.num_epochs):
            print(f"\nEpoch {epoch+1}/{args.num_epochs}")

            # Set epoch for distributed sampler
            if isinstance(train_sampler, DistributedSampler):
                try:
                    train_sampler.set_epoch(epoch)
                except Exception as e:
                    print(f"Error setting epoch for sampler: {str(e)}")
            
            # Track memory before training if requested
            if args.track_memory and is_main and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                start_mem = torch.cuda.memory_allocated() / 1e9  # GB
                print(f"Memory before training: {start_mem:.2f}GB")

            # Train for one epoch
            train_metrics = train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epoch=epoch,
                scaler=scaler,
                grad_clip=args.grad_clip,
                mixed_precision=args.mixed_precision
            )

            # Track memory after training if requested
            if args.track_memory and is_main and torch.cuda.is_available():
                current_mem = torch.cuda.memory_allocated() / 1e9  # GB
                peak_mem = torch.cuda.max_memory_allocated() / 1e9  # GB
                print(f"Training memory stats - Current: {current_mem:.2f}GB, Peak: {peak_mem:.2f}GB")

            # Update EMA parameters
            if ema is not None:
                ema.update()
            
            # Print training metrics
            if is_main:
                print(f"Train loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}, "
                    f"Time: {train_metrics['time']:.2f}s")
            
            # Initialize val_metrics with safe defaults BEFORE validation
            val_metrics = {
                'loss': float('inf'),
                'dice': 0.0,
                'iou': 0.0,
                'class_dice': {}
            }
            
            if not args.skip_validation:
            # Add barriers for synchronization
                if world_size > 1:
                    try:
                        dist.barrier()  # Synchronize before validation
                    except Exception as e:
                        print(f"Error in barrier before validation: {e}")

                # Validation
                try:
                    if ema is not None:
                        with ema.average_parameters():
                            if is_main:
                                val_metrics = validate(
                                    model=model,
                                    val_loader=val_loader,
                                    criterion=criterion,
                                    device=device,
                                    epoch=epoch,
                                    output_dir=args.output_dir,
                                    idx_to_class=train_dataset.idx_to_class,
                                    class_colors=train_dataset.colors,
                                    mixed_precision=args.mixed_precision
                                )
                    else:
                        if is_main:
                            val_metrics = validate(
                                model=model,
                                val_loader=val_loader,
                                criterion=criterion,
                                device=device,
                                epoch=epoch,
                                output_dir=args.output_dir,
                                idx_to_class=train_dataset.idx_to_class,
                                class_colors=train_dataset.colors
                            )

                    # Track memory after validation if requested
                    if args.track_memory and is_main and torch.cuda.is_available():
                        current_mem = torch.cuda.memory_allocated() / 1e9  # GB
                        peak_mem = torch.cuda.max_memory_allocated() / 1e9  # GB
                        print(f"Validation memory stats - Current: {current_mem:.2f}GB, Peak: {peak_mem:.2f}GB")
                        torch.cuda.reset_peak_memory_stats()  # Reset for next epoch
                except Exception as e:
                    print(f"Rank {rank}: Error during validation: {str(e)}")
                    traceback.print_exc()
                    # Set default metrics
                    val_metrics = {
                        'loss': float('inf'),
                        'dice': 0.0,
                        'iou': 0.0,
                        'class_dice': {}
                    }

                # Ensure all processes reach this point regardless of validation success
                if world_size > 1:
                    try:
                        dist.barrier()  # Synchronize after validation
                    except Exception as e:
                        print(f"Error in barrier after validation: {e}")
            
            else:
                print("Skipping validation as requested")
                val_metrics = {
                    'loss': 0.0,
                    'dice': 0.0,
                    'iou': 0.0,
                    'class_dice': {}
                }

            # Broadcast validation metrics to all processes
            if world_size > 1:
                try:
                    # Use a shorter timeout
                    timeout_seconds = 60
                    
                    # Create smaller tensors for basic metrics
                    val_tensor = torch.tensor([val_metrics['loss'], val_metrics['dice']], 
                                        dtype=torch.float32, device=device)
                    
                    # Execute broadcast with timeout
                    req = dist.broadcast(val_tensor, 0, async_op=True)
                    
                    # Wait with timeout
                    start_time = time.time()
                    while not req.is_completed():
                        if time.time() - start_time > timeout_seconds:
                            print(f"Broadcast timeout on rank {rank}")
                            break
                        time.sleep(0.1)
                    
                except Exception as e:
                    print(f"Error in broadcast: {e}")

            # Now all ranks can safely access val_metrics
            print(f"Val Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")
            
            # Print per-class dice scores for validation (only on main process)
            if is_main and 'class_dice' in val_metrics and val_metrics['class_dice']:
                print("Per-class Dice scores:")
                for class_idx, dice in sorted(val_metrics['class_dice'].items()):
                    class_name = train_dataset.idx_to_class.get(class_idx, f"Unknown_{class_idx}")
                    print(f"  {class_name}: {dice:.4f}")
            
            # Update learning rate scheduler
            lr_scheduler.step(val_metrics['loss'])

            # Save checkpoint
            is_best_dice = val_metrics['dice'] > best_dice
            is_best_iou = val_metrics['iou'] > best_iou
            is_best = is_best_dice or is_best_iou  # Save if either metric improves

            best_dice = max(val_metrics['dice'], best_dice)
            best_iou = max(val_metrics['iou'], best_iou)

            # Update print statement
            if is_best:
                selection_metric = "Dice" if is_best_dice else "IoU"
                print(f"New best model found! Best {selection_metric} score: {best_dice:.4f}, Best IoU: {best_iou:.4f}")

            # Save checkpoint
            if is_best or (epoch+1) % args.save_every == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'best_dice': best_dice,
                    'best_iou': best_iou,
                    'val_metrics': val_metrics,
                }

                if ema is not None:
                    checkpoint['ema'] = ema.state_dict()
                
                # Save periodic checkpoint
                if is_main and ((epoch + 1) % args.save_every == 0 or is_best):
                    checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pth")
                    torch.save(checkpoint, checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")
                
                # Save best model
                if is_best:
                    best_model_path = os.path.join(args.output_dir, "best_model.pth")
                    torch.save(checkpoint, best_model_path)
                    print(f"Saved best model to {best_model_path} with the Dice score: {best_dice:.4f}")

                    # Save best model with ema weights if available
                    if ema is not None:
                        with ema.average_parameters():
                            best_ema_model_path = os.path.join(args.output_dir, "best_model_ema.pth")
                            torch.save({
                                'epoch': epoch + 1,
                                'model_state_dict': model.state_dict(),
                                'best_dice': best_dice,
                            }, best_ema_model_path)
                            print(f"Saved best EMA model to {best_ema_model_path}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
                    torch.cuda.reset_accumulated_memory_stats()
                print(f"Memory after epoch {epoch+1}: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                
                # On CUDA OOM, try to recover
                import gc
                gc.collect()    
        
        print(f"Training completed. Best validation Dice score: {best_dice:.4f}")

        # Finish wandb run
        if wandb.run is not None:
            wandb.finish()
        
    finally:
        # Always clean up distributed process group
        if world_size > 1:
            try:
                dist.destroy_process_group()
            except Exception as e:
                print(f"Error destroying process group: {str(e)}")

if __name__ == "__main__":
    main()