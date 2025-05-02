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
    Initialize distributed process group with proper error handling
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        # Environment variables to improve stability
        os.environ['NCCL_BLOCKING_WAIT'] = '1'  
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
        os.environ['NCCL_DEBUG'] = 'INFO'
        os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'  # Adjust based on your network interface
        
        # Use a much shorter timeout
        timeout = datetime.timedelta(seconds=60)  # 60 seconds instead of 600
        
        dist.init_process_group(
            backend="nccl", 
            rank=rank, 
            world_size=world_size,
            timeout=timeout
        )
        
        # Set device
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        
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
    Calculate the segmentation metrics with improved dimension handling
    """
    # Print shapes for debugging
    print(f"Debug - pred_masks shape: {pred_masks.shape}, gt_masks shape: {gt_masks.shape}")
    
    # Ensure batch dimensions match
    if pred_masks.shape[0] != gt_masks.shape[0]:
        if gt_masks.shape[0] > pred_masks.shape[0]:
            gt_masks = gt_masks[:pred_masks.shape[0]]
        else:
            # Repeat gt_masks to match batch_size (rarely needed)
            repeats = [1] * len(gt_masks.shape)
            repeats[0] = pred_masks.shape[0] // gt_masks.shape[0] + 1
            gt_masks = gt_masks.repeat(*repeats)[:pred_masks.shape[0]]
    
    pred_binary = (torch.sigmoid(pred_masks) > threshold).float()
    gt_binary = (gt_masks > threshold).float()
    
    try:
        # Calculate Dice coefficient
        intersection = torch.sum(pred_binary * gt_binary, dim=(2,3))
        union = torch.sum(pred_binary, dim=(2,3)) + torch.sum(gt_binary, dim=(2,3))
        
        # Add small value to avoid division by zero
        dice = (2.0 * intersection) / (union + 1e-6)
        
        # Calculate IoU (Jaccard)
        iou = intersection / (union - intersection + 1e-6)
        
        # Average over batch and classes
        mean_dice = dice.mean().item()
        mean_iou = iou.mean().item()
        
        # Calculate per class metrics
        class_dice = dice.mean(dim=0).cpu().numpy()
        
    except Exception as e:
        print(f"Error in metric calculation: {str(e)}")
        print(f"pred_binary shape: {pred_binary.shape}, gt_binary shape: {gt_binary.shape}")
        # Return default values on error
        return {
            'mean_dice': 0.0,
            'mean_iou': 0.0,
            'class_dice': np.array([0.0] * pred_masks.shape[1])
        }
    
    return {
        'mean_dice': mean_dice,
        'mean_iou': mean_iou,
        'class_dice': class_dice
    }

def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch,
                    scaler=None, log_interval=10, grad_clip=1.0, mixed_precision=True):
    """
    Train for one epoch.

    Args:
        mode: The neural network model
        train_loader: DataLoader for training data
        optimizer: Optimizer for updating weights
        criterion: Loss function
        device: Device to train on
        epoch: Current epoch number
        scaler: GradScaler for mixed precision training
        log_interval: How often to log process
        grad_clip: Gradient clipping value
        mixed_precision: Weather to use mixed precision training
    
    Returns:
        dict: Dictionary with the training metrics
    """
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    batch_count = 0

    start_time = time.time()

    # Wrap with tqdm for progress bar
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} [Train]")

    for  batch_idx, (images, targets) in pbar:
        # Move data to device
        images = images.to(device)
        for k in targets:
            if isinstance(targets[k], torch.Tensor):
                targets[k] = targets[k].to(device)
        
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass with mixed precision
        if mixed_precision and scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)

                # Generate ground truth masks for metrics
                gt_masks = criterion._segments_to_masks(
                    targets['segments'],
                    targets['cls'],
                    outputs.shape[2:],
                    targets['original_size']
                )

                # Calculate metrics
                metrics = calculate_metrics(outputs.detach().cpu(), gt_masks)
                total_dice += metrics['mean_dice']
            
            # Backward pass with scaling
            scaler.scale(loss).backward()

            # Clip gradients
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            # Update weights with scaling
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard precision training
            outputs = model(images)
            loss = criterion(outputs, targets)

            # Generate ground truth masks for metrics
            gt_masks = criterion._segments_to_masks(
                targets['segments'],
                targets['cls'],
                outputs.shape[2:],
                targets['original_size']   
            )

            # Calculate metrics
            metrics = calculate_metrics(outputs, gt_masks)
            total_dice += metrics['mean_dice']

            # Backward pass
            loss.backward()

            # Clip gradient
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # Update weights
            optimizer.step()
        
        # Update running statistics
        total_loss += loss.item()
        batch_count += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'dice': f"{metrics['mean_dice']:.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
        })

        # Log to wandb
        if wandb.run is not None and batch_idx % log_interval == 0:
            wandb.log({
                'train/batch_loss': loss.item(),
                'train/batch_dice': metrics['mean_dice'],
                'train/batch_iou': metrics['mean_iou'],
                'train/learning_rate': optimizer.param_groups[0]['lr'],
                'train/batch': epoch * len(train_loader) + batch_idx
            })
        
    # Calculate average metrics
    avg_loss = total_loss / batch_count
    avg_dice = total_dice / batch_count

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    return {
        'loss': avg_loss,
        'dice': avg_dice,
        'time': elapsed_time
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
    total_iou = 0.0
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
                for k in targets:
                    if isinstance(targets[k], torch.Tensor):
                        targets[k] = targets[k].to(device)
                
                # Forward pass with optional mixed precision
                if mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                else:
                    outputs = model(images)
                
                print(f"Model output shape: {outputs.shape}")
                
                # Generate ground truth masks for loss and metrics ON CPU
                # Keep targets on GPU since they're small
                gt_masks = criterion._segments_to_masks(
                    targets['segments'],
                    targets['cls'],
                    outputs.shape[2:],
                    targets['original_size']
                )  # This returns CPU tensor
                
                print(f"Ground truth mask shape: {gt_masks.shape}")
                
                # Ensure batch dimensions match before moving to GPU
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
                
                # Move a copy to GPU for loss calculation only
                gt_masks_gpu = gt_masks.to(device, non_blocking=True)
                
                # Calculate loss
                if mixed_precision:
                    with torch.cuda.amp.autocast():
                        loss = criterion(outputs, gt_masks_gpu)
                else:
                    loss = criterion(outputs, gt_masks_gpu)

                # Free GPU memory for ground truth masks
                del gt_masks_gpu
                torch.cuda.empty_cache()
                
                # Calculate metrics on CPU to save GPU memory
                outputs_cpu = outputs.cpu()
                
                # Free more GPU memory
                if batch_idx < len(val_loader) - 1:  # Keep last batch outputs for visualization
                    del outputs
                    torch.cuda.empty_cache()
                
                # Calculate metrics on CPU
                metrics = calculate_metrics(outputs_cpu, gt_masks)
                
                # Update running statistics
                total_loss += loss.item()
                total_dice += metrics['mean_dice']
                total_iou += metrics['mean_iou']
                batch_count += 1
                
                # Update per class dice scores
                for c in range(len(metrics['class_dice'])):
                    if c not in class_dice_scores:
                        class_dice_scores[c] = []
                    if not np.isnan(metrics['class_dice'][c]):
                        class_dice_scores[c].append(metrics['class_dice'][c])
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'dice': f"{metrics['mean_dice']:.4f}"
                })

                # Save visualization for select batches
                if batch_idx % visualize_every == 0 and batch_idx < max_visualizations * visualize_every:
                    try:
                        for i in range(min(images.size(0), 2)):  # Visualize max 2 images per batch
                            # Ensure index is valid for gt_masks
                            gt_idx = min(i, gt_masks.shape[0] - 1)
                            
                            # Move images back to CPU for visualization
                            img_cpu = images[i].cpu()
                            
                            # Do visualization on CPU
                            visualize_prediction(
                                img_cpu,
                                outputs_cpu[i],
                                gt_masks[gt_idx],
                                class_colors,
                                idx_to_class,
                                f"{batch_idx}_{i}",
                                epoch,
                                output_dir
                            )
                            
                            # Clean up
                            del img_cpu
                    except Exception as vis_error:
                        print(f"Error visualizing prediction: {str(vis_error)}")
                        traceback.print_exc()
                        continue
                
                # Clean up to save memory
                del outputs_cpu, gt_masks
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
    if wandb.run is not None:
        log_dict = {
            'val/loss': avg_loss,
            'val/dice': avg_dice,
            'val/iou': avg_iou,
            'val/epoch': epoch
        }

        # Log per class dice
        for c, dice in avg_class_dice.items():
            class_name = idx_to_class.get(c, f"class_{c}")
            log_dict[f'val/dice_{class_name}'] = dice
        
        wandb.log(log_dict)
    
    return {
        'loss': avg_loss,
        'dice': avg_dice,
        'iou': avg_iou,
        'class_dice': avg_class_dice
    }

def main():
    parser = argparse.ArgumentParser(description="Train brain segmentation model with custom training loop")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--yolo_weights', type=str, required=True, help='Path to YOLOv9 weights')
    parser.add_argument('--img_size', type=int, default=1024, help='Image size for training')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--save_every', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='brain-segmentation', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity name')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--skip_validation', action='store_true', help='Skip validation for debugging')
    parser.add_argument('--use_ddp', action='store_true', help='Force using DDP even on single GPU')
    parser.add_argument('--val_img_size', type=int, default=None, help='Validation image size (default: same as training)')
    parser.add_argument('--val_batch_size', type=int, default=1, help='Validation batch size (default: 1)')
    parser.add_argument('--memory_efficient', action='store_true', help='Use memory efficient operations')
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
    
    val_img_size = args.val_img_size if args.val_img_size else args.img_size
    
    # Create dataset:
    train_dataset = BrainSegmentationDataset(
        data_dir=os.path.join(args.data_dir, "train"),
        transform=get_transforms(is_train=True),
        img_size=args.img_size
    )

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
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

    # Initialize mixed precision scaler if requested
    scaler = GradScaler() if args.mixed_precision else None

    # Load base YOLO model
    base_model = YOLO("/storage/siddhant/work_my/yolo_custom_training_loop/trained_yolov9e-seg.pt")
    print("Loaded base YOLO model: trained_yolov9e-seg.pt")

    # Create model
    model = PatchFusionYOLO(
        base_model=base_model.model,
        num_classes=25,
        learn_fusion_weights=True
    ).to(device)

    if use_distributed:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

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
            ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
            print("Using EMA for model parameters.")
        except ImportError:
            print("torch-ema not found, skipping EMA")
    
    # Resume the checkpoint if specified
    start_epoch = 0
    best_dice = 0.0
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
            if use_distributed:
                train_sampler.set_epoch(epoch)

            if isinstance(train_sampler, DistributedSampler):
                train_sampler.set_epoch(epoch)

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

            # Update EMA parameters
            if ema is not None:
                ema.update()
            
            # Print training metrics
            if is_main:
                print(f"Train loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}, "
                    f"Time: {train_metrics['time']:.2f}s")
            
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
                # Create tensors for basic metrics
                val_tensor = torch.tensor([val_metrics['loss'], val_metrics['dice'], val_metrics['iou']], 
                                      dtype=torch.float32, device=device)
                
                # Broadcast tensors
                dist.broadcast(val_tensor, 0)
                
                # Update metrics from tensor for non-rank-0 processes
                if not is_main:
                    val_metrics['loss'] = val_tensor[0].item()
                    val_metrics['dice'] = val_tensor[1].item()
                    val_metrics['iou'] = val_tensor[2].item()
                
                # Synchronize
                dist.barrier()

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
            is_best = val_metrics['dice'] > best_dice
            best_dice = max(val_metrics['dice'], best_dice)

            # Save checkpoint
            if is_best or (epoch+1) % args.save_every == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'best_dice': best_dice,
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