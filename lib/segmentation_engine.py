"""
Segmentation training and evaluation functions for SAM3 fine-tuning.
"""
import torch
import torch.nn.functional as F
import math
import sys
import random
from typing import Iterable, Optional
from lib import utils


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Dice loss for segmentation.
    
    Args:
        inputs: (B, C, H, W) predictions (logits or probabilities)
        targets: (B, C, H, W) target masks
        smooth: Smoothing constant
    
    Returns:
        Dice loss scalar
    """
    inputs = torch.sigmoid(inputs)
    
    intersection = (inputs * targets).sum()
    union = inputs.sum() + targets.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice


def iou_metric(inputs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute Intersection over Union (IoU) metric.
    
    Args:
        inputs: (B, C, H, W) predictions (logits)
        targets: (B, C, H, W) target masks (0 or 1)
        threshold: Threshold for binarizing predictions
    
    Returns:
        IoU score (0-1)
    """
    inputs = (torch.sigmoid(inputs) > threshold).float()
    
    intersection = (inputs * targets).sum()
    union = (inputs + targets).sum() - intersection
    
    iou = (intersection + 1e-8) / (union + 1e-8)
    return iou.item()


def dice_metric(inputs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute Dice coefficient metric.
    
    Args:
        inputs: (B, C, H, W) predictions (logits)
        targets: (B, C, H, W) target masks (0 or 1)
        threshold: Threshold for binarizing predictions
    
    Returns:
        Dice score (0-1)
    """
    inputs = (torch.sigmoid(inputs) > threshold).float()
    
    intersection = (inputs * targets).sum()
    union = inputs.sum() + targets.sum()
    
    dice = (2.0 * intersection + 1e-8) / (union + 1e-8)
    return dice.item()


def train_one_epoch_segmentation(
    model: torch.nn.Module,
    criterion_bce: torch.nn.Module,
    criterion_dice: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[object] = None,
    amp: bool = True,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
):
    """
    Train for one epoch on segmentation task.
    
    Args:
        model: SAM3 segmentation model
        criterion_bce: Binary Cross-Entropy loss
        criterion_dice: Dice loss
        data_loader: Segmentation data loader
        optimizer: Optimizer
        device: torch device
        epoch: Current epoch
        loss_scaler: Loss scaler for AMP
        max_norm: Gradient clipping norm
        model_ema: Exponential moving average model
        amp: Use automatic mixed precision
        bce_weight: Weight for BCE loss
        dice_weight: Weight for Dice loss
    
    Returns:
        Dict of metrics
    """
    model.train()
    
    random.seed(epoch)
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Segmentation Epoch: [{epoch}]'
    print_freq = 10
    
    for images, masks, boxes in metric_logger.log_every(data_loader, print_freq, header):
        
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        boxes = boxes.to(device, non_blocking=True)
        
        if amp:
            with torch.cuda.amp.autocast():
                outputs, iou_preds = model(images, boxes)
                loss_bce = criterion_bce(outputs, masks)
                loss_dice = criterion_dice(outputs, masks)
                loss = bce_weight * loss_bce + dice_weight * loss_dice
        else:
            outputs, iou_preds = model(images, boxes)
            loss_bce = criterion_bce(outputs, masks)
            loss_dice = criterion_dice(outputs, masks)
            loss = bce_weight * loss_bce + dice_weight * loss_dice
        
        loss_value = loss.item()
        
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)
        
        optimizer.zero_grad()
        
        if amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                       parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            if max_norm is not None and max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
        
        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        
        metric_logger.update(loss=loss_value, loss_bce=loss_bce.item(), loss_dice=loss_dice.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    metric_logger.synchronize_between_processes()
    print("Segmentation training - Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_segmentation(
    data_loader: Iterable,
    model: torch.nn.Module,
    device: torch.device,
    amp: bool = True,
    threshold: float = 0.5,
):
    """
    Evaluate segmentation model.
    
    Args:
        data_loader: Validation data loader
        model: SAM3 segmentation model
        device: torch device
        amp: Use automatic mixed precision
        threshold: Threshold for binarizing predictions
    
    Returns:
        Dict of metrics (dice, iou)
    """
    criterion_bce = torch.nn.BCEWithLogitsLoss()
    criterion_dice = dice_loss
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Segmentation Validation:'
    model.eval()
    
    for images, masks, boxes in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        boxes = boxes.to(device, non_blocking=True)
        
        if amp:
            with torch.cuda.amp.autocast():
                outputs, iou_preds = model(images, boxes)
                loss_bce = criterion_bce(outputs, masks)
                loss_dice = criterion_dice(outputs, masks)
        else:
            outputs, iou_preds = model(images, boxes)
            loss_bce = criterion_bce(outputs, masks)
            loss_dice = criterion_dice(outputs, masks)
        
        batch_size = images.shape[0]
        dice_score = dice_metric(outputs, masks, threshold=threshold)
        iou_score = iou_metric(outputs, masks, threshold=threshold)
        
        metric_logger.update(loss_bce=loss_bce.item())
        metric_logger.update(loss_dice=loss_dice.item())
        metric_logger.update(dice=dice_score)
        metric_logger.update(iou=iou_score)
    
    metric_logger.synchronize_between_processes()
    print(f'* Segmentation Dice {metric_logger.dice.global_avg:.3f} '
          f'IoU {metric_logger.iou.global_avg:.3f}')
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def get_sensitivity_segmentation(
    model: torch.nn.Module,
    criterion_bce: torch.nn.Module,
    criterion_dice: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    amp: bool = True,
    dataset: str = None,
    low_rank_dim: int = 8,
    structured_vector: bool = True,
    exp_name: str = None,
    structured_type: str = 'lora',
    alpha: float = 5.0,
    beta: float = 5.0,
    structured_only: bool = False,
    sensitivity_batch_num: int = 16,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
):
    """
    Get parameter sensitivity for segmentation task (image encoder only).
    
    Args:
        model: SAM3 model with image encoder
        criterion_bce: BCE loss
        criterion_dice: Dice loss
        data_loader: Data loader for sensitivity analysis
        device: torch device
        amp: Use AMP
        dataset: Dataset name for saving
        low_rank_dim: LoRA/Adapter rank
        structured_vector: Whether to structurally tune vectors
        exp_name: Experiment name
        structured_type: 'lora' or 'adapter'
        alpha: Threshold for matrix tuning
        beta: Threshold for vector tuning
        structured_only: Only structural tuning
        sensitivity_batch_num: Number of batches for sensitivity
        bce_weight: Weight for BCE loss
        dice_weight: Weight for Dice loss
    """
    print(f'Ratio for structural tuning matrices: {alpha}, vectors: {beta}')
    
    model.train()
    random.seed(0)
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Getting segmentation sensitivity, batch'
    print_freq = 10
    
    # Accumulate gradients on image encoder only
    grad_dict = {}
    image_encoder = model.get_image_encoder()
    
    for name, param in image_encoder.named_parameters():
        grad_dict[name] = 0.0
    
    for idx, (images, masks, boxes) in enumerate(data_loader):
        
        print(f'===== {header}: {idx}')
        if idx >= sensitivity_batch_num:
            break
        
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        boxes = boxes.to(device, non_blocking=True)
        
        model.zero_grad()
        
        if amp:
            with torch.cuda.amp.autocast():
                outputs, iou_preds = model(images, boxes)
                loss_bce = criterion_bce(outputs, masks)
                loss_dice = criterion_dice(outputs, masks)
                loss = bce_weight * loss_bce + dice_weight * loss_dice
        else:
            outputs, iou_preds = model(images, boxes)
            loss_bce = criterion_bce(outputs, masks)
            loss_dice = criterion_dice(outputs, masks)
            loss = bce_weight * loss_bce + dice_weight * loss_dice
        
        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping")
            continue
        
        loss.backward()
        
        # Accumulate squared gradients from image encoder
        with torch.no_grad():
            for name, param in image_encoder.named_parameters():
                if param.grad is not None:
                    grad_dict[name] += (param.grad ** 2).sum().item()
    
    # Normalize accumulated gradients
    for key in grad_dict:
        grad_dict[key] /= (idx + 1)
    
    print('Segmentation sensitivity analysis complete')
    print('Grad norms (image encoder):', grad_dict)
    
    metric_logger.synchronize_between_processes()
    return {k: v for k, v in grad_dict.items()}
