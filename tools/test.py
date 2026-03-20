"""
Model validation script
This script validates a model on the validation set, supporting distributed training and EMA evaluation.
"""

import os
import torch
import torch.nn as nn
import logging
import time
from torch.nn.parallel import DistributedDataParallel as DDP

from lib.models.builder import build_model
from lib.dataset.builder import build_dataloader
from lib.utils.args import parse_args
from lib.utils.dist_utils import init_dist, init_logger
from lib.utils.misc import accuracy, AverageMeter, CheckpointManager
from lib.utils.model_ema import ModelEMA
from lib.utils.measure import get_params, get_flops

# Enable cuDNN benchmark for faster convolutions
torch.backends.cudnn.benchmark = True

# Initialize logger
logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    """
    Main function for model validation
    """
    # Parse command line arguments
    args, args_text = parse_args()
    args.exp_dir = f'experiments/{args.experiment}'

    # Initialize distributed training
    init_dist(args)
    init_logger(args)

    # Build validation data loader
    _, val_dataset, _, val_loader = build_dataloader(args)

    # Build model
    model = build_model(args, args.model)

    # Log model information
    logger.info(f"Model architecture:\n{model}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Parameters: {get_params(model) / 1e6:.3f}M")
    logger.info(f"FLOPs: {get_flops(model, input_shape=args.input_shape) / 1e9:.3f}G")

    # Move model to GPU
    model = model.cuda()

    # Wrap model with DDP for distributed training
    if args.distributed:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            find_unused_parameters=args.find_unused_parameters
        )

    # Define loss function
    criterion = nn.CrossEntropyLoss().cuda()

    # Initialize Model EMA if enabled
    if args.model_ema:
        model_ema = ModelEMA(model, decay=args.model_ema_decay)
    else:
        model_ema = None

    # Initialize checkpoint manager
    ckpt_manager = CheckpointManager(
        model=model,
        ema_model=model_ema,
        optimizer=None,
        scheduler=None,
        save_dir=args.exp_dir,
        rank=args.rank
    )

    # Load checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = ckpt_manager.load(args.resume)
        logger.info(f"Loaded checkpoint from: {args.resume}")
        logger.info(f"Starting from epoch: {start_epoch}")

    # Validate base model
    logger.info("Validating base model...")
    base_metrics = validate_model(
        args=args,
        epoch=start_epoch,
        model=model,
        data_loader=val_loader,
        criterion=criterion,
        log_suffix=""
    )
    log_validation_results(base_metrics, prefix="Base Model")

    # Validate EMA model if available
    if model_ema is not None:
        logger.info("Validating EMA model...")
        ema_metrics = validate_model(
            args=args,
            epoch=start_epoch,
            model=model_ema.module,
            data_loader=val_loader,
            criterion=criterion,
            log_suffix="(EMA)"
        )
        log_validation_results(ema_metrics, prefix="EMA Model")


def validate_model(args, epoch, model, data_loader, criterion, log_suffix=""):
    """
    Validate model performance on validation set

    Args:
        args: Command line arguments
        epoch: Current epoch number
        model: Model to validate
        data_loader: Validation data loader
        criterion: Loss function
        log_suffix: Suffix for logging

    Returns:
        Dictionary containing validation metrics
    """
    # Initialize metrics trackers
    batch_time = AverageMeter(dist=args.distributed)
    data_time = AverageMeter(dist=args.distributed)
    losses = AverageMeter(dist=args.distributed)
    top1_acc = AverageMeter(dist=args.distributed)
    top5_acc = AverageMeter(dist=args.distributed)

    # Switch to evaluation mode
    model.eval()

    end_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            # Measure data loading time
            data_time.update(time.time() - end_time)

            # Move data to GPU
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Compute accuracy
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

            # Update metrics
            batch_size = inputs.size(0)
            losses.update(loss.item(), batch_size)
            top1_acc.update(acc1.item(), batch_size)
            top5_acc.update(acc5.item(), batch_size)

            # Measure batch processing time
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            # Log progress periodically
            if batch_idx % args.log_interval == 0 or batch_idx == len(data_loader) - 1:
                logger.info(
                    f"Validation{log_suffix}: "
                    f"Epoch: {epoch} [{batch_idx:4d}/{len(data_loader):4d}] "
                    f"Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s) "
                    f"Data: {data_time.val:.3f}s ({data_time.avg:.3f}s) "
                    f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                    f"Top-1: {top1_acc.val:.2f}% ({top1_acc.avg:.2f}%) "
                    f"Top-5: {top5_acc.val:.2f}% ({top5_acc.avg:.2f}%)"
                )

    # Return validation metrics
    return {
        'loss': losses.avg,
        'top1': top1_acc.avg,
        'top5': top5_acc.avg,
        'batch_time': batch_time.avg,
        'data_time': data_time.avg
    }


def log_validation_results(metrics, prefix=""):
    """
    Log validation results in a formatted way

    Args:
        metrics: Dictionary containing validation metrics
        prefix: Prefix for log message
    """
    logger.info(f"{prefix} Results:")
    logger.info(f"  Loss: {metrics['loss']:.4f}")
    logger.info(f"  Top-1 Accuracy: {metrics['top1']:.2f}%")
    logger.info(f"  Top-5 Accuracy: {metrics['top5']:.2f}%")
    logger.info(f"  Average Batch Time: {metrics['batch_time']:.3f}s")
    logger.info(f"  Average Data Time: {metrics['data_time']:.3f}s")


if __name__ == '__main__':
    main()