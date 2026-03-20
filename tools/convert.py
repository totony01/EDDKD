"""
Model conversion script for DBB/DyRep deployment
This script converts training-time models with DBB/DyRep branches to deployment-time models.
"""

import os
import torch
import torch.nn as nn
import logging
import time
from torch.nn.parallel import DistributedDataParallel as DDP

from lib.models.builder import build_model
from lib.models.loss import CrossEntropyLabelSmooth
from lib.models.utils.dbb.dbb_block import DiverseBranchBlock
from lib.dataset.builder import build_dataloader
from lib.utils.args import parse_args
from lib.utils.dist_utils import init_dist, init_logger
from lib.utils.misc import accuracy, AverageMeter, CheckpointManager
from lib.utils.model_ema import ModelEMA
from lib.utils.measure import get_params, get_flops

# Enable cuDNN benchmark for faster inference
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
    Main function for model conversion
    Converts DBB/DyRep models to inference models by merging branches
    """
    # Parse command line arguments
    args, args_text = parse_args()

    # Checkpoint path must be provided
    if not args.resume:
        logger.error("Checkpoint path (--resume) must be provided")
        return

    # Set experiment directory for converted model
    checkpoint_dir = os.path.dirname(args.resume)
    args.exp_dir = os.path.join(checkpoint_dir, 'converted')
    os.makedirs(args.exp_dir, exist_ok=True)

    # Initialize distributed training
    init_dist(args)
    init_logger(args)

    # Build data loader for validation
    train_dataset, val_dataset, train_loader, val_loader = build_dataloader(args)

    # Build model
    model = build_model(args)

    # Define loss function
    if args.smoothing == 0.:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = CrossEntropyLabelSmooth(
            num_classes=args.num_classes,
            epsilon=args.smoothing
        ).cuda()

    logger.info(f"Model: {args.model}")
    logger.info(f"Parameters: {get_params(model):,}")
    logger.info(f"FLOPs: {get_flops(model, input_shape=args.input_shape):,}")

    # Convert to Diverse Branch Blocks if specified
    if args.dbb:
        from lib.models.utils.dbb_converter import convert_to_dbb
        convert_to_dbb(model)
        logger.info("Model converted to DBB blocks")
        logger.info(f"Parameters after DBB conversion: {get_params(model):,}")
        logger.info(f"FLOPs after DBB conversion: {get_flops(model, input_shape=args.input_shape):,}")

    # Move model to GPU
    model = model.cuda()

    # Wrap model with DDP if distributed training is enabled
    if args.distributed:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            find_unused_parameters=False
        )

    # Initialize Model EMA if enabled
    if args.model_ema:
        model_ema = ModelEMA(model, decay=args.model_ema_decay)
    else:
        model_ema = None

    # Initialize DyRep for dynamic re-parameterization
    if args.dyrep:
        from lib.models.utils.dyrep import DyRep
        dyrep = DyRep(
            model=model.module,
            optimizer=None
        )
        logger.info("DyRep initialized")
    else:
        dyrep = None

    # Initialize checkpoint manager
    ckpt_manager = CheckpointManager(
        model=model,
        ema_model=model_ema,
        save_dir=args.exp_dir,
        rank=args.rank,
        additional_objects={'dyrep': dyrep}
    )

    # Load checkpoint
    logger.info(f"Loading checkpoint from: {args.resume}")
    epoch = ckpt_manager.load(args.resume)

    if args.dyrep:
        # Reinitialize DDP for DyRep compatibility
        model = DDP(
            model.module,
            device_ids=[args.local_rank],
            find_unused_parameters=True
        )

    logger.info(f"Checkpoint loaded successfully, epoch: {epoch}")

    # Validate original model
    logger.info("Validating original model...")
    original_metrics = validate_model(
        args=args,
        epoch=epoch,
        model=model,
        data_loader=val_loader,
        criterion=criterion,
        log_suffix=" (original)"
    )

    logger.info(f"Original model - Loss: {original_metrics['test_loss']:.4f}, "
                f"Top-1: {original_metrics['top1']:.2f}%, "
                f"Top-5: {original_metrics['top5']:.2f}%")

    # Convert DBB/DyRep model to inference model
    logger.info("Converting DBB/DyRep model to inference model...")
    model.eval()

    # Switch all DBB blocks to deployment mode
    dbb_blocks_converted = 0
    for module in model.module.modules():
        if isinstance(module, DiverseBranchBlock):
            module.switch_to_deploy()
            dbb_blocks_converted += 1

    logger.info(f"Converted {dbb_blocks_converted} DBB blocks to deployment mode")

    # Log converted model architecture
    logger.info(f"Converted model architecture:\n{model.module}")
    logger.info(f"Converted model parameters: {get_params(model):,}")
    logger.info(f"Converted model FLOPs: {get_flops(model, input_shape=args.input_shape):,}")

    # Validate converted model
    logger.info("Validating converted model...")
    converted_metrics = validate_model(
        args=args,
        epoch=epoch,
        model=model,
        data_loader=val_loader,
        criterion=criterion,
        log_suffix=" (converted)"
    )

    logger.info(f"Converted model - Loss: {converted_metrics['test_loss']:.4f}, "
                f"Top-1: {converted_metrics['top1']:.2f}%, "
                f"Top-5: {converted_metrics['top5']:.2f}%")

    # Save converted checkpoint
    if args.rank == 0:
        save_path = os.path.join(args.exp_dir, 'model_deployed.pt')

        # Save model state dictionary
        torch.save(model.module.state_dict(), save_path)

        # Save complete model information
        model_info = {
            'epoch': epoch,
            'model_name': args.model,
            'state_dict': model.module.state_dict(),
            'original_metrics': original_metrics,
            'converted_metrics': converted_metrics,
            'conversion_params': {
                'dbb_blocks_converted': dbb_blocks_converted,
                'input_shape': args.input_shape,
                'num_classes': args.num_classes
            }
        }

        # Save complete model checkpoint
        checkpoint_path = os.path.join(args.exp_dir, 'deployment_checkpoint.pt')
        torch.save(model_info, checkpoint_path)

        logger.info(f"Saved converted model state_dict to: {save_path}")
        logger.info(f"Saved complete deployment checkpoint to: {checkpoint_path}")

        # Log performance comparison
        logger.info("\n" + "=" * 60)
        logger.info("PERFORMANCE COMPARISON")
        logger.info("=" * 60)
        logger.info(f"{'Metric':<20} {'Original':<15} {'Converted':<15} {'Diff':<10}")
        logger.info("-" * 60)
        logger.info(f"{'Loss':<20} {original_metrics['test_loss']:<15.4f} "
                    f"{converted_metrics['test_loss']:<15.4f} "
                    f"{converted_metrics['test_loss'] - original_metrics['test_loss']:<+10.4f}")
        logger.info(f"{'Top-1 Accuracy':<20} {original_metrics['top1']:<15.2f}% "
                    f"{converted_metrics['top1']:<15.2f}% "
                    f"{converted_metrics['top1'] - original_metrics['top1']:<+10.2f}%")
        logger.info(f"{'Top-5 Accuracy':<20} {original_metrics['top5']:<15.2f}% "
                    f"{converted_metrics['top5']:<15.2f}% "
                    f"{converted_metrics['top5'] - original_metrics['top5']:<+10.2f}%")
        logger.info("=" * 60)


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
        Dictionary with validation metrics
    """
    batch_time = AverageMeter(dist=args.distributed)
    losses = AverageMeter(dist=args.distributed)
    top1_acc = AverageMeter(dist=args.distributed)
    top5_acc = AverageMeter(dist=args.distributed)

    model.eval()
    end_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
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

            # Log validation progress
            if batch_idx % args.log_interval == 0 or batch_idx == len(data_loader) - 1:
                logger.info(
                    f"Validation{log_suffix}: Epoch {epoch} "
                    f"[{batch_idx:4d}/{len(data_loader):4d}] "
                    f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                    f"Top-1: {top1_acc.val:.2f}% ({top1_acc.avg:.2f}%) "
                    f"Top-5: {top5_acc.val:.2f}% ({top5_acc.avg:.2f}%) "
                    f"Time: {batch_time.val:.3f}s"
                )

    return {
        'test_loss': losses.avg,
        'top1': top1_acc.avg,
        'top5': top5_acc.avg,
        'batch_time': batch_time.avg
    }


if __name__ == '__main__':
    main()