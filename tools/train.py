"""
Model training and validation script
This script supports distributed training, knowledge distillation, model EMA, and DyRep dynamic re-parameterization.
"""

import os
import torch
import torch.nn as nn
import logging
import time
import random
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP

from lib.models.builder import build_model
from lib.models.losses import CrossEntropyLabelSmooth, SoftTargetCrossEntropy
from lib.dataset.builder import build_dataloader
from lib.utils.optim import build_optimizer
from lib.utils.scheduler import build_scheduler
from lib.utils.args import parse_args
from lib.utils.dist_utils import init_dist, init_logger
from lib.utils.misc import accuracy, AverageMeter, CheckpointManager, AuxiliaryOutputBuffer
from lib.utils.model_ema import ModelEMA
from lib.utils.measure import get_params, get_flops

# Enable cuDNN benchmark for faster training
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
    Main training function
    """
    # Parse command line arguments
    args, args_text = parse_args()
    args.exp_dir = f'experiments/{args.experiment}'

    # Initialize distributed training
    init_dist(args)
    init_logger(args)

    # Save arguments to file
    logger.info(f"Arguments: {args}")
    if args.rank == 0:
        with open(os.path.join(args.exp_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    # Fix random seed for reproducibility
    seed = args.seed + args.rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Build data loaders
    train_dataset, val_dataset, train_loader, val_loader = build_dataloader(args)

    # Build model
    model = build_model(args, args.model)

    # Log model information
    logger.info(f"Model architecture:\n{model}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Parameters: {get_params(model) / 1e6:.3f}M")
    logger.info(f"FLOPs: {get_flops(model, input_shape=args.input_shape) / 1e9:.3f}G")

    # Convert to Diverse Branch Blocks if specified
    if args.dbb:
        from lib.models.utils.dbb_converter import convert_to_dbb
        convert_to_dbb(model)
        logger.info("Model converted to DBB blocks")
        logger.info(f"Parameters after DBB: {get_params(model) / 1e6:.3f}M")
        logger.info(f"FLOPs after DBB: {get_flops(model, input_shape=args.input_shape) / 1e9:.3f}G")

    # Move model to GPU
    model = model.cuda()

    # Initialize knowledge distillation if specified
    teacher_model = None
    if args.kd:
        # Build teacher model
        teacher_model = build_model(
            args,
            args.teacher_model,
            args.teacher_pretrained,
            args.teacher_ckpt
        )
        teacher_model = teacher_model.cuda()
        logger.info(f"Teacher model: {args.teacher_model}")
        logger.info(f"Teacher parameters: {get_params(teacher_model) / 1e6:.3f}M")
        logger.info(f"Teacher FLOPs: {get_flops(teacher_model, input_shape=args.input_shape) / 1e9:.3f}G")

        # Validate teacher model
        teacher_metrics = validate(
            args, 0, teacher_model, val_loader,
            nn.CrossEntropyLoss().cuda(),
            log_suffix=' (teacher)'
        )
        logger.info(f"Teacher Top-1 accuracy: {teacher_metrics['top1']:.2f}%")

        # Build KD loss
        from lib.models.losses.kd_loss import KDLoss
        loss_fn = KDLoss(
            student_model=None,  # Will be set after DDP wrapper
            teacher_model=teacher_model,
            student_name=args.model,
            teacher_name=args.teacher_model,
            base_loss=nn.CrossEntropyLoss().cuda(),
            kd_type=args.kd,
            ori_weight=args.ori_loss_weight,
            kd_weight=args.kd_loss_weight,
            kd_kwargs=args.kd_loss_kwargs
        )
    else:
        # Build standard loss function
        if args.mixup > 0. or args.cutmix > 0 or args.cutmix_minmax is not None:
            loss_fn = SoftTargetCrossEntropy()
        elif args.smoothing == 0.:
            loss_fn = nn.CrossEntropyLoss().cuda()
        else:
            loss_fn = CrossEntropyLabelSmooth(
                num_classes=args.num_classes,
                epsilon=args.smoothing
            ).cuda()

    # Validation loss function
    val_loss_fn = nn.CrossEntropyLoss().cuda() if not args.kd else loss_fn

    # Wrap model with DDP
    if args.distributed:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            find_unused_parameters=args.find_unused_parameters
        )

    # Set student model in KD loss
    if args.kd:
        loss_fn.student = model

    logger.info(f"Final model architecture:\n{model}")

    # Initialize Model EMA
    if args.model_ema:
        model_ema = ModelEMA(model, decay=args.model_ema_decay)
    else:
        model_ema = None

    # Build optimizer
    optimizer = build_optimizer(
        name=args.opt,
        model=model.module,
        lr=args.lr,
        eps=args.opt_eps,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        filter_bias_and_bn=not args.opt_no_filter,
        nesterov=not args.sgd_no_nesterov,
        sort_params=args.dyrep
    )

    # Build scheduler
    steps_per_epoch = len(train_loader)
    warmup_steps = args.warmup_epochs * steps_per_epoch
    decay_steps = args.decay_epochs * steps_per_epoch
    total_steps = args.epochs * steps_per_epoch

    scheduler = build_scheduler(
        name=args.sched,
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        warmup_lr=args.warmup_lr,
        decay_steps=decay_steps,
        decay_rate=args.decay_rate,
        total_steps=total_steps,
        steps_per_epoch=steps_per_epoch,
        decay_by_epoch=args.decay_by_epoch,
        min_lr=args.min_lr
    )

    # Initialize DyRep for dynamic re-parameterization
    dyrep = None
    if args.dyrep:
        from lib.models.utils.dyrep import DyRep
        from lib.models.utils.recal_bn import recal_bn

        dyrep = DyRep(
            model=model.module,
            optimizer=optimizer,
            recal_bn_fn=lambda m: recal_bn(
                model.module,
                train_loader,
                args.dyrep_recal_bn_iters,
                m
            ),
            filter_bias_and_bn=not args.opt_no_filter
        )
        logger.info("DyRep initialized")

    # Initialize automatic mixed precision
    if args.amp:
        loss_scaler = torch.cuda.amp.GradScaler()
    else:
        loss_scaler = None

    # Initialize checkpoint manager
    ckpt_manager = CheckpointManager(
        model=model,
        optimizer=optimizer,
        ema_model=model_ema,
        save_dir=args.exp_dir,
        rank=args.rank,
        additional_objects={
            'scaler': loss_scaler,
            'dyrep': dyrep
        }
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = ckpt_manager.load(args.resume) + 1
        if start_epoch > args.warmup_epochs:
            scheduler.finished = True
        scheduler.step(start_epoch * steps_per_epoch)

        if args.dyrep:
            # Reinitialize DDP for DyRep
            model = DDP(
                model.module,
                device_ids=[args.local_rank],
                find_unused_parameters=True
            )

        logger.info(f"Resumed from checkpoint: {args.resume}")
        logger.info(f"Starting training from epoch: {start_epoch}")

    # Initialize auxiliary output buffer
    if args.auxiliary:
        auxiliary_buffer = AuxiliaryOutputBuffer(
            model=model,
            weight=args.auxiliary_weight
        )
    else:
        auxiliary_buffer = None

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Set epoch for distributed sampler
        if args.distributed:
            train_loader.loader.sampler.set_epoch(epoch)

        # Adjust drop path rate linearly
        if args.drop_path_rate > 0. and args.drop_path_strategy == 'linear':
            if hasattr(model.module, 'drop_path_rate'):
                model.module.drop_path_rate = args.drop_path_rate * epoch / args.epochs

        # Training phase
        train_metrics = train_epoch(
            args=args,
            epoch=epoch,
            model=model,
            model_ema=model_ema,
            data_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler,
            auxiliary_buffer=auxiliary_buffer,
            dyrep=dyrep,
            loss_scaler=loss_scaler
        )

        # Validation phase
        logger.info(f"Validating base model at epoch {epoch}...")
        val_metrics = validate(
            args=args,
            epoch=epoch,
            model=model,
            data_loader=val_loader,
            loss_fn=val_loss_fn
        )

        # Validate EMA model if available
        if model_ema is not None:
            logger.info(f"Validating EMA model at epoch {epoch}...")
            ema_metrics = validate(
                args=args,
                epoch=epoch,
                model=model_ema.module,
                data_loader=val_loader,
                loss_fn=val_loss_fn,
                log_suffix='(EMA)'
            )

        # DyRep adjustments
        if dyrep is not None:
            if epoch < args.dyrep_max_adjust_epochs:
                if (epoch + 1) % args.dyrep_adjust_interval == 0:
                    logger.info("DyRep: Adjusting model...")
                    dyrep.adjust_model()
                    logger.info(f"Model parameters: {get_params(model)/1e6:.3f}M")
                    logger.info(f"Model FLOPs: {get_flops(model, input_shape=args.input_shape)/1e9:.3f}G")

                    # Reinitialize DDP
                    model = DDP(
                        model.module,
                        device_ids=[args.local_rank],
                        find_unused_parameters=True
                    )

                    # Revalidate after adjustment
                    val_metrics = validate(
                        args=args,
                        epoch=epoch,
                        model=model,
                        data_loader=val_loader,
                        loss_fn=val_loss_fn
                    )
                elif args.dyrep_recal_bn_every_epoch:
                    logger.info("DyRep: Recalibrating BN...")
                    from lib.models.utils.recal_bn import recal_bn
                    recal_bn(model.module, train_loader, 200)

                    val_metrics = validate(
                        args=args,
                        epoch=epoch,
                        model=model,
                        data_loader=val_loader,
                        loss_fn=val_loss_fn
                    )

        # Save checkpoint
        all_metrics = {**train_metrics, **val_metrics}
        saved_checkpoints = ckpt_manager.update(epoch, all_metrics)

        logger.info("Checkpoints saved:")
        for ckpt, score in saved_checkpoints:
            logger.info(f"  {ckpt}: {score:.3f}%")


def train_epoch(args, epoch, model, model_ema, data_loader, optimizer, loss_fn,
                scheduler, auxiliary_buffer=None, dyrep=None, loss_scaler=None):
    """
    Train model for one epoch

    Args:
        args: Command line arguments
        epoch: Current epoch number
        model: Model to train
        model_ema: EMA model
        data_loader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        scheduler: Learning rate scheduler
        auxiliary_buffer: Auxiliary output buffer
        dyrep: DyRep object for dynamic re-parameterization
        loss_scaler: Gradient scaler for mixed precision training

    Returns:
        Dictionary with training metrics
    """
    batch_time = AverageMeter(dist=args.distributed)
    data_time = AverageMeter(dist=args.distributed)
    losses = AverageMeter(dist=args.distributed)

    model.train()
    end_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end_time)

        # Move data to GPU
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        # Zero gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward pass with mixed precision if enabled
        with torch.cuda.amp.autocast(enabled=loss_scaler is not None):
            if not args.kd:
                # Standard training
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
            else:
                # Knowledge distillation
                loss = loss_fn(inputs, targets)

            # Add auxiliary loss if enabled
            if auxiliary_buffer is not None:
                aux_loss = loss_fn(auxiliary_buffer.output, targets)
                loss += aux_loss * auxiliary_buffer.loss_weight

        # Backward pass
        if loss_scaler is None:
            loss.backward()
        else:
            loss_scaler.scale(loss).backward()

        # Gradient clipping
        if args.clip_grad_norm:
            if loss_scaler is not None:
                loss_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.clip_grad_max_norm
            )

        # Record metrics for DyRep
        if dyrep is not None:
            dyrep.record_metrics()

        # Optimizer step
        if loss_scaler is None:
            optimizer.step()
        else:
            loss_scaler.step(optimizer)
            loss_scaler.update()

        # Update EMA model
        if model_ema is not None:
            model_ema.update(model)

        # Update metrics
        batch_size = inputs.size(0)
        losses.update(loss.item(), batch_size)

        # Measure batch processing time
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        # Log training progress
        if batch_idx % args.log_interval == 0 or batch_idx == len(data_loader) - 1:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(
                f"Train: Epoch {epoch} [{batch_idx:4d}/{len(data_loader):4d}] "
                f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                f"LR: {current_lr:.3e} "
                f"Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s) "
                f"Data: {data_time.val:.3f}s"
            )

        # Update learning rate
        scheduler.step(epoch * len(data_loader) + batch_idx + 1)

    return {'train_loss': losses.avg}


def validate(args, epoch, model, data_loader, loss_fn, log_suffix=''):
    """
    Validate model performance on validation set

    Args:
        args: Command line arguments
        epoch: Current epoch number
        model: Model to validate
        data_loader: Validation data loader
        loss_fn: Loss function
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
            loss = loss_fn(outputs, targets)

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
                    f"Test{log_suffix}: Epoch {epoch} [{batch_idx:4d}/{len(data_loader):4d}] "
                    f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                    f"Top-1: {top1_acc.val:.2f}% ({top1_acc.avg:.2f}%) "
                    f"Top-5: {top5_acc.val:.2f}% ({top5_acc.avg:.2f}%) "
                    f"Time: {batch_time.val:.3f}s"
                )

    return {
        'test_loss': losses.avg,
        'top1': top1_acc.avg,
        'top5': top5_acc.avg
    }


if __name__ == '__main__':
    main()