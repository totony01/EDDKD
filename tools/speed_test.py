"""
Model inference speed benchmarking script
This script measures the throughput (samples/second) of a model on GPU.
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
from lib.utils.args import parse_args
from lib.utils.dist_utils import init_dist, init_logger
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
    Main function for model speed benchmarking
    """
    # Parse command line arguments
    args, args_text = parse_args()

    # Set default input shape for benchmarking
    if not hasattr(args, 'input_shape') or args.input_shape is None:
        args.input_shape = (3, 224, 224)

    # Fix random seed for reproducibility
    if hasattr(args, 'seed'):
        seed = args.seed
    else:
        seed = 42

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Initialize distributed training (if needed)
    if hasattr(args, 'distributed') and args.distributed:
        init_dist(args)
        init_logger(args)

    # Build model
    model = build_model(args)

    # Log model information
    logger.info(f"Model: {args.model}")
    logger.info(f"Parameters: {get_params(model):,}")
    logger.info(f"FLOPs: {get_flops(model, input_shape=args.input_shape):,}")

    # Convert to Diverse Branch Blocks if specified
    if hasattr(args, 'dbb') and args.dbb:
        from lib.models.utils.dbb_converter import convert_to_dbb
        convert_to_dbb(model)
        logger.info("Model converted to DBB blocks")
        logger.info(f"Parameters after DBB conversion: {get_params(model):,}")
        logger.info(f"FLOPs after DBB conversion: {get_flops(model, input_shape=args.input_shape):,}")

    # Run speed test
    batch_size = args.batch_size if hasattr(args, 'batch_size') else 128
    benchmark_model_speed(
        model=model,
        batch_size=batch_size,
        input_shape=args.input_shape,
        device='cuda',
        warmup_iterations=100,
        test_iterations=1000
    )


def benchmark_model_speed(model, batch_size, input_shape, device='cuda',
                          warmup_iterations=100, test_iterations=1000):
    """
    Benchmark the inference speed of a model

    Args:
        model: Model to benchmark
        batch_size: Batch size for inference
        input_shape: Input tensor shape (C, H, W)
        device: Device to run benchmark on ('cuda' or 'cpu')
        warmup_iterations: Number of warmup iterations
        test_iterations: Number of test iterations
    """
    # Set device
    device = torch.device(device)

    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()

    # Create random input tensor
    input_tensor = torch.randn(
        (batch_size, *input_shape),
        device=device,
        dtype=torch.float32
    )

    # Warmup phase
    logger.info(f"Starting warmup phase with {warmup_iterations} iterations...")
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(input_tensor)

    # Synchronize GPU before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Main benchmarking phase
    logger.info(f"Starting speed test with {test_iterations} iterations...")
    logger.info(f"Batch size: {batch_size}, Input shape: {input_shape}")

    start_time = time.time()

    with torch.no_grad():
        for iteration in range(test_iterations):
            _ = model(input_tensor)

            # Log progress every 10% of iterations
            if (iteration + 1) % (test_iterations // 10) == 0:
                logger.info(f"  Completed {iteration + 1}/{test_iterations} iterations")

    # Synchronize GPU after timing
    if device.type == 'cuda':
        torch.cuda.synchronize()

    end_time = time.time()

    # Calculate statistics
    total_time = end_time - start_time
    total_samples = batch_size * test_iterations
    throughput = total_samples / total_time

    # Calculate latency statistics
    average_latency = total_time / test_iterations
    batch_latency = average_latency
    per_sample_latency = average_latency / batch_size

    # Log results
    logger.info("\n" + "=" * 50)
    logger.info("SPEED BENCHMARK RESULTS")
    logger.info("=" * 50)
    logger.info(f"Device: {device}")
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Warmup iterations: {warmup_iterations}")
    logger.info(f"Test iterations: {test_iterations}")
    logger.info("-" * 50)
    logger.info(f"Total time: {total_time:.3f} seconds")
    logger.info(f"Total samples processed: {total_samples:,}")
    logger.info(f"Throughput: {throughput:.2f} samples/second")
    logger.info(f"Average batch latency: {batch_latency:.4f} seconds/batch")
    logger.info(f"Average per-sample latency: {per_sample_latency:.6f} seconds/sample")
    logger.info(f"Average FPS: {batch_size / average_latency:.2f} frames/second")

    # Additional GPU memory info if available
    if device.type == 'cuda':
        try:
            allocated_memory = torch.cuda.memory_allocated() / 1024 ** 3
            cached_memory = torch.cuda.memory_reserved() / 1024 ** 3
            logger.info(f"GPU memory allocated: {allocated_memory:.2f} GB")
            logger.info(f"GPU memory cached: {cached_memory:.2f} GB")
        except:
            logger.info("GPU memory information not available")

    logger.info("=" * 50)

    return {
        'throughput_samples_per_sec': throughput,
        'average_batch_latency_sec': batch_latency,
        'average_per_sample_latency_sec': per_sample_latency,
        'fps': batch_size / average_latency,
        'total_time_sec': total_time,
        'total_samples': total_samples
    }


if __name__ == '__main__':
    main()