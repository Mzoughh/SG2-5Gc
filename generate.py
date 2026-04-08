###############################################################################
# Adaptation of NVIDIA StyleGAN2-ada Code to generate images in a 5Gc context
# Copyright (c) 2021, NVIDIA CORPORATION.  
###############################################################################

import argparse
import os
import re
from typing import List, Optional
import dnnlib
import numpy as np
import PIL.Image
import torch
import legacy


def num_range(s: str) -> List[int]:
    """Accept either 'a,b,c' or a range 'a-c' and return a list of ints."""
    m = re.match(r'^(\d+)-(\d+)$', s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    return [int(x) for x in s.split(',')]

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description='Generate images using pretrained network pickle.')
    parser.add_argument('--network', dest='network_pkl', required=True, help='Network pickle filename')
    parser.add_argument('--seeds', type=num_range, required=True, help='List of random seeds')
    parser.add_argument('--num-images', dest='num_images', type=int, default=10, help='Number of images to generate per seed (default: 10)')
    parser.add_argument('--trunc', dest='truncation_psi', type=float, default=1.0, help='Truncation psi')
    parser.add_argument('--noise-mode', dest='noise_mode', choices=['const', 'random', 'none'], default='const', help='Noise mode')
    parser.add_argument('--device', dest='device', default='cpu', help='Torch device (default: cuda). E.g. cuda, cuda:0, cpu')
    parser.add_argument('--outdir', type=str, required=True, metavar='DIR', help='Where to save the output images')
    args = parser.parse_args(argv)

    # Init the device for the futur loading
    if str(args.device).startswith('cuda') and not torch.cuda.is_available():
        raise SystemExit('CUDA device requested but CUDA is not available. Use --device=cpu or run with GPU support.')
    device = torch.device(args.device)
    
    # Check if outdir exists 
    os.makedirs(args.outdir, exist_ok=True)

    # Loading the SG2-ada model 
    print(f'Loading network from "{args.network_pkl}"...')
    with dnnlib.util.open_url(args.network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  

    # Evaluation mode for inference only 
    G.eval()
    torch.set_grad_enabled(False)
    # CPU robustness:
    # StyleGAN2-ADA can internally switch to FP16 (`use_fp16`) unless we force FP32.
    # On CPU, some conv kernels are not implemented for float16, so we must avoid FP16.
    force_fp32 = (device.type == 'cpu')
    if force_fp32:
        G = G.to(dtype=torch.float32)

    # This deployment assumes an unconditional network.
    if G.c_dim != 0:
        raise SystemExit('This script expects an unconditional network (c_dim=0).')

    # Generation
    for seed_idx, seed in enumerate(args.seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(args.seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(args.num_images, G.z_dim)).to(device)
        if force_fp32:
            z = z.to(dtype=torch.float32)
        label = torch.zeros([z.shape[0], G.c_dim], device=device)
        if force_fp32:
            label = label.to(dtype=torch.float32)
        with torch.inference_mode():
            img = G(z, label, truncation_psi=args.truncation_psi, noise_mode=args.noise_mode, force_fp32=force_fp32)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        for i in range(img.shape[0]):
            PIL.Image.fromarray(img[i].cpu().numpy(), 'RGB').save(f'{args.outdir}/seed{seed:04d}_img{i:02d}.png')


if __name__ == '__main__':
    main()

#----------------------------------------------------------------------------
