# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import time
from tryon_dataset import CPDataset, CPDataLoader
from networks import GMM, UnetGenerator, load_checkpoint
from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images, save_images

# Helper function to determine the device (GPU/CPU)
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get options from command line
def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="GMM")
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument("--dataroot", default="data")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--stage", default="GMM")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--result_dir', type=str, default='result', help='save result infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for test')
    parser.add_argument("--display_count", type=int, default=1)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt

# Test function for GMM model
def test_gmm(opt, test_loader, model, board):
    device = get_device()
    model.to(device)
    model.eval()

    base_name = os.path.basename(opt.checkpoint)
    save_dir = os.path.join(opt.result_dir, base_name, opt.datamode)
    os.makedirs(save_dir, exist_ok=True)
    warp_cloth_dir = os.path.join(save_dir, 'warp-cloth')
    os.makedirs(warp_cloth_dir, exist_ok=True)
    warp_mask_dir = os.path.join(save_dir, 'warp-mask')
    os.makedirs(warp_mask_dir, exist_ok=True)

    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()

        # Move inputs to device
        c_names = inputs['c_name']
        im = inputs['image'].to(device)
        im_pose = inputs['pose_image'].to(device)
        im_h = inputs['head'].to(device)
        shape = inputs['shape'].to(device)
        agnostic = inputs['agnostic'].to(device)
        c = inputs['cloth'].to(device)
        cm = inputs['cloth_mask'].to(device)
        im_c = inputs['parse_cloth'].to(device)
        im_g = inputs['grid_image'].to(device)

        grid, theta = model(agnostic, c)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')

        visuals = [[im_h, shape, im_pose],
                   [c, warped_cloth, im_c],
                   [warped_grid, (warped_cloth + im) * 0.5, im]]

        save_images(warped_cloth, c_names, warp_cloth_dir)
        save_images(warped_mask * 2 - 1, c_names, warp_mask_dir)

        if (step + 1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step + 1)
            t = time.time() - iter_start_time
            print(f'step: {step + 1:8d}, time: {t:.3f}', flush=True)

# Test function for TOM model
def test_tom(opt, test_loader, model, board):
    device = get_device()
    model.to(device)
    model.eval()

    base_name = os.path.basename(opt.checkpoint)
    save_dir = os.path.join(opt.result_dir, base_name, opt.datamode)
    os.makedirs(save_dir, exist_ok=True)
    try_on_dir = os.path.join(save_dir, 'try-on')
    os.makedirs(try_on_dir, exist_ok=True)

    print(f'Dataset size: {len(test_loader.dataset):05d}!', flush=True)
    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()

        # Move inputs to device
        im_names = inputs['im_name']
        im = inputs['image'].to(device)
        im_pose = inputs['pose_image'].to(device)
        im_h = inputs['head'].to(device)
        shape = inputs['shape'].to(device)
        agnostic = inputs['agnostic'].to(device)
        c = inputs['cloth'].to(device)
        cm = inputs['cloth_mask'].to(device)

        outputs = model(torch.cat([agnostic, c], 1))
        p_rendered, m_composite = torch.split(outputs, 3, 1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        visuals = [[im_h, shape, im_pose],
                   [c, 2 * cm - 1, m_composite],
                   [p_rendered, p_tryon, im]]

        save_images(p_tryon, im_names, try_on_dir)
        if (step + 1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step + 1)
            t = time.time() - iter_start_time
            print(f'step: {step + 1:8d}, time: {t:.3f}', flush=True)

# Main function to initialize everything
def main():
    opt = get_opt()
    print(opt)
    print(f"Start to test stage: {opt.stage}, named: {opt.name}!")

    # Create dataset and dataloader
    train_dataset = CPDataset(opt)
    train_loader = CPDataLoader(opt, train_dataset)

    # Create tensorboard writer
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))

    # Create model and load checkpoint
    if opt.stage == 'GMM':
        model = GMM(opt)
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_gmm(opt, train_loader, model, board)
    elif opt.stage == 'TOM':
        model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_tom(opt, train_loader, model, board)
    else:
        raise NotImplementedError(f'Model [{opt.stage}] is not implemented')

    print(f'Finished test {opt.stage}, named: {opt.name}!')

# Run the main function
if __name__ == "__main__":
    main()
