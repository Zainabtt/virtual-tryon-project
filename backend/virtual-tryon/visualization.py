
from PIL import Image
import os
import torch
from torchvision.utils import save_image

# Improved normalization: only apply scaling if data is in [-1, 1]
def tensor_for_board(img_tensor):
    tensor = img_tensor.clone()
    if tensor.min() < 0:  # likely in [-1, 1] range
        tensor = (tensor + 1) * 0.5
    tensor = tensor.cpu().clamp(0, 1)

    if tensor.size(1) == 1:
        tensor = tensor.repeat(1, 3, 1, 1)

    return tensor

def tensor_list_for_board(img_tensors_list):
    grid_h = len(img_tensors_list)
    grid_w = max(len(img_tensors) for img_tensors in img_tensors_list)

    batch_size, channel, height, width = tensor_for_board(img_tensors_list[0][0]).size()
    canvas_h = grid_h * height
    canvas_w = grid_w * width
    canvas = torch.FloatTensor(batch_size, channel, canvas_h, canvas_w).fill_(0.5)

    for i, img_tensors in enumerate(img_tensors_list):
        for j, img_tensor in enumerate(img_tensors):
            offset_h = i * height
            offset_w = j * width
            tensor = tensor_for_board(img_tensor)
            canvas[:, :, offset_h:offset_h + height, offset_w:offset_w + width].copy_(tensor)

    return canvas

def save_images(img_tensors, img_names, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for img_tensor, img_name in zip(img_tensors, img_names):
        tensor = img_tensor.clone()
        if tensor.min() < 0:
            tensor = (tensor + 1) * 0.5
        tensor = tensor.clamp(0, 1)
        path = os.path.join(save_dir, os.path.splitext(img_name)[0] + ".png")
        save_image(tensor, path)

def board_add_image(board, tag_name, img_tensor, step_count):
    tensor = tensor_for_board(img_tensor)
    for i, img in enumerate(tensor):
        board.add_image(f'{tag_name}/{i:03d}', img, step_count)

def board_add_images(board, tag_name, img_tensors_list, step_count):
    tensor = tensor_list_for_board(img_tensors_list)
    for i, img in enumerate(tensor):
        board.add_image(f'{tag_name}/{i:03d}', img, step_count)
