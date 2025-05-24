import numpy as np
from PIL import Image, ImageDraw
import cv2

def create_agnostic(person_img, pose_img, parse_img):
   
    
    person_np = np.array(person_img)
    parse_np = np.array(parse_img.convert('L'))

   
    mask = np.isin(parse_np, [1, 2])
    person_np[mask] = [255, 255, 255] 

    agnostic_image = Image.fromarray(person_np)
    return agnostic_image

def create_cloth_mask(cloth_img):
    
    
    gray = cloth_img.convert('L')
    np_gray = np.array(gray)

    _, mask = cv2.threshold(np_gray, 250, 255, cv2.THRESH_BINARY_INV)

    mask_image = Image.fromarray(mask)
    return mask_image
