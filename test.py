import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from models import CausalSR
from utils.metrics import calculate_psnr, calculate_ssim

def test(model_path, image_path, scale=4):
    # Load model
    model = CausalSR(scale_factor=scale)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()
    model.eval()
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).cuda()
    
    # Get CLIP features
    clip_model, preprocess = clip.load("ViT-B/32", device="cuda")
    with torch.no_grad():
        clip_features = clip_model.encode_image(preprocess(img).unsqueeze(0))
    
    # Super-resolve
    with torch.no_grad():
        sr_tensor = model(img_tensor, clip_features)
        
    # Convert to image
    sr_img = transforms.ToPILImage()(sr_tensor.squeeze(0).cpu())
    
    # Calculate metrics
    psnr = calculate_psnr(sr_tensor, img_tensor)
    ssim = calculate_ssim(sr_tensor, img_tensor)
    
    print(f'PSNR: {psnr:.2f}')
    print(f'SSIM: {ssim:.4f}')
    
    return sr_img

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--scale', type=int, default=4)
    args = parser.parse_args()
    
    sr_img = test(args.model, args.input, args.scale)
    sr_img.save('output.png')