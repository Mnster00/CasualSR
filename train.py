import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
from models import CausalSR
from utils.dataset import SRDataset
from utils.losses import CausalSRLoss

def train(config):
    # Create model
    model = CausalSR(
        scale_factor=config['model']['scale_factor'],
        num_channels=config['model']['num_channels'],
        num_blocks=config['model']['num_blocks']
    ).cuda()
    
    # Create datasets and dataloaders
    train_dataset = SRDataset(
        config['data']['train_path'],
        patch_size=config['training']['patch_size'],
        scale=config['model']['scale_factor'],
        is_train=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['lr'],
        betas=(config['training']['beta1'], config['training']['beta2'])
    )
    
    # Create loss function
    criterion = CausalSRLoss(config['loss_weights'])
    
    # Training loop
    for epoch in range(config['training']['num_epochs']):
        model.train()
        
        for batch in tqdm(train_loader):
            # Move data to GPU
            lr = batch['lr'].cuda()
            hr = batch['hr'].cuda()
            lr_cf = batch['lr_cf'].cuda()
            clip_features = batch['clip_features'].cuda()
            
            # Forward pass
            outputs = model(lr, clip_features)
            
            # Calculate loss
            loss, loss_dict = criterion(outputs, {
                'hr': hr,
                'hr_cf': hr,
                'degradation_params': batch['degradation_params']
            })
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'checkpoints/model_epoch_{epoch+1}.pth')
            
if __name__ == '__main__':
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train(config)