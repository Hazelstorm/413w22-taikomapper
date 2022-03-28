
import torch, os, random, gc
import torch.nn as nn
import matplotlib.pyplot as plt
from helper import *
from tqdm import tqdm, trange
from loader import get_dataloader

random.seed(87)

def load_data(npy_path, song):
    song_path = os.path.join(npy_path, song)
    audio_data, notes_data = get_npy_data(song_path)
    audio_data, notes_data = torch.Tensor(audio_data), torch.Tensor(notes_data)
    audio_data, notes_data = torch.unsqueeze(audio_data, 0), torch.unsqueeze(notes_data, 0)
    mask = torch.zeros(audio_data.shape[1], audio_data.shape[1])
    
    return audio_data, notes_data, mask

def train(model, lr=1e-4, wd=0, num_epochs=20, num_iters=20):
    """
    Parameters
    ----------
    lr: learning rate
    wd: weight decay
    num_epochs: how many sets of iterations to train for
    num_iters: how many iterations to train in each epoch
    
    Additional Notes
    -----
    -One iteration consists of training on a randomly selected song
    
    """
    
    npy_path = os.path.join("data", "npy", "futsuu")
    train_loader = get_dataloader(npy_path, 0, 0.8)
    val_loader = get_dataloader(npy_path, 0.8, 0.9)
    
    losses = []
    
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss = nn.CrossEntropyLoss(reduction="mean")
    
    for e in range(num_epochs):
        
        for audio_data, notes_data in tqdm(train_loader):
            
            model.train()
        
            mask = torch.Tensor(audio_data.shape[1], audio_data.shape[1])
            if torch.cuda.is_available():
                audio_data = audio_data.cuda()
                notes_data = notes_data.cuda()
                mask = mask.cuda()
                
            # training step
            optim.zero_grad()
            model_out = model(audio_data, mask)
            model_loss = loss(model_out, notes_data)
            model_loss.backward()
            optim.step()
            
            model.eval()
        
        # loss evaluation
        epoch_loss = 0
        with torch.no_grad():
            for audio_data, notes_data in tqdm(val_loader):
            
                mask = torch.Tensor(audio_data.shape[1], audio_data.shape[1])
                if torch.cuda.is_available():
                    audio_data = audio_data.cuda()
                    notes_data = notes_data.cuda()
                    mask = mask.cuda()
                    
                model_out = model(audio_data, mask)
                model_loss = loss(model_out, notes_data)
                epoch_loss += model_loss.item()
            
        # final iteration loss
        epoch_loss /= len(val_loader)
        losses.append(epoch_loss)
        print(f"Epoch {e} | Loss: {epoch_loss}")
        
    return losses
