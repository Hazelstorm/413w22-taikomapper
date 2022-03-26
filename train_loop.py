
import torch, os, random, gc
import torch.nn as nn
import matplotlib.pyplot as plt
from model import GeneratorModel, DiscriminatorModel
from helper import *
from tqdm import tqdm, trange

random.seed(87)

def load_data(npy_path, song):
    song_path = os.path.join(npy_path, song)
    audio_data, notes_data = get_npy_data(song_path)
    audio_data, notes_data = torch.Tensor(audio_data), torch.Tensor(notes_data)
    audio_data, notes_data = torch.unsqueeze(audio_data, 0), torch.unsqueeze(notes_data, 0)
    mask = torch.zeros(audio_data.shape[1], audio_data.shape[1])
    
    return audio_data, notes_data, mask

def train_gan(dis, gen, lr=5e-5, wd=0, num_epochs=20, d_iters=50, g_iters=10):
    """
    Parameters
    ----------
    lr: learning rate
    wd: weight decay
    num_epochs: how many sets of iterations to train for
    d_iter: how many iterations the discriminator trains in each epoch
    g_iter: how many iterations the generator trains in each epoch
    
    Additional Notes
    -----
    -One iteration consists of training on a randomly selected song
    -d_iters should be larger than g_iters
    
    """
    
    npy_path = os.path.join("data", "npy")
    npy_lib = os.listdir(npy_path)
    
    train_lib = npy_lib[-20:]
    test_lib = npy_lib[-20:]
    
    # wasserstein gan loss https://agustinus.kristia.de/techblog/2017/02/04/wasserstein-gan/
    dis_losses = []
    gen_losses = []
    
    dis_optim = torch.optim.RMSprop(dis.parameters(), lr=lr, weight_decay=wd)
    gen_optim = torch.optim.RMSprop(gen.parameters(), lr=lr, weight_decay=wd)

    for e in range(num_epochs):
        
        dis.train()
        gen.eval()
        for d in range(d_iters):
            
            # load data
            song = random.choice(train_lib)
            audio_data, notes_data, mask = load_data(npy_path, song)
            if torch.cuda.is_available():
                audio_data = audio_data.cuda()
                notes_data = notes_data.cuda()
                mask = mask.cuda()
            
            # discriminator training step
            dis_optim.zero_grad()
            dis_real = dis(audio_data, notes_data, mask)
            dis_fake = dis(audio_data, gen(audio_data, mask), mask)
            dis_loss = dis_fake - dis_real
            dis_loss.backward()
            dis_optim.step()
            
            # clip parameters
            for p in dis.parameters():
                p.data.clamp_(-0.01, 0.01)
                    
        dis.eval()
        gen.train()
        for g in range(g_iters):
            
            # load data
            song = random.choice(train_lib)
            audio_data, notes_data, mask = load_data(npy_path, song)
            if torch.cuda.is_available():
                audio_data = audio_data.cuda()
                notes_data = notes_data.cuda()
                mask = mask.cuda()
            
            # generator training step
            gen_optim.zero_grad()
            dis_fake = dis(audio_data, gen(audio_data, mask), mask)
            gen_loss = -dis_fake
            gen_loss.backward()
            gen_optim.step()
        
        dis.eval()
        gen.eval()
        dis_loss = 0
        gen_loss = 0
        with torch.no_grad():
            for song in tqdm(test_lib):
                
                # load data
                audio_data, notes_data, mask = load_data(npy_path, song)
                if torch.cuda.is_available():
                    audio_data = audio_data.cuda()
                    notes_data = notes_data.cuda()
                    mask = mask.cuda()
                
                # eval losses
                dis_real = dis(audio_data, notes_data, mask)
                dis_fake = dis(audio_data, gen(audio_data, mask), mask)
                dis_loss += (dis_fake - dis_real).item()
                gen_loss += (-dis_fake).item()
            
        # final epoch losses
        dis_losses.append(dis_loss / len(test_lib))
        gen_losses.append(gen_loss / len(test_lib))
        
        print(f"Epoch {e} | dis_loss: {dis_loss} | gen_loss: {gen_loss}")
        
    return dis_losses, gen_losses

# testing non-gan training
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
    
    npy_path = os.path.join("data", "npy")
    npy_lib = os.listdir(npy_path)
    
    train_lib = npy_lib[-1:]
    test_lib = npy_lib[-1:]
    
    audio_data, notes_data, mask = load_data(npy_path, train_lib[0]) # TODO remove
    if torch.cuda.is_available():
        audio_data = audio_data.cuda()
        notes_data = notes_data.cuda()
        mask = mask.cuda()
    
    losses = []
    
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss = nn.CrossEntropyLoss()
    
    for e in range(num_epochs):
        
        model.train()
        
        for i in range(num_iters):
        
            # load data
            """
            song = random.choice(train_lib)
            audio_data, notes_data, mask = load_data(npy_path, song)
            if torch.cuda.is_available():
                audio_data = audio_data.cuda()
                notes_data = notes_data.cuda()
                mask = mask.cuda()
            """
                
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
            for song in test_lib:
                
                # load data
                audio_data, notes_data, mask = load_data(npy_path, song)
                if torch.cuda.is_available():
                    audio_data = audio_data.cuda()
                    notes_data = notes_data.cuda()
                    mask = mask.cuda()
                    
                model_out = model(audio_data, mask)
                model_loss = loss(model_out, notes_data)
                epoch_loss += model_loss.item()
                
        # final iteration loss
        epoch_loss /= len(test_lib)
        losses.append(epoch_loss)
        print(f"Epoch {e} | Loss: {epoch_loss}")
        
    return losses
