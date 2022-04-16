from email.mime import audio
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import tqdm
import matplotlib.pyplot as plt
from rnn import notePresenceRNN, noteColourRNN, noteFinisherRNN
import random
import helper
import hyper_param

os.environ['KMP_DUPLICATE_LIB_OK']='True'

SEED = 88

def note_presence_loss(model_output, notes_data):
    nonzero_notes = torch.minimum(notes_data, torch.ones_like(notes_data))
    weight = torch.tensor([hyper_param.note_presence_weight])
    if torch.cuda.is_available():
        weight = weight.cuda()
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    loss = bce(model_output, nonzero_notes)

    return loss

def note_colour_loss(model_output, notes_data):
    # filter out all zero entries in notes_data, and also filter the same entries in model_output
    nonzero_entries = notes_data > 0
    notes_data = notes_data[nonzero_entries]
    model_output = model_output[nonzero_entries]

    kat_notes = torch.eq(notes_data, torch.mul(2, torch.ones_like(notes_data))) \
                    + torch.eq(notes_data, torch.mul(4, torch.ones_like(notes_data)))
    kat_notes = kat_notes.to(dtype=torch.float32)
    bce = torch.nn.BCEWithLogitsLoss()
    loss = bce(model_output, kat_notes)
    return loss

def note_finisher_loss(model_output, notes_data):
    nonzero_entries = notes_data > 0
    notes_data = notes_data[nonzero_entries]
    model_output = model_output[nonzero_entries]
    finisher_notes = torch.eq(notes_data, torch.mul(3, torch.ones_like(notes_data))) \
                    + torch.eq(notes_data, torch.mul(4, torch.ones_like(notes_data)))
    finisher_notes = finisher_notes.to(dtype=torch.float32)
    bce = torch.nn.BCEWithLogitsLoss()
    loss = bce(model_output, finisher_notes)
    return loss

def model_compute_note_presence(model: notePresenceRNN, audio_windows, notes_data):
    return model(audio_windows)

def model_compute_note_colour(model: noteColourRNN, audio_windows, notes_data):
    return model(audio_windows, notes_data)

def model_compute_note_finisher(model: noteFinisherRNN, audio_windows, notes_data):
    return model(audio_windows, notes_data)

TRAIN_PATH = os.path.join("data", "npy", "futsuu")

class MapDataset(Dataset): 
    def __init__(self, start, stop):
        self.path = TRAIN_PATH
        self.dir = os.listdir(self.path)
        random.seed(SEED)
        random.shuffle(self.dir)
        self.dir = self.dir[round(len(self.dir)*start) : round(len(self.dir)*stop)]

    def __len__(self):
        return len(self.dir)

    def __getitem__(self, idx):
        song_path = os.path.join(self.path, self.dir[idx])
        audio_data, timing_data, notes_data = helper.get_npy_data(song_path)
        notes_data = notes_data.astype('int32')
        audio_data, notes_data = torch.Tensor(audio_data), torch.Tensor(notes_data)
        return audio_data, timing_data, notes_data

class baselineModel():
    def __init__(self):
        self.is_cuda = False
        
    def __call__(self, x):
        return self.forward(x)
        
    def cuda(self):
        self.is_cuda = True
        return self
    
    def forward(self, x):
        out = torch.zeros(1, x.shape[1], 5)
        out[:,:,0] = 1
        if self.is_cuda:
            out = out.cuda()
        return out

"""
Trains the RNN.
Arguments:
- checkpoint_path: path to save checkpoint files. {} needs to appear to store the iteration number (e.g. "ckpt-{}.pt").
- plot: Plot the training and validation curves.
"""
def train_rnn_network(model, model_compute, criterion, num_epochs=100, learning_rate=1e-3, wd=0, 
    checkpoint_path=None, plot=False):
    print(f"Beginning training (lr={learning_rate})")
    
    train_dataset = MapDataset(0, 0.6)
    val_dataset = MapDataset(0.6, 0.8)
    train_loader = DataLoader(train_dataset, shuffle=True)
    val_loader = DataLoader(val_dataset, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)

    train_losses = []
    val_losses = []
    val_iters = []

    # training loop
    for epoch_num in range(num_epochs):
        train_loss = []
        model.train() 
        
        for audio_data, timing_data, notes_data in tqdm.tqdm(train_loader, desc=f"Epoch {epoch_num}"):
            if torch.cuda.is_available():
                audio_data = audio_data.cuda()
                notes_data = notes_data.cuda()
            optimizer.zero_grad()
            bar_len = timing_data["bar_len"].item()
            offset = timing_data["offset"].item()
            audio_windows = helper.get_audio_around_snaps(torch.squeeze(audio_data, dim=0), bar_len, offset, hyper_param.window_size)
            audio_windows = torch.flatten(audio_windows, start_dim=1)
            model_out = model_compute(model, audio_windows, notes_data)
            notes_data = torch.squeeze(notes_data, dim=0)
            model_loss = criterion(model_out, notes_data)
            model_loss.backward()
            optimizer.step()
            train_loss.append(model_loss.item())
        
        train_loss = sum(train_loss) / len(train_loss)
        train_losses.append(train_loss)
            
        # validation loss
        if (epoch_num % 100 == 0):
            model.eval()
            val_loss = []
            with torch.no_grad(): # disable gradient computation to save memory
                for audio_data, timing_data, notes_data in val_loader:
                    if torch.cuda.is_available():
                        audio_data = audio_data.cuda()
                        notes_data = notes_data.cuda()
                    bar_len = timing_data["bar_len"].item()
                    offset = timing_data["offset"].item()
                    audio_windows = helper.get_audio_around_snaps(torch.squeeze(audio_data, dim=0), bar_len, offset, hyper_param.window_size)
                    audio_windows = torch.flatten(audio_windows, start_dim=1)
                    model_out = model_compute(model, audio_windows, notes_data)
                    model_out = torch.squeeze(model_out, dim=0)
                    notes_data = torch.squeeze(notes_data, dim=0)
                    model_loss = criterion(model_out, notes_data)
                    val_loss.append(model_loss.item())
            val_loss = sum(val_loss) / len(val_loss)
            val_losses.append(val_loss)
            val_iters.append(epoch_num)

            print(f"Epoch {epoch_num + 1}/{num_epochs}" + 
                  f" | Train Loss: {'{:.4f}'.format(train_losses[-1])}" + 
                  f" | Val Loss: {'{:.4f}'.format(val_losses[-1])}")
        else:
            print(f"Epoch {epoch_num + 1}/{num_epochs}" + 
                  f" | Train Loss: {'{:.4f}'.format(train_losses[-1])}")
    
        if checkpoint_path and (epoch_num % 100) == 0:
            torch.save(model.state_dict(), checkpoint_path.format(epoch_num,learning_rate))
    
    if plot:
        plt.plot(train_losses, label=f"Training Loss")
        plt.plot(val_iters, val_losses, label=f"Validation Loss")
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()

    return train_losses, val_losses, val_iters

if __name__ == "__main__":
    # model = notePresenceRNN()
    # if torch.cuda.is_available():
    #     model = model.cuda()
    # train_rnn_network(model, model_compute_note_presence, note_presence_loss, learning_rate=1e-3, num_epochs=5, wd=0, checkpoint_path=None, plot=True)
    model = noteColourRNN()
    if torch.cuda.is_available():
        model = model.cuda()
    train_rnn_network(model, model_compute_note_colour, note_colour_loss, learning_rate=1e-4, num_epochs=1000, wd=0, checkpoint_path=None)