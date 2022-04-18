from email.mime import audio
import torch
from torch.utils.data import Dataset, DataLoader
import os
import tqdm
import matplotlib.pyplot as plt
from rnn import notePresenceRNN, noteColourRNN, noteFinisherRNN
import random
import helper
import hyper_param
import csv
import signal, sys

os.environ['KMP_DUPLICATE_LIB_OK']='True'

train_losses = []
val_losses = []
val_iters = []

# Locations for dumping the training/validation curves in a .csv file
train_loss_dump = open('losses.csv','a', encoding='utf8')
val_loss_dump = open('validation.csv','a', encoding='utf8')

SEED = 88 # Random seed used during training time to shuffle dataset

"""
Creates a plot of the training/validation losses.
<val_iters>: Iterations where the validation losses are evaluated. For example,
             if validation loss is evalutaed every 10 epochs, val_iters=[0, 10, 20, ...]
"""
def plot(train_losses, val_losses, val_iters):
    plt.title(f"Loss Curves")
    plt.plot(train_losses, label=f"Training Loss")
    plt.plot(val_iters, val_losses, label=f"Validation Loss")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()

"""
Outputs the training/validation losses into a csv file.
The training and validation loss is dumped into train_loss_dump and val_loss_dump respectively.
"""
def dump_losses(train_losses, val_losses, val_iters, graph_name):
    global train_loss_dump, val_loss_dump
    
    train_header = ['training losses', graph_name]
    train_writer = csv.writer(train_loss_dump)
    train_writer.writerow(train_header)
    
    for i in (train_losses):
        train_writer.writerow([i])
    
    
    val_header = ['val_losses', 'val_iters', graph_name]    
    val_writer = csv.writer(val_loss_dump)
    val_writer.writerow(val_header)
    
    for j, k in zip(val_losses, val_iters):
        val_writer.writerow([j,k])

"""
Enable this signal handler to plots training and validation loss upon keyboard interrupt.
Useful if you want to stop training.
"""
def signal_handler(sig, frame):
    global train_losses, val_losses, val_iters
    plot(train_losses, val_losses, val_iters)
    dump_losses(train_losses, val_losses, val_iters, None)
    train_loss_dump.close()
    val_loss_dump.close()
    sys.exit(0)


# Model computation signatures
def model_compute_note_presence(model: notePresenceRNN, audio_windows, notes_data):
    return model(audio_windows)

def model_compute_note_colour(model: noteColourRNN, audio_windows, notes_data):
    return model(audio_windows, notes_data)

def model_compute_note_finisher(model: noteFinisherRNN, audio_windows, notes_data):
    return model(audio_windows, notes_data)


# Data Loader
TRAIN_PATH = os.path.join("data", "npy", "kantan")
# Where to split the training set into train/validation/test
SPLIT = [0.8, 0.9] 

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


# Loss functions for each model
note_presence_weight = 5 
note_finisher_weight = 3

"""
Compute the average ratio of present notes to non-present notes on snaps, and update
note_presence_weight accordingly.
"""
def compute_note_presence_weight():
    train_dataset = MapDataset(0, SPLIT[0])
    train_loader = DataLoader(train_dataset)
    total_notes = 0
    total_no_notes = 0
    for _, _, notes_data in train_loader:
        notes, no_notes = helper.get_note_ratio(notes_data)
        total_notes += notes
        total_no_notes += no_notes
    global note_presence_weight
    note_presence_weight = total_no_notes / total_notes

def note_presence_loss(model_output, notes_data, pos_weight=note_presence_weight):
    nonzero_notes = torch.minimum(notes_data, torch.ones_like(notes_data))
    pos_weight = torch.tensor(pos_weight)
    if torch.cuda.is_available():
        pos_weight = pos_weight.cuda()
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss = bce(model_output, nonzero_notes)
    return loss

def note_colour_loss(model_output, notes_data):
    # Filter out all zero entries in notes_data, and also filter the same entries in model_output
    nonzero_entries = notes_data > 0
    notes_data = notes_data[nonzero_entries]
    model_output = model_output[nonzero_entries]
    kat_notes = torch.eq(notes_data, torch.mul(2, torch.ones_like(notes_data))) \
                    + torch.eq(notes_data, torch.mul(4, torch.ones_like(notes_data)))
    kat_notes = kat_notes.to(dtype=torch.float32)
    bce = torch.nn.BCEWithLogitsLoss()
    loss = bce(model_output, kat_notes)
    return loss

def note_finisher_loss(model_output, notes_data, pos_weight=note_finisher_weight):
    nonzero_entries = notes_data > 0
    notes_data = notes_data[nonzero_entries]
    model_output = model_output[nonzero_entries]
    finisher_notes = torch.eq(notes_data, torch.mul(3, torch.ones_like(notes_data))) \
                    + torch.eq(notes_data, torch.mul(4, torch.ones_like(notes_data)))
    finisher_notes = finisher_notes.to(dtype=torch.float32)
    pos_weight = torch.tensor(pos_weight)
    if torch.cuda.is_available():
        pos_weight = pos_weight.cuda()
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss = bce(model_output, finisher_notes)
    return loss


"""
Trains the RNN.
Arguments:
- model_compute: The forward function for model (model_compute_note_presence, model_compute_note_colour, model_compute_note_finisher).
- criterion: The loss function for the model (note_presence_loss, note_colour_loss, note_finisher_loss).
- num_epochs: Number of epochs to train this model for.
- lr: Learning rate.
- wd: Weight decay (regularization).
- checkpoint_path: Path to save checkpoint files. {} needs to appear to store the iteration number (e.g. "ckpt-{}.pt").
- augment_noise: Determines how much noise is added to each audio spectrogram. Setting this to None adds no noise.
- compute_valid_loss_every: Validation loss is computed every <compute_valid_loss_every> epochs.
"""
def train_rnn_network(model, model_compute, criterion, num_epochs=100, lr=1e-3, wd=0, 
    checkpoint_path=None, augment_noise=None, compute_valid_loss_every=10):
    print(f"Beginning training (lr={lr}).")

    # Reset arrays used for plotting
    global train_losses
    global val_losses
    global val_iters

    train_losses = []
    val_losses = []
    val_iters = []
    
    train_dataset = MapDataset(0, SPLIT[0])
    val_dataset = MapDataset(SPLIT[0], SPLIT[1])
    train_loader = DataLoader(train_dataset, shuffle=True)
    val_loader = DataLoader(val_dataset, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Main training loop
    for epoch_num in range(num_epochs):
        train_loss = []
        model.train() 
        
        for audio_data, timing_data, notes_data in tqdm.tqdm(train_loader, desc=f"Epoch {epoch_num}"):
            if torch.cuda.is_available():
                audio_data = audio_data.cuda()
                notes_data = notes_data.cuda()
                
            optimizer.zero_grad()
            
            # train_loader provides a spectrogram. We convert this spectrogram into audio windows around each snap.
            bar_len = timing_data["bar_len"].item()
            offset = timing_data["offset"].item()
            audio_windows = helper.get_audio_around_snaps(torch.squeeze(audio_data, dim=0), 
                                    bar_len, offset, hyper_param.window_size)
            if augment_noise:
                if torch.cuda.is_available():
                    noise = torch.normal(0,augment_noise,size=audio_windows.size()).cuda()
                else:
                    noise = torch.normal(0,augment_noise,size=audio_windows.size())
                audio_windows += noise
    
            model_out = model_compute(model, audio_windows, notes_data)
            notes_data = torch.squeeze(notes_data, dim=0)
            model_loss = criterion(model_out, notes_data)
            model_loss.backward()
            optimizer.step()
            train_loss.append(model_loss.item())
        
        train_loss = sum(train_loss) / len(train_loss)
        train_losses.append(train_loss)
            
        # Compute validation loss
        if (epoch_num % compute_valid_loss_every == 0):
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
    
        if checkpoint_path and (epoch_num % 5) == 0:
            torch.save(model.state_dict(), checkpoint_path.format(epoch_num,lr))

    return train_losses, val_losses, val_iters

if __name__ == "__main__":
    print("Computing note presence weight...")
    compute_note_presence_weight()
    signal.signal(signal.SIGINT, signal_handler) # Plot upon SIGINT

    checkpoint_dir = "checkpoints"
    if not (os.path.isdir("checkpoints")):
        os.mkdir("checkpoints")

    # Train notePresenceRNN
    # presence_model = notePresenceRNN()
    # if torch.cuda.is_available():
    #     presence_model = presence_model.cuda()
    #     presence_model.load_state_dict(torch.load("...", map_location=torch.device('cuda')))
    # presence_model.load_state_dict(torch.load("...", map_location=torch.device('cpu')))
    # train_losses, val_losses, val_iters = \
    #     train_rnn_network(presence_model, model_compute_note_presence, note_presence_loss, 
    #             lr=1e-5, num_epochs=1001, wd=0, checkpoint_path=checkpoint_dir+"/notePresenceRNN-iter{}.pt", 
    #             augment_noise=5, compute_valid_loss_every=1)
    # dump_losses(train_losses, val_losses, val_iters)
    
    # Train noteColourRNN
    # colour_model = noteColourRNN()
    # if torch.cuda.is_available():
    #     colour_model = colour_model.cuda()
    #     colour_model.load_state_dict(torch.load("...", map_location=torch.device('cuda')))
    # colour_model.load_state_dict(torch.load("...", map_location=torch.device('cpu')))
    # train_losses, val_losses, val_iters = \
    #     train_rnn_network(colour_model, model_compute_note_colour, note_colour_loss, 
    #             lr=1e-5, num_epochs=1001, wd=0, checkpoint_path=checkpoint_dir+"/noteColourRNN-iter{}.pt", 
    #             augment_noise=5, compute_valid_loss_every=1)
    # dump_losses(train_losses, val_losses, val_iters)

    # Train noteFinisher
    # finisher_model = noteFinisherRNN()
    # if torch.cuda.is_available():
    #     finisher_model = finisher_model.cuda()
    #     finisher_model.load_state_dict(torch.load("...", map_location=torch.device('cuda')))
    # finisher_model.load_state_dict(torch.load("...", map_location=torch.device('cpu')))
    # train_losses, val_losses, val_iters = \
    #     train_rnn_network(finisher_model, model_compute_note_finisher, note_finisher_loss, 
    #             lr=1e-5, num_epochs=1001, wd=0, checkpoint_path=checkpoint_dir+"/noteFinisherRNN-iter{}.pt", 
    #             augment_noise=5, compute_valid_loss_every=1)
    # dump_losses(train_losses, val_losses, val_iters)
