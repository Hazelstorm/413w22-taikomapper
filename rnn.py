import torch, os
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from helper import *
import hyper_param
from tqdm import tqdm

HIDDEN_SIZE = 10

WEIGHTS = torch.tensor([0.01, 1, 1, 2, 2])

class MapDataset2(Dataset): # overfitting to one song
    def __init__(self):
        self.song_path = "data\\npy\\muzukashii\\data 2013 100019 Owl City & Carly Rae Jepsen - Good Time"
        self.audio_data, self.notes_data = get_npy_data(self.song_path)
        self.audio_data = torch.from_numpy(self.audio_data)
        self.notes_data = torch.from_numpy(self.notes_data)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.audio_data, self.notes_data

# Modified from RNN notebook
class taikoRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = HIDDEN_SIZE
        self.rnn = nn.RNN(input_size=hyper_param.n_mels, hidden_size=self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, 5)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out

def train_rnn_network(model, num_iters=100, learning_rate=1e-3, wd=0, checkpoint_path=None):
    dataset = MapDataset2()
    train_loader = DataLoader(dataset)
    criterion = nn.CrossEntropyLoss(weight=WEIGHTS)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)

    losses = []

    for iter_num in range(num_iters):
        for audio_data, notes_data in train_loader:
            if torch.cuda.is_available():
                audio_data = audio_data.cuda()
                notes_data = notes_data.cuda()

            optimizer.zero_grad()
            model_out = torch.squeeze(model(audio_data), dim=0)
            notes_data = torch.squeeze(notes_data, dim=0)
            model_loss = criterion(model_out, notes_data)
            model_loss.backward()
            optimizer.step()
            losses.append(float(model_loss))
            print(f"Iteration {iter_num + 1} | Loss: {model_loss}")
    
        if checkpoint_path and iter_num % 100 == 0:
            torch.save(model.state_dict(), checkpoint_path.format(iter_num))

model = taikoRNN()
train_rnn_network(model, learning_rate = 1e-4, num_iters=10000, wd=0, checkpoint_path="checkpoint/iter-{}.pt")
