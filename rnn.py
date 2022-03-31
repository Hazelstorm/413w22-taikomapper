import torch, os
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from helper import *
import hyper_param
import matplotlib.pyplot as plt

HIDDEN_SIZE = 20
WEIGHTS = torch.tensor([0.01, 1, 1, 2, 2])
if torch.cuda.is_available():
    WEIGHTS = WEIGHTS.cuda()

TRAIN_PATH = os.path.join("data", "npy", "muzukashii")

class MapDataset2(Dataset): # overfitting to one song
    def __init__(self):
        self.path = TRAIN_PATH
        self.dir = os.listdir(self.path)

    def __len__(self):
        return len(self.dir)

    def __getitem__(self, idx):
        song_path = os.path.join(self.path, self.dir[idx])
        audio_data, notes_data = get_npy_data(song_path)
        audio_data, notes_data = torch.Tensor(audio_data), torch.Tensor(notes_data)
        return audio_data, notes_data

# Modified from RNN notebook
class taikoRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = HIDDEN_SIZE
        self.rnn = nn.GRU(input_size=hyper_param.n_mels, hidden_size=self.hidden_size)
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
    
        if checkpoint_path:
            torch.save(model.state_dict(), checkpoint_path.format(iter_num))
            
    return losses

model = taikoRNN()
if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load("checkpoint\\check.pt", map_location=torch.device('cpu')))
losses = train_rnn_network(model, learning_rate = 4e-4, num_iters=10000, wd=0, checkpoint_path="checkpoint/iter-{}.pt")
plt.plot(losses)
plt.show()
