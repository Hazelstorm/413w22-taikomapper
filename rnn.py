import torch, os, random, copy
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from helper import *
import hyper_param
import matplotlib.pyplot as plt
from preprocessing_helpers import get_map_audio


SEED = 88
HIDDEN_SIZE = 50

WEIGHTS = torch.tensor([0.01, 1, 1, 2, 2])
if torch.cuda.is_available():
    WEIGHTS = WEIGHTS.cuda()

TRAIN_PATH = os.path.join("data", "npy", "muzukashii")

class MapDataset2(Dataset): 
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
        audio_data, timing_data, notes_data = get_npy_data(song_path)
        audio_data, notes_data = torch.Tensor(audio_data), torch.Tensor(notes_data)
        return audio_data, timing_data, notes_data

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
    
    

def train_rnn_network(model, baseline, num_epochs=100, learning_rate=1e-3, wd=0, checkpoint_path=None):
    print(f"Beginning training (lr={learning_rate})")
    
    train_dataset = MapDataset2(0, 0.6)
    val_dataset = MapDataset2(0.6, 0.8)
    test_dataset = MapDataset2(0.8, 1)
    train_loader = DataLoader(train_dataset, shuffle=True)
    val_loader = DataLoader(val_dataset, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=True)
    criterion = nn.CrossEntropyLoss(weight=WEIGHTS)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)

    train_losses = []
    val_losses = []
    baseline_losses = []
    
    num_iters = len(train_loader)
      
    # compute baseline loss
    baseline_loss = []
    
    for audio_data, timing_data, notes_data in train_loader:
        
        if torch.cuda.is_available():
            audio_data = audio_data.cuda()
            notes_data = notes_data.cuda()
    
        model_out = torch.squeeze(baseline(audio_data), dim=0)
        notes_data = torch.squeeze(notes_data, dim=0)
        y,t = filter_model_output(model_out, notes_data, timing_data)
        model_loss = criterion(y, t)
        baseline_loss.append(model_loss.item())
    
    baseline_loss = sum(baseline_loss) / len(baseline_loss)

    # training loop
    for epoch_num in range(num_epochs):
        
        # train loss
        train_loss = []
        model.train() 
        
        for audio_data, timing_data, notes_data in train_loader:
            
            if torch.cuda.is_available():
                audio_data = audio_data.cuda()
                notes_data = notes_data.cuda()

            optimizer.zero_grad()
            model_out = torch.squeeze(model(audio_data), dim=0)
            notes_data = torch.squeeze(notes_data, dim=0)
            y,t = filter_model_output(model_out, notes_data, timing_data)
            model_loss = criterion(y, t)
            model_loss.backward()
            optimizer.step()
            train_loss.append(model_loss.item())
        
        train_loss = sum(train_loss) / len(train_loss)
        train_losses.append(train_loss)
            
        # validation loss
        val_loss = []
        model.eval() 
        
        with torch.no_grad(): # disable gradient computation to save memory
            for audio_data, timing_data, notes_data in val_loader:
    
                if torch.cuda.is_available():
                    audio_data = audio_data.cuda()
                    notes_data = notes_data.cuda()
                    
                model_out = torch.squeeze(model(audio_data), dim=0)
                notes_data = torch.squeeze(notes_data, dim=0)
                y,t = filter_model_output(model_out, notes_data, timing_data)
                model_loss = criterion(y, t)
                val_loss.append(model_loss.item())
            
            val_loss = sum(val_loss) / len(val_loss)
            val_losses.append(val_loss)
        
        print(f"Epoch {epoch_num + 1}/{num_epochs}" + 
              f" | Train Loss: {'{:.4f}'.format(train_losses[-1])}" + 
              f" | Val Loss: {'{:.4f}'.format(val_losses[-1])}")
    
        if checkpoint_path:
            torch.save(model.state_dict(), checkpoint_path.format(iter_num))
            
    return train_losses, val_losses, baseline_loss
        
# model = taikoRNN()
# baseline = baselineModel()
# if torch.cuda.is_available():
#     model = model.cuda()
#     baseline = baseline.cuda()

# for lr in [1e-4, 1e-5, 1e-6]:
#     model_copy = copy.deepcopy(model)
    
#     train_losses, val_losses, baseline_loss = train_rnn_network(model_copy, baseline, learning_rate=lr, num_epochs=1000, wd=0, checkpoint_path=None)
#     plt.plot(train_losses, label=f"Training Loss (lr={lr})")
#     plt.plot(val_losses, label=f"Validation Loss (lr={lr})")
    
# plt.title(f"RNN Hyperparameter Tuning")
# plt.legend()
# plt.xlabel("Iteration")
# plt.ylabel("Cross-Entropy Loss")
# plt.show()

# print(f"Baseline Loss: {baseline_loss}")

