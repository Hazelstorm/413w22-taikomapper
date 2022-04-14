import torch, os, random, copy
from torch import nn
from torch.utils.data import Dataset, DataLoader
from helper import *
import hyper_param
import matplotlib.pyplot as plt
import tqdm

os.environ['KMP_DUPLICATE_LIB_OK']='True'

SEED = 88
HIDDEN_SIZE = 50

WEIGHTS = torch.tensor([0.1, 1, 1, 2, 2])
WEIGHT_MATRIX = torch.tensor([
    [0, 1, 1, 2, 2],
    [1, 0, .5, .5, .5],
    [1, .5, 0, .5, .5],
    [2, .5, .5, 0, .5],
    [2, .5, .5, .5, 0],
    ])

if torch.cuda.is_available():
    WEIGHTS = WEIGHTS.cuda()
    WEIGHT_MATRIX = WEIGHT_MATRIX.cuda()

def custom_loss(z, t):
    y = torch.softmax(z, dim=1)
    log_y = torch.log(y)
    loss_matrix = -1 * torch.matmul(torch.transpose(log_y, 0, 1), t)
    weighted_loss_matrix = torch.mul(loss_matrix, WEIGHT_MATRIX)
    return torch.sum(weighted_loss_matrix)

TRAIN_PATH = os.path.join("data", "npy", "muzukashii")

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

def validation_loss(model, val_loader: DataLoader, criterion):
    val_loss = []
    with torch.no_grad(): # disable gradient computation to save memory
        for audio_data, timing_data, notes_data in val_loader:
            if torch.cuda.is_available():
                audio_data = audio_data.cuda()
                notes_data = notes_data.cuda()
            model_out = torch.squeeze(model(audio_data), dim=0)
            notes_data = torch.squeeze(notes_data, dim=0)
            z = filter_model_output(model_out, timing_data["bar_len"].item(), timing_data["offset"].item(), unsnap_tolerance=0)
            t = filter_model_output(notes_data, timing_data["bar_len"].item(), timing_data["offset"].item(), unsnap_tolerance=2)
            model_loss = criterion(z, t)
            val_loss.append(model_loss.item())
    return sum(val_loss) / len(val_loss)

"""
Trains the RNN.
Arguments:
- checkpoint_path: path to save checkpoint files. {} needs to appear to store the iteration number (e.g. "ckpt-{}.pt").
- compute_baseline: Compute and print the baseline (all zeros) loss at the beginning.
- plot: Plot the training and validation curves.
- criterion: Loss Criterion.
"""
def train_rnn_network(model, num_epochs=100, learning_rate=1e-3, wd=0, 
    checkpoint_path=None, compute_baseline=False, plot=False, criterion=nn.CrossEntropyLoss(weight=WEIGHTS)):
    print(f"Beginning training (lr={learning_rate})")
    
    train_dataset = MapDataset(0, 0.6)
    val_dataset = MapDataset(0.6, 0.8)
    train_loader = DataLoader(train_dataset, shuffle=True)
    val_loader = DataLoader(val_dataset, shuffle=True)
    # criterion = nn.CrossEntropyLoss(weight=WEIGHTS)
    criterion = custom_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)

    train_losses = []
    val_losses = []
      
    if compute_baseline:
        baseline = baselineModel()
        if torch.cuda.is_available():
            baseline = baseline.cuda()
        print(f"Baseline Validation Loss: {validation_loss(baseline, val_loader, criterion)}")

    # training loop
    for epoch_num in range(num_epochs):
        # train loss
        train_loss = []
        model.train() 
        
        for audio_data, timing_data, notes_data in tqdm.tqdm(train_loader, desc=f"Epoch {epoch_num}"):
            if torch.cuda.is_available():
                audio_data = audio_data.cuda()
                notes_data = notes_data.cuda()
            optimizer.zero_grad()
            model_out = torch.squeeze(model(audio_data), dim=0)
            notes_data = torch.squeeze(notes_data, dim=0)
            z = filter_model_output(model_out, timing_data["bar_len"].item(), timing_data["offset"].item(), unsnap_tolerance=0)
            t = filter_model_output(notes_data, timing_data["bar_len"].item(), timing_data["offset"].item(), unsnap_tolerance=2)
            model_loss = criterion(z, t)
            model_loss.backward()
            optimizer.step()
            train_loss.append(model_loss.item())
        
        train_loss = sum(train_loss) / len(train_loss)
        train_losses.append(train_loss)
            
        # validation loss
        model.eval()
        val_loss = validation_loss(model, val_loader, criterion)
        val_losses.append(val_loss)

        print(f"Epoch {epoch_num + 1}/{num_epochs}" + 
              f" | Train Loss: {'{:.4f}'.format(train_losses[-1])}" + 
              f" | Val Loss: {'{:.4f}'.format(val_losses[-1])}")
    
        if checkpoint_path and (epoch_num % 100) == 0:
            torch.save(model.state_dict(), checkpoint_path.format(epoch_num,learning_rate))
    
    if plot:
        plt.plot(train_losses, label=f"Training Loss")
        plt.plot(val_losses, label=f"Validation Loss")
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()

    return train_losses, val_losses
        
model = taikoRNN()
if torch.cuda.is_available():
    model = model.cuda()

train_rnn_network(model, learning_rate=1e-4, num_epochs=1000, wd=0, checkpoint_path=None)

"""        
model = taikoRNN()
baseline = baselineModel()
if torch.cuda.is_available():
    model = model.cuda()
    baseline = baseline.cuda()

for lr in [1e-4, 1e-5, 1e-6]:
    model_copy = copy.deepcopy(model)
    
    train_losses, val_losses, baseline_loss = train_rnn_network(model_copy, baseline, learning_rate=lr, num_epochs=1000, wd=0, checkpoint_path="C:\\Users\\Natal\\413w22-taikomapper\\checkpoints\\ckpt-{}-{}.pk")
    plt.plot(train_losses, label=f"Training Loss (lr={lr})")
    plt.plot(val_losses, label=f"Validation Loss (lr={lr})")
    
plt.title(f"RNN Hyperparameter Tuning")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Cross-Entropy Loss")
plt.show()

print(f"Baseline Loss: {baseline_loss}")
"""
