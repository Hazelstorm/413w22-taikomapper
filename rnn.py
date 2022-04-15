import torch
from torch import nn
import hyper_param
from helper import filter_to_snaps

WINDOW_LENGTH = 2 * hyper_param.window_size + 1

# Modified from RNN notebook
class notePresenceRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 50
        self.rnn = nn.GRU(input_size=(hyper_param.n_mels*WINDOW_LENGTH + 1), hidden_size=self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, 1)
    
    def forward(self, spectro, bar_len, offset):
        out, _ = self.rnn(spectro)
        out = self.fc(out)
        out = torch.squeeze(out, dim=2)
        out = torch.squeeze(out, dim=0)
        out = filter_to_snaps(out, bar_len, offset, 0)
        return out

class noteColourRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 50
        self.rnn = nn.GRU(input_size=(hyper_param.n_mels*WINDOW_LENGTH + 1), hidden_size=self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, 1)
    
    def forward(self, spectro, bar_len, offset, notes_data):
        spectro = torch.flatten(spectro, start_dim=1)
        notes_data = torch.transpose(notes_data, 0, 1)
        augmented_spectro = torch.cat((spectro, notes_data), 1)
        augmented_spectro = torch.unsqueeze(augmented_spectro, dim=0)
        out, _ = self.rnn(augmented_spectro)
        out = self.fc(out)
        out = torch.squeeze(out, dim=2)
        out = torch.squeeze(out, dim=0)
        # Allow the model to only colour notes that are present in notes_data
        notes_present = notes_data > 0
        out = torch.mul(out, torch.squeeze(notes_present, dim=1))
        return out

class noteFinisherRNN(noteColourRNN):
    def __init__(self):
        super().__init__()
