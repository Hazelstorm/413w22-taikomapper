import torch
from torch import nn
import hyper_param

WINDOW_LENGTH = 2 * hyper_param.window_size + 1

# Modified from RNN notebook
class notePresenceRNN(nn.Module):
    def __init__(self, emb_size=hyper_param.notePresenceRNN_embedding_size, 
            hidden_size=hyper_param.notePresenceRNN_hidden_size):
        super().__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.embedding = nn.Linear(hyper_param.n_mels*WINDOW_LENGTH, self.emb_size)
        self.rnn = nn.GRU(input_size=emb_size, hidden_size=self.hidden_size, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size * 2, 1)
    
    def forward(self, audio_windows):
        audio_windows = torch.unsqueeze(audio_windows, dim=0)  # Batch size is always 1
        out = self.embedding(audio_windows)
        out, _ = self.rnn(out)
        out = self.fc(out)
        out = torch.squeeze(out, dim=2)
        out = torch.squeeze(out, dim=0)
        return out

class noteColourRNN(nn.Module):
    def __init__(self, emb_size=hyper_param.noteColourRNN_embedding_size, 
            hidden_size=hyper_param.noteColourRNN_hidden_size):
        super().__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.embedding = nn.Linear(hyper_param.n_mels*WINDOW_LENGTH + 1, self.emb_size)
        self.rnn = nn.GRU(input_size=emb_size, hidden_size=self.hidden_size, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size * 2, 1)
    
    def forward(self, audio_windows, notes_data):
        # Concatenate notes_data to the end of audio_windows
        notes_data = torch.transpose(notes_data, 0, 1)
        augmented_spectro = torch.cat((audio_windows, notes_data), 1)
        augmented_spectro = torch.unsqueeze(augmented_spectro, dim=0)
        
        out = self.embedding(augmented_spectro)
        out, _ = self.rnn(out)
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
