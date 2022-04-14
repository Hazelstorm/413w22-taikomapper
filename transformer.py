import torch, os, random, copy, math
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
from helper import *
import hyper_param
import matplotlib.pyplot as plt

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


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(hyper_param.max_ms).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, hyper_param.max_ms, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.1):
        super().__init__()
        self.model_type = 'Transformer'
        self.embedding = nn.Linear(hyper_param.n_mels, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, 5)

    def forward(self, src, src_mask):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [batch_size, seq_len, 5]
        """
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


def train_transformer_network(model, baseline, num_epochs=100, learning_rate=1e-3, wd=0, checkpoint_path=None):
    print(f"Beginning training (lr={learning_rate})")
    
    train_dataset = MapDataset(0, 0.6)
    val_dataset = MapDataset(0.6, 0.8)
    test_dataset = MapDataset(0.8, 1)
    train_loader = DataLoader(train_dataset, shuffle=True)
    val_loader = DataLoader(val_dataset, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=True)
    criterion = nn.CrossEntropyLoss(weight=WEIGHTS)
    # criterion = custom_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)

    train_losses = []
    val_losses = []
      
    # compute baseline loss
    baseline_loss = []
    
    for audio_data, timing_data, notes_data in train_loader:
        
        if torch.cuda.is_available():
            audio_data = audio_data.cuda()
            notes_data = notes_data.cuda()
    
        model_out = torch.squeeze(baseline(audio_data), dim=0)
        notes_data = torch.squeeze(notes_data, dim=0)
        z = filter_model_output(model_out, timing_data["bar_len"].item(), timing_data["offset"].item(), unsnap_tolerance=0)
        t = filter_model_output(notes_data, timing_data["bar_len"].item(), timing_data["offset"].item(), unsnap_tolerance=2)
        model_loss = criterion(z, t)
        baseline_loss.append(model_loss.item())
    
    baseline_loss = sum(baseline_loss) / len(baseline_loss)

    # training loop
    for epoch_num in range(num_epochs):
        
        # train loss
        train_loss = []
        model.train() 
        
        for audio_data, timing_data, notes_data in train_loader:
            src_mask = torch.zeros(audio_data.shape[1])
            
            if torch.cuda.is_available():
                audio_data = audio_data.cuda()
                notes_data = notes_data.cuda()

            optimizer.zero_grad()
            model_out = torch.squeeze(model(audio_data, src_mask), dim=0)
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
        val_loss = []
        model.eval() 
        
        with torch.no_grad(): # disable gradient computation to save memory
            for audio_data, timing_data, notes_data in val_loader:
                src_mask = torch.zeros(audio_data.shape[1])
    
                if torch.cuda.is_available():
                    audio_data = audio_data.cuda()
                    notes_data = notes_data.cuda()
                    
                model_out = torch.squeeze(model(audio_data, src_mask), dim=0)
                notes_data = torch.squeeze(notes_data, dim=0)
                z = filter_model_output(model_out, timing_data["bar_len"].item(), timing_data["offset"].item(), unsnap_tolerance=0)
                t = filter_model_output(notes_data, timing_data["bar_len"].item(), timing_data["offset"].item(), unsnap_tolerance=2)
                model_loss = criterion(z, t)
                val_loss.append(model_loss.item())
            
            val_loss = sum(val_loss) / len(val_loss)
            val_losses.append(val_loss)
        
        print(f"Epoch {epoch_num + 1}/{num_epochs}" + 
              f" | Train Loss: {'{:.4f}'.format(train_losses[-1])}" + 
              f" | Val Loss: {'{:.4f}'.format(val_losses[-1])}")
    
        if checkpoint_path and (epoch_num % 100) == 0:
            torch.save(model.state_dict(), checkpoint_path.format(epoch_num,learning_rate))
            
    return train_losses, val_losses, baseline_loss
        
# model = TransformerModel(d_model: 512, nhead: 8, d_hid: 2048, nlayers: 6)
# baseline = baselineModel()
# if torch.cuda.is_available():
#     model = model.cuda()
#     baseline = baseline.cuda()

# train_losses, val_losses, baseline_loss = train_rnn_network(model, baseline, learning_rate=1e-4, num_epochs=1000, wd=0, checkpoint_path=None)
