import os
import torch
from torch.nn import nn

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.word_embedding = nn.Embeding( self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM( input_size = self.embed_size,
                            hidden_size = self.hidden_size,
                            num_layers = self.num_layers,
                            batch_first = True
                            )
        self.fc = nn.Linear( self.hidden_size, self.vocab_size)
        
    
    def init_hidden(self, batch_first):
        return ( torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros( self.num_layers, batch_size, self.hidden_size).to(device))
        
    def forward(self, features, caption):
        captions = caption[:,:-1]
        self.batch_size = features.shape[0]
        self.hidden = self.init_hidden(self.batch_size)
        embeds = self.word_embedding(captions)
        inputs = torch.cat((features.unsqueeze(dim=1), embeds), dim=1)
        lstm_out, self.hidden = self.lstm(inputs, self.hidden)
        outputs = self.fc(lstm_out)
        return outputs