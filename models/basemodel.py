import torch
import torch.nn as nn
import torch.nn.functional as F

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__() 
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self):
        super(BertEmbeddings, self).__init__()
        self.embedding = nn.Embedding(21128, 512)
        self.word_embeddings = nn.Embedding(21128, 512)
        self.position_embeddings = nn.Embedding(512, 512)
        self.token_type_embeddings = nn.Embedding(2, 512)
        self.LayerNorm = BertLayerNorm(512, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    

class BaseModel(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(BaseModel, self).__init__()
        self.embedding = BertEmbeddings()
        # Transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=2),
            num_layers=2
        )
        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv1d(embedding_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1)
        )
        # LSTM layers
        self.lstm = nn.LSTM(128, 64, num_layers=2, batch_first=True)

        
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x, token_type_ids=None):
        B,_,N = x.shape
        x = self.embedding(x, token_type_ids)
        x = x.reshape(B,N,-1)
        
        # Transformer layers
        transformer_out = self.transformer(x)

        # CNN layers
        cnn_output = self.cnn(transformer_out.transpose(1, 2))

        # LSTM layers
        
        lstm_output, _ = self.lstm(cnn_output.transpose(1, 2))
        lstm_output = lstm_output[:, -1, :]
        
        # Fully connected layer
        out = self.fc(lstm_output)
        return out