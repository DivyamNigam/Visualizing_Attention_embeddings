# Visualizing_Attention_embeddings
Visualizing embeddings from attention and without attention
Tokenizing sentence- "I would like sit at river bank and get money from bank.", with and without attention mechanism and plottong words on a 3d graph.

Using `google-bert/bert-base-uncased` model to tokenize text and for creating embeddings of dimension 3 then applying self attention to it
`class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.query_linear = nn.Linear(768, 768)
        self.key_linear = nn.Linear(768, 768)
        self.value_linear = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.1)

    def forward(self, embeddings):
        query = self.query_linear(embeddings)
        key = self.key_linear(embeddings)
        value = self.value_linear(embeddings)
        
        # Transpose key
        key_transposed = key.permute(0, 2, 1)
        
        attention_scores = torch.matmul(query, key_transposed) / math.sqrt(768)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, value)
        return context

attention_model = Attention()`

![newplot (1)](https://github.com/user-attachments/assets/25ea7720-6d9f-4bd8-b0e4-85f688fbcf79)
![newplot](https://github.com/user-attachments/assets/cf727f38-1355-40f0-bc45-9a77418624a6)

