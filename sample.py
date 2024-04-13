import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_heads, num_encoder_layers, num_decoder_layers):
        super(TransformerModel, self).__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.transformer = nn.Transformer(
            d_model=embedding_size,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers
        )
        self.fc = nn.Linear(embedding_size, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        output = self.transformer(src, tgt)
        output = self.fc(output)
        return output

# Define the training process
def train(model, optimizer, criterion, src, tgt):
    optimizer.zero_grad()
    output = model(src, tgt)
    loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
    loss.backward()
    optimizer.step()
    return loss.item()

# Define a simple tokenization function
def tokenize(text):
    return text.split()

# Example usage
text_data = ["Hello, how are you?", "What is your name?", "How is life?"]
tokenizer = get_tokenizer("basic_english")
vocab = build_vocab_from_iterator(map(tokenize, text_data), specials=["<pad>", "<bos>", "<eos>", "<unk>"])
tokens_to_text = {idx:txt for idx, txt in enumerate(vocab.get_itos())}
print(tokens_to_text)
vocab_size = len(vocab)
embedding_size = 128
num_heads = 8
num_encoder_layers = 4
num_decoder_layers = 4

model = TransformerModel(vocab_size, embedding_size, num_heads, num_encoder_layers, num_decoder_layers)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])

# Training loop
for epoch in range(100):
    total_loss = 0
    for text in text_data:
        tokens = ["<bos>"] + tokenize(text) + ["<eos>"]
        src = torch.tensor([vocab[token] for token in tokens[:-1]])
        tgt = torch.tensor([vocab[token] for token in tokens[1:]])
        loss = train(model, optimizer, criterion, src.unsqueeze(0), tgt.unsqueeze(0))
        total_loss += loss
    print(f"Epoch {epoch+1}, Loss: {total_loss}")

    
    # Text generation after each epoch
    

    print(f"Epoch {epoch+1}, Loss: {total_loss}")


with torch.no_grad():
    for initial_text in ["Hello,", "What", "How are"]:
        input_tokens = ["<bos>"] + tokenize(initial_text)
        input_tensor = torch.tensor([vocab[token] for token in input_tokens]).unsqueeze(0)
        
        # Generate text by predicting one token at a time
        max_length = 10
        for _ in range(max_length):
            output = model(input_tensor, input_tensor)  # Using same input for src and tgt for autoregressive generation
            predicted_token = output.argmax(dim=-1)[:,-1].item()

            input_tensor = torch.cat([input_tensor, torch.tensor([[predicted_token]])], dim=-1)
            if predicted_token == vocab["<eos>"]:
                break
        
        # Convert the generated token indices back to text
        output_tokens = list(input_tensor.squeeze().detach().cpu().numpy())
        # print(output_tokens)
        # predicted_text = tokens_to_text(output_tokens, token_to_index)
        predicted_text = [tokens_to_text[t] for t in output_tokens]
        print(predicted_text)
        print(f"Initial Text: {initial_text}, Predicted Text: {predicted_text}")