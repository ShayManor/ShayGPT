import torch
from torch import nn
from torch.utils.data import DataLoader

from src.TextDataset import TextDataset
from src.TransformerEncoderModel import TransformerEncoderModel
from src.clean import full_clean
from src.tokenizer import tokenizer


def train(data, epochs):
    model = TransformerEncoderModel(tokenizer.get_vocab_size())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))
    dataset = TextDataset(data, max_len=128)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for batch_ids, batch_att in loader:
            # [:, :-1] removes last column
            input = batch_ids[:, :-1]
            tgt = batch_ids[:, 1:]
            att = batch_att[:, :-1]
            out = model(input, att)
            # tie weights for LM head
            logits = out @ model.embedding.weight.T
            loss = criterion(logits.view(-1, logits.size(-1)), tgt.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}  Loss {loss.item():.4f}")

    emb_matrix = model.embedding.weight.data.cpu().numpy()
    torch.save(emb_matrix, "embed_matrix.pth")
    torch.save(model.state_dict(), "transformer_encoder.pth")


if __name__ == '__main__':
    train('data/AI_Human.csv', 6)
