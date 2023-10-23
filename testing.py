from sarcasm_dataset import SarcasmDataset
from torch.utils.data import DataLoader
from util import custom_collate_fn
from gru import GRU
import torch
from tqdm import tqdm


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loaded device: {device}")

    dataset_val = SarcasmDataset("val_data.json", "word2vec/word2vec.model", pool_sequence=False)
    loader_val = DataLoader(dataset_val, batch_size=16, collate_fn=custom_collate_fn)

    model = GRU(embedding_size=100)
    model.load_state_dict(torch.load("models/GRU+BCELoss+Adam+20-Oct_12-50/epoch10.pt"))
    model.eval()

    corrects = 0

    for x_batch, y_batch in tqdm(loader_val):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        output = model(x_batch)
        pred = torch.round(output)

        corrects += torch.sum(pred == y_batch)

    print("Accuracy: ", corrects/len(dataset_val))
