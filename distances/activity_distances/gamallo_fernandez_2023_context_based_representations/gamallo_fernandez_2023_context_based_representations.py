import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import lightning as pl
from torch.utils.data import DataLoader, Dataset


# Updated Dataset Class for One-Hot Encoding
class AEracDataset(Dataset):
    def __init__(self, sequences, num_categories, win_size):
        #input:dataset: pd.DataFrame
        #input: num categories = len of alphabet?
        #self.dataset = dataset.reset_index()  # Not drop index

        self.num_categories = num_categories
        self.win_size = win_size
        self.samples = self._generate_samples(sequences)

    def _generate_samples(self, sequences):
        samples = []
        for seq in sequences:
            for i in range(len(seq)):
                context = []
                for j in range(-self.win_size, self.win_size + 1):
                    if j != 0 and 0 <= i + j < len(seq):
                        context.append(seq[i + j])
                    elif j != 0:
                        context.append(self.num_categories)  # Padding index

                samples.append((context, seq[i]))  # Context, target

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        context, target = self.samples[idx]
        context_tensor = torch.zeros((len(context), self.num_categories + 1))

        for i, event in enumerate(context):
            context_tensor[i, event] = 1  # One-hot encoding

        return context_tensor, target


# Updated Encoder with Separate Linear Layers
class Encoder(nn.Module):
    def __init__(self, num_categories, win_size, emb_size):
        super().__init__()
        self.num_categories = num_categories
        self.win_size = win_size
        self.emb_size = emb_size

        self.input_layers = nn.ModuleList([
            nn.Linear(num_categories + 1, emb_size, bias=False) for _ in range(2 * win_size)
        ])
        self.hidden_layer = nn.Linear((2 * win_size) * emb_size, emb_size, bias=False)

    def forward(self, inputs):
        x = torch.cat([layer(inputs[:, i]) for i, layer in enumerate(self.input_layers)], dim=1)
        return self.hidden_layer(x)


# Updated Decoder with Separate Output Layers
class Decoder(nn.Module):
    def __init__(self, num_categories, win_size, emb_size):
        super().__init__()
        self.num_categories = num_categories
        self.win_size = win_size
        self.emb_size = emb_size

        self.hidden_layer = nn.Linear(emb_size, (2 * win_size) * emb_size, bias=False)
        self.output_layers = nn.ModuleList([
            nn.Linear((2 * win_size) * emb_size, num_categories + 1, bias=False) for _ in range(2 * win_size)
        ])

    def forward(self, code):
        h = self.hidden_layer(code)
        return torch.cat([layer(h).unsqueeze(1) for layer in self.output_layers], dim=1)


# Updated AEracModel with BCE Loss
class AEracModel(pl.LightningModule):
    def __init__(self, num_categories, win_size, emb_size):
        super().__init__()
        self.encoder = Encoder(num_categories, win_size, emb_size)
        self.decoder = Decoder(num_categories, win_size, emb_size)
        self.loss_function = nn.BCEWithLogitsLoss()

    def forward(self, inputs):
        code = self.encoder(inputs)
        return self.decoder(code)

    def training_step(self, batch, batch_idx):
        inputs, _ = batch
        reconstruction = self.forward(inputs)
        loss = self.loss_function(reconstruction, inputs)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.01)


# Function to Train Model
def train_model(eventlog, num_categories, win_size, emb_size, batch_size=32, epochs=50):
    dataset = AEracDataset(eventlog, num_categories, win_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AEracModel(num_categories, win_size, emb_size)

    # Select GPU if available, MPS for Apple, else CPU
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1
    elif torch.backends.mps.is_available():  # Mac M1/M2 support
        accelerator = "mps"
        devices = "auto"
    else:
        accelerator = "cpu"
        devices = "auto"

    trainer = pl.Trainer(max_epochs=epochs, accelerator=accelerator, devices=devices)
    trainer.fit(model, dataloader)


# Example Usage
activity_sequences = [[0, 1, 2], [1, 2, 3, 4]]  # Example sequences (numeric categories)
num_activities = 5  # Assume 5 unique activities (0-4)

train_model(activity_sequences, num_activities, win_size=2, emb_size=64)
