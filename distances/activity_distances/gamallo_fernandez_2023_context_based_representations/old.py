import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Step 1: Preprocess Data
def preprocess_data(sequences, window_size):
    """Generate sliding window samples from activity sequences."""
    unique_activities = sorted(set(a for seq in sequences for a in seq))
    activity_to_idx = {act: idx for idx, act in enumerate(unique_activities)}
    idx_to_activity = {idx: act for act, idx in activity_to_idx.items()}

    padding_idx = len(activity_to_idx)  # Padding index

    samples = []
    for seq in sequences:
        seq_indices = [activity_to_idx[act] for act in seq]
        for i, target in enumerate(seq_indices):
            context = []
            for j in range(-window_size, window_size + 1):
                if j != 0 and 0 <= i + j < len(seq_indices):
                    context.append(seq_indices[i + j])
                elif j != 0:
                    context.append(padding_idx)  # Valid padding index
            samples.append((context, target))

    return samples, activity_to_idx, idx_to_activity, padding_idx



class Autoencoder(nn.Module):
    def __init__(self, num_activities, embedding_size, window_size, padding_idx):
        super(Autoencoder, self).__init__()
        self.num_activities = num_activities
        self.embedding_size = embedding_size
        self.context_size = 2 * window_size  # Fix the context size to match input

        # Embedding layer for activities
        self.activity_embedding = nn.Embedding(num_activities + 1, embedding_size, padding_idx=padding_idx)

        # Encoder and decoder
        self.encoder = nn.Linear(self.context_size * embedding_size, embedding_size)
        self.decoder = nn.Linear(embedding_size, self.context_size * embedding_size)

    def forward(self, context):
        # Get embeddings for the context
        context_embeddings = self.activity_embedding(context)  # Shape: (batch_size, context_size, embedding_size)

        batch_size, context_size, embedding_size = context_embeddings.size()
        assert context_size == self.context_size, f"Expected context size {self.context_size}, got {context_size}"
        assert embedding_size == self.embedding_size, f"Expected embedding size {self.embedding_size}, got {embedding_size}"

        context_embeddings = context_embeddings.view(batch_size, -1)  # Flatten

        # Encode and decode
        encoded = self.encoder(context_embeddings)
        decoded = self.decoder(encoded)

        # Reshape output back to context size
        decoded = decoded.view(batch_size, self.context_size, self.embedding_size)
        return decoded


def train_autoencoder(samples, model, epochs, batch_size, learning_rate):
    # Prepare data
    context_data = []
    target_data = []
    for context, target in samples:
        context_data.append(context)
        target_data.append(target)

    context_tensor = torch.tensor(context_data, dtype=torch.long)
    target_tensor = torch.tensor(target_data, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(context_tensor, target_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for context_batch, target_batch in dataloader:
            optimizer.zero_grad()
            output = model(context_batch)  # Shape: (batch_size, context_size, embedding_size)

            # Reshape output properly to match the target shape
            batch_size, context_size, embedding_dim = output.shape
            output_flat = output.view(batch_size * context_size,
                                      embedding_dim)  # Shape: (batch_size * context_size, embedding_dim)

            target_batch_flat = context_batch.view(-1)  # Flatten target batch

            # Ensure sizes match
            assert output_flat.shape[0] == target_batch_flat.shape[0], \
                f"Mismatch: output_flat {output_flat.shape}, target_batch_flat {target_batch_flat.shape}"

            # Compute loss
            loss = criterion(output_flat, target_batch_flat)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

# Step 4: Extract Embeddings
def extract_embeddings(model, idx_to_activity):
    embeddings = model.activity_embedding.weight.data.numpy()
    activity_embeddings = {idx_to_activity[idx]: embeddings[idx] for idx in range(len(idx_to_activity))}
    return activity_embeddings



# Example activity sequences
activity_sequences = [["a", "b", "c"], ["b", "c", "d", "e"]]

# Hyperparameters
embedding_size = 64  # Size of the embeddings
window_size = 2     # Context window size
learning_rate = 0.01
epochs = 100
batch_size = 32


samples, activity_to_idx, idx_to_activity, padding_idx = preprocess_data(activity_sequences, window_size)
num_activities = len(activity_to_idx)

# Initialize model
model = Autoencoder(num_activities=num_activities, embedding_size=embedding_size, window_size=window_size, padding_idx=padding_idx)
train_autoencoder(samples, model, epochs, batch_size, learning_rate)

activity_embeddings = extract_embeddings(model, idx_to_activity)
print("Activity Embeddings:", activity_embeddings)
