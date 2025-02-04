from model import VQVAE
import torch
import torch.optim as optim
import numpy as np
from utils import PNGImageDataset, compute_dataset_variance
from torch.utils.data import DataLoader
import torchvision.transforms as T

transform = T.Compose([
    T.ToTensor()           # scales [0, 255] -> [0, 1] for each channel
])



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VQVAE(h_dim=256, n_embeddings=32, embedding_dim=32).to(device)

optimizer = optim.Adam(model.parameters(), lr=2e-4, amsgrad=True)

model.train()

n_updates = 10000

results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': [],
}

log_interval = 100
save = False
train_dataset = PNGImageDataset(folder_path='./images', transform=transform)
x_train_var = 1.0
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)


def train():
    update_count = 0
    while update_count < n_updates:
        for x in train_loader:
            # x is a batch of images with shape (batch_size, 3, height, width)
            x = x.to(device)

            optimizer.zero_grad()

            # Forward pass: run through VQ-VAE
            embedding_loss, x_hat, perplexity = model(x)
            
            # Reconstruction loss
            recon_loss = torch.mean((x_hat - x)**2) / x_train_var
            
            # Total loss
            loss = recon_loss + embedding_loss

            # Backprop
            loss.backward()
            optimizer.step()

            # Record results
            results["recon_errors"].append(recon_loss.item())
            results["perplexities"].append(perplexity.item())
            results["loss_vals"].append(loss.item())
            results["n_updates"] = update_count

            # Logging
            if update_count % log_interval == 0:
                if save:
                    model.save_model("model.pth")  # or your chosen file path

                print(
                    f'Update #{update_count} | '
                    f'Recon Error: {np.mean(results["recon_errors"][-log_interval:]):.4f} | '
                    f'Loss: {np.mean(results["loss_vals"][-log_interval:]):.4f} | '
                    f'Perplexity: {np.mean(results["perplexities"][-log_interval:]):.4f}'
                )

            update_count += 1
            if update_count >= n_updates:
                break  # Stop if we've reached the desired number of updates

    return results

if __name__ == "__main__":
    train()