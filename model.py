import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):

    def __init__(self, in_channels, hidden_channels, scale_factor=1):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1)
        )
        conv_layers = []
        size = in_channels
        for i in range(scale_factor):
            conv_layers.append(
                nn.Conv2d(size, hidden_channels // (2**(scale_factor-i)), kernel_size=4, stride=2, padding=1)
            )
            conv_layers.append(nn.ReLU(inplace=True))
            size = hidden_channels // (2 ** (scale_factor-i))
        conv_layers += [
            nn.Conv2d(hidden_channels // 2, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        ]
        self.conv = nn.Sequential(*conv_layers)
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels, hidden_channels // 4, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(hidden_channels // 4, hidden_channels//2, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(hidden_channels // 2, hidden_channels, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        # )

    def forward(self, x):

        x = self.conv(x)

        x = x + self.res_block(x)

        x = x + self.res_block(x)

        return x

class Decoder(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, scale_factor=1):
        super().__init__()

        self.initial = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
        )
        self.res_block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1)
        )
        conv_layers=[]
        size=hidden_channels
        for i in range(scale_factor):
            conv_layers.append(
                nn.ConvTranspose2d(size, hidden_channels // (2**(i)), kernel_size=4, stride=2, padding=1)
            )
            conv_layers.append(nn.ReLU(inplace=True))
            size = hidden_channels // (2**i)
        
        conv_layers += [
            nn.ConvTranspose2d(hidden_channels // (2**i), out_channels, kernel_size=4, stride =2, padding=1),
        ]
        self.conv = nn.Sequential(*conv_layers)

        # self.conv = nn.Sequential(
        #     nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(hidden_channels // 2, out_channels=hidden_channels//4, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(hidden_channels // 4, out_channels, kernel_size=4, stride =2, padding=1),
        # )

    def forward(self, x ):
        x = self.initial(x)

        x = x + self.res_block(x)

        x = x + self.res_block(x)

        x = self.conv(x)

        return x
    

#https://github.com/MishaLaskin/vqvae/blob/master/models/quantizer.py
class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta = 0.25):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        # first torch.sum produces (B*H*W,1) tensor, second torch.sum produces (n_e,) tensor, according to 
        #broadcasting rules, produces (B*H*W, n_e) tensor
        # matmul between (B*H*W, e_dim) and (e_dim, n_e) produces (B*H*W, n_e) tensor)
        #outputs a tensor of shape (B*H*W, n_e)
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients 
        #pass through basically
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices




class VQVAE(nn.Module):
    def __init__(self, h_dim,
                 n_embeddings, embedding_dim, scale_factor=1, beta=0.25, save_img_embedding_map=False):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(in_channels=3, hidden_channels=h_dim, scale_factor=scale_factor)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(in_channels=embedding_dim, hidden_channels=h_dim, out_channels=3, scale_factor=scale_factor)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False):

        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(
            z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity
    
    def save_model(self, file_path: str):
        """
        Save the model's state_dict to the specified file path.

        Args:
            file_path (str): Where to save the model (.pth or .pt file).
        """
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path:str):
        self.load_state_dict(torch.load(file_path))
        print(f'model loaded from {file_path}')




