import torch
from model import mlp_bloc

class MNISTGeneratorLatent(torch.nn.Module) :
    # We use same architeccture to provide fair comparaison
    def __init__(self, latent_dim) :
        super().__init__()
        self.latent_dim = latent_dim

        self.model = torch.nn.Sequential(*[
            mlp_bloc(latent_dim, 1024, torch.nn.SiLU()),
            mlp_bloc(1024, 2048, torch.nn.SiLU()),
            mlp_bloc(2048, 1024, torch.nn.SiLU()),
            mlp_bloc(1024, latent_dim, torch.nn.Identity()),
        ])

        self.t_injection = torch.nn.Sequential(torch.nn.Linear(1, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 2048))

    def forward(self, x, t) :
        x_first_layer = self.model[0](x.reshape(-1, self.latent_dim))
        t_injected    = self.t_injection(t)

        x0 = x_first_layer*(1+t_injected[..., :1024]) + t_injected[..., 1024:2048]
        x1 = self.model[1](x0)
        x  = self.model[2](x1)
        return self.model[3](x+x0)
    
    def get_gradients_updates(self) :
        """
        Assumes loss.backward before, we dont take in acccount the momentums nor the lr, just the raw gradients
        """
        gradients = []
        for layer in self.model :
            gradients.append(layer.model[0].weight.grad.norm().item() / layer.model[0].weight.norm().item())
        return gradients
