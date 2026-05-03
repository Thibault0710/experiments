import torch



# class MNISTGenerator(torch.nn.Module):

#     def __init__(self, depth=3):
#         super().__init__()

#         self.conv_model = torch.nn.Sequential(
#             torch.nn.Conv2d(1,4,3,stride=2,padding=1),  # 28→14
#             torch.nn.SiLU(),
#             torch.nn.BatchNorm2d(4),
#             torch.nn.Conv2d(4,4,3,stride=2,padding=1),  # 14→7
#             torch.nn.SiLU(),
#             torch.nn.BatchNorm2d(4),
#             torch.nn.Conv2d(4,4,3,padding=1)
#         )

#         layers = [mlp_bloc(196,256,torch.nn.SiLU())]

#         for _ in range(depth):
#             layers.append(mlp_bloc(256,256,torch.nn.SiLU()))

#         layers.append(torch.nn.Linear(256,196))

#         self.mlp = torch.nn.Sequential(*layers)

#         self.unconv_model = torch.nn.Sequential(

#             torch.nn.ConvTranspose2d(4,8,3,stride=2,padding=1,output_padding=1),  # 7 → 14
#             torch.nn.SiLU(),

#             torch.nn.ConvTranspose2d(8,4,3,stride=2,padding=1,output_padding=1),  # 14 → 28
#             torch.nn.SiLU(),

#             torch.nn.Conv2d(4,1,3,padding=1)
#         )

#         self.adaln = torch.nn.Sequential(torch.nn.Linear(1, 128), torch.nn.ReLU(), torch.nn.Linear(128, 196*2))

#     def forward(self,x, t):

#         B = x.shape[0]

#         x = x.unsqueeze(1)
#         x_conv = self.conv_model(x).reshape(B,196)
#         cond = self.adaln(t)

#         x_conv = torch.nn.functional.layer_norm(x_conv, [196])

#         x_conv = x_conv*(1+cond[..., 196:]) + cond[..., :196]
#         x_unconv = self.mlp(x_conv)

#         return self.unconv_model(x_unconv.reshape(B,4,7,7)).squeeze(1)

class mlp_bloc(torch.nn.Module) :
    def __init__(self, input_dim, output_dim, act) :
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            act
        )
    
    def forward(self, x) :
        return self.model(x)

class SymLog(torch.nn.Module):
    def forward(self, x):
        return torch.sign(x) * torch.log1p(torch.abs(x))

class MNISTGenerator(torch.nn.Module):
    def __init__(self) :
        super().__init__()

        # self.batch_norm = torch.nn.BatchNorm1d(784)
        self.first_layer = mlp_bloc(784, 1024, torch.nn.SiLU())

        self.model = torch.nn.Sequential(*[
            mlp_bloc(1024, 2048, torch.nn.SiLU()),
            mlp_bloc(2048, 1024, torch.nn.SiLU()),
            mlp_bloc(1024, 784, torch.nn.Identity()),
        ])

        self.t_injection = torch.nn.Sequential(torch.nn.Linear(1, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 2048))

    def forward(self, x, t) :
        x_first_layer = self.first_layer(x.reshape(-1, 784))
        t_injected  = self.t_injection(t)

        x0 = x_first_layer*(1+t_injected[..., :1024]) + t_injected[..., 1024:2048]
        x1 = self.model[0](x0)
        x  = self.model[1](x1)
        return self.model[2](x+x0).reshape(-1, 28, 28)

    def get_gradients_updates(self) :
        """
        Assumes loss.backward before, we dont take in acccount the momentums nor the lr, just the raw gradients
        """
        gradients = [self.first_layer.model[0].weight.grad.norm().item() / self.first_layer.model[0].weight.norm().item()]
        for layer in self.model :
            gradients.append(layer.model[0].weight.grad.norm().item() / layer.model[0].weight.norm().item())
        return gradients