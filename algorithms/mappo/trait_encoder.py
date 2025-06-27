import torch
import torch.optim as optim
from torch import nn as nn
from torch.nn import functional as F
from algorithms.mappo.encoder_trait import Hero_Attention1

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Trait_Map(nn.Module):
    # create trait-map
    def __init__(self, args, layer_norm = True, lr=5e-5, device=torch.device("cuda")):
        super(Trait_Map, self).__init__()
        # args
        self.args = args
        self.hidden_size = self.args.hidden_size
        self.device = device
        self.pro_map = None

        self.layer1 = nn.Linear(self.args.num_trait, self.args.hidden_size * 2)
        self.layer2 = nn.Linear(self.args.hidden_size * 2, self.args.hidden_size)
        self.layer3 = nn.Linear(self.args.hidden_size, self.args.trait_dim)
        self.layernorm = nn.LayerNorm(self.args.trait_dim, eps=1e-6)

        self.layer_atten = nn.Linear(self.args.hidden_size * 2, self.args.trait_dim)

        self.apply(weights_init_)
        self.lr = lr
        self.layer_norm = layer_norm
        
        # hero encoder
        # self.hero_atten = Hero_Attention(self.args, device=self.device)
        self.hero_atten = Hero_Attention1(self.args, self.args.num_trait, device=self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)


    def forward(self, feature):
        # encode trait
        feature = feature.reshape(self.args.n_rollout_threads, self.args.num_trait, self.args.num_agent)
        feature = feature.permute(0,2,1)

        x = torch.relu(self.layer1(feature))
        x_ = torch.relu(self.layer2(x))
        trait_feature = self.layer3(x_)

        trait_feature = torch.sigmoid(trait_feature)
        return  trait_feature
    
    def update_network(self, target_var, pred_var):
        loss = F.mse_loss(target_var, pred_var, reduction='none')
        loss = loss.sum(dim=-1, keepdim=True)
        loss = loss.reshape(-1,1).mean().to(self.device)

        # self.optimizer.zero_grad()
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
        # self.optimizer.step()

        return loss