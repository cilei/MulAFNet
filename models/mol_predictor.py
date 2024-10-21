import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
dtype = torch.float32


def create_var(tensor, device, requires_grad=None):
    if requires_grad is None:
        return Variable(tensor).to(device)
    else:
        return Variable(tensor, requires_grad=requires_grad).to(device)

class MolPredictor(nn.Module):
    def __init__(self, encoder, latent_dim=256, num_tasks=1, **kwargs):
        super().__init__()

        self.encoder = encoder
        self.predictor = nn.Linear(latent_dim, num_tasks)

    def forward(self, mol):
        emb = self.encoder(mol)
        out = self.predictor(emb)
        return out

    def from_pretrained(self, model_path, device):
        pre_model = torch.load(model_path, map_location=device)
        self.encoder.load_state_dict(pre_model)

class Classification(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, dropout=0.1):
        super(Classification, self).__init__()
        self.Linear1 = nn.Linear(in_size, hidden_size)
        self.Relu = nn.ReLU()
        self.Dropout = nn.Dropout(dropout)
        self.Linear2 = nn.Linear(hidden_size, out_size)

        torch.nn.init.xavier_uniform_(self.Linear1.weight)
        torch.nn.init.xavier_uniform_(self.Linear2.weight)

    def forward(self, src):
        hid = self.Linear1(src)
        hid = self.Relu(hid)
        hid = self.Dropout(hid)
        return self.Linear2(hid)

class MolPredictorAllM(nn.Module):
    def __init__(self, encoder1,encoder2, latent_dim=1280, num_tasks=1, **kwargs):
        super().__init__()

        self.encoder1 = encoder1
        self.encoder2 = encoder2

        self.featureFusion = FeatureFusionModule2(global_dim=128,local_dim=128)

        self.featureFusion2 = FeatureFusionModule3(global_dim=1024,local_dim=256)
        
        self.predictor = nn.Linear(latent_dim, num_tasks)

    def forward(self, mol):
        emb1 = self.encoder1(mol)
        emb2 = self.encoder2(mol)
        emb3 = mol.seqEncoderTensor

        output = self.featureFusion(emb1,emb2)
        
        outputall = self.featureFusion2(emb3,output)
        # embconcat1 = torch.concat([emb1,emb2],dim=-1)
        # embconcat2 = torch.concat([output,emb3],dim=-1)

        out = self.predictor(outputall)
        return out

    def from_pretrained_encoder1(self, model_path, device):
        pre_model = torch.load(model_path, map_location=device)
        self.encoder1.load_state_dict(pre_model)
    
    def from_pretrained_encoder2(self, model_path, device):
        pre_model = torch.load(model_path, map_location=device)
        self.encoder2.load_state_dict(pre_model)


class Atom_Model_decoder(nn.Module):
    def __init__(self,encoder1, hidden_size, device, dropout=0.2):
        super(Atom_Model_decoder, self).__init__()
        self.encoder1 = encoder1
        self.hidden_size = hidden_size
        self.device = device

        self.atom_num_s = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//4),
            nn.Softplus(),
            nn.Linear(hidden_size//4, 1)
            )
        
        self.bond_num_s = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//4),
            nn.Softplus(),
            nn.Linear(hidden_size//4, 1)
            )
        self.atom_num_pred_loss = nn.SmoothL1Loss(reduction="mean")
        self.bond_num_pred_loss = nn.SmoothL1Loss(reduction="mean")

        
    
    def forward(self, mol):
        emb1 = self.encoder1(mol)
        atom_num_loss, bond_num_loss = 0, 0

        atom_num_pred = self.atom_num_s(emb1).squeeze(-1)
        atom_num_target = mol.num_atoms

        atom_num_loss += self.atom_num_pred_loss(atom_num_pred, atom_num_target) / len(mol)
        atom_num_rmse = torch.sqrt(torch.sum((atom_num_pred - atom_num_target) ** 2)).item() / len(mol)

        bond_num_pred = self.bond_num_s(emb1).squeeze(-1)
        bond_num_target = mol.num_bond
        bond_num_loss += self.bond_num_pred_loss(bond_num_pred, bond_num_target) / len(mol)
        bond_num_rmse = torch.sqrt(torch.sum((bond_num_pred - bond_num_target) ** 2)).item() / len(mol)

        loss_tur = [atom_num_loss, bond_num_loss]

        loss_weight = create_var(torch.rand(2),self.device, requires_grad=True)
        loss_wei = torch.softmax(loss_weight, dim=-1)

        loss = 0
        for index in range(len(loss_tur)):
            loss += loss_tur[index] * loss_wei[index]
        
        return loss
    
#多头注意力生成流场，然后模态融合
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        values = values.reshape(N, -1, self.heads, self.head_dim)
        keys = keys.reshape(N, -1, self.heads, self.head_dim)
        queries = query.reshape(N, -1, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, -1, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out

class FeatureFusionModule2(nn.Module):
    def __init__(self, global_dim, local_dim, num_heads=8):
        super(FeatureFusionModule2, self).__init__()
        self.global_dim = global_dim
        self.local_dim = local_dim

        self.local_dim_adjust = nn.Linear(local_dim, global_dim)

        # Adjust the fc_out in MultiHeadAttention to match the adjusted local dimension
        self.flow_field_generator = MultiHeadAttention(global_dim * 2, num_heads)
        self.flow_field_generator.fc_out = nn.Linear(num_heads * (global_dim * 2 // num_heads), global_dim)

        self.output_network = nn.Sequential(
            nn.Linear(global_dim * 2, global_dim + local_dim),
            nn.ReLU(),
            nn.Linear(global_dim + local_dim, global_dim + local_dim)
        )

    def forward(self, global_features, local_features):
        local_features_adjusted = self.local_dim_adjust(local_features)

        combined_features = torch.cat([global_features, local_features_adjusted], dim=-1)

        flow_field = self.flow_field_generator(combined_features, combined_features, combined_features)

        refined_local_features = local_features_adjusted + flow_field.reshape(-1,128)

        final_features = torch.cat([refined_local_features, global_features], dim=-1)
        output = self.output_network(final_features)

        return output
    



#多头注意力生成流场，然后模态融合
class MultiHeadAttention3(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention3, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        values = values.reshape(N, -1, self.heads, self.head_dim)
        keys = keys.reshape(N, -1, self.heads, self.head_dim)
        queries = query.reshape(N, -1, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, -1, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out

class FeatureFusionModule3(nn.Module):
    def __init__(self, global_dim, local_dim, num_heads=8):
        super(FeatureFusionModule3, self).__init__()
        self.global_dim = global_dim
        self.local_dim = local_dim

        self.local_dim_adjust = nn.Linear(local_dim, global_dim)

        # Adjust the fc_out in MultiHeadAttention to match the adjusted local dimension
        self.flow_field_generator = MultiHeadAttention(global_dim * 2, num_heads)
        self.flow_field_generator.fc_out = nn.Linear(num_heads * (global_dim * 2 // num_heads), global_dim)

        self.output_network = nn.Sequential(
            nn.Linear(global_dim * 2, global_dim + local_dim),
            nn.ReLU(),
            nn.Linear(global_dim + local_dim, global_dim + local_dim)
        )

    def forward(self, global_features, local_features):
        local_features_adjusted = self.local_dim_adjust(local_features)

        combined_features = torch.cat([global_features, local_features_adjusted], dim=-1)

        flow_field = self.flow_field_generator(combined_features, combined_features, combined_features)

        refined_local_features = local_features_adjusted + flow_field.reshape(-1,1024)

        final_features = torch.cat([refined_local_features, global_features], dim=-1)
        output = self.output_network(final_features)

        return output
    