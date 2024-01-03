import torch
import torch.nn as nn
from typing import List


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int):
        
        super(Encoder, self).__init__()
        self.dims:List[int] = [input_dim] + hidden_dims + [output_dim]
        self.linear_layers = nn.ModuleList()
        for ii in range(1, len(self.dims)):
            self.linear_layers.append( nn.Linear(self.dims[ii-1], self.dims[ii]) )
    
    
    def forward(self, x: torch.tensor):
        for ii in range(len(self.dims) - 2):
            x = self.linear_layers[ii](x)
            x = torch.nn.functional.relu(x)
        x = self.linear_layers[-1](x)
        return x

    
    def check(self):
        print("Encoder:")
        for ii in range(len(self.dims) - 1):
            print(self.linear_layers[ii])
     
            
class Decoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int):
        
        super(Decoder, self).__init__()
        self.dims:List[int] = [input_dim] + hidden_dims + [output_dim]
        self.linear_layers = nn.ModuleList()
        for ii in range(1, len(self.dims)):
            self.linear_layers.append( nn.Linear(self.dims[ii-1], self.dims[ii]) )
        
    
    def forward(self, x: torch.tensor):
        for ii in range(len(self.dims) - 2):
            x = self.linear_layers[ii](x)
            x = torch.nn.functional.relu(x)
        x = self.linear_layers[-1](x)
        return x
    
    
    def check(self):
        print("Decoder:")
        for ii in range(len(self.dims)-1):
            print(self.linear_layers[ii])
            
            
class AutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int):
        super(AutoEncoder, self).__init__()
        self.input_dim: int = input_dim
        self.hidden_dims: List[int] = hidden_dims
        self.output_dim: int = output_dim
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.encoders.append(
            Encoder(
                input_dim=input_dim, 
                hidden_dims=hidden_dims,
                output_dim=output_dim)
        )
        self.decoders.append(
            Decoder(
                input_dim=output_dim,
                hidden_dims=hidden_dims[::-1],
                output_dim=input_dim
            )
        )
    
    
    def forward(self, x: torch.tensor):
        x = self.encoders[0](x)
        x = self.decoders[0](x)
        return x
    
        
    def check(self):
        print("AutoEncoder:")
        self.encoders[0].check()
        self.decoders[0].check()
    
    
    def get_checkpoint_info(self):
        input_dim: int = self.input_dim
        hidden_dims: List[int] = self.hidden_dims
        output_dim: int = self.output_dim
        state_dict = self.state_dict()
        
        checkpoint = {
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "output_dim": output_dim,
            "@state_dict": state_dict
        }
        
        return checkpoint
    

def save_checkpoint(
    model: AutoEncoder, 
    filename: str):
    torch.save(
        obj=model.get_checkpoint_info(),
        f=filename)


def load_model(
    filename: str,
    map_location:torch.device):
    checkpoint: dict = torch.load(
        f=filename, 
        map_location=map_location)
    d: dict = {}
    for k, v in checkpoint.items():
        if not k.startswith("@"):
            d.update({k: v})
    model = AutoEncoder(**d)
    model.load_state_dict(state_dict=checkpoint["@state_dict"])
    
    return model