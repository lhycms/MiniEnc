import unittest
import torch
from typing import List
from minienc.model.autoEncoder import (
    Encoder, 
    Decoder, 
    AutoEncoder,
    save_checkpoint,
    load_model
)


class AutoEncoderTest(unittest.TestCase):
    def test_encoder(self):
        input_dim: int = 10
        hidden_dims: List[int] = [20, 10, 5]
        output_dim: int = 2
        encoder = Encoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim
        )
        encoder.check()

        x: torch.tensor = torch.ones((10, input_dim))
        assert encoder(x).size()[0] == 10
        assert encoder(x).size()[1] == output_dim
        

    def test_decoder(self):
        input_dim: int = 2
        hidden_dims: List[int] = [5, 10, 20]
        output_dim: int = 10
        decoder = Decoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim
        )
        decoder.check()
        
        x: torch.tensor = torch.ones((10, input_dim))
        assert decoder(x).size()[0] == 10
        assert decoder(x).size()[1] == output_dim
        
        
    def test_autoencoder(self):
        input_dim: int = 10
        hidden_dims: List[int] = [20, 10, 5]
        output_dim: int = 2
        auto_encoder = AutoEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim
        )
        auto_encoder.check()
        #print(auto_encoder.get_checkpoint_info())

    def test_save_checkpoint(self):
        filename = "checkpoint.pt"
        input_dim: int = 10
        hidden_dims: List[int] = [20, 10, 5]
        output_dim: int = 2
        auto_encoder = AutoEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim
        )
        save_checkpoint(auto_encoder, filename)
        
        model = load_model(filename, torch.device("cpu"))
        print(model)
        

if __name__ == "__main__":
    unittest.main()