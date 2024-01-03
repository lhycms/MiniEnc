import unittest
import torch
from minienc.dataset.blobs import BlobsDataset


class BlobsTest(unittest.TestCase):
    def test_all(self):
        n_samples: int = 300
        n_features: int = 10
        centers: int = 3
        random_state: int = 42
        
        blobs_dataset = BlobsDataset(
            n_samples=n_samples,
            n_features=n_features,
            centers=centers,
            random_state=random_state
        )
        self.assertEqual(len(blobs_dataset), n_samples)
        assert torch.eq(blobs_dataset[0][0], blobs_dataset[0][1]).all().item() 
        
    
        

if __name__ == "__main__":
    unittest.main()