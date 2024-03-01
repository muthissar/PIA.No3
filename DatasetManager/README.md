# DatasetManager
Dataset manager to easily load symbolic music datasets

## Installation
Clone repository and then 
`pip install -e .` in root directory.

## Usage
```python
from DatasetManager.dataset_manager import DatasetManager
from DatasetManager.chorale_dataset import ChoraleDataset
dataset_manager = DatasetManager()

# parameters of the ChoraleDataset class
# (name and chorale_corpus_it_gen need not be specified)
chorale_dataset_kwargs = {'voice_ids': [0, 1, 2, 3],
        'metadatas': [],
        'sequences_size': 8,
        'subdivision': 4
}

bach_chorales_dataset: ChoraleDataset = dataset_manager.get_dataset(
    name='bach_chorales_test',
    **chorale_dataset_kwargs
)
	
(train_dataloader, val_dataloader, test_dataloader) = (
    bach_chorales_dataset.data_loaders(
        batch_size=128,
        split=(0.85, 0.10)
))
	
# get first element in the dataset
# elements are couples tensor_chorale, metadata
# torch.LongTensor of the form (batch_size, num_voices, chorale_length)   
print(next(train_dataloader.__iter__()))
```

