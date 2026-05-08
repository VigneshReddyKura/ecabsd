from data.dataset import BindingSiteDataset, collate_fn
from torch.utils.data import DataLoader

dataset = BindingSiteDataset('data/processed', 'data/splits.csv', split='train')
sample = dataset[0]
print('Sample keys:', sample.keys() if hasattr(sample, 'keys') else dir(sample))
print('data_a:', sample['data_a'])
print('data_b:', sample['data_b'])
print('labels shape:', sample['labels'].shape)