import torch
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')

class MoADataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets


    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        if self.targets is not None:
            dct = {
                'x': torch.tensor(self.features[idx, :], dtype=torch.float).to(DEVICE),
                'y': torch.tensor(self.targets[idx, :], dtype=torch.float).to(DEVICE)
            }
        else:
            dct = {
                'x': torch.tensor(self.features[idx, :], dtype=torch.float).to(DEVICE),
            }
        # dct['x'] = self.normalize(dct['x'].unsqueeze(0).unsqueeze(0)).squeeze()
        # dct['x'] = (dct['x'] - torch.min(dct['x'])) / (torch.max(dct['x']) - torch.min(dct['x']))
        return dct

