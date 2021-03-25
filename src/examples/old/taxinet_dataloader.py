class TaxiNetFull(torch.utils.data.Dataset):
    def __init__(self, split='train',N=1000, shuffle=True):
        N = N
        idx = None
        if split == 'train':
            split = 'exp1_train'
            start_idx = 0
            end_idx = 30000
            idx = np.arange(start_idx, end_idx)
        elif split == 'val':
            split = 'exp1_train'
            start_idx = 30000
            end_idx = 37051
            idx = np.arange(start_idx, end_idx)            
        
        folder = os.path.join(DATASET_PATH,  split)
        if not os.path.exists(folder):
            print(folder + " does not exist, check split argument")
            raise AttributeError
        
        y = np.load(os.path.join(folder, "Y.npy"))
        n = y.shape[0]
        
        if idx is None:
            if shuffle:
                idx = np.random.choice(n, N, replace=False)
            else:
                idx = range(N)
        
        self.X = np.load(os.path.join(folder, "X.npy"))[idx]
        self.y = y[idx]
        
    def __len__(self):
        return self.X.shape[0]
        
    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx,:,:,:]*1./256).float().permute(2,0,1)
        y = torch.from_numpy(self.y[idx,:]).float()
        
        return x,y

