from HW1.DNN import COVID19Dataset
from torch.utils.data import DataLoader
from HW1.DNN import NeuralNet

dataset = COVID19Dataset('covid.test.csv', 'test')
data_loader = DataLoader(dataset, 270, shuffle=False, drop_last=False, num_workers=0, pin_memory=True)
tt_set = data_loader
model = NeuralNet(tt_set.dataset.dim).to('cuda')
model.eval()
preds = []
for x in tt_set:
    print(x)
