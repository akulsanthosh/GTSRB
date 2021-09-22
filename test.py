import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from transform import *
import pytorch_colors as pc
from augment import image_loader, save_data

batch_size = 32
momentum = 0.9
lr = 0.01
epochs = 0
log_interval = 10

augment = False
gc = transforms.Grayscale(num_output_channels=1)

class MyDataset(Dataset):

    def __init__(self, X_path="X.pt", y_path="y.pt", transform=None):
        self.transform = transform
        self.X = torch.load(X_path).squeeze(1)
        self.y = torch.load(y_path).squeeze(1)
    
    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        im = self.X[idx] / 2 + 0.5
        if self.transform is not None:
            im = self.transform(im)
        im_gc = gc.forward(im)
        im = (im-0.5)*2
        return im, self.y[idx]

train_dataset = []
for i in transform_list:
    print(i)
    train_dataset.append(MyDataset(X_path="train/X.pt", y_path="train/y.pt", transform=i))

train_dataset = torch.utils.data.ConcatDataset(train_dataset)
val_dataset = MyDataset(X_path="validation/X.pt", y_path="validation/y.pt")

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

if not augment:
    print(torch.cuda.get_device_name(0))

    nclasses = 43 # GTSRB has 43 classes

    keep_prob = 1
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(4, 100, kernel_size=5),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.BatchNorm2d(100),
                nn.Dropout2d()
                )

            self.layer2 = nn.Sequential(
                nn.Conv2d(100, 150, kernel_size=3),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.BatchNorm2d(150),
                nn.Dropout2d()
                )

            self.layer3 = nn.Sequential(
                nn.Conv2d(150, 250, kernel_size=3),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.BatchNorm2d(250),
                nn.Dropout2d()
                )

            self.layer4 = torch.nn.Sequential(
                nn.Linear(1000,350),
                nn.ReLU(),
                nn.Dropout())

            self.layer5 = torch.nn.Sequential(
                nn.Linear(350,nclasses))

            # CNN layers
            # self.conv1 = nn.Conv2d(4, 100, kernel_size=5)
            # self.bn1 = nn.BatchNorm2d(100)
            # self.conv2 = nn.Conv2d(100, 150, kernel_size=3)
            # self.bn2 = nn.BatchNorm2d(150)
            # self.conv3 = nn.Conv2d(150, 250, kernel_size=3)
            # self.bn3 = nn.BatchNorm2d(250)
            # self.conv_drop = nn.Dropout2d()
            # self.fc1 = nn.Linear(250*2*2, 350)
            # self.fc2 = nn.Linear(350, nclasses)

        def forward(self, x):
            im = x / 2 + 0.5
            im_gc = gc.forward(im)
            im = (im-0.5)*2
            im = torch.cat([im,im_gc], dim=1)
            x = self.layer1(im)
            x = self.layer2(x)
            x = self.layer3(x)
            x = x.view(-1, 1000)
            x = self.layer4(x)
            x = self.layer5(x)
            return F.log_softmax(x, dim=1)

        # def forward(self, x):

        #     im = x / 2 + 0.5
        #     im_gc = gc.forward(im)
        #     im = (im-0.5)*2
        #     im = torch.cat([im,im_gc], dim=1)
        #     x = self.bn1(F.max_pool2d(F.leaky_relu(self.conv1(im)),2))
        #     x = self.conv_drop(x)
        #     x = self.bn2(F.max_pool2d(F.leaky_relu(self.conv2(x)),2))
        #     x = self.conv_drop(x)
        #     x = self.bn3(F.max_pool2d(F.leaky_relu(self.conv3(x)),2))
        #     x = self.conv_drop(x)
        #     x = x.view(-1, 250*2*2)
        #     x = F.relu(self.fc1(x))
        #     x = F.dropout(x, training=self.training)
        #     x = self.fc2(x)
        #     return F.log_softmax(x, dim=1)
        
    model = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    def validation():
        model.eval()
        validation_loss = 0
        correct = 0
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            validation_loss += F.nll_loss(output, target, reduction="sum").item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        validation_loss /= len(val_loader.dataset)
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
            validation_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))


    for epoch in range(1, epochs + 1):
        train(epoch)
        validation()
        model_file = 'model_' + str(epoch) + '.pth'
        torch.save(model.state_dict(), model_file)
        print('\nSaved model to ' + model_file + '.')

    import pickle
    import pandas as pd

    outfile = 'gtsrb_kaggle.csv'

    output_file = open(outfile, "w")
    dataframe_dict = {"Filename" : [], "ClassId": []}

    test_data = torch.load('testing/test.pt')
    file_ids = pickle.load(open('testing/file_ids.pkl', 'rb'))
    model.cpu()
    model.eval()
    for i, data in enumerate(test_data):
        data = data.unsqueeze(0)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1].item()
        file_id = file_ids[i][0:5]
        dataframe_dict['Filename'].append(file_id)
        dataframe_dict['ClassId'].append(pred)

    df = pd.DataFrame(data=dataframe_dict)
    df.to_csv(outfile, index=False)
    print("Written to csv file {}".format(outfile))