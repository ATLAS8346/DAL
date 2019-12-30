import torch.nn as nn
import torch.nn.functional as F


class Feature_base(nn.Module):
    def __init__(self):
        super(Feature_base, self).__init__()
        self.features = nn.Sequential(
            # layer 1
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=3, padding=1),
            # layer 2
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=3, padding=1),
            # layer 3
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 8192)
        return x


class Feature_disentangle(nn.Module):
    def __init__(self):
        super(Feature_disentangle, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(8192, 3072),
            nn.BatchNorm1d(3072),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(3072, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.features(x)


class Feature_discriminator(nn.Module):
    def __init__(self):
        super(Feature_discriminator, self).__init__()
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        return x


class Reconstructor(nn.Module):
    def __init__(self):
        super(Reconstructor, self).__init__()
        self.fc = nn.Linear(4096, 8192)

    def forward(self,x):
        x = self.fc(x)
        return x


class Mine(nn.Module):
    def __init__(self):
        super(Mine, self).__init__()
        self.fc1_x = nn.Linear(2048, 512)
        self.fc1_y = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x, y):
        h1 = F.leaky_relu(self.fc1_x(x) + self.fc1_y(y))
        h2 = self.fc2(h1)
        return h2


class Predictor(nn.Module):
    def __init__(self, prob=0.5):
        super(Predictor, self).__init__()
        self.fc3 = nn.Linear(2048, 10)
        # self.bn_fc3 = nn.BatchNorm1d(10)
        # self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        # x = F.relu(self.bn2_fc(self.fc2(x)))
        x = self.fc3(x)
        return x
