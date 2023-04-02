from torch import nn


class AnomalyClassifier(nn.Module):
    def __init__(self, input_dim=4096, num_classes=6):
        super(AnomalyClassifier, self).__init__()
        # self.fc1 = nn.Linear(input_dim, 512)
        self.fc1 = nn.Linear(input_dim, 1024)
        self.dropout1 = nn.Dropout(0.3)
        # self.fc2 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.3)
        # self.fc3 = nn.Linear(64, num_classes)
        self.fc3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(256, num_classes)

        self.relu = nn.ReLU()

        self.softmax = nn.Softmax(dim=1)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.xavier_normal_(self.fc4.weight)

    def forward(self, x):
        x = self.relu(self.dropout1(self.fc1(x)))
        x = self.relu(self.dropout2(self.fc2(x)))
        x = self.relu(self.dropout3(self.fc3(x)))
        x = self.softmax(self.fc4(x))
        return x

