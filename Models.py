import torch
import torch.nn as nn
import torch.nn.functional as F

''' This file includes models that we will train and make the cross-subject test later. The model list
includes LSTM, 1D-CNN, 2D-CNN, CNNNet, CNN-LSTM, CNN-GRU, EEGNet_8_2, EEGNet-Attention, and CNNNex model.'''

class SingleLSTM(nn.Module):
    def __init__(self, n_features = 22, n_outputs = 4):
        '''This model includes three layers of LSTM, after which we include a dropout layer with
        proportion 0.5 and two fully connected layers. 
        For other datasets, n_features needs to be set as the number of electrodes (channels)
        n_outputs needs to be set as the number of motion classes.'''

        super(SingleLSTM, self).__init__()

        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=100, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=100, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=100, hidden_size=100, batch_first=True)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, n_outputs)

        # Activation function
        self.elu = nn.ELU()

    def forward(self, x):
        # LSTM layers - handling sequences manually
        x, _ = self.lstm1(x)     
        x, _ = self.lstm2(x)
        x, (h_n, _) = self.lstm3(x)  # Getting the last hidden state

        # Using the last hidden state
        x = h_n[-1]  # Get the last layer's last hidden state

        # Dropout and fully connected layers
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        return x





class OneD_CNN(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs):
        super(OneD_CNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=64)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=64)
        self.dropout = nn.Dropout(0.5)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (n_timesteps // 2), 100)
        self.fc2 = nn.Linear(100, n_outputs)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # PyTorch expects (batch, channels, length), so make sure input is shaped correctly
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.conv2(x)))
        x = F.elu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        return(x)
    




class TwoD_CNN(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs):
        super(TwoD_CNN, self).__init__()

        # Adjust the in_channels to 1 and set the height to n_features (22 in your case)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(n_features, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(num_features=64)

        # Since you're reducing the height to 1 after the first conv layer, adjust following conv layers accordingly
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(num_features=64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), padding=(0, 1))
        self.dropout = nn.Dropout(0.5)
        # Adjust the pooling layer as well
        self.pool = nn.AvgPool2d(kernel_size=(1, 2), padding=(0, 1))

        self._to_linear = None
        self._dummy_x = torch.rand(32, 1, n_features, n_timesteps)
        self._dummy_forward()

        self.fc1 = nn.Linear(self._to_linear, 100)
        self.fc2 = nn.Linear(100, n_outputs)

    def _dummy_forward(self):
        x = self._dummy_x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        self._to_linear = torch.flatten(x, 1).size(1)

    def forward(self, x):
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.conv2(x)))
        x = F.elu(self.conv3(x))
        x = self.dropout(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        return x
    




class CNN_LSTM(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs):
        super(CNN_LSTM, self).__init__()
        self.n_timesteps = n_timesteps

        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=64)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=64)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=64)

        self.dropout1 = nn.Dropout(0.5)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=64 * (n_timesteps // 2), hidden_size=480, batch_first=True)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(480, 100)
        self.fc2 = nn.Linear(100, n_outputs)

    def forward(self, x):
        # Convolutional layers expect (batch, channels, length), so permute
        x = x.permute(0, 2, 1)

        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.conv2(x)))
        x = F.elu(self.bn3(self.conv3(x)))

        x = self.dropout1(x)
        x = self.pool(x)
        
        # LSTM expects (batch, seq_len, features), so flatten conv features and permute
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(x.size(0), -1, 64 * (self.n_timesteps // 2))

        x, (hn, cn) = self.lstm(x)
        x = self.dropout2(x[:, -1, :])  # Use the last output for classification
        
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    




class CNN_GRU(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs):
        super(CNN_GRU, self).__init__()
        self.n_timesteps = n_timesteps

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=64)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=64)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=64)

        self.dropout1 = nn.Dropout(0.5)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # GRU layer
        self.gru = nn.GRU(input_size=64 * (n_timesteps // 2), hidden_size=480, batch_first=True)

        # Fully connected layers
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(480, 100)
        self.fc2 = nn.Linear(100, n_outputs)

    def forward(self, x):
        # Permute to match PyTorch's expected input dimensions for Conv1d: (batch, channels, length)
        x = x.permute(0, 2, 1)

        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.conv2(x)))
        x = F.elu(self.bn3(self.conv3(x)))

        x = self.dropout1(x)
        x = self.pool(x)

        # Permute and reshape to match GRU's expected input dimensions: (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(x.size(0), -1, 64 * (self.n_timesteps // 2))

        x, _ = self.gru(x)
        x = self.dropout2(x[:, -1, :])  # Take the last GRU output for classification

        x = F.elu(self.fc1(x))
        x = self.fc2(x)

        return x





class EEGNet_8_2(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs):
        super(EEGNet_8_2, self).__init__()
        self.n_features = n_features
        self.n_timesteps = n_timesteps
        
        # Define convolutional layers
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(1, 64), bias=False, padding='same')
        self.bn1 = nn.BatchNorm2d(8)
        self.depthwise = nn.Conv2d(8, 8 * 2, kernel_size=(n_features, 1), groups=8, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(8 * 2)
        self.depthwise_activation = nn.ELU()
        self.depthwise_pooling = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(0.5)
        self.separable = nn.Conv2d(8 * 2, 16, kernel_size=(1, 16), bias=False, padding='same', groups=8*2)
        self.pointwise = nn.Conv2d(16, 16, kernel_size=1, bias=False)
        self.sep_bn = nn.BatchNorm2d(16)
        self.sep_activation = nn.ELU()
        self.sep_pooling = nn.AvgPool2d(kernel_size=(1, 8), padding=(0, 3))
        self.dropout2 = nn.Dropout(0.5)
        self.flatten = nn.Flatten()

        # Pseudo forward to get the correct number of features
        dummy_input = torch.randn(1, 1, n_features, n_timesteps)
        output_features = self.pseudo_forward(dummy_input)

        # Define the linear layer with the correct number of input features
        self.dense = nn.Linear(output_features, n_outputs)

    def pseudo_forward(self, x):
        # Exact same operations as in the forward method, but only to compute output size
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.depthwise_bn(x)
        x = self.depthwise_activation(x)
        x = self.depthwise_pooling(x)
        x = self.dropout1(x)
        x = self.separable(x)
        x = self.pointwise(x)
        x = self.sep_bn(x)
        x = self.sep_activation(x)
        x = self.sep_pooling(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        return x.size(1)

    def forward(self, x):
        # Perform the same operations as in the pseudo_forward
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.depthwise_bn(x)
        x = self.depthwise_activation(x)
        x = self.depthwise_pooling(x)
        x = self.dropout1(x)
        x = self.separable(x)
        x = self.pointwise(x)
        x = self.sep_bn(x)
        x = self.sep_activation(x)
        x = self.sep_pooling(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x





class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(22, input_dim)
        self.key = nn.Linear(22, input_dim)
        self.value = nn.Linear(22, input_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        out = torch.bmm(attention_weights, V)
        return out, attention_weights

class EEGNet_8_2_att(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs):
        super(EEGNet_8_2_att, self).__init__()
        
        self.sep_pooling = nn.AvgPool2d(kernel_size=(1, 8), padding=(0, 3))
        self.dropout2 = nn.Dropout(0.5)
        
        self.attention = SelfAttention(input_dim=n_features)  # Assuming each feature is an input dimension

        # Using a dummy input to determine the size for the linear layer
        dummy_input = torch.randn(1, 1, n_features, n_timesteps)
        dummy_output = self.pseudo_forward(dummy_input)
        num_features_after_attention = dummy_output.size(-1) * dummy_output.size(-2)
        
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(num_features_after_attention, n_outputs)
        
    def pseudo_forward(self, x):
        x = self.sep_pooling(x)
        x = self.dropout2(x)
        x = x.view(x.size(0), x.size(3), x.size(1) * x.size(2))  # Adjust shape for attention
        x, _ = self.attention(x)
        return x

    def forward(self, x):
        x = self.sep_pooling(x)
        x = self.dropout2(x)
        x = x.view(x.size(0), x.size(3), x.size(1) * x.size(2))  # Flatten for attention
        x, _ = self.attention(x)
        x = x.view(x.size(0), -1)  # Flatten for the dense layer
        x = self.dense(x)
        return x





class EEGNeX_8_32(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs):
        super(EEGNeX_8_32, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=(1, 64), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.elu = nn.ELU()
        
        self.conv2 = nn.Conv2d(8, 32, kernel_size=(1, 64), padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        self.depthwise = nn.Conv2d(32, 32 * 2, kernel_size=(n_features, 1), groups=32, bias=False)
        self.bn3 = nn.BatchNorm2d(32 * 2)

        self.avgpool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(0.5)

        self.conv3 = nn.Conv2d(32 * 2, 32, kernel_size=(1, 16), padding='same', bias=False, dilation=(1, 2))
        self.bn4 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 8, kernel_size=(1, 16), padding='same', bias=False, dilation=(1, 4))
        self.bn5 = nn.BatchNorm2d(8)
        
        self.avgpool2 = nn.AvgPool2d(kernel_size=(1, 4), padding=(0,1))
        self.dropout2 = nn.Dropout(0.5)

        self.flatten = nn.Flatten()

        # Using a dummy input to determine the correct number of features for the Linear layer
        dummy_input = torch.randn(1, 1, n_features, n_timesteps)
        output_features = self.pseudo_forward(dummy_input)
        output_features = output_features.shape[1]

        self.fc = nn.Linear(output_features, n_outputs)  # Initialize with the dynamically calculated size

    def pseudo_forward(self, x):
        x = self.elu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.elu(self.bn3(self.depthwise(x)))
        x = self.avgpool1(x)
        x = self.dropout1(x)
        x = self.bn4(self.conv3(x))
        x = self.elu(self.bn5(self.conv4(x)))
        x = self.avgpool2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        return x

    def forward(self, x):
        x = self.elu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.elu(self.bn3(self.depthwise(x)))
        x = self.avgpool1(x)
        x = self.dropout1(x)
        x = self.bn4(self.conv3(x))
        x = self.elu(self.bn5(self.conv4(x)))
        x = self.avgpool2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

