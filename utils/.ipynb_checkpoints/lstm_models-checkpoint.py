import torch
from torch import nn

class VanillaLSTM(nn.Module):

    def __init__(self, input_size=3, hidden_size=36, output_size=3, seq_length=4, num_layers=1, dropout=0):

        super(VanillaLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        self.hidden_output = (torch.randn(num_layers,1,hidden_size),
                              torch.randn(num_layers,1,hidden_size))
    

    def reset_hidden_state(self):
        self.hidden_output = (torch.randn(self.num_layers,1,self.hidden_size),
                              torch.randn(self.num_layers,1,self.hidden_size))

    def forward(self, sequence):
        lstm_out, self.hidden_output = self.lstm(sequence.view(len(sequence), 1, -1), self.hidden_output)
        prediction = self.linear(lstm_out)

        return prediction[-1]
    
    def train(self, train_data, test_data, loss_f, optimizer_f):
 
        accumulated_loss_test = None

        for seq, y in train_data:
            
            accumulated_loss_train = 0
            
            self.zero_grad()
            self.reset_hidden_state()
            
            y_pred = self.forward(seq)
            loss_train = loss_f(y_pred.squeeze(), y)
            accumulated_loss_train += loss_train.item()
            
            loss_train.backward()
            optimizer_f.step()
            
        if test_data != None: 

            for seq, y in test_data:

                accumulated_loss_test = 0

                with torch.no_grad():

                    y_pred = self.forward(seq)
                    loss_test = loss_f(y_pred.squeeze(), y)
                    accumulated_loss_test += loss_test.item()
                
        return accumulated_loss_train, accumulated_loss_test
    
    def predict(self, data):
        
        accumulated_loss = 0
        predictions = []
        
        with torch.no_grad():
        
            for seq, y in data:

                y_pred = self.forward(seq).squeeze()
                accumulated_loss += ((y_pred - y)**2).numpy()
                predictions.append(y_pred.numpy())

        return predictions, accumulated_loss/len(data)