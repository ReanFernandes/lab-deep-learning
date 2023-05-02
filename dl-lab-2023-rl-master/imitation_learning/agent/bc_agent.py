import torch
from agent.networks import CNN
import torch.nn.functional as F
class BCAgent:
    
    def __init__(self,history_length, batch_size, lr,output_classes =4):
        # TODO: Define network, loss function, optimizer
        self.history_length = history_length
        self.output_classes = output_classes
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = CNN(self.history_length, self.output_classes, self.batch_size).to(self.device)
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        

    def update(self, X_batch, y_batch):
        # TODO: transform input to tensors
        # TODO: forward + backward + optimize
        #check if inputs are tensor, because in training i convert them to tensors
        if not torch.is_tensor(X_batch):
            X_batch = torch.from_numpy(X_batch).to(self.device)
        if not torch.is_tensor(y_batch):
            y_batch = torch.from_numpy(y_batch).to(self.device)
        # zero the parameter gradients
        self.optimizer.zero_grad()
        
        # forward pass
        output = self.net.forward(X_batch)
        
        # calculate loss
        loss = self.loss(output, y_batch)
        
        # backward pass
        loss.backward()
        
        # optimize
        self.optimizer.step()

        return loss

    def predict(self, X):
        # TODO: forward pass
        if not torch.is_tensor(X):
            X = torch.from_numpy(X).float().to(self.device)
        outputs = self.net(X)
        return outputs
    
    def _get_predictions(self, output_distro):
        # gets the predicted labels from the output distribution, output distro will be (batch_size, output_classes)
        
        return torch.argmax(output_distro, dim=1)
        
    def get_accuracy_and_loss(self, input_data, truth_labels):
        # calculates the accuracy and loss of the model on the given data
        # output_distro = self.predict(input_data)
        with torch.no_grad():
            output_distro = self.predict(input_data)
            if not torch.is_tensor(truth_labels):
                truth_labels = torch.from_numpy(truth_labels).to(self.device)
            predictions = self._get_predictions(output_distro)
            accuracy = torch.mean(predictions.eq(truth_labels).float())
            loss = self.loss(output_distro, truth_labels)
        return accuracy, loss
    

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name))


    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)

