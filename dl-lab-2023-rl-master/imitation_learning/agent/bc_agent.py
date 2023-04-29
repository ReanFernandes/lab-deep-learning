import torch
from agent.networks import CNN
import torch.nn.functional as F
class BCAgent:
    
    def __init__(self,history_length, output_classes, batch_size, lr):
        # TODO: Define network, loss function, optimizer
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = CNN(history_length, output_classes, batch_size).to(self.device)
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        

    def update(self, X_batch, y_batch):
        # TODO: transform input to tensors
        # TODO: forward + backward + optimize
        
        X_batch = torch.from_numpy(X_batch).to(self.device)
        y_batch = torch.from_numpy(y_batch).to(self.device)
        self.optimizer.zero_grad()
        # forward pass
        output = self.net.forward(X_batch)

        loss = self.loss(output, y_batch)
        # backward pass
        loss.backward()
        # optimize
        self.optimizer.step()

        return loss

    def predict(self, X):
        # TODO: forward pass
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
            truth_labels = torch.from_numpy(truth_labels).to(self.device)
            predictions = self._get_predictions(output_distro)
            accuracy = torch.mean(predictions.eq(truth_labels).float())
            loss = self.loss(output_distro, truth_labels)
        return accuracy, loss
    

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name))


    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)

