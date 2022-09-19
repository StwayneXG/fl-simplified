import torch
import torch.nn as nn
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output    

""" Training the Model """
""" args:
        model: Model to be Trained
        training_data_loader: Prepared Dataset Loader for Training Data
        epochs: Number of Epochs to Run
        criterion: Loss Type
        optimizer: Optimizer """
""" return: None """ 
def train_model(model, train_dataloader, loss_fn, optimizer, epochs = 10, batch_view_dims = (-1, 1, 28, 28)):
    model.train()
    # enumerate epochs
    for epoch in range(epochs):
        samples_trained = 0
        total_samples = len(train_dataloader.dataset)
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dataloader):
            b_x = Variable(inputs.view(batch_view_dims))
            b_y = Variable(targets)

            # compute the model output
            output = model(b_x)
            # calculate loss
            loss = loss_fn(output, b_y)
            # clear the gradients
            optimizer.zero_grad()
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
            
            samples_trained += len(targets)
            if i % 10 == 0:
                print(f"train epoch: [{epoch+1} / {epochs}], samples trained: [{samples_trained} / {total_samples}], current loss: [{loss.item():.6f}]")


""" Evaluating the Model """
""" args:
        model: Model to be Evaluated
        test_data_loader: Prepared Dataset Loader for Testing Data """
""" return: None """
def evaluate_model(model, test_dataloader, batch_view_dims = (-1, 1, 28, 28)):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, targets in test_dataloader:
            b_x = Variable(inputs.view(batch_view_dims))
            b_y = Variable(targets)
            
            output = model(b_x)
            pred_y = torch.max(output, 1).indices
            total += len(targets)
            correct += (pred_y == targets).sum()
        accuracy = float(correct) / float(total)

    print(f'Accuracy = {100*accuracy:.2f}')