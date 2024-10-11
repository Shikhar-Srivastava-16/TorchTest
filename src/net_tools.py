import torch
from torch import nn

# Import torchvision 
import torchvision
from torchvision import datasets as ds
from torchvision.transforms import ToTensor

# model with only linear modules
class LinModel(nn.Module):
    def __init__(self, 
                 in_shape: int, 
                 hidden_layer_nodes: int, 
                 out_shape: int
                ):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), # neural networks like their inputs in vector form
            nn.Linear(in_features=in_shape, out_features=hidden_layer_nodes), # in_features = number of features in a data sample (784 pixels)
            nn.Linear(in_features=hidden_layer_nodes, out_features=out_shape)
        )
    
    def forward(self, x):
        return self.layer_stack(x)
    
    
# model with non-linear and linear layers
class ReluModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), # flatten inputs into single vector
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)
    
class ConvNet(nn.Module):  
    def __init__(self, in_channels: int, hidden_layer_nodes: int, out_channels: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=hidden_layer_nodes,
                      kernel_size=3, # how big is the square that's going over the image?
                      padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_layer_nodes,
                      out_channels=hidden_layer_nodes,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_layer_nodes, hidden_layer_nodes, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_layer_nodes, hidden_layer_nodes, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from?
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=hidden_layer_nodes*7*7,
                      out_features=out_channels)
        )

    def forward(self, x: torch.Tensor):
        return self.out(self.block_2(self.block_1(x)))
               
def train(model: torch.nn.Module,
            data_loader: torch.utils.data.DataLoader,
            loss_fn: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            accuracy_fn,
            device: torch.device = "cuda") -> float:
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")
    return train_acc

def test(data_loader: torch.utils.data.DataLoader, 
         model: torch.nn.Module,
         loss_fn: torch.nn.Module,
         accuracy_fn,
        device: torch.device = 'cuda') -> float:
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred = model(X)
            
            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                y_pred=test_pred.argmax(dim=1) # Go from logits -> pred labels
            )
        
        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")
        return test_acc 
    
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc