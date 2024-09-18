from kan.utils import create_dataset
from kan.utils import ex_round
from MatrixKAN import *
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_util import *

torch.set_default_dtype(torch.float64)

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# create dataset f(x,y) = exp(sin(pi*x)+y^2)
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
dataset = create_dataset(f, n_var=2, device=device, train_num=100000, test_num=100000)

train_dataset = NewDataSet(dataset["train_input"], dataset["train_label"])
test_dataset = NewDataSet(dataset["test_input"], dataset["test_label"])

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

model = MatrixKAN([2, 5, 1], base_activation=nn.Identity, grid_eps=1, device=device)
model.to(device)

#optimizer = torch.optim.LBFGS(model.parameters(), lr=1)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

criterion = nn.MSELoss()
#criterion = nn.CrossEntropyLoss()
for epoch in range(100):
    # Train
    model.train()
    with (tqdm(train_loader) as pbar):
        for i, (inputs, labels) in enumerate(pbar):
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels.to(device))
            loss.backward()
            optimizer.step()
            # accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
            accuracy = (output == labels.to(device)).float().mean()
            # accuracy = (labels.to(device) == labels.to(device)).float().mean()
            pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])

    # Validate
    model.eval()
    test_loss = 0
    test_accuracy = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            output = model(inputs)
            test_loss += criterion(output, labels.to(device)).item()
            test_accuracy += (
                (output.argmax(dim=1) == labels.to(device)).float().mean().item()
            )
    test_loss /= len(test_loader)
    test_accuracy /= len(test_loader)

    # Update learning rate
    # scheduler.step()

    print(
        f"Epoch {epoch + 1}, Val Loss: {test_loss}, Val Accuracy: {test_accuracy}"
    )