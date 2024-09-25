from kan import LBFGS
from kan.utils import create_dataset
from MatrixKAN import *
import ekan
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_util import *

# CONFIGS
EPOCHS = 100
OPTIMIZER = "LBFGS"                 # "Adam", "AdamW" or "LBFGS"
LEARNING_RATE = 1.
UPDATE_GRID = True
UPDATE_GRID_FREQ = 10               # Number of epochs between grid update

torch.set_default_dtype(torch.float64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# create dataset f(x,y) = exp(sin(pi*x)+y^2)
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
dataset = create_dataset(f, n_var=2, device=device) #, train_num=6, test_num=6)

train_dataset = NewDataSet(dataset["train_input"], dataset["train_label"])
test_dataset = NewDataSet(dataset["test_input"], dataset["test_label"])

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

model = MatrixKAN([2, 5, 1], grid_size=3, spline_order=2, grid_eps=1, device=device)
# model = ekan.KAN([2, 5, 1], grid_size=3, spline_order=2, grid_eps=1)
model.to(device)

# DEFINE OPTIMIZER
if OPTIMIZER == "AdamW":
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
elif OPTIMIZER == "LBFGS":
    optimizer = LBFGS(model.parameters(), lr=LEARNING_RATE, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)
    def closure():
        global train_loss, reg_
        optimizer.zero_grad()
        pred = model(dataset['train_input'][train_id], update_grid_now)
        train_loss = loss_fn(pred, dataset['train_label'][train_id])
        train_loss = torch.nan_to_num(train_loss)
        reg_ = torch.tensor(0.)
        objective = train_loss
        objective.backward()
        return objective
else:
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    if OPTIMIZER != "Adam":
        print("Optimizer not supported. Using default--Adam.")

# DEFINE SCHEDULER
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

# DEFINE LOSS FUNCTION
# loss_fn = loss_fn_eval = nn.MSELoss()
# loss_fn = loss_fn_eval = nn.CrossEntropyLoss()
loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)

pbar = tqdm(range(EPOCHS), desc='description', ncols=100)

for _ in pbar:
    # Train
    model.train()

    train_id = np.random.choice(dataset['train_input'].shape[0], dataset['train_input'].shape[0], replace=False)
    test_id = np.random.choice(dataset['test_input'].shape[0], dataset['test_input'].shape[0], replace=False)

    update_grid_now = False
    if UPDATE_GRID and _ % UPDATE_GRID_FREQ == 0:
        update_grid_now = True

    if OPTIMIZER == "LBFGS":
        optimizer.step(closure)
    else:
        pred = model(dataset['train_input'][train_id], update_grid=update_grid_now)
        loss = train_loss = loss_fn(pred, dataset['train_label'][train_id])
        reg_ = torch.tensor(0.)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validate
    model.eval()
    pred = model(dataset['test_input'][test_id])
    test_loss = loss_fn_eval(pred, dataset['test_label'][test_id].to(device))

    pbar.set_description("| train_loss: %.2e | test_loss: %.2e | reg: %.2e | " % (
    torch.sqrt(train_loss).cpu().detach().numpy(), torch.sqrt(test_loss).cpu().detach().numpy(),
    reg_.cpu().detach().numpy()))

    # Update learning rate
    # scheduler.step()