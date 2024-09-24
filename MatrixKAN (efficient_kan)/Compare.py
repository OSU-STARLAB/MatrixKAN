from MatrixKAN_test import *
from kan_test import *
from kan import *
import torch.nn as nn
import torch.optim as optim

torch.set_default_dtype(torch.float64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# create KAN models
model_matrix = MatrixKAN([2, 5, 1], grid_size=2, spline_order=2, grid_range=[-1, 1], device=device)
model_reg = KAN_Reg(layers_hidden=[2,5,1], grid_size=2, spline_order=2, grid_range=[-1, 1])

# create dataset f(x,y) = exp(sin(pi*x)+y^2)
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
dataset = create_dataset(f, n_var=2, test_num=1000, train_num=1000)


# Set to eval mode
model_matrix.eval()
model_reg.eval()

# Inference
with torch.inference_mode():

    # Setup data
    n_epochs = 10  # number of epochs to run
    batch_size = 10  # size of each batch
    batches_per_epoch = len(dataset['train_input']) // batch_size

    for epoch in range(n_epochs):
        for i in range(batches_per_epoch):
            start = i * batch_size
            # take a batch
            Xbatch = dataset['train_input'][start:start + batch_size]
            ybatch = dataset['train_label'][start:start + batch_size]
            # print(f"X Values: {Xbatch}")
            # forward pass
            y_pred_reg = model_reg(Xbatch)
            y_pred_matrix = model_matrix(Xbatch)
            # print(f"Matrix Results: {y_pred_matrix}")
            # print(f"Reg Results: {y_pred_reg}")
            print(f"Differences: {y_pred_reg - y_pred_matrix}")
