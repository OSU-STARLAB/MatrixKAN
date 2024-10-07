from kan import *
from MatrixKAN import *
from kan.utils import create_dataset
from kan.utils import ex_round
from feynman import *

torch.set_default_dtype(torch.float64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print(device)

# create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
# model = MatrixKAN(width=[2,2,1,1], grid=3, k=2, seed=42, device=device, grid_eps=1)
model = MultKAN(width=[2,2,1,1], grid=3, k=2, seed=42, device=device, grid_eps=1)

# create dataset f(x,y) = exp(sin(pi*x)+y^2)
# f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
symbol, expr, f, ranges = get_feynman_dataset(2)
dataset = create_dataset(f, n_var=len(symbol), device=device) #, train_num=6)

# plot KAN at initialization
model(dataset['train_input'])
model.plot()

# train the model
model.fit(dataset, opt="LBFGS", steps=100, update_grid=True, lamb=0.005)

model.plot()

model = model.prune()
model.plot()

model.fit(dataset, opt="LBFGS", steps=75, update_grid=True)

model = model.refine(10)

model.fit(dataset, opt="LBFGS", steps=75, update_grid=True)

mode = "auto" # "manual"

if mode == "manual":
    # manual mode
    model.fix_symbolic(0,0,0,'sin')
    model.fix_symbolic(0,1,0,'x^2')
    model.fix_symbolic(1,0,0,'exp')
elif mode == "auto":
    # automatic mode
    lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
    model.auto_symbolic(lib=lib)

model.fit(dataset, opt="LBFGS", steps=75, update_grid=True)

ex_round(model.symbolic_formula()[0][0],4)
