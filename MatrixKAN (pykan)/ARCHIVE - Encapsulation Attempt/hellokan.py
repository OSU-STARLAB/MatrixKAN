from MatrixKAN import *
from pykan.kan.utils import create_dataset
from pykan.kan.utils import ex_round

torch.set_default_dtype(torch.float64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
model = MatrixKAN(width=[2,5,1], grid=3, k=2, seed=42, device=device, spline_matrix=True)

# create dataset f(x,y) = exp(sin(pi*x)+y^2)
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
dataset = create_dataset(f, n_var=2, device=device)

# plot KAN at initialization
model(dataset['train_input'])
model.plot()

# train the model
model.fit(dataset, opt="LBFGS", steps=75, lamb=0.001, update_grid=False)

model.plot()

model = model.prune()
model.plot()

model.fit(dataset, opt="LBFGS", steps=75, update_grid=False)

model = model.refine(10)

model.fit(dataset, opt="LBFGS", steps=75, update_grid=False)

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

model.fit(dataset, opt="LBFGS", steps=75, update_grid=False)

ex_round(model.symbolic_formula()[0][0],4)