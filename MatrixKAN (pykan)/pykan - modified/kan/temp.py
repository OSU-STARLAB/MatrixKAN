import torch

def normalize(x, center, scalar):
    norm = torch.nn.Tanh()
    x_norm = x - center
    x_norm = x_norm / (scalar / 2)
    x_norm = norm(x_norm)
    x_norm = x_norm * scalar
    x_norm = x_norm + center
    return x_norm

value = -4.5
input = torch.tensor([value], dtype=torch.float64)
num_range = [-10, 1]
center = (num_range[0] + num_range[1]) / 2
scalar = (num_range[1] - num_range[0]) / 2
print(normalize(input, center, scalar))