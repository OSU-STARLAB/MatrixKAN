import torch
import torch.nn as nn
import numpy as np
import kan
from kan.spline import *
from kan.utils import sparse_mask


class MatrixKANLayer(kan.KANLayer, nn.Module):
    """
    MatrixKANLayer class
    

    Attributes:
    -----------
        in_dim: int
            input dimension
        out_dim: int
            output dimension
        num: int
            the number of grid intervals
        k: int
            the piecewise polynomial order of splines
        noise_scale: float
            spline scale at initialization
        coef: 2D torch.tensor
            coefficients of B-spline bases
        scale_base_mu: float
            magnitude of the residual function b(x) is drawn from N(mu, sigma^2), mu = sigma_base_mu
        scale_base_sigma: float
            magnitude of the residual function b(x) is drawn from N(mu, sigma^2), mu = sigma_base_sigma
        scale_sp: float
            mangitude of the spline function spline(x)
        base_fun: fun
            residual function b(x)
        mask: 1D torch.float
            mask of spline functions. setting some element of the mask to zero means setting the corresponding activation to zero function.
        grid_eps: float in [0,1]
            a hyperparameter used in update_grid_from_samples. When grid_eps = 1, the grid is uniform; when grid_eps = 0, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes.
            the id of activation functions that are locked
        device: str
            device
    """

    def __init__(self, in_dim=3, out_dim=2, num=5, k=3, noise_scale=0.5, scale_base_mu=0.0, scale_base_sigma=1.0, scale_sp=1.0, base_fun=torch.nn.SiLU(), grid_eps=0.02, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True, save_plot_data = True, device='cpu', sparse_init=False):
        """
        initialize a MatrixKANLayer
        
        Args:
        -----
            in_dim : int
                input dimension. Default: 2.
            out_dim : int
                output dimension. Default: 3.
            num : int
                the number of grid intervals = G. Default: 5.
            k : int
                the order of piecewise polynomial. Default: 3.
            noise_scale : float
                the scale of noise injected at initialization. Default: 0.1.
            scale_base_mu : float
                the scale of the residual function b(x) is intialized to be N(scale_base_mu, scale_base_sigma^2).
            scale_base_sigma : float
                the scale of the residual function b(x) is intialized to be N(scale_base_mu, scale_base_sigma^2).
            scale_sp : float
                the scale of the base function spline(x).
            base_fun : function
                residual function b(x). Default: torch.nn.SiLU()
            grid_eps : float
                When grid_eps = 1, the grid is uniform; when grid_eps = 0, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes.
            grid_range : list/np.array of shape (2,)
                setting the range of grids. Default: [-1,1].
            sp_trainable : bool
                If true, scale_sp is trainable
            sb_trainable : bool
                If true, scale_base is trainable
            device : str
                device
            sparse_init : bool
                if sparse_init = True, sparse initialization is applied.
            
        Returns:
        --------
            self
        """
        nn.Module.__init__(self)
        # size 
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k

        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1)[None,:].expand(self.in_dim, num+1)
        grid = extend_grid(grid, k_extend=k)
        self.grid = torch.nn.Parameter(grid).requires_grad_(False)
        noises = (torch.rand(self.num+1, self.in_dim, self.out_dim) - 1/2) * noise_scale / num

        self.grid_range = torch.tensor(grid_range, device=device).unsqueeze(0).expand(in_dim, -1)
        self.grid_range = self.grid_range.clone().to(dtype=torch.float64)
        self.grid_range = torch.nn.Parameter(self.grid_range).requires_grad_(False)

        self.grid_intervals = ((self.grid_range[:, 1] - self.grid_range[:, 0]) / num)
        self.grid_intervals = torch.nn.Parameter(self.grid_intervals).requires_grad_(False)

        self.coef = torch.nn.Parameter(curve2coef(self.grid[:,k:-k].permute(1,0), noises, self.grid, k))
        
        if sparse_init:
            self.mask = torch.nn.Parameter(sparse_mask(in_dim, out_dim)).requires_grad_(False)
        else:
            self.mask = torch.nn.Parameter(torch.ones(in_dim, out_dim)).requires_grad_(False)
        
        self.scale_base = torch.nn.Parameter(scale_base_mu * 1 / np.sqrt(in_dim) + \
                         scale_base_sigma * (torch.rand(in_dim, out_dim)*2-1) * 1/np.sqrt(in_dim)).requires_grad_(sb_trainable)
        self.scale_sp = torch.nn.Parameter(torch.ones(in_dim, out_dim) * scale_sp * self.mask).requires_grad_(sp_trainable)  # make scale trainable
        self.base_fun = base_fun

        self.device = device

        self.basis_matrix = self.calculate_basis_matrix()
        self.basis_matrix = torch.nn.Parameter(self.basis_matrix).requires_grad_(False)
        
        self.grid_eps = grid_eps
        
        self.to(device)

    def __getattribute__(self, name):
        """Dynamically replaces KANLayer calls with calls to MatrixKANLayer."""
        if name == "KANLayer":
            return MatrixKANLayer
        return super().__getattribute__(name)

    def calculate_basis_matrix(self):
        """
        Compute the basis matrix for a uniform B-spline with a given spline degree.

        Returns:
            torch.Tensor: Basis matrix tensor.
        """

        basis_matrix = torch.tensor([
            [1]
        ], dtype=torch.float32, device=self.device)

        scalar = 1

        k = 2

        while k <= self.k + 1:
            term_1 = torch.nn.functional.pad(basis_matrix, (0, 0, 0, 1), "constant", 0)
            term_3 = torch.nn.functional.pad(basis_matrix, (0, 0, 1, 0), "constant", 0)

            term_2 = torch.zeros((k - 1, k), device=self.device, dtype=term_1.dtype)
            term_4 = torch.zeros((k - 1, k), device=self.device, dtype=term_3.dtype)
            for i in range(k - 1):
                term_2[i, i] = i + 1
                term_2[i, i + 1] = k - (i + 2)

                term_4[i, i] = -1
                term_4[i, i + 1] = 1

            basis_matrix = torch.matmul(term_1, term_2) + torch.matmul(term_3, term_4)
            scalar *= 1 / (k - 1)
            k += 1

        basis_matrix *= scalar

        return basis_matrix.to(dtype=torch.float64)

    def power_bases(self, x: torch.Tensor):
        """
        Compute power bases for the given input tensor.

        Args:
            x (torch.Tensor):                   Input tensor.

        Returns:
            u (torch.Tensor):                   Power bases tensor.
            x_intervals (torch.Tensor):         Tensor representing the applicable grid interval for each input value.
        """

        # Determine applicable grid interval boundary values
        grid_floors = self.grid[:, 0]
        grid_floors = grid_floors.unsqueeze(0).expand(x.shape[0], -1)

        x = x.unsqueeze(dim=2)
        grid = self.grid.unsqueeze(dim=0)

        x.to(self.device)
        x_intervals = (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])
        x_interval_floor = torch.argmax(x_intervals.to(torch.int), dim=-1, keepdim=True)
        x_interval_floor = x_interval_floor.squeeze(-1)
        x_interval_floor = ((x_interval_floor * self.grid_intervals) + grid_floors)
        x_interval_ceiling = x_interval_floor + self.grid_intervals

        x = x.squeeze(2)

        # Calculate power bases
        u1_numerator = x - x_interval_floor
        u1_denominator = x_interval_ceiling - x_interval_floor
        u1 = (u1_numerator / u1_denominator).unsqueeze(-1)
        ones = torch.ones(u1.shape, dtype=x.dtype, device=self.device)
        u = torch.cat((ones, u1), -1)
        for i in range(2, self.k + 1):
            base = u1 ** i
            u = torch.cat((u, base), -1)

        return u, x_intervals

    def b_splines_matrix(self, x):
        """
        Computes the b-spline output based on the given input tensor.

        Args:
            x (torch.Tensor):       Input tensor.

        Returns:
            result (torch.Tensor):   Tensor representing the outputs of each basis function.
        """

        # Calculate power bases and applicable grid intervals
        power_bases, x_intervals = self.power_bases(x)

        # Pad basis matrix per input
        basis_matrices = torch.nn.functional.pad(self.basis_matrix, (self.k + self.num, self.k + self.num),
                                                 mode='constant', value=0)
        basis_matrices = basis_matrices.unsqueeze(0).unsqueeze(0)
        basis_matrices = basis_matrices.expand(power_bases.size(0), self.in_dim, -1, -1)

        # Calculate applicable grid intervals
        out_of_bounds_interval = torch.zeros((x_intervals.size(0), x_intervals.size(1), 1), dtype=torch.bool).to(self.device)
        x_intervals = torch.cat((out_of_bounds_interval, x_intervals), -1)

        # Identify and gather applicable basis functions
        basis_func_floor_indices = torch.argmax(x_intervals.to(torch.int), dim=-1, keepdim=True)
        basis_func_floor_indices = (2 * self.k) + self.num - basis_func_floor_indices + 1
        basis_func_indices = torch.arange(0, self.k + self.num, 1).unsqueeze(0).unsqueeze(0).to(self.device)
        basis_func_indices = basis_func_indices.expand(
            basis_matrices.size(0),
            basis_matrices.size(1),
            basis_matrices.size(2),
            -1
        )
        basis_func_indices = basis_func_indices.clone()
        basis_func_indices += basis_func_floor_indices.unsqueeze(-2).expand(-1, -1, basis_func_indices.size(-2), -1)

        basis_matrices = torch.gather(basis_matrices, -1, basis_func_indices)

        # Calculate basis function outputs
        power_bases = power_bases.unsqueeze(-2)
        result = torch.matmul(power_bases, basis_matrices)
        result = result.squeeze(-2)

        # in case grid is degenerate
        result = torch.nan_to_num(result)
        return result

    def b_splines_matrix_output(self, x: torch.Tensor):
        """
        Computes b-spline output based on the given input tensor and spline coefficients.

        Args:
            x (torch.Tensor):   Input tensor.

        Returns:
            result (torch.Tensor):   Tensor representing the outputs of each B-spline.
        """

        # Calculate basis function outputs
        basis_func_outputs = self.b_splines_matrix(x)

        # Calculate spline outputs
        result = torch.einsum('ijk,jlk->ijl', basis_func_outputs, self.coef)

        return result

    def forward(self, x):
        """
        MatrixKANLayer forward given input x
        
        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            y : 2D torch.float
                outputs, shape (number of samples, output dimension)
            preacts : 3D torch.float
                fan out x into activations, shape (number of sampels, output dimension, input dimension)
            postacts : 3D torch.float
                the outputs of activation functions with preacts as inputs
            postspline : 3D torch.float
                the outputs of spline functions with preacts as inputs
        """

        batch = x.shape[0]
        preacts = x[:,None,:].clone().expand(batch, self.out_dim, self.in_dim)
            
        base = self.base_fun(x) # (batch, in_dim)
        y = self.b_splines_matrix_output(x)
        
        postspline = y.clone().permute(0,2,1)
            
        y = self.scale_base[None,:,:] * base[:,:,None] + self.scale_sp[None,:,:] * y
        y = self.mask[None,:,:] * y
        
        postacts = y.clone().permute(0,2,1)
            
        y = torch.sum(y, dim=1)
        return y, preacts, postacts, postspline

    def update_grid_from_samples(self, x, mode='sample'):
        """
        update grid from samples
        
        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            None
        """
        
        batch = x.shape[0]
        x_pos = torch.sort(x, dim=0)[0]
        y_eval = coef2curve(x_pos, self.grid, self.coef, self.k)
        num_interval = self.grid.shape[1] - 1 - 2 * self.k
        
        def get_grid(num_interval):
            ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
            grid_adaptive = x_pos[ids, :].permute(1,0)
            h = (grid_adaptive[:,[-1]] - grid_adaptive[:,[0]])/num_interval
            grid_uniform = grid_adaptive[:,[0]] + h * torch.arange(num_interval+1,)[None, :].to(x.device)
            grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
            return grid
        
        grid = get_grid(num_interval)
        
        if mode == 'grid':
            sample_grid = get_grid(2*num_interval)
            x_pos = sample_grid.permute(1,0)
            y_eval = coef2curve(x_pos, self.grid, self.coef, self.k)

        self.grid_range[:, 0], self.grid_range[:, 1] = grid[:, 0], grid[:, -1]
        self.grid_intervals.data = (self.grid_range[:, 1] - self.grid_range[:, 0]) / self.num
        
        self.grid.data = extend_grid(grid, k_extend=self.k)
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k)

    def initialize_grid_from_parent(self, parent, x, mode='sample'):
        """
        update grid from a parent MatrixKANLayer & samples
        
        Args:
        -----
            parent : MatrixKANLayer
                a parent MatrixKANLayer (whose grid is usually coarser than the current model)
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            None
        """
        
        batch = x.shape[0]
        
        x_pos = torch.sort(x, dim=0)[0]
        y_eval = coef2curve(x_pos, parent.grid, parent.coef, parent.k)
        num_interval = self.grid.shape[1] - 1 - 2*self.k
        
        def get_grid(num_interval):
            ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
            grid_adaptive = x_pos[ids, :].permute(1,0)
            h = (grid_adaptive[:,[-1]] - grid_adaptive[:,[0]])/num_interval
            grid_uniform = grid_adaptive[:,[0]] + h * torch.arange(num_interval+1,)[None, :].to(x.device)
            grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
            return grid
        
        grid = get_grid(num_interval)
        
        if mode == 'grid':
            sample_grid = get_grid(2*num_interval)
            x_pos = sample_grid.permute(1,0)
            y_eval = coef2curve(x_pos, parent.grid, parent.coef, parent.k)

        self.grid_range[:, 0], self.grid_range[:, 1] = grid[:, 0], grid[:, -1]
        self.grid_intervals.data = (self.grid_range[:, 1] - self.grid_range[:, 0]) / self.num
        
        grid = extend_grid(grid, k_extend=self.k)
        self.grid.data = grid
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k)

