import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import layer_norm

from .spline import *
from .utils import sparse_mask


class MatrixKANLayer(nn.Module):
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
        ''''
        initialize a KANLayer
        
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
            
        Example
        -------
        >>> from kan.MatrixKANLayer import *
        >>> model = MatrixKANLayer(in_dim=3, out_dim=5)
        >>> (model.in_dim, model.out_dim)
        '''
        super(MatrixKANLayer, self).__init__()
        # size 
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k

        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1)[None,:].to(torch.float64).expand(self.in_dim, num+1)
        # grid = torch.linspace(grid_range[0], grid_range[1], steps=k)[None, :].expand(self.in_dim, k)
        grid = extend_grid(grid, k_extend=k)
        self.grid = torch.nn.Parameter(grid).requires_grad_(False)
        noises = (torch.rand(self.num+1, self.in_dim, self.out_dim) - 1/2) * noise_scale / num

        self.coef = torch.nn.Parameter(curve2coef(self.grid[:,k:-k].permute(1,0), noises, self.grid, k))
        
        if sparse_init:
            self.mask = torch.nn.Parameter(sparse_mask(in_dim, out_dim)).requires_grad_(False)
        else:
            self.mask = torch.nn.Parameter(torch.ones(in_dim, out_dim)).requires_grad_(False)
        
        self.scale_base = torch.nn.Parameter(scale_base_mu * 1 / np.sqrt(in_dim) + \
                         scale_base_sigma * (torch.rand(in_dim, out_dim)*2-1) * 1/np.sqrt(in_dim)).requires_grad_(sb_trainable)
        self.scale_sp = torch.nn.Parameter(torch.ones(in_dim, out_dim) * scale_sp * self.mask).requires_grad_(sp_trainable)  # make scale trainable
        self.base_fun = base_fun
        # self.grid_range = grid_range
        self.grid_range = torch.tensor(grid_range, device=device).unsqueeze(0).expand(in_dim, -1)
        self.grid_range = self.grid_range.clone().to(dtype=torch.float64)
        # self.grid_interval = (grid_range[1] - grid_range[0]) / num
        self.grid_intervals = ((self.grid_range[:,1] - self.grid_range[:,0]) / num)
        self.device = device

        self.layer_norm = torch.nn.Tanh()
        # self.layer_norm = torch.nn.Hardtanh()
        # self.layer_norm_scalars = torch.abs(self.grid_range[:,-1])
        # self.layer_norm_scalars = self.grid_range[:,-1]
        self.layer_norm_shifts = (self.grid_range[:,0] + self.grid_range[:,-1]) / 2
        self.layer_norm_scalars = (self.grid_range[:,-1] - self.grid_range[:,0]) / 2

        # Initialize Basis Matrix
        self.basis_matrix = self.basis_matrix()

        self.grid_eps = grid_eps
        
        self.to(device)
        
    def to(self, device):
        super(MatrixKANLayer, self).to(device)
        self.device = device    
        return self

    def forward(self, x):
        '''
        KANLayer forward given input x
        
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
        
        Example
        -------
        >>> from kan.MatrixKANLayer import *
        >>> model = MatrixKANLayer(in_dim=3, out_dim=5)
        >>> x = torch.normal(0,1,size=(100,3))
        >>> y, preacts, postacts, postspline = model(x)
        >>> y.shape, preacts.shape, postacts.shape, postspline.shape
        '''
        batch = x.shape[0]

        x = x - self.layer_norm_shifts
        x = x / (self.layer_norm_scalars / 2)
        x = self.layer_norm(x)
        x = x * self.layer_norm_scalars
        x = x + self.layer_norm_shifts

        preacts = x[:, None, :].clone().expand(batch, self.out_dim, self.in_dim)
            
        base = self.base_fun(x) # (batch, in_dim)
        y = self.b_spline_matrix(x)
        
        postspline = y.clone().permute(0,2,1)
            
        y = self.scale_base[None,:,:] * base[:,:,None] + self.scale_sp[None,:,:] * y
        y = self.mask[None,:,:] * y
        
        postacts = y.clone().permute(0,2,1)
            
        y = torch.sum(y, dim=1)
        return y, preacts, postacts, postspline

    def basis_matrix(self):
        """
        Compute the basis matrix for a uniform B-spline with a given spline degree.

        Returns:
            torch.Tensor: Basis matrix tensor of shape (spline_order + 1, spline_order + 1).
        """

        basis_matrix = torch.tensor([
            [1]
        ], dtype=torch.float32, device=self.device)

        scalar = 1

        k = 2

        while k <= self.k + 1:
            term_1 = torch.nn.functional.pad(basis_matrix, (0, 0, 0, 1), "constant", 0)
            term_3 = torch.nn.functional.pad(basis_matrix, (0, 0, 1, 0), "constant", 0)

            term_2 = torch.zeros((k - 1, k), dtype=term_1.dtype, device=self.device)
            term_4 = torch.zeros((k - 1, k), dtype=term_3.dtype, device=self.device)
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

        :Args:
            x (torch.Tensor):                   Input tensor of shape (batch_size, sequence length, in_features).

        Returns:
            u (torch.Tensor):                   Power bases tensor of shape
                                                (batch_size, sequence length, in_features, spline_order + 1).

            grid_interval_floor (torch.Tensor): self.grid expanded per input and masked to represent lower bound
                                                of applicable knot interval

            grid_interval_floor_indices
            (torch.Tensor):                     Indices of lower bound of applicable knot interval in self.grid.
        """

        # Determine applicable grid interval values (lower-bound)
        grid_floors = self.grid[:,0]
        grid_floors = grid_floors.unsqueeze(0).expand(x.shape[0], -1)
        x_pos = (x - grid_floors)
        x_interval_floor = torch.floor(x_pos / self.grid_intervals)
        x_interval_floor = ((x_interval_floor * self.grid_intervals) + grid_floors)

        # Determine applicable grid interval values (upper-bound)
        x_interval_ceiling = x_interval_floor + self.grid_intervals

        # Calculate grid indices of interval floor
        """
        if self.grid.shape[-1] != x_interval_floor.shape[-1]:
            x_interval_floor = x_interval_floor.unsqueeze(-1)
        grid_interval_floor = torch.isclose(self.grid, x_interval_floor)
        """
        x = x.unsqueeze(dim=2)
        grid = self.grid.unsqueeze(dim=0)
        grid_interval_floor = (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])
        # equal = grid_interval_floor[:, :, :-1] == grid_interval_floor2
        x = x.squeeze(dim=2)
        # grid_interval_floor = self.grid == x_interval_floor


        # Calculate index position of the lower knot in the applicable knot span.
        # This is later used to calculate the applicable control points / basis functions.
        grid_interval_floor_indices = torch.nonzero(grid_interval_floor, as_tuple=True)
        grid_interval_floor_indices = grid_interval_floor_indices[-1]
        if x.size(dim=0) * self.in_dim != grid_interval_floor_indices.size(dim=0):
            print("help")

        grid_interval_floor_indices = grid_interval_floor_indices.reshape(x.shape)

        grid_interval_floor_indices = torch.clamp(grid_interval_floor_indices, min=self.k, max=self.k + self.num - 1)
        """
        out_of_bounds = ~((grid_interval_floor_indices >= 3) * (grid_interval_floor_indices <= 4))                                                       ############## DEBUG DELETE #############
        out_of_bounds = torch.nonzero(out_of_bounds, as_tuple=True)
        if out_of_bounds[-1].size(dim=-1) > 0:
            bad_val = x[out_of_bounds[0][0]][out_of_bounds[1][0]]
            print("Out of bounds:")
            print(out_of_bounds)
        """

        # Calculate power bases
        if x.shape[-1] != x_interval_floor.shape[-1]:
            x_interval_floor = x_interval_floor.reshape(x.shape)
        u1_numerator = x - x_interval_floor
        u1_denominator = x_interval_ceiling - x_interval_floor
        u1 = (u1_numerator / u1_denominator).unsqueeze(-1)
        ones = torch.ones(u1.shape, dtype=x.dtype, device=self.device)
        u = torch.cat((ones, u1), -1)
        for i in range(2, self.k + 1):
            base = u1 ** i
            u = torch.cat((u, base), -1)

        return u, grid_interval_floor_indices

    def b_spline_matrix(self, x: torch.Tensor, normalized=True):
        """
        Compute spline output for the given input tensor.

        :Args:
            x (torch.Tensor):   Input tensor of shape (batch_size, sequence length, in_features).

        Returns:
            torch.Tensor:       Power bases tensor of shape (batch_size, sequence length, out_features).
        """


        if not normalized:
            x = x - self.layer_norm_shifts
            x = x / (self.layer_norm_scalars / 2)
            x = self.layer_norm(x)
            x = x * self.layer_norm_scalars
            x = x + self.layer_norm_shifts

        # Calculate power bases
        power_bases, grid_interval_floor_indices = self.power_bases(x)

        # Calculate applicable control points
        if self.num == 1:
            # If grid_size == 1, only one curve defined for spline / all control points apply.
            control_point_floor_indices = torch.zeros(grid_interval_floor_indices.shape,
                                                      dtype=torch.int64,
                                                      device=self.device)
        else:
            # If grid_size > 1, multiple curves defined for spline / calculate applicable control points per input
            # For knot interval [u(i), u(i+1)), applicable control points are P(i-p) ... P(i)
            control_point_floor_indices = (grid_interval_floor_indices - self.k)

        control_point_floor_indices = control_point_floor_indices.unsqueeze(-1)

        control_point_indices = torch.arange(0, self.k + 1, 1).unsqueeze(0).unsqueeze(0).to(self.device)
        control_point_indices = control_point_indices.expand(
            control_point_floor_indices.size(0),
            control_point_floor_indices.size(1),
            -1
        )
        control_point_indices = control_point_indices.clone()
        control_point_indices += control_point_floor_indices
        control_point_indices = control_point_indices.unsqueeze(2).expand(-1, -1, self.out_dim, -1)

        control_points = self.coef.unsqueeze(0).expand(
            control_point_indices.size(0), -1, -1, -1)
        # out_of_bounds = ~((control_point_floor_indices >= 0) * (control_point_floor_indices <= 1))               ############## DEBUG DELETE #############
        # out_of_bounds = torch.nonzero(out_of_bounds, as_tuple=True)
        # if out_of_bounds[-1].size(dim=-1) > 0:
        #     bad_val = x[out_of_bounds[0][0]][out_of_bounds[1][0]]
        #     print("Out of bounds:")
        #     print(out_of_bounds)
        control_points = torch.gather(control_points, -1, control_point_indices)
        control_points = control_points.view(
            control_point_indices.size(0),
            control_point_indices.size(1),
            control_point_indices.size(2),
            -1)

        # Calculate spline outputs
        prod1 = torch.matmul(power_bases, self.basis_matrix)
        prod1 = prod1.view(x.size(0), x.size(1), -1).unsqueeze(-2)
        control_points = control_points.view(control_points.size(0), control_points.size(1), -1, control_points.size(2))
        result = torch.matmul(prod1, control_points).squeeze(-2)

        return result

    def update_grid_from_samples_uniform(self, x, mode='sample'):
        '''
        update grid from samples (uniform)

        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)

        Returns:
        --------
            None

        Example
        -------
        >>> model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(model.grid.data)
        >>> x = torch.linspace(-3,3,steps=100)[:,None]
        >>> model.update_grid_from_samples(x)
        >>> print(model.grid.data)
        '''

        batch = x.shape[0]
        x_pos = torch.sort(x, dim=0)[0]
        # y_eval = self.b_spline_matrix((self.layer_norm(x_pos) * self.layer_norm_scalars) + self.layer_norm_shifts)
        y_eval = coef2curve(x_pos, self.grid, self.coef, self.k)
        num_interval = self.grid.shape[1] - 1 - 2 * self.k

        def get_grid(num_interval):
            grid_max = torch.max(torch.abs(x_pos[[0, -1], :].permute(1, 0)), dim=1)[0]
            h = ((grid_max * 2 )/ num_interval).unsqueeze(-1)
            grid_max = grid_max.unsqueeze(-1)
            grid_uniform = (h * torch.arange(num_interval + 1, )[None, :].to(x.device)) - grid_max
            return grid_uniform

        grid = get_grid(num_interval)

        if mode == 'grid':
            sample_grid = get_grid(2 * num_interval)
            x_pos = sample_grid.permute(1, 0)
            y_eval = coef2curve(x_pos, self.grid, self.coef, self.k)
            # y_eval = self.b_spline_matrix(x_pos, normalized=False)

        self.grid_range[:,0], self.grid_range[:,1] = grid[:,0], grid[:,-1]
        self.grid_intervals = (self.grid_range[:, 1] - self.grid_range[:, 0]) / self.num

        # Determine shift and scalar for new layer_norm and update
        # self.layer_norm_scalars = torch.abs(self.grid_range[:,-1])
        # self.layer_norm_scalars = self.grid_range[:,-1]
        self.layer_norm_shifts = (self.grid_range[:, 0] + self.grid_range[:, -1]) / 2
        self.layer_norm_scalars = (self.grid_range[:, -1] - self.grid_range[:, 0]) / 2

        self.grid.data = extend_grid(grid, k_extend=self.k)
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k)

    def update_grid_from_samples(self, x, mode='sample'):
        '''
        update grid from samples
        
        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            None
        
        Example
        -------
        >>> model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(model.grid.data)
        >>> x = torch.linspace(-3,3,steps=100)[:,None]
        >>> model.update_grid_from_samples(x)
        >>> print(model.grid.data)
        '''
        
        batch = x.shape[0]
        #x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, ).to(self.device)).reshape(batch, self.size).permute(1, 0)
        x_pos = torch.sort(x, dim=0)[0]
        y_eval = coef2curve(x_pos, self.grid, self.coef, self.k)
        # y_eval = self.b_spline_matrix(x_pos, normalized=False)
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
            y_eval = coef2curve(x_pos, self.grid, self.coef, self.k)
            # y_eval = self.b_spline_matrix(x_pos, normalized=False)

        self.grid_range[:, 0], self.grid_range[:, 1] = grid[:, 0], grid[:, -1]
        self.grid_intervals = (self.grid_range[:, 1] - self.grid_range[:, 0]) / self.num

        # Determine scalar for new layer_norm and update
        # self.layer_norm_scalars = torch.abs(self.grid_range[:, -1])
        # self.layer_norm_scalars = self.grid_range[:, -1]
        self.layer_norm_shifts = (self.grid_range[:, 0] + self.grid_range[:, -1]) / 2
        self.layer_norm_scalars = (self.grid_range[:, -1] - self.grid_range[:, 0]) / 2

        self.grid.data = extend_grid(grid, k_extend=self.k)
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k)

    def initialize_grid_from_parent(self, parent, x, mode='sample'):
        '''
        update grid from a parent KANLayer & samples
        
        Args:
        -----
            parent : KANLayer
                a parent KANLayer (whose grid is usually coarser than the current model)
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            None
          
        Example
        -------
        >>> batch = 100
        >>> parent_model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(parent_model.grid.data)
        >>> model = KANLayer(in_dim=1, out_dim=1, num=10, k=3)
        >>> x = torch.normal(0,1,size=(batch, 1))
        >>> model.initialize_grid_from_parent(parent_model, x)
        >>> print(model.grid.data)
        '''
        
        batch = x.shape[0]
        
        x_pos = torch.sort(x, dim=0)[0]
        # y_eval = coef2curve(x_pos, parent.grid, parent.coef, parent.k)
        y_eval = self.b_spline_matrix(x_pos, normalized=False)
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
            # y_eval = coef2curve(x_pos, parent.grid, parent.coef, parent.k)
            y_eval = self.b_spline_matrix(x_pos, normalized=False)

        self.grid_range[:, 0], self.grid_range[:, 1] = grid[:, 0], grid[:, -1]
        self.grid_intervals = (self.grid_range[:, 1] - self.grid_range[:, 0]) / self.num

        # Determine scalar for new layer_norm and update
        # self.layer_norm_scalars = torch.abs(self.grid_range[:, -1])
        # self.layer_norm_scalars = self.grid_range[:, -1]
        self.layer_norm_shifts = (self.grid_range[:, 0] + self.grid_range[:, -1]) / 2
        self.layer_norm_scalars = (self.grid_range[:, -1] - self.grid_range[:, 0]) / 2
        
        grid = extend_grid(grid, k_extend=self.k)
        self.grid.data = grid
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k)

    def get_subset(self, in_id, out_id):
        '''
        get a smaller KANLayer from a larger KANLayer (used for pruning)
        
        Args:
        -----
            in_id : list
                id of selected input neurons
            out_id : list
                id of selected output neurons
            
        Returns:
        --------
            spb : KANLayer
            
        Example
        -------
        >>> kanlayer_large = KANLayer(in_dim=10, out_dim=10, num=5, k=3)
        >>> kanlayer_small = kanlayer_large.get_subset([0,9],[1,2,3])
        >>> kanlayer_small.in_dim, kanlayer_small.out_dim
        (2, 3)
        '''
        spb = MatrixKANLayer(len(in_id), len(out_id), self.num, self.k, base_fun=self.base_fun)
        spb.grid.data = self.grid[in_id]
        spb.coef.data = self.coef[in_id][:,out_id]
        spb.scale_base.data = self.scale_base[in_id][:,out_id]
        spb.scale_sp.data = self.scale_sp[in_id][:,out_id]
        spb.mask.data = self.mask[in_id][:,out_id]

        spb.in_dim = len(in_id)
        spb.out_dim = len(out_id)
        return spb
    
    
    def swap(self, i1, i2, mode='in'):
        '''
        swap the i1 neuron with the i2 neuron in input (if mode == 'in') or output (if mode == 'out') 
        
        Args:
        -----
            i1 : int
            i2 : int
            mode : str
                mode = 'in' or 'out'
            
        Returns:
        --------
            None
            
        Example
        -------
        >>> from kan.KANLayer import *
        >>> model = KANLayer(in_dim=2, out_dim=2, num=5, k=3)
        >>> print(model.coef)
        >>> model.swap(0,1,mode='in')
        >>> print(model.coef)
        '''
        with torch.no_grad():
            def swap_(data, i1, i2, mode='in'):
                if mode == 'in':
                    data[i1], data[i2] = data[i2].clone(), data[i1].clone()
                elif mode == 'out':
                    data[:,i1], data[:,i2] = data[:,i2].clone(), data[:,i1].clone()

            if mode == 'in':
                swap_(self.grid.data, i1, i2, mode='in')
            swap_(self.coef.data, i1, i2, mode=mode)
            swap_(self.scale_base.data, i1, i2, mode=mode)
            swap_(self.scale_sp.data, i1, i2, mode=mode)
            swap_(self.mask.data, i1, i2, mode=mode)

