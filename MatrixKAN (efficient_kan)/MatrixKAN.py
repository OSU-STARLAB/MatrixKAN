import torch
import math
import torch.nn.functional as F


class MatrixKANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            device,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1]
    ):
        super(MatrixKANLinear, self).__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Initialize knots
        self.grid_interval = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * self.grid_interval
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.grid = grid.to(self.device)

        self.grid_range = torch.tensor(grid_range, device=device).unsqueeze(0).expand(self.in_features, -1)
        self.grid_range = self.grid_range.clone().to(dtype=torch.float64)
        self.grid_intervals = ((self.grid_range[:,1] - self.grid_range[:,0]) / self.grid_size)

        # self.layer_norm = torch.nn.Tanh()
        self.layer_norm = torch.nn.Hardtanh()
        self.layer_norm_shifts = (self.grid_range[:, 0] + self.grid_range[:, -1]) / 2
        self.layer_norm_scalars = (self.grid_range[:, -1] - self.grid_range[:, 0]) / 2

        # Initialized Trainable Parameters
        self.base_weight = torch.nn.Parameter(torch.tensor(torch.ones(out_features, in_features), dtype=torch.float64))
        self.spline_weight = torch.nn.Parameter(
            torch.tensor(torch.ones(in_features, out_features, grid_size + spline_order), dtype=torch.float64)
        )

        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(in_features, out_features)
            )

        # Initialize Basis Matrix
        self.basis_matrix = self.basis_matrix()

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

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

        while k <= self.spline_order + 1:
            term_1 = torch.nn.functional.pad(basis_matrix, (0, 0, 0, 1), "constant", 0)
            term_3 = torch.nn.functional.pad(basis_matrix, (0, 0, 1, 0), "constant", 0)

            term_2 = torch.zeros((k - 1, k), device=self.device, dtype=torch.float32)
            term_4 = torch.zeros((k - 1, k), device=self.device, dtype=torch.float32)
            for i in range(k - 1):
                term_2[i, i] = i + 1
                term_2[i, i + 1] = k - (i + 2)

                term_4[i, i] = -1
                term_4[i, i + 1] = 1

            basis_matrix = torch.matmul(term_1, term_2) + torch.matmul(term_3, term_4)
            scalar *= 1 / (k - 1)
            k += 1

        basis_matrix *= scalar

        return basis_matrix.to(torch.float64)

    def power_bases(self, x: torch.Tensor):
        """
        Compute power bases for the given input tensor.

        Args:
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
        grid_floors = self.grid[:, 0]
        grid_floors = grid_floors.unsqueeze(0).expand(x.shape[0], -1)
        x_interval_floor = torch.floor((x - grid_floors) / self.grid_interval)
        x_interval_floor = ((x_interval_floor * self.grid_interval) + grid_floors)

        # Determine applicable grid interval values (upper-bound)
        x_interval_ceiling = x_interval_floor + self.grid_interval

        # Calculate grid indices of interval floor
        x = x.unsqueeze(dim=2)
        grid = self.grid.unsqueeze(dim=0)
        grid_interval_floor = (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])
        x = x.squeeze(dim=2)

        # Calculate index position of the lower knot in the applicable knot span.
        # This is later used to calculate the applicable control points / basis functions.
        grid_interval_floor_indices = torch.nonzero(grid_interval_floor, as_tuple=True)
        grid_interval_floor_indices = grid_interval_floor_indices[-1]
        grid_interval_floor_indices = torch.clamp(grid_interval_floor_indices, min=self.spline_order, max=self.spline_order + self.grid_size - 1)
        if grid_interval_floor_indices.size(0) < (x.size(dim=0) * x.size(dim=1)):
            print("help")                                                                                                 ############### DEBUG ###################
        grid_interval_floor_indices = grid_interval_floor_indices.reshape(x.shape)

        # Calculate power bases
        u1_numerator = x - x_interval_floor
        u1_denominator = x_interval_ceiling - x_interval_floor  ############# REPLACE WITH GRID_INTERVALS ##########
        u1 = (u1_numerator / u1_denominator).unsqueeze(-1)
        ones = torch.ones(u1.shape, dtype=x.dtype, device=self.device)
        u = torch.cat((ones, u1), -1)
        for i in range(2, self.spline_order + 1):
            base = u1 ** i
            u = torch.cat((u, base), -1)

        return u, grid_interval_floor_indices

    def b_splines_matrix(self, power_bases):
        """
        Computes the b-spline output based on the given input tensor.

        Args:
            power_bases (torch.Tensor):   Input tensor of shape (batch_size, sequence length, in_features).

        Returns:
            torch.Tensor:       Power bases tensor of shape (batch_size, sequence length, out_features).
        """

        return torch.matmul(power_bases, self.basis_matrix)

    def b_splines_matrix_output(self, x: torch.Tensor):
        """
        Computes b-spline output based on the given input tensor and spline coefficients.

        Args:
            x (torch.Tensor):   Input tensor of shape (batch_size, sequence length, in_features).

        Returns:
            torch.Tensor:       Power bases tensor of shape (batch_size, sequence length, out_features).
        """

        # Calculate power bases
        power_bases, grid_interval_floor_indices = self.power_bases(x)
        b_splines = self.b_splines_matrix(power_bases)

        # Calculate applicable control points
        if self.grid_size == 1:
            # If grid_size == 1, only one curve defined for spline / all control points apply.
            control_point_floor_indices = torch.zeros(grid_interval_floor_indices.shape,
                                                      dtype=torch.int64,
                                                      device=self.device)
        else:
            # If grid_size > 1, multiple curves defined for spline / calculate applicable control points per input
            # For knot interval [u(i), u(i+1)), applicable control points are P(i-p) ... P(i)
            control_point_floor_indices = (grid_interval_floor_indices - self.spline_order)

        control_point_floor_indices = control_point_floor_indices.unsqueeze(-1)

        control_point_indices = torch.arange(0, self.spline_order + 1, 1).unsqueeze(0).unsqueeze(0).to(self.device)
        control_point_indices = control_point_indices.expand(
            control_point_floor_indices.size(0),
            control_point_floor_indices.size(1),
            -1
        )
        control_point_indices = control_point_indices.clone()
        control_point_indices += control_point_floor_indices
        control_point_indices = control_point_indices.unsqueeze(1).expand(-1, self.out_features, -1, -1)

        control_points = self.spline_weight.view(self.out_features, self.in_features, -1).unsqueeze(0).expand(
            control_point_indices.size(0), -1, -1, -1)
        control_points = torch.gather(control_points, -1, control_point_indices)
        control_points = control_points.view(
            control_point_indices.size(0),
            control_point_indices.size(1),
            control_point_indices.size(2),
            -1)

        # Calculate spline outputs
        # prod1 = torch.matmul(power_bases, self.basis_matrix)
        b_splines = b_splines.view(x.size(0), x.size(1), -1).unsqueeze(-2)
        control_points = control_points.view(b_splines.size(0), b_splines.size(1), b_splines.size(-1), -1)
        result = torch.matmul(b_splines, control_points)
        result = result.squeeze(-2)

        return result

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff_orig(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                ).view(self.in_features, self.out_features, -1)
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    # Original
    """
    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.reshape(*original_shape[:-1], self.out_features)
        return output
    """

    # MatrixKAN
    def forward(self, x: torch.Tensor, mask):
        assert x.size(-1) == self.in_features

        # x = self.normalize_input(x)
        x, mask = self.mask_input(x, mask)

        base_activations = self.base_activation(x)
        base_output = torch.matmul(base_activations, self.base_weight.transpose(-2, -1))
        spline_output = torch.tensor([])
        if x.size(0) != 0:
            spline_output = self.b_splines_matrix_output(x)
            spline_output = torch.sum(spline_output, dim=-2)
        output = base_output + spline_output

        return output, mask

    # B_spline_matrix() version
    """
    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        x_sorted = torch.sort(x, dim=0)[0]
        # grid_max = torch.max(torch.abs(x_sorted[[0, -1], :].permute(1, 0)), dim=1)[0].unsqueeze(-1)
        uniform_step = (x_sorted[-1] - x_sorted[0]) / self.grid_size
        self.update_grid_rescale(uniform_step)

        unreduced_spline_output = self.b_spline_matrix(x)  # (batch, in, out)

        # sort each channel individually to collect data distribution
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid_range[:, 0], self.grid_range[:, 1] = grid_uniform.T[:, 0], grid_uniform.T[:, -1]
        self.grid_intervals.data = (self.grid_range[:, 1] - self.grid_range[:, 0]) / self.grid_size
        self.layer_norm_shifts.data = (self.grid_range.data[:, 0] + self.grid_range.data[:, -1]) / 2
        self.layer_norm_scalars.data = (self.grid_range.data[:, -1] - self.grid_range.data[:, 0]) / 2

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))
    """

    # Original version
    """
    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        
        # x_sorted = torch.sort(x, dim=0)[0]
        # grid_max = torch.max(torch.abs(x_sorted[[0, -1], :].permute(1, 0)), dim=1)[0].unsqueeze(-1)
        # grid_range = x_sorted.T[:, [0, -1]]
        # intervals = (x_sorted[-1] - x_sorted[0]) / self.grid_size
        # scalar = (x_sorted[-1] - x_sorted[0]) / 2
        # self.update_grid_rescale(grid_range, intervals, scalar)

        splines = self.b_splines(x)  # (batch, in, coeff)
        # splines = self.b_splines_matrix(self.power_bases(x)[0])  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )
        
        # self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))
        
        self.grid_range[:, 0], self.grid_range[:, 1] = grid_uniform.T[:, 0], grid_uniform.T[:, -1]
        self.grid_intervals.data = (self.grid_range[:, 1] - self.grid_range[:, 0]) / self.grid_size
        self.layer_norm_shifts.data = (self.grid_range.data[:, 0] + self.grid_range.data[:, -1]) / 2
        self.layer_norm_scalars.data = (self.grid_range.data[:, -1] - self.grid_range.data[:, 0]) / 2


        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))
    """

    # Pykan version

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features

        batch = x.shape[0]
        # x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, ).to(self.device)).reshape(batch, self.size).permute(1, 0)
        x_pos = torch.sort(x, dim=0)[0]
        y_eval = self.coef2curve(x_pos, self.grid, self.spline_weight, self.spline_order)
        num_interval = self.grid.shape[1] - 1 - 2 * self.spline_order

        def get_grid(num_interval):
            ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
            grid_adaptive = x_pos[ids, :].permute(1, 0)
            h = (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]]) / num_interval
            grid_uniform = grid_adaptive[:, [0]] + h * torch.arange(num_interval + 1, )[None, :].to(x.device)
            grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
            return grid

        grid = get_grid(num_interval)

        # if mode == 'grid':
        #     sample_grid = get_grid(2 * num_interval)
        #     x_pos = sample_grid.permute(1, 0)
        #     y_eval = coef2curve(x_pos, self.grid, self.coef, self.k)
        
        self.grid_range[:, 0], self.grid_range[:, 1] = grid[:, 0], grid[:, -1]
        self.grid_intervals.data = (self.grid_range[:, 1] - self.grid_range[:, 0]) / self.grid_size
        self.layer_norm_shifts.data = (self.grid_range.data[:, 0] + self.grid_range.data[:, -1]) / 2
        self.layer_norm_scalars.data = (self.grid_range.data[:, -1] - self.grid_range.data[:, 0]) / 2

        self.grid.data = self.extend_grid(grid, k_extend=self.spline_order)
        self.spline_weight.data = self.curve2coeff(x_pos, y_eval, self.grid, self.spline_order)


    def update_grid_rescale(self, grid_range, intervals, scalar):
        """
        Preliminarily updates the grid to allow for the un-normalized x values to be calculated using b_spline_matrix().
        """

        intervals = intervals.unsqueeze(-1)
        scalar = scalar.unsqueeze(-1)
        self.grid = ((self.grid - self.layer_norm_shifts.unsqueeze(-1)) / self.layer_norm_scalars.unsqueeze(-1)) * scalar
        self.grid_range = grid_range
        # self.grid_range = ((self.grid_range - self.layer_norm_shifts.unsqueeze(-1)) / self.layer_norm_scalars.unsqueeze(-1)) * scalar
        self.grid_intervals.data = intervals
        self.layer_norm_scalars.data = scalar

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )

    def normalize_input(self, x):
        """
        Normalizes the input values to fall between the defined range of each spline.
        """

        x = x - self.layer_norm_shifts
        x = x / self.layer_norm_scalars
        x = self.layer_norm(x)
        x = x * self.layer_norm_scalars
        x = x + self.layer_norm_shifts

        return x

    def mask_input(self, x, prior_mask=None):
        """
        Masks input values that are outside the defined range of each spline.
        """

        # Removing inputs with out-of-range values
        mask = torch.all((x > self.grid_range[:, 0]) & (x < self.grid_range[:, 1]), dim=-1, keepdim=True).squeeze(-1)
        x_masked = x[mask]
        if prior_mask is not None:
            prior_mask_true = prior_mask.nonzero(as_tuple=True)[0]
            new_mask = prior_mask.clone()
            new_mask[prior_mask_true] = mask
            mask = new_mask
        return x_masked, mask



    def coef2curve(self, x_eval, grid, coef, k, device="cpu"):
        '''
        converting B-spline coefficients to B-spline curves. Evaluate x on B-spline curves (summing up B_batch results over B-spline basis).

        Args:
        -----
            x_eval : 2D torch.tensor
                shape (batch, in_dim)
            grid : 2D torch.tensor
                shape (in_dim, G+2k). G: the number of grid intervals; k: spline order.
            coef : 3D torch.tensor
                shape (in_dim, out_dim, G+k)
            k : int
                the piecewise polynomial order of splines.
            device : str
                devicde

        Returns:
        --------
            y_eval : 3D torch.tensor
                shape (number of samples, in_dim, out_dim)

        '''

        b_splines = self.b_splines(x_eval)
        y_eval = torch.einsum('ijk,jlk->ijl', b_splines, coef.to(b_splines.device))

        return y_eval

    # Original version
    def curve2coeff_orig(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A.to(self.device), B.to(self.device)
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    # pykan version
    def curve2coeff(self, x_eval, y_eval, grid, k, lamb=1e-8):
        '''
        converting B-spline curves to B-spline coefficients using least squares.

        Args:
        -----
            x_eval : 2D torch.tensor
                shape (in_dim, out_dim, number of samples)
            y_eval : 2D torch.tensor
                shape (in_dim, out_dim, number of samples)
            grid : 2D torch.tensor
                shape (in_dim, grid+2*k)
            k : int
                spline order
            lamb : float
                regularized least square lambda

        Returns:
        --------
            coef : 3D torch.tensor
                shape (in_dim, out_dim, G+k)
        '''
        batch = x_eval.shape[0]
        in_dim = x_eval.shape[1]
        out_dim = y_eval.shape[2]
        n_coef = grid.shape[1] - k - 1
        mat = self.b_splines(x_eval)
        mat = mat.permute(1, 0, 2)[:, None, :, :].expand(in_dim, out_dim, batch, n_coef)
        y_eval = y_eval.permute(1, 2, 0).unsqueeze(dim=3)
        device = mat.device

        # coef = torch.linalg.lstsq(mat, y_eval,
        # driver='gelsy' if device == 'cpu' else 'gels').solution[:,:,:,0]

        XtX = torch.einsum('ijmn,ijnp->ijmp', mat.permute(0, 1, 3, 2), mat)
        Xty = torch.einsum('ijmn,ijnp->ijmp', mat.permute(0, 1, 3, 2), y_eval)
        n1, n2, n = XtX.shape[0], XtX.shape[1], XtX.shape[2]
        identity = torch.eye(n, n)[None, None, :, :].expand(n1, n2, n, n).to(device)
        A = XtX + lamb * identity
        B = Xty
        coef = (A.pinverse() @ B)[:, :, :, 0]

        return coef

    def extend_grid(self, grid, k_extend=0):
        '''
        extend grid
        '''
        h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

        for i in range(k_extend):
            grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
            grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)

        return grid


class MatrixKAN(torch.nn.Module):
    def __init__(
            self,
            layers_hidden,
            device,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1]
    ):
        super(MatrixKAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                MatrixKANLinear(
                    in_features,
                    out_features,
                    device,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        mask = None
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x, margin=0)
            x, mask = layer(x, mask)
        return x, mask

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
