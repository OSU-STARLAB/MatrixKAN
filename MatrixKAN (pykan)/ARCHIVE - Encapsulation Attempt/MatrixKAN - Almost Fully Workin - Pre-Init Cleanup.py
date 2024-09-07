import pykan.kan as pykan
from pykan.kan.spline import *
from pykan.kan.utils import *
import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import yaml

class MatrixKAN(pykan.MultKAN):
    """
    MatrixKAN class

    Attributes:
    -----------
        grid : int
            the number of grid intervals
        k : int
            spline order
        act_fun : a list of KANLayers
        symbolic_fun: a list of Symbolic_KANLayer
        depth : int
            depth of KAN
        width : list
            number of neurons in each layer.
            Without multiplication nodes, [2,5,5,3] means 2D inputs, 3D outputs, with 2 layers of 5 hidden neurons.
            With multiplication nodes, [2,[5,3],[5,1],3] means besides the [2,5,53] KAN, there are 3 (1) mul nodes in layer 1 (2).
        mult_arity : int, or list of int lists
            multiplication arity for each multiplication node (the number of numbers to be multiplied)
        grid : int
            the number of grid intervals
        k : int
            the order of piecewise polynomial
        base_fun : fun
            residual function b(x). an activation function phi(x) = sb_scale * b(x) + sp_scale * spline(x)
        symbolic_fun : a list of Symbolic_KANLayer
            Symbolic_KANLayers
        symbolic_enabled : bool
            If False, the symbolic front is not computed (to save time). Default: True.
        width_in : list
            The number of input neurons for each layer
        width_out : list
            The number of output neurons for each layer
        base_fun_name : str
            The base function b(x)
        grip_eps : float
            The parameter that interpolates between uniform grid and adaptive grid (based on sample quantile)
        node_bias : a list of 1D torch.float
        node_scale : a list of 1D torch.float
        subnode_bias : a list of 1D torch.float
        subnode_scale : a list of 1D torch.float
        symbolic_enabled : bool
            when symbolic_enabled = False, the symbolic branch (symbolic_fun) will be ignored in computation (set to zero)
        affine_trainable : bool
            indicate whether affine parameters are trainable (node_bias, node_scale, subnode_bias, subnode_scale)
        sp_trainable : bool
            indicate whether the overall magnitude of splines is trainable
        sb_trainable : bool
            indicate whether the overall magnitude of base function is trainable
        save_act : bool
            indicate whether intermediate activations are saved in forward pass
        node_scores : None or list of 1D torch.float
            node attribution score
        edge_scores : None or list of 2D torch.float
            edge attribution score
        subnode_scores : None or list of 1D torch.float
            subnode attribution score
        cache_data : None or 2D torch.float
            cached input data
        acts : None or a list of 2D torch.float
            activations on nodes
        auto_save : bool
            indicate whether to automatically save a checkpoint once the model is modified
        state_id : int
            the state of the model (used to save checkpoint)
        ckpt_path : str
            the folder to store checkpoints
        round : int
            the number of times rewind() has been called
        device : str
    """

    def __init__(self, width=None, grid=3, k=3, mult_arity = 2, noise_scale=0.3, scale_base_mu=0.0, scale_base_sigma=1.0, base_fun='silu', symbolic_enabled=True, affine_trainable=False, grid_eps=0.02, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True, seed=1, save_act=True, sparse_init=False, auto_save=True, first_init=True, ckpt_path='./model', state_id=0, round=0, device='cpu', spline_matrix=False):
        """
        Initalize a MatrixKAN model

        Args:
        -----
            width : list of int
                Without multiplication nodes: :math:`[n_0, n_1, .., n_{L-1}]` specify the number of neurons in each layer (including inputs/outputs)
                With multiplication nodes: :math:`[[n_0,m_0=0], [n_1,m_1], .., [n_{L-1},m_{L-1}]]` specify the number of addition/multiplication nodes in each layer (including inputs/outputs)
            grid : int
                number of grid intervals. Default: 3.
            k : int
                order of piecewise polynomial. Default: 3.
            mult_arity : int, or list of int lists
                multiplication arity for each multiplication node (the number of numbers to be multiplied)
            noise_scale : float
                initial injected noise to spline.
            base_fun : str
                the residual function b(x). Default: 'silu'
            symbolic_enabled : bool
                compute (True) or skip (False) symbolic computations (for efficiency). By default: True.
            affine_trainable : bool
                affine parameters are updated or not. Affine parameters include node_scale, node_bias, subnode_scale, subnode_bias
            grid_eps : float
                When grid_eps = 1, the grid is uniform; when grid_eps = 0, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes.
            grid_range : list/np.array of shape (2,))
                setting the range of grids. Default: [-1,1]. This argument is not important if fit(update_grid=True) (by default updata_grid=True)
            sp_trainable : bool
                If true, scale_sp is trainable. Default: True.
            sb_trainable : bool
                If true, scale_base is trainable. Default: True.
            device : str
                device
            seed : int
                random seed
            save_act : bool
                indicate whether intermediate activations are saved in forward pass
            sparse_init : bool
                sparse initialization (True) or normal dense initialization. Default: False.
            auto_save : bool
                indicate whether to automatically save a checkpoint once the model is modified
            state_id : int
                the state of the model (used to save checkpoint)
            ckpt_path : str
                the folder to store checkpoints. Default: './model'
            round : int
                the number of times rewind() has been called
            device : str
            spline_matrix : bool
                indicate whether to use parallelized matrix calculation method for splines
        Returns:
        --------
            self

        Example
        -------
        >>> from kan import *                                             ################ FIX THIS ####################
        >>> model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
        checkpoint directory created: ./model
        saving model version 0.0
        """

        self.spline_matrix = spline_matrix

        super(MatrixKAN, self).__init__(width=width, grid=grid, k=k, mult_arity=mult_arity, noise_scale=noise_scale,
                                        scale_base_mu=scale_base_mu, scale_base_sigma=scale_base_sigma,
                                        base_fun=base_fun, symbolic_enabled=symbolic_enabled,
                                        affine_trainable=affine_trainable, grid_eps=grid_eps, grid_range=grid_range,
                                        sp_trainable=sp_trainable, sb_trainable=sb_trainable, seed=seed,
                                        save_act=save_act, sparse_init=sparse_init, auto_save=auto_save,
                                        first_init=first_init, ckpt_path=ckpt_path, state_id=state_id, round=round,
                                        device=device)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        ### initializeing the numerical front ###

        # self.act_fun = []
        self.depth = len(width) - 1

        for i in range(len(width)):
            if type(width[i]) == int:
                width[i] = [width[i], 0]

        self.width = width

        # if mult_arity is just a scalar, we extend it to a list of lists
        # e.g, mult_arity = [[2,3],[4]] means that in the first hidden layer, 2 mult ops have arity 2 and 3, respectively;
        # in the second hidden layer, 1 mult op has arity 4.
        if isinstance(mult_arity, int):
            self.mult_homo = True  # when homo is True, parallelization is possible
        else:
            self.mult_homo = False  # when home if False, for loop is required.
        self.mult_arity = mult_arity

        width_in = self.width_in
        width_out = self.width_out

        self.base_fun_name = base_fun
        if base_fun == 'silu':
            base_fun = torch.nn.SiLU()
        elif base_fun == 'identity':
            base_fun = torch.nn.Identity()
        elif base_fun == 'zero':
            base_fun = lambda x: x * 0.

        self.grid_eps = grid_eps
        self.grid_range = grid_range

        for l in range(self.depth):
            self.act_fun.pop(0)

        for l in range(self.depth):
            # splines
            if spline_matrix:
                sp_batch = MatrixKANLayer(in_dim=width_in[l], out_dim=width_out[l + 1], num=grid, k=k,
                                          noise_scale=noise_scale, scale_base_mu=scale_base_mu,
                                          scale_base_sigma=scale_base_sigma, scale_sp=1., base_fun=base_fun,
                                          grid_eps=grid_eps, grid_range=grid_range, sp_trainable=sp_trainable,
                                          sb_trainable=sb_trainable, sparse_init=sparse_init, device=device)
            else:
                sp_batch = pykan.KANLayer(in_dim=width_in[l], out_dim=width_out[l + 1], num=grid, k=k,
                                    noise_scale=noise_scale, scale_base_mu=scale_base_mu,
                                    scale_base_sigma=scale_base_sigma, scale_sp=1., base_fun=base_fun,
                                    grid_eps=grid_eps, grid_range=grid_range, sp_trainable=sp_trainable,
                                    sb_trainable=sb_trainable, sparse_init=sparse_init, device=device)
            self.act_fun.append(sp_batch)

        self.node_bias = []
        self.node_scale = []
        self.subnode_bias = []
        self.subnode_scale = []

        globals()['self.node_bias_0'] = torch.nn.Parameter(torch.zeros(3, 1)).requires_grad_(False)
        exec('self.node_bias_0' + " = torch.nn.Parameter(torch.zeros(3,1)).requires_grad_(False)")

        for l in range(self.depth):
            exec(
                f'self.node_bias_{l} = torch.nn.Parameter(torch.zeros(width_in[l+1])).requires_grad_(affine_trainable)')
            exec(
                f'self.node_scale_{l} = torch.nn.Parameter(torch.ones(width_in[l+1])).requires_grad_(affine_trainable)')
            exec(
                f'self.subnode_bias_{l} = torch.nn.Parameter(torch.zeros(width_out[l+1])).requires_grad_(affine_trainable)')
            exec(
                f'self.subnode_scale_{l} = torch.nn.Parameter(torch.ones(width_out[l+1])).requires_grad_(affine_trainable)')
            exec(f'self.node_bias.append(self.node_bias_{l})')
            exec(f'self.node_scale.append(self.node_scale_{l})')
            exec(f'self.subnode_bias.append(self.subnode_bias_{l})')
            exec(f'self.subnode_scale.append(self.subnode_scale_{l})')

        self.act_fun = nn.ModuleList(self.act_fun)
        self.layer_norm = torch.nn.Tanh()

        self.grid = grid
        self.k = k
        self.base_fun = base_fun

        ### initializing the symbolic front ###
        # self.symbolic_fun = []
        # for l in range(self.depth):
        #     sb_batch = pykan.Symbolic_KANLayer(in_dim=width_in[l], out_dim=width_out[l + 1])
        #     self.symbolic_fun.append(sb_batch)

        # self.symbolic_fun = nn.ModuleList(self.symbolic_fun)
        self.symbolic_enabled = symbolic_enabled
        self.affine_trainable = affine_trainable
        self.sp_trainable = sp_trainable
        self.sb_trainable = sb_trainable

        self.save_act = save_act

        self.node_scores = None
        self.edge_scores = None
        self.subnode_scores = None

        self.cache_data = None
        self.acts = None

        self.auto_save = auto_save
        self.state_id = 0
        self.ckpt_path = ckpt_path
        self.round = round

        self.device = device
        self.to(device)

        if auto_save:
            if first_init:
                if not os.path.exists(ckpt_path):
                    # Create the directory
                    os.makedirs(ckpt_path)
                print(f"checkpoint directory created: {ckpt_path}")
                print('saving model version 0.0')

                history_path = self.ckpt_path + '/history.txt'
                with open(history_path, 'w') as file:
                    file.write(f'### Round {self.round} ###' + '\n')
                    file.write('init => 0.0' + '\n')
                self.saveckpt(path=self.ckpt_path + '/' + '0.0')
            else:
                self.state_id = state_id

        self.input_id = torch.arange(self.width_in[0], )

    def refine(self, new_grid):
        '''
        grid refinement

        Args:
        -----
            new_grid : init
                the number of grid intervals after refinement

        Returns:
        --------
            a refined model : MatrixKAN

        Example
        -------
        >>> from kan import *                                             ################ FIX THIS ####################
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
        >>> print(model.grid)
        >>> x = torch.rand(100,2)
        >>> model.get_act(x)
        >>> model = model.refine(10)
        >>> print(model.grid)
        checkpoint directory created: ./model
        saving model version 0.0
        5
        saving model version 0.1
        10
        '''

        model_new = MatrixKAN(width=self.width,
                            grid=new_grid,
                            k=self.k,
                            mult_arity=self.mult_arity,
                            base_fun=self.base_fun_name,
                            symbolic_enabled=self.symbolic_enabled,
                            affine_trainable=self.affine_trainable,
                            grid_eps=self.grid_eps,
                            grid_range=self.grid_range,
                            sp_trainable=self.sp_trainable,
                            sb_trainable=self.sb_trainable,
                            ckpt_path=self.ckpt_path,
                            auto_save=True,
                            first_init=False,
                            state_id=self.state_id,
                            round=self.round,
                            device=self.device,
                            spline_matrix=self.spline_matrix)

        model_new.initialize_from_another_model(self, self.cache_data)
        model_new.cache_data = self.cache_data
        model_new.grid = new_grid

        self.log_history('refine')
        model_new.state_id += 1

        return model_new.to(self.device)

    def saveckpt(self, path='model'):
        '''
        save the current model to files (configuration file and state file)

        Args:
        -----
            path : str
                the path where checkpoints are saved

        Returns:
        --------
            None

        Example
        -------
        >>> from kan import *                                             ################ FIX THIS ####################
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
        >>> model.saveckpt('./mark')
        # There will be three files appearing in the current folder: mark_cache_data, mark_config.yml, mark_state
        '''

        model = self

        dic = dict(
            width=model.width,
            grid=model.grid,
            k=model.k,
            mult_arity=model.mult_arity,
            base_fun_name=model.base_fun_name,
            symbolic_enabled=model.symbolic_enabled,
            affine_trainable=model.affine_trainable,
            grid_eps=model.grid_eps,
            grid_range=model.grid_range,
            sp_trainable=model.sp_trainable,
            sb_trainable=model.sb_trainable,
            state_id=model.state_id,
            auto_save=model.auto_save,
            ckpt_path=model.ckpt_path,
            round=model.round,
            device=str(model.device),
            spline_matrix=model.spline_matrix
        )

        for i in range(model.depth):
            dic[f'symbolic.funs_name.{i}'] = model.symbolic_fun[i].funs_name

        with open(f'{path}_config.yml', 'w') as outfile:
            yaml.dump(dic, outfile, default_flow_style=False)

        torch.save(model.state_dict(), f'{path}_state')
        torch.save(model.cache_data, f'{path}_cache_data')

    @staticmethod
    def loadckpt(path='model'):
        '''
        load checkpoint from path

        Args:
        -----
            path : str
                the path where checkpoints are saved

        Returns:
        --------
            MatrixKAN

        Example
        -------
        >>> from kan import *                                             ################ FIX THIS ####################
        >>> model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
        >>> model.saveckpt('./mark')
        >>> KAN.loadckpt('./mark')
        '''
        with open(f'{path}_config.yml', 'r') as stream:
            config = yaml.safe_load(stream)

        state = torch.load(f'{path}_state')

        model_load = MatrixKAN(width=config['width'],
                             grid=config['grid'],
                             k=config['k'],
                             mult_arity=config['mult_arity'],
                             base_fun=config['base_fun_name'],
                             symbolic_enabled=config['symbolic_enabled'],
                             affine_trainable=config['affine_trainable'],
                             grid_eps=config['grid_eps'],
                             grid_range=config['grid_range'],
                             sp_trainable=config['sp_trainable'],
                             sb_trainable=config['sb_trainable'],
                             state_id=config['state_id'],
                             auto_save=config['auto_save'],
                             first_init=False,
                             ckpt_path=config['ckpt_path'],
                             round=config['round'] + 1,
                             device=config['device'],
                             spline_matrix=config['spline_matrix'])

        model_load.load_state_dict(state)
        model_load.cache_data = torch.load(f'{path}_cache_data')

        depth = len(model_load.width) - 1
        for l in range(depth):
            out_dim = model_load.symbolic_fun[l].out_dim
            in_dim = model_load.symbolic_fun[l].in_dim
            funs_name = config[f'symbolic.funs_name.{l}']
            for j in range(out_dim):
                for i in range(in_dim):
                    fun_name = funs_name[j][i]
                    model_load.symbolic_fun[l].funs_name[j][i] = fun_name
                    model_load.symbolic_fun[l].funs[j][i] = pykan.utils.SYMBOLIC_LIB[fun_name][0]
                    model_load.symbolic_fun[l].funs_sympy[j][i] = pykan.utils.SYMBOLIC_LIB[fun_name][1]
                    model_load.symbolic_fun[l].funs_avoid_singularity[j][i] = pykan.utils.SYMBOLIC_LIB[fun_name][3]
        return model_load

    def forward(self, x, singularity_avoiding=False, y_th=10.):
        '''
        forward pass

        Args:
        -----
            x : 2D torch.tensor
                inputs
            singularity_avoiding : bool
                whether to avoid singularity for the symbolic branch
            y_th : float
                the threshold for singularity

        Returns:
        --------
            None

        Example1
        --------
        >>> from kan import *                                             ################ FIX THIS ####################
        >>> model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
        >>> x = torch.rand(100,2)
        >>> model(x).shape

        Example2
        --------
        >>> from kan import *                                             ################ FIX THIS ####################
        >>> model = KAN(width=[1,1], grid=5, k=3, seed=0)
        >>> x = torch.tensor([[1],[-0.01]])
        >>> model.fix_symbolic(0,0,0,'log',fit_params_bool=False)
        >>> print(model(x))
        >>> print(model(x, singularity_avoiding=True))
        >>> print(model(x, singularity_avoiding=True, y_th=1.))
        '''
        x = x[:, self.input_id.long()]
        assert x.shape[1] == self.width_in[0]

        # cache data
        self.cache_data = x

        self.acts = []  # shape ([batch, n0], [batch, n1], ..., [batch, n_L])
        self.acts_premult = []
        self.spline_preacts = []
        self.spline_postsplines = []
        self.spline_postacts = []
        self.acts_scale = []
        self.acts_scale_spline = []
        self.subnode_actscale = []
        self.edge_actscale = []
        # self.neurons_scale = []

        self.acts.append(x)  # acts shape: (batch, width[l])

        for l in range(self.depth):

            x_norm = x
            if self.spline_matrix:
                x_norm = self.layer_norm(x)

            x_numerical, preacts, postacts_numerical, postspline = self.act_fun[l](x_norm)

            if self.symbolic_enabled == True:
                x_symbolic, postacts_symbolic = self.symbolic_fun[l](x_norm, singularity_avoiding=singularity_avoiding,
                                                                     y_th=y_th)
            else:
                x_symbolic = 0.
                postacts_symbolic = 0.

            x = x_numerical + x_symbolic

            if self.save_act:
                # save subnode_scale
                self.subnode_actscale.append(torch.std(x, dim=0).detach())

            # subnode affine transform
            x = self.subnode_scale[l][None, :] * x + self.subnode_bias[l][None, :]

            if self.save_act:
                postacts = postacts_numerical + postacts_symbolic

                # self.neurons_scale.append(torch.mean(torch.abs(x), dim=0))
                # grid_reshape = self.act_fun[l].grid.reshape(self.width_out[l + 1], self.width_in[l], -1)
                input_range = torch.std(preacts, dim=0) + 0.1
                output_range_spline = torch.std(postacts_numerical,
                                                dim=0)  # for training, only penalize the spline part
                output_range = torch.std(postacts,
                                         dim=0)  # for visualization, include the contribution from both spline + symbolic
                # save edge_scale
                self.edge_actscale.append(output_range)

                self.acts_scale.append((output_range / input_range).detach())
                self.acts_scale_spline.append(output_range_spline / input_range)
                self.spline_preacts.append(preacts.detach())
                self.spline_postacts.append(postacts.detach())
                self.spline_postsplines.append(postspline.detach())

                self.acts_premult.append(x.detach())

            # multiplication
            dim_sum = self.width[l + 1][0]
            dim_mult = self.width[l + 1][1]

            if self.mult_homo == True:
                for i in range(self.mult_arity - 1):
                    if i == 0:
                        x_mult = x[:, dim_sum::self.mult_arity] * x[:, dim_sum + 1::self.mult_arity]
                    else:
                        x_mult = x_mult * x[:, dim_sum + i + 1::self.mult_arity]

            else:
                for j in range(dim_mult):
                    acml_id = dim_sum + np.sum(self.mult_arity[l + 1][:j])
                    for i in range(self.mult_arity[l + 1][j] - 1):
                        if i == 0:
                            x_mult_j = x[:, [acml_id]] * x[:, [acml_id + 1]]
                        else:
                            x_mult_j = x_mult_j * x[:, [acml_id + i + 1]]

                    if j == 0:
                        x_mult = x_mult_j
                    else:
                        x_mult = torch.cat([x_mult, x_mult_j], dim=1)

            if self.width[l + 1][1] > 0:
                x = torch.cat([x[:, :dim_sum], x_mult], dim=1)

            # x = x + self.biases[l].weight
            # node affine transform
            x = self.node_scale[l][None, :] * x + self.node_bias[l][None, :]

            self.acts.append(x.detach())

        return x

    def fit(self, dataset, opt="LBFGS", steps=100, log=1, lamb=0., lamb_l1=1., lamb_entropy=2., lamb_coef=0.,
            lamb_coefdiff=0., update_grid=True, grid_update_num=10, loss_fn=None, lr=1., start_grid_update_step=-1,
            stop_grid_update_step=50, batch=-1,
            metrics=None, save_fig=False, in_vars=None, out_vars=None, beta=3, save_fig_freq=1, img_folder='./video',
            singularity_avoiding=False, y_th=1000., reg_metric='edge_forward_spline_n', display_metrics=None):
        '''
        training

        Args:
        -----
            dataset : dic
                contains dataset['train_input'], dataset['train_label'], dataset['test_input'], dataset['test_label']
            opt : str
                "LBFGS" or "Adam"
            steps : int
                training steps
            log : int
                logging frequency
            lamb : float
                overall penalty strength
            lamb_l1 : float
                l1 penalty strength
            lamb_entropy : float
                entropy penalty strength
            lamb_coef : float
                coefficient magnitude penalty strength
            lamb_coefdiff : float
                difference of nearby coefficits (smoothness) penalty strength
            update_grid : bool
                If True, update grid regularly before stop_grid_update_step
            grid_update_num : int
                the number of grid updates before stop_grid_update_step
            start_grid_update_step : int
                no grid updates before this training step
            stop_grid_update_step : int
                no grid updates after this training step
            loss_fn : function
                loss function
            lr : float
                learning rate
            batch : int
                batch size, if -1 then full.
            save_fig_freq : int
                save figure every (save_fig_freq) steps
            singularity_avoiding : bool
                indicate whether to avoid singularity for the symbolic part
            y_th : float
                singularity threshold (anything above the threshold is considered singular and is softened in some ways)
            reg_metric : str
                regularization metric. Choose from {'edge_forward_spline_n', 'edge_forward_spline_u', 'edge_forward_sum', 'edge_backward', 'node_backward'}
            metrics : a list of metrics (as functions)
                the metrics to be computed in training
            display_metrics : a list of functions
                the metric to be displayed in tqdm progress bar

        Returns:
        --------
            results : dic
                results['train_loss'], 1D array of training losses (RMSE)
                results['test_loss'], 1D array of test losses (RMSE)
                results['reg'], 1D array of regularization
                other metrics specified in metrics

        Example
        -------
        >>> from kan import *
        >>> model = KAN(width=[2,5,1], grid=5, k=3, noise_scale=0.3, seed=2)
        >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
        >>> dataset = create_dataset(f, n_var=2)
        >>> model.fit(dataset, opt='LBFGS', steps=20, lamb=0.001);
        >>> model.plot()
        # Most examples in toturals involve the fit() method. Please check them for useness.
        '''

        if lamb > 0. and not self.save_act:
            print('setting lamb=0. If you want to set lamb > 0, set self.save_act=True')

        old_save_act, old_symbolic_enabled = self.disable_symbolic_in_fit(lamb)

        pbar = tqdm(range(steps), desc='description', ncols=100)

        if loss_fn == None:
            loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)
        else:
            loss_fn = loss_fn_eval = loss_fn

        grid_update_freq = int(stop_grid_update_step / grid_update_num)

        if opt == "Adam":
            optimizer = torch.optim.Adam(self.get_params(), lr=lr)
        elif opt == "LBFGS":
            optimizer = pykan.LBFGS(self.get_params(), lr=lr, history_size=10, line_search_fn="strong_wolfe",
                              tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)

        results = {}
        results['train_loss'] = []
        results['test_loss'] = []
        results['reg'] = []
        if metrics != None:
            for i in range(len(metrics)):
                results[metrics[i].__name__] = []

        if batch == -1 or batch > dataset['train_input'].shape[0]:
            batch_size = dataset['train_input'].shape[0]
            batch_size_test = dataset['test_input'].shape[0]
        else:
            batch_size = batch
            batch_size_test = batch

        global train_loss, reg_

        def closure():
            global train_loss, reg_
            optimizer.zero_grad()
            pred = self.forward(dataset['train_input'][train_id], singularity_avoiding=singularity_avoiding, y_th=y_th)
            train_loss = loss_fn(pred, dataset['train_label'][train_id])
            if self.save_act:
                if reg_metric == 'edge_backward':
                    self.attribute()
                if reg_metric == 'node_backward':
                    self.node_attribute()
                reg_ = self.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
            else:
                reg_ = torch.tensor(0.)
            objective = train_loss + lamb * reg_
            objective.backward()
            return objective

        if save_fig:
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)

        for _ in pbar:

            if _ == steps - 1 and old_save_act:
                self.save_act = True

            train_id = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
            test_id = np.random.choice(dataset['test_input'].shape[0], batch_size_test, replace=False)

            if _ % grid_update_freq == 0 and _ < stop_grid_update_step and update_grid and _ >= start_grid_update_step:
                self.update_grid(dataset['train_input'][train_id])                                                        ########### REVISE REFERENCE IF NECESSARY #############

            if opt == "LBFGS":
                optimizer.step(closure)

            if opt == "Adam":
                pred = self.forward(dataset['train_input'][train_id], singularity_avoiding=singularity_avoiding,
                                    y_th=y_th)
                train_loss = loss_fn(pred, dataset['train_label'][train_id])
                if self.save_act:
                    if reg_metric == 'edge_backward':
                        self.attribute()
                    if reg_metric == 'node_backward':
                        self.node_attribute()
                    reg_ = self.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
                else:
                    reg_ = torch.tensor(0.)
                loss = train_loss + lamb * reg_
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            test_loss = loss_fn_eval(self.forward(dataset['test_input'][test_id]), dataset['test_label'][test_id])

            if metrics is not None:
                for i in range(len(metrics)):
                    results[metrics[i].__name__].append(metrics[i]().item())

            results['train_loss'].append(torch.sqrt(train_loss).cpu().detach().numpy())
            results['test_loss'].append(torch.sqrt(test_loss).cpu().detach().numpy())
            results['reg'].append(reg_.cpu().detach().numpy())

            if _ % log == 0:
                if display_metrics is None:
                    pbar.set_description("| train_loss: %.2e | test_loss: %.2e | reg: %.2e | " % (
                    torch.sqrt(train_loss).cpu().detach().numpy(), torch.sqrt(test_loss).cpu().detach().numpy(),
                    reg_.cpu().detach().numpy()))
                else:
                    string = ''
                    data = ()
                    for metric in display_metrics:
                        string += f' {metric}: %.2e |'
                        try:
                            results[metric]
                        except:
                            raise Exception(f'{metric} not recognized')
                        data += (results[metric][-1],)
                    pbar.set_description(string % data)

            if save_fig and _ % save_fig_freq == 0:
                self.plot(folder=img_folder, in_vars=in_vars, out_vars=out_vars, title="Step {}".format(_), beta=beta)
                plt.savefig(img_folder + '/' + str(_) + '.jpg', bbox_inches='tight', dpi=200)
                plt.close()

        self.log_history('fit')
        # revert back to original state
        self.symbolic_enabled = old_symbolic_enabled
        return results

KAN = MatrixKAN

class MatrixKANLayer(pykan.KANLayer):
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

    def __init__(self, in_dim=3, out_dim=2, num=5, k=3, noise_scale=0.5, scale_base_mu=0.0, scale_base_sigma=1.0,
                 scale_sp=1.0, base_fun=torch.nn.SiLU(), grid_eps=0.02, grid_range=[-1, 1], sp_trainable=True,
                 sb_trainable=True, save_plot_data=True, device='cpu', sparse_init=False):
        """
        Initialize a MatrixKANLayer

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
        >>> from kan.MatrixKANLayer import *                                     ############## FIX THIS ##############
        >>> model = MatrixKANLayer(in_dim=3, out_dim=5)
        >>> (model.in_dim, model.out_dim)
        """
        super(MatrixKANLayer, self).__init__(in_dim=in_dim, out_dim=out_dim, num=num, k=k, noise_scale=noise_scale, scale_base_mu=scale_base_mu, scale_base_sigma=scale_base_sigma,
                 scale_sp=scale_sp, base_fun=base_fun, grid_eps=grid_eps, grid_range=grid_range, sp_trainable=sp_trainable,
                 sb_trainable=sb_trainable, save_plot_data=save_plot_data, device=device, sparse_init=sparse_init)

        # size
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k

        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1)[None, :].expand(self.in_dim, num + 1)
        grid = extend_grid(grid, k_extend=k)
        self.grid = torch.nn.Parameter(grid).requires_grad_(False)
        noises = (torch.rand(self.num + 1, self.in_dim, self.out_dim) - 1 / 2) * noise_scale / num

        self.coef = torch.nn.Parameter(curve2coef(self.grid[:, k:-k].permute(1, 0), noises, self.grid, k))

        if sparse_init:
            self.mask = torch.nn.Parameter(sparse_mask(in_dim, out_dim)).requires_grad_(False)
        else:
            self.mask = torch.nn.Parameter(torch.ones(in_dim, out_dim)).requires_grad_(False)

        self.scale_base = torch.nn.Parameter(scale_base_mu * 1 / np.sqrt(in_dim) + \
                                             scale_base_sigma * (torch.rand(in_dim, out_dim) * 2 - 1) * 1 / np.sqrt(
            in_dim)).requires_grad_(sb_trainable)
        self.scale_sp = torch.nn.Parameter(torch.ones(in_dim, out_dim) * scale_sp * self.mask).requires_grad_(
            sp_trainable)  # make scale trainable
        self.base_fun = base_fun
        self.grid_range = grid_range
        self.grid_interval = (grid_range[1] - grid_range[0]) / num
        self.device = device

        # Initialize Basis Matrix
        self.basis_matrix = self.basis_matrix()

        self.grid_eps = grid_eps

        self.to(device)

    def to(self, device):
        super(MatrixKANLayer, self).to(device)
        self.device = device
        return self

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

        Example
        -------
        >>> from kan.MatrixKANLayer import *                                     ############## FIX THIS ##############
        >>> model = MatrixKANLayer(in_dim=3, out_dim=5)
        >>> x = torch.normal(0,1,size=(100,3))
        >>> y, preacts, postacts, postspline = model(x)
        >>> y.shape, preacts.shape, postacts.shape, postspline.shape
        """

        batch = x.shape[0]
        preacts = x[:, None, :].clone().expand(batch, self.out_dim, self.in_dim)

        base = self.base_fun(x)  # (batch, in_dim)
        y = self.b_spline_matrix(x)

        postspline = y.clone().permute(0, 2, 1)

        y = self.scale_base[None, :, :] * base[:, :, None] + self.scale_sp[None, :, :] * y
        y = self.mask[None, :, :] * y

        postacts = y.clone().permute(0, 2, 1)

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

        return basis_matrix.to(dtype=torch.float64)                                   ##### MIGHT BE ABLE TO REMOVE THE DATATYPE CASTING ##########

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
        x_interval_floor = torch.floor((x - self.grid_range[0]) / self.grid_interval)
        x_interval_floor = ((x_interval_floor * self.grid_interval) + self.grid_range[0])

        # Determine applicable grid interval values (upper-bound)
        x_interval_ceiling = x_interval_floor + self.grid_interval

        # Calculate grid indices of interval floor
        if self.grid.shape[-1] != x_interval_floor.shape[-1]:
            x_interval_floor = x_interval_floor.unsqueeze(-1)
        grid_interval_floor = torch.isclose(self.grid, x_interval_floor)

        # Calculate index position of the lower knot in the applicable knot span.
        # This is later used to calculate the applicable control points / basis functions.
        grid_interval_floor_indices = torch.nonzero(grid_interval_floor, as_tuple=True)
        grid_interval_floor_indices = grid_interval_floor_indices[-1]
        grid_interval_floor_indices = grid_interval_floor_indices.reshape(x.shape)

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

    def b_spline_matrix(self, x: torch.Tensor):
        """
        Compute spline output for the given input tensor.

        :Args:
            x (torch.Tensor):   Input tensor of shape (batch_size, sequence length, in_features).

        Returns:
            torch.Tensor:       Power bases tensor of shape (batch_size, sequence length, out_features).
        """

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

        Example
        -------
        >>> model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(model.grid.data)
        >>> x = torch.linspace(-3,3,steps=100)[:,None]
        >>> model.update_grid_from_samples(x)
        >>> print(model.grid.data)
        """

        batch = x.shape[0]
        x_pos = torch.sort(x, dim=0)[0]
        y_eval = self.b_spline_matrix(x_pos)
        num_interval = self.grid.shape[1] - 1 - 2 * self.k

        def get_grid(num_interval):
            ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
            grid_adaptive = x_pos[ids, :].permute(1, 0)
            h = (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]]) / num_interval
            grid_uniform = grid_adaptive[:, [0]] + h * torch.arange(num_interval + 1, )[None, :].to(x.device)
            grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
            return grid

        grid = get_grid(num_interval)

        if mode == 'grid':
            sample_grid = get_grid(2 * num_interval)
            x_pos = sample_grid.permute(1, 0)
            y_eval = self.b_spline_matrix(x_pos)

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
        y_eval = self.b_spline_matrix(x_pos)
        num_interval = self.grid.shape[1] - 1 - 2 * self.k

        def get_grid(num_interval):
            ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
            grid_adaptive = x_pos[ids, :].permute(1, 0)
            h = (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]]) / num_interval
            grid_uniform = grid_adaptive[:, [0]] + h * torch.arange(num_interval + 1, )[None, :].to(x.device)
            grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
            return grid

        grid = get_grid(num_interval)

        if mode == 'grid':
            sample_grid = get_grid(2 * num_interval)
            x_pos = sample_grid.permute(1, 0)
            y_eval = self.b_spline_matrix(x_pos)

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
        spb.coef.data = self.coef[in_id][:, out_id]
        spb.scale_base.data = self.scale_base[in_id][:, out_id]
        spb.scale_sp.data = self.scale_sp[in_id][:, out_id]
        spb.mask.data = self.mask[in_id][:, out_id]

        spb.in_dim = len(in_id)
        spb.out_dim = len(out_id)
        return spb

