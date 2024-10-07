from kan import *
from MatrixKAN import *
from kan.utils import create_dataset
from kan.utils import ex_round
from feynman import *
import random
import json

torch.set_default_dtype(torch.float64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print(device)

FUNCTIONS = [2]                              ### NOTE: Numbers correspond to values assigned to functions in feynman.py
SPLINE_DEGREES = [2] #[3, 5, 7, 9]
KAN_DEPTHS = [2] #[2, 3, 4, 5, 6]
GRID_SIZES = [30] #[2, 5, 10, 20, 50, 100, 200]
LAMBDAS = [0.005] #, [0.01, 0.005, 0.001, 0.0005, 0.0001]
STEPS = 50 #200

results_all = {}

for func in FUNCTIONS:
    # create dataset
    symbol, expr, f, ranges = get_feynman_dataset(func)
    dataset = create_dataset(f, n_var=len(symbol), device=device)

    results_func = []
    optimal_result = {
        "degree": None,
        "width": None,
        "lambda": None,
        "seed": None,
        "train_loss_hist": None,
        "test_loss_hist": None,
        "test_loss_best": None
    }
    for k in SPLINE_DEGREES:
        for depth in KAN_DEPTHS:
            for l in LAMBDAS:
                optimal_result_candidate = {
                    "degree"            : None,
                    "width"             : None,
                    "lambda"            : None,
                    "seed"              : None,
                    "train_loss_hist"   : None,
                    "test_loss_hist"    : None,
                    "test_loss_best"    : None
                }
                for i in range(3): # Tests same parameters with 3 different random seeds.
                    train_losses = []
                    test_losses = []
                    seed = random.randint(0,1000000)
                    for index, g in enumerate(GRID_SIZES):
                        if index == 0:
                            width_inner = [5 for n in range(depth-1)]
                            width = [len(symbol)] + width_inner + [1]

                            model = MatrixKAN(width=list(width), grid=g, k=k, seed=seed, device=device, grid_eps=1)
                            # model = MultKAN(width=list(width), grid=g, k=k, seed=seed, device=device, grid_eps=1)
                        else:
                            model = model.prune()
                            model = model.refine(g)
                        results = model.fit(dataset, opt="LBFGS", steps=STEPS, update_grid=True, lamb=l)
                        train_losses += [results["train_loss"][0].item()]
                        test_losses += [results["test_loss"][0].item()]
                        if (optimal_result_candidate["test_loss_best"] is None) or (results["test_loss"][0].item() <= optimal_result_candidate["test_loss_best"]):
                            optimal_result_candidate["degree"] = k
                            optimal_result_candidate["width"] = width
                            optimal_result_candidate["lambda"] = l
                            optimal_result_candidate["seed"] = seed
                            optimal_result_candidate["test_loss_best"] = results["test_loss"][0].item()
                    if i == 0 or optimal_result_candidate["seed"] == seed:
                        optimal_result_candidate["train_loss_hist"] = train_losses
                        optimal_result_candidate["test_loss_hist"] = test_losses
                results_func.append(optimal_result_candidate)
                if (optimal_result["test_loss_best"] is None) or (optimal_result_candidate["test_loss_best"] <= optimal_result["test_loss_best"]):
                    optimal_result["degree"] = optimal_result_candidate["degree"]
                    optimal_result["width"] = optimal_result_candidate["width"]
                    optimal_result["lambda"] = optimal_result_candidate["lambda"]
                    optimal_result["seed"] = optimal_result_candidate["seed"]
                    optimal_result["test_loss_best"] = optimal_result_candidate["test_loss_best"]
                    optimal_result["train_loss_hist"] = optimal_result_candidate["train_loss_hist"]
                    optimal_result["test_loss_hist"] = optimal_result_candidate["test_loss_hist"]
    results_all[func] = {
        "results"           : results_func,
        "optimal_result"    : optimal_result
    }

with open('test_results.json', 'w') as outfile:
    json.dump(results_all, outfile)