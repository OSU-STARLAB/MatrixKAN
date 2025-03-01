# MatrixKAN
Kolmogorov-Arnold Networks (KAN) are a new class of neural network architecture representing a promising alternative to the Multilayer Perceptron (MLP), demonstrating improved expressiveness and interpretability.  However, KANs suffer from slow training and inference speeds relative to MLPs due in part to the recursive nature of the underlying B-spline calculations.  This issue is particularly apparent with respect to KANs utilizing high-degree B-splines, as the number of required non-parallelizable recursions is proportional to B-spline degree.

We solve this issue by proposing MatrixKAN, a novel optimization that parallelizes B-spline calculations with matrix representation and operations, thus significantly improving effective computation time for models utilizing high-degree B-splines.  MatrixKAN is a modified version of [pykan](https://github.com/KindXiaoming/pykan/) and works with most existing KAN features. In our experiments, MatrixKAN demonstrate speedups of approximately 40x relative to KAN, with significant additional speedup potential for larger datasets or higher spline degrees.

## Installation
MatrixKAN can be installed directly from GitHub. 

**Pre-requisites:**

```
Python 3.11 or higher
pip
```

**GitHub Installation**

```
git clone https://github.com/OSU-STARLAB/MatrixKAN.git
cd MatrixKAN
pip install -e .
```

Requirements

```python
# python==3.11
colorama==0.4.6
filelock==3.13.1
fsspec==2024.2.0
iniconfig==2.0.0
Jinja2==3.1.3
MarkupSafe==2.1.5
mpmath==1.3.0
networkx==3.2.1
packaging==24.1
pluggy==1.5.0
pykan==0.2.6
pytest==8.3.3
sympy==1.12
torch==2.4.0+cu124
typing_extensions==4.9.0
```

Requirements are located in Requirements.txt and can be installed as follows:
```python
pip install -r Requirements.txt
```

## KNOWN ISSUES

prune() - Pruning of MatrixKAN networks sometimes results in NaN parameter values.

## CHANGELOG

All notable changes to this project will be documented in [Changelog](COMPLETE ME).

<ins>Current Version</ins>: 1.1.0

## Citation
```python
@misc{coffman2025matrixkanparallelizedkolmogorovarnoldnetwork,
      title={MatrixKAN: Parallelized Kolmogorov-Arnold Network}, 
      author={Cale Coffman and Lizhong Chen},
      year={2025},
      eprint={2502.07176},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.07176}, 
}
```

## Credits

We would like to thank the creators of [pykan](https://github.com/KindXiaoming/pykan/) for developing the KAN architecture that is optimized here.
