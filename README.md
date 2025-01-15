# Time-Varying Convex Optimization: A Contraction and Equilibrium Tracking Approach
This repository contains supporting material for our paper ***Time-Varying Convex Optimization: A Contraction and Equilibrium Tracking Approach***, available on arxiv at https://arxiv.org/abs/2305.15595.

We provide Python code to reproduce the experiments in Section V. We have Jupyter notebooks for the time-varying [equality-constrained](https://github.com/davydovalexander/time-varying-convex/blob/main/Equality-constrained.ipynb) minimization and the time-varying [inequality-constrained](https://github.com/davydovalexander/time-varying-convex/blob/main/Inequality-constrained.ipynb) minimization problems. These notebooks reproduce the figures in Section V.A. and Section V.B., respectively. 

We additionally provide Python code for experiments in the [Robotarium](https://www.robotarium.gatech.edu/) which supplement our experimental results in Section V.C. The Python simulator for Robotarium is available at https://github.com/robotarium/robotarium_python_simulator. The videos supplementing these experiments are available [here](https://bit.ly/TimeVaryingConvex).

The following dependencies are required for the code:
- NumPy (http://www.numpy.org)
- matplotlib (http://matplotlib.org/index.html)
- CVXPY (https://www.cvxpy.org)
- SciPy (https://scipy.org)
- Robotarium.

If you would like to cite our paper or use our code in your paper, please cite the following arXiv preprint:
```
@misc{davydov2024timevarying,
title={Time-Varying Convex Optimization: A Contraction and Equilibrium Tracking Approach}, 
author={Alexander Davydov and Veronica Centorrino and Anand Gokhale and Giovanni Russo and Francesco Bullo},
year={2024},
eprint={2305.15595},
archivePrefix={arXiv},
primaryClass={math.OC},
url={https://arxiv.org/abs/2305.15595}, 
}
```
