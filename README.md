# Fortified Test Functions

The [run_bumpy_branin.py](run_bumpy_branin.py) Python file reproduces the results presented in `Weaponizing Favorite Test Functions for Testing Global Optimization Algorithms: An illustration with the Branin-HooFunction` as seen in AIAA Aviation 2020. 

This work demonstrates that optimization test functions can be made more challenging by adding small radial basis functions to the local optima. This is illustrated here for the Branin-Hoo function, which has three local optima which are also global optima.

The optimization parameters are controlled by these variables at the top of the script.

```python
n_runs = 1000  # number of optimization runs
pop_size = 10  # DE population size
max_iter = 20  # number of DE iterations
use_bfgs = False  # whether to use BFGS after DE
```

# Requirements

Python with numpy and scipy. If you are new to Python, we suggested grabbing the latest [Anaconda](https://www.anaconda.com/products/individual) installation. 

```
scipy > 1.0.0
numpy > 0.14.0
```
