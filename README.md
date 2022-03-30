# Partitioned Active Learning
This is the code for implementation of Partitioned Active Learning (from [Lee et al. (2021)](https://arxiv.org/abs/2105.08547)).

The package contains the following scripts:
- `active_learning.py`: active learning class that initiates with a model, a strategy.
- `doe.py`: contains some functions for initial dataset genearation such as Latin Hypercube Design, factor two approximation.
- `partitioned_gp.py`: Partitioned GP class that is built upon `scikit-learn`.
- `sim_funcs.py`: contains some simulation functions.
- `strategy_lib.py`: a library of active learning strategies can be plugged into the active learning module.

The code requires `scikit-learn > 1.0`.

A simple implementation example is as follows.
```
my_al = ActiveLearningMachine(<model>, <strategy>, <testing_dataset>)
my_al.init_fit(<X_init>, <y_init>)
x_new = my_al.query(<X_pool>)
# get label of x_new -> y_new
pal.update(x_new, y_new)
```
The active learning module automatically stores the learning progress (if `<testing_dataset>` exists) and the time for querying.
