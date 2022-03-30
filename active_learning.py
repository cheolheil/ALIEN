import numpy as np
import time
import strategy_lib

from sklearn.metrics import mean_squared_error


class ActiveLearningMachine:
    def __init__(self, model, strategy, budget, test_dataset=None, eval_metric=mean_squared_error):
        self.model = model
        self.strategy = strategy
        self.budget = budget
        if test_dataset != None:
            self.Xtest, self.ytest = test_dataset
            self.prog = []
        else:
            pass
        self.eval_metric = eval_metric
        self.time = 0

    def init_fit(self, X, y):
        self.model.fit(X, y)
        self.Xtrain = X.copy()
        self.ytrain = y.copy()

        if hasattr(self, 'prog'):
            self.prog.append(self.eval_metric(self.model.predict(self.Xtest), self.ytest))
        else:
            pass

    def query(self, X_cand, return_index=False, *args):
        tic = time.time()
        info_vals = self.strategy(X_cand, self.model, *args)
        max_idx = np.argmax(info_vals)
        tac = time.time()
        self.time += tac - tic
        if return_index:
            return max_idx
        else:
            return np.atleast_2d(X_cand[max_idx])

    def update(self, X, y):
        if len(self.prog) == 0:
            raise Exception('You must call init_fit before update')
        else:
            self.Xtrain = np.vstack((self.Xtrain, X))
            self.ytrain = np.hstack((self.ytrain, y))
            self.model.fit(self.Xtrain, self.ytrain)
            if hasattr(self, 'prog'):
                self.prog.append(self.eval_metric(self.model.predict(self.Xtest), self.ytest))
            else:
                pass