import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi
from sklearn.cluster import MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC

class agglomerative_cluster:
    def __init__(self, n_cluster=2, linkage='ward', dist_metric='euclidean'):
        self.n_cluster = n_cluster
        self.linkage = linkage
        self.dist_metric = dist_metric
    
    def dissimilarity(self, c1, c2):
        X1, y1 = np.split(c1, [-1], axis=1)
        X2, y2 = np.split(c2, [-1], axis=1)
        X_cdist = cdist(X1, X2, metric=self.dist_metric)
        
        if self.linakge == 'ward':
            n1 = len(X1)
            n2 = len(X2)
            return n1 * n2 / (n1 + n2) * np.square(y1.mean() - y2.mean()) / X_cdist.mean()
        
        else:
            y_cdist = cdist(y1, y2, metric=self.dist_metric)
            D = y_cdist / X_cdist

            if self.self.linakge == 'single':
                return D.min()

            elif self.linakge == 'complete':
                return D.max()

            elif self.linakge == 'average':
                return D.mean()
                
            else:
                raise ValueError('Unknown dissimilarity criterion')
    
    def initialize_(self, X, y):
        self.X = X.copy()
        self.y = y.copy()

        # generate the Voronoi tessellation (add_point is allowed)
        self.vor = Voronoi(self.X, incremental=True)

        # note that pairs are undirected sets of clusters
        # the orders of A and D must be preserved
        self.adjacent_list = [set(pairs) for pairs in self.vor.ridge_points]
        self.dissimilarity_array = np.zeros(len(self.adjacent_list))

        # concatenate input and output
        # initial labels are generated in the order of data
        self.Z = np.column_stack((self.X, self.y))
        self.labels = np.arange(len(self.Z))
    
    def fit(self, X, y):
        # initialize clustering
        self.initialize_(X, y)

        # agglomerative clustering
        for i in range(len(self.X) - self.n_cluster):
            # calculate the dissimilarity between each pair of adjacent clusters
            for j, (p, q) in enumerate(self.adjacent_list):
                # for the initial iteration, calculate the dissimilarity for every pair
                if i == 0:
                    self.dissimilarity_array[j] = self.dissimilarity(self.Z[self.labels==p], self.Z[self.labels==q])
                # after the initial iteration, only update the dissimilarities of r_min
                else:
                    if p != r_min and q != r_min:
                        continue
                    else:
                        self.dissimilarity_array[j] = self.dissimilarity(self.Z[self.labels==p], self.Z[self.labels==q])
            
            min_id = np.argmin(self.dissimilarity_array)            # find the minimum dissimilarity
            r_min, r_max = sorted(self.adjacent_list[min_id])       # get the two clusters
            self.labels[self.labels==r_max] = r_min                 # update labels (r_min -> r_max)

            # merge the two clusters
            del self.adjacent_list[min_id]
            self.dissimilarity_array = np.delete(self.dissimilarity_array, min_id)

            for i, pair in enumerate(self.adjacent_list):
                # if the pair is associated with r_max, update with r_min
                if r_max in pair:
                    new_pair = pair.copy()
                    new_pair.remove(r_max)
                    new_pair.add(r_min)
                    # if the merged pair is already in the adjacency list, make it a dummy pair
                    # otherwise, add the merged pair to the adjacency list
                    if new_pair in self.adjacent_list:
                        self.adjacent_list[i] = -1
                        self.dissimilarity_array[i] = -1
                    else:
                        self.adjacent_list[i] = new_pair
                else:
                    pass
            self.adjacent_list = [pair for pair in self.adjacent_list if pair != -1]
            self.dissimilarity_array = np.delete(self.dissimilarity_array, np.where(self.dissimilarity_array == -1))
        
        # renumber the labels
        for i, label in enumerate(np.unique(self.labels)):
            self.labels[self.labels==label] = i

    def predict(self, X):
        return self.labels[np.argmin(cdist(X, self.X, metric=self.dist_metric), axis=1)]

class pal_region_classifier:
    def __init__(self, n_partitions=2, seed=None, n_iter=300):
        self.n_partitions = n_partitions
        self.seed = seed
        self.n_iter = n_iter

    def fit(self, X, y, method='meanshift', criterion='average', bandwidth_bounds=(1e-5, 1e3), gamma=1.1):
        self.X = X.copy()
        self.y = y.copy()
        self.gamma = gamma

        self.grads = self.crude_grad(self.X, self.y, criterion=criterion)

        if method == 'meanshift':
            if self.n_partitions == None:
                cluster = MeanShift()
                self.labels = cluster.fit_predict(np.column_stack((self.X, self.grads)))
            else:
                cluster = self.jumping_grid_search(np.column_stack((self.X, self.grads)), bandwidth_bounds)
                self.labels = cluster.predict(np.column_stack((self.X, self.grads)))

        elif method == 'mixture':
            cluster = GaussianMixture(self.n_partitions)
            self.labels = cluster.fit_predict(np.column_stack((self.X, self.grads)))        

        # wrap up with SVC
        self.clf = SVC(C=1e1, random_state=self.seed, probability=True)
        self.clf.fit(self.X, self.labels)
    
    def jumping_grid_search(self, data, bandwidth_bounds):
        h_trj = np.array(bandwidth_bounds)
        M_trj = np.zeros_like(h_trj)

        for i, h in enumerate(h_trj):
            ms = MeanShift(bandwidth=h)
            labels = ms.fit_predict(data)
            M_trj[i] = len(np.unique(labels))
            if M_trj[i] == self.n_partitions:
                return ms
            else:
                pass
        
        for i in range(self.n_iter):
            new_ind = np.where(M_trj < self.n_partitions)[0][0]
            h_new = (h_trj[new_ind] + h_trj[new_ind-1]) / 2
            h_trj = np.insert(h_trj, new_ind, h_new)
            ms = MeanShift(bandwidth=h_new)
            labels = ms.fit_predict(data)
            M_trj = np.insert(M_trj, new_ind, len(np.unique(labels)))
            if M_trj[new_ind] == self.n_partitions:
                ms_old = ms
                # last iteration for balancing
                h_lb = h_trj[new_ind]
                h_ub = h_trj[new_ind+1]
                for j in range(self.n_iter):
                    h_new = np.random.uniform(h_lb, h_ub)
                    ms_new = MeanShift(bandwidth=h_new)
                    labels = ms_new.fit_predict(data)
                    if len(np.unique(labels)) == self.n_partitions:
                        return ms_new
                    else:
                        return ms_old
            else:
                pass
            if i == self.n_iter - 1:
                raise ValueError('No suitable bandwidth found')
            else:
                pass
    
    def crude_grad(self, X, y, criterion='average'):
        if np.ndim(X) == 1:
            X = X[np.newaxis, :]
    
        # define radius for closed ball with the maximum pairwise distance
        Xdist = cdist(X, X)
        np.fill_diagonal(Xdist, np.inf)
        
        row = np.argmin(Xdist, axis=0)
        col = np.argmax(np.min(Xdist, axis=0))
        r_max = self.gamma * Xdist[col, row[col]]

        grads = np.empty(len(X))
        
        for i in range(len(X)):
            Xdist_i = Xdist[i, np.where(Xdist[i] <= r_max)]
            ydist_i = abs(y[np.where(Xdist[i] <= r_max)] - y[i])
            
            if criterion == 'average':
                grads[i] = (ydist_i / Xdist_i).mean()
            elif criterion == 'max':
                grads[i] = (ydist_i / Xdist_i).max()
            elif criterion == 'median':
                grads[i] = np.median(ydist_i / Xdist_i)
        return grads
    
    def predict(self, X):
        return self.clf.predict(X)
    

class PGP:
    def __init__(self, kernel, kde_method='meanshift', n_partitions=None, n_restarts=10, gamma=1.1, seed=None, n_iter=300):
        self.kernel = kernel
        self.kde_method = kde_method
        self.n_partitions = n_partitions
        self.n_restarts = n_restarts
        self.gamma = gamma
        self.seed = seed
        self.n_iter = n_iter

    def fit(self, X, y):
        # fit the region classifier
        if hasattr(self, 'region_classifier'):
            pass
        else:
            self.region_classifier = pal_region_classifier(n_partitions=self.n_partitions, seed=self.seed, n_iter=self.n_iter)
            self.region_classifier.fit(X, y, method=self.kde_method, criterion='average', bandwidth_bounds=(1e-5, 1e3), gamma=self.gamma)

        # self.classes = np.unique(self.region_classifier.labels)
        self.local_gp = {}
        self.X_local = {}
        self.y_local = {}
        for m in range(self.n_partitions): 
            gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=self.n_restarts)
            self.local_gp[m] = gp

        m_train = self.region_classifier.predict(X)
        # self.unknown_classes = self.classes.copy()
        for m in range(self.n_partitions):
            # if m not in m_train:
            #     pass
            # else:
            self.X_local[m] = X[m_train == m]
            self.y_local[m] = y[m_train == m]
            self.local_gp[m].fit(self.X_local[m], self.y_local[m])
                # self.unknown_classes = np.delete(self.unknown_classes, np.where(self.unknown_classes==c))

    def predict(self, X_new, return_std=False):
        m_new = self.region_classifier.predict(X_new)
        mu = np.zeros(len(X_new))
        std = np.zeros(len(X_new))
        for m in np.unique(m_new):
            # if c in self.unknown_classes:
            #     pass
            # else:
            mu[m_new==m], std[m_new==m] = self.local_gp[m].predict(X_new[m_new==m], return_std=True)
        if return_std:
            return mu, std
        else:
            return mu
