import numpy as np

from sklearn.preprocessing import StandardScaler as SS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score as bas

from sklearn.utils import check_X_y

import tensorflow_probability as tfp
from scipy.stats import multivariate_normal, norm
from scipy.linalg import polar

from joblib import Parallel, delayed

def sample_continuous(a, p=0.1):
    if p ==1 or p == len(a):
        return a

    if isinstance(p, int):
        n = p
    else:
        n = int(len(a) * p)
        
    start_index=np.random.randint(0, len(a) - n)
    
    return a[start_index: start_index + n]

def even_sample_inds(y, p=0.8, continuous=False):
    unique, counts = np.unique(y, return_counts=True)
    by_unique = []
    min_ = min(counts)
    n_to_sample = int(np.ceil(p*min_))
    
    for c in unique:
        if continuous:
            inds = sample_continuous(np.where(y == c)[0], p=n_to_sample)
        else:
            inds = np.random.choice(np.where(y == c)[0], size=n_to_sample, replace=False)
        by_unique.append(inds)
                
    return np.concatenate(by_unique).astype(int)


def get_stratified_train_test_inds(y, p_train=0.8):
    unique, counts = np.unique(y, return_counts=True)
    
    n_samples_by_class = [int(np.max([1, np.math.floor(p_train * c)])) for c in counts]
    
    train = []
    
    for i, c in enumerate(unique):
        train.append(np.random.choice(np.where(y == c)[0], size=n_samples_by_class[i], replace=False))
        
    train = np.concatenate(train)
    test = [i for i in range(len(y)) if i not in train]
    
    return train, test


class TranslateStandardScale:
    def __init__(self):
        pass

    def fit(self, X, y):
        means_ = np.array([np.mean(X[np.where(y == c)[0]], axis=0) for c in np.unique(y)])

        self.mid_point = np.mean(means_, axis=0)
        self.scale = np.linalg.norm(means_[0] - self.mid_point)
        
        return self

    def transform(self, X):
        return (X - self.mid_point) / self.scale 

    def fit_transform(self, X, y):
        self.fit(X,y)
        return self.transform(X)


class FLD:
    def __init__(self, priors=None, zero_intersection=True, cov=None):
        self.is_fitted=False
        self.zero_intersection = zero_intersection

        self.priors_=priors
        self.cov_ = cov

    def fit(self, X, y):
        X, y = check_X_y(X,y)

        d = X.shape[1]
        
        self.classes_, counts = np.unique(y, return_counts=True)
        if len(self.classes_) != 2:
            raise ValueError("2 class only.")

        y0 = self.classes_[0]
        y1 = self.classes_[1]

        X_0 = X[y==y0]
        X_1 = X[y==y1]

        # estimate class means
        m_0 = np.mean(X_0, axis=0)
        m_1 = np.mean(X_1, axis=0)

        self.means_ = np.array([m_0, m_1])

        # estimate class covariance matrices
        if self.cov_ is None:
            n_0, n_1 = len(X_0), len(X_1)
            cov_0 = np.cov(X_0, rowvar=False)
            cov_1 = np.cov(X_1, rowvar=False)
            self.cov_ = ((n_0 - 1) * cov_0 + (n_1 - 1) * cov_1) / (n_0 + n_1 - 2) # pooled variance
    
        # estimate class priors

        if self.priors_ is None:
            self.priors_ = len(y[y==y1])/len(y)

        # estimate projection and threshold terms
        self.projection_ = np.linalg.pinv(2 * self.cov_) @ (m_1 - m_0)
        self.projection_ /= np.linalg.norm(self.projection_)
        
        c_0 = 2 * np.log((1-self.priors_)/self.priors_)
        nu_hat = (m_1 - m_0)/2

        self.fitted=True

        return self
    

    def predict(self, X):
        if self.zero_intersection:
            mid_point = 0.5 * (self.means_[0] + self.means_[1])
            X -= mid_point

        projected = X @ self.projection_
        
        return self.classes_[(projected > self.threshold).astype(int)]


def estimate_mu(dir_vectors):
    mu_hat = np.mean(dir_vectors, axis=0)
    mu_hat /= np.linalg.norm(mu_hat)
    
    return mu_hat


def estimate_kappa(dir_vectors):
    J = len(dir_vectors)
    d = len(dir_vectors[0])

    R_bar = np.linalg.norm(np.mean(dir_vectors, axis=0))
    
    num = R_bar * (d - R_bar**2)
    den = 1 - R_bar**2
    
    return num / den


def expected_risk(w_alpha, Sigma_t, nu):
    "Compute the expected accuracy of the combined hypothesis analytically"
    w_alpha = np.expand_dims(w_alpha, -1)
    nom = - w_alpha.T @ nu
    denom = np.sqrt(w_alpha.T @ Sigma_t @ w_alpha)
    risk = norm.cdf(nom / denom)
    return risk


def get_optimal_alpha(n, target_class_1_mean, source_projections, 
    alpha_grid_size=0.01, n_samples_to_estimate_risk=300, 
    cov=None, variance_from_target=None, variance_from_combined=None,
    return_risk=False, n_jobs=1
    ):

    alpha_grid = np.arange(0, 1+alpha_grid_size, alpha_grid_size)
    
    expected_risks = np.zeros(len(alpha_grid))

    for i, alpha in enumerate(alpha_grid):
        expected_risks[i] = compute_analytical_risk(n, target_class_1_mean, source_projections, alpha, n_samples_to_estimate_risk, cov, variance_from_target, variance_from_combined, n_jobs)

    optimal_alpha = alpha_grid[np.argmin(expected_risks)]
    minimum_risk = expected_risks[np.argmin(expected_risks)]

    if return_risk:
        return optimal_alpha, minimum_risk
    else:
        return optimal_alpha
    

def compute_analytical_risk(n, target_class_1_mean, source_projections, alpha, n_samples_to_estimate_risk=300, cov=None, variance_from_target=None, variance_from_combined=None, n_jobs=1):
    J = len(source_projections)
    d = len(source_projections[0])

    target_projection = np.linalg.pinv(cov) @ target_class_1_mean
    target_projection /= np.linalg.norm(target_projection)
        
    combined_projection = estimate_mu(source_projections)
    
    mu_omega = alpha * target_projection + (1-alpha) * combined_projection

    pi=0.5
        
    if variance_from_target is None:
        variance_from_target = get_target_variance_nonidentity(n, target_class_1_mean, cov)

    if variance_from_combined is None:
        variance_from_combined = get_average_source_std_error(source_projections)

    sigma_omega = alpha**2 * variance_from_target + (1-alpha)**2 * variance_from_combined
    c = 2 * (alpha + (1-alpha) * J) * np.log((1-pi)/pi)

    func = lambda x: expected_risk(x, cov, target_class_1_mean)
    risks = np.array(Parallel(n_jobs=n_jobs)(delayed(func)(x) for x in multivariate_normal.rvs(mean=mu_omega, cov=sigma_omega, size=n_samples_to_estimate_risk)))
    
    return np.mean(risks)


def get_target_variance_nonidentity(n, mu, cov):
    mu = mu.reshape(len(mu), 1)
    inv_cov = np.linalg.pinv(cov)
    cov_w = ((1 + mu.T @ inv_cov @ mu) * inv_cov - inv_cov @ mu @ mu.T @ inv_cov) / n
    return cov_w



def get_average_source_std_error(source_projections):
    "Compute the estimated standard error of the average source projection vector"
    if isinstance(source_projections, list):
        source_projections = np.array(source_projections)

    avg_ws = np.mean(source_projections, axis=0)
    R = np.linalg.norm(avg_ws)
    avg_ws /= R
    J = len(source_projections)
    d = 1 - np.mean((avg_ws @ source_projections.T)**2)
    sigma = np.sqrt(d / (J * R**2))
    return sigma**2 * np.eye(source_projections.shape[1])


def get_target_variance(n, pi, cov):
    denom = 4 * n * pi * (1-pi)
    
    return cov / denom


def calculate_variance_of_mu(source_projections):
    #-from VMF wiki
    N = len(source_projections)
    d=len(source_projections[0])

    mean_ = np.mean(source_projections, axis=0)
    norm_mean = np.linalg.norm(mean_)
    
    mu = mean_ / norm_mean
    
    num = np.mean([(np.dot(mu, s))**2 for s in source_projections], axis=0)
    num = 1 - num
    
    denom = N * norm_mean**2
    
    variance = num / denom
    
    return variance * np.eye(d)


def bootstrap_projection_vectors(type_, X, y, number_of_bootstraps):
    if isinstance(type_, str):
        if type_.lower() not in ['par', 'nonpar']:
            raise ValueError()
        if type_.lower() == 'par':
            return parametric_bootstrap_projection_vectors(X, y, number_of_bootstraps)
        else:
            return nonparametric_bootstrap_projection_vectors(X, y, number_of_bootstraps)

def parametric_bootstrap_projection_vectors(X, y, number_of_bootstraps=25):
    n, d = X.shape

    mean = np.mean(X[np.where(y == 0)[0]],axis=0)

    inds = [np.where(y == c)[0] for c in np.unique(y)]
    covs_ = np.array([np.cov(X[inds_].T) for inds_ in inds])
    cov = np.mean(covs_, axis=0)
    n0 = n // 2
    n1 = n - n0

    bootstrap_vectors = np.zeros((number_of_bootstraps, d))
    for b in range(number_of_bootstraps):

        X = np.vstack([np.random.multivariate_normal(mean, cov, size=n0), 
                            np.random.multivariate_normal(-1 * mean, cov, size=n1)
                           ])
        y = np.hstack([np.zeros(n0), np.ones(n1)])

        lda = FLD(cov=None)
        lda.fit(X,y)
        bootstrap_vectors[b] = lda.projection_

    return bootstrap_vectors

def nonparametric_bootstrap_projection_vectors(X, y, number_of_bootstraps, cov=None):
    n, d = X.shape

    bootstrap_vectors = np.zeros((number_of_bootstraps, d))
    
    for i in range(number_of_bootstraps):
        bootstrap_inds = np.random.choice(n, size=2*n, replace=True)
        fld = FLD(cov=cov).fit(X[bootstrap_inds], y[bootstrap_inds])
        bootstrap_vectors[i] = fld.projection_
        
    return bootstrap_vectors

def bootstrap_scaler_and_projection(X, y, cov=None, number_of_bootstraps=25):
    n, d = X.shape
    
    inds = [np.where(y == c)[0] for c in np.unique(y)]    
    bootstrap_vectors = np.zeros((number_of_bootstraps, d))
    
    for b in range(number_of_bootstraps):
        train_inds = np.hstack([np.random.choice(ind, size=n//2, replace=True) for ind in inds])
        scaler = TranslateStandardScale().fit(X[train_inds], y[train_inds])
    
        X_ = scaler.transform(X[train_inds])    
        fld = FLD(cov=cov).fit(X_[train_inds], y[train_inds])
        bootstrap_vectors[b] = fld.projection_
        
    return bootstrap_vectors


def get_source_tasks_to_keep(bootstrap_vectors, source_vectors, threshold=0.2):    
    mu_bootstrap = estimate_mu(bootstrap_vectors)
    null_distribution = np.array([1-abs(np.dot(mu_bootstrap, bv)) for bv in bootstrap_vectors])
    
    test_distances = np.array([1-abs(np.dot(mu_bootstrap, s)) for s in source_vectors])
    p_values = np.array([np.mean(d <= null_distribution) for d in test_distances])
        
    return p_values > threshold


def get_cosine_variance(vectors):
    J, d = vectors.shape
    mu = estimate_mu(vectors)
    variance = np.mean([((1 - np.dot(mu, vec)))**2 for vec in vectors], axis=0)
    return variance * np.eye(d)


def bootstrap_mus(vectors, number_of_bootstraps=25):
    J = len(vectors)
    d = len(vectors[0])

    bootstrap_mus = np.zeros((number_of_bootstraps, d))
    for b in range(number_of_bootstraps):
        inds = np.random.choice(J, size=J, replace=True)
        bootstrap_mus[b] = estimate_mu([vectors[ind] for ind in inds])

    return bootstrap_mus


def bootstrap_mu_variance(vectors):
    J = len(vectors)
    d = len(vectors[0])
    
    bootstrap_mu = bootstrap_mus(vectors)
    
    return get_cosine_variance(bootstrap_mu)