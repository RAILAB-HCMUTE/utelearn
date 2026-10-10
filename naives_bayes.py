# naives_bayes.py

import numpy as np 

def gaussian_pdf(x, mean, var): 
    """Calculate Gaussian probability density function."""
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x - mean) ** 2 / (2 * var)) 
 
def train_naive_bayes(X_train, y_train): 
    """Train the Naive Bayes classifier and return means, variances, and prior probabilities."""
    classes = np.unique(y_train) 
    means = {} 
    vars = {} 
    priors = {} 
 
    for c in classes: 
        X_c = X_train[y_train == c] 
        means[c] = np.mean(X_c, axis=0) 
        vars[c] = np.var(X_c, axis=0) 
        priors[c] = X_c.shape[0] / X_train.shape[0] 
     
    return means, vars, priors 
 
def predict_naive_bayes(X, means, vars, priors): 
    """Predict class labels for given features using the trained Naive Bayes parameters."""
    probs = [] 
     
    for c in means: 
        # Start with the prior probability (log space)
        prob = np.log(priors[c]) 
         
        # Add the log of the likelihoods for each feature 
        for i in range(X.shape[1]): 
            prob += np.log(gaussian_pdf(X[:, i], means[c][i], vars[c][i])) 
         
        probs.append(prob) 
     
    # Return the class with the highest probability 
    return np.argmax(probs, axis=0) 
 
def generate_synthetic_data(n_points, n_classes, n_features): 
    """Generate synthetic multivariate normal data."""
    means = np.random.uniform(-10, 10, (n_classes, n_features)) 
    covs = np.array([np.eye(n_features) for _ in range(n_classes)])  # Identity covariance matrix 
     
    points = [] 
    labels = [] 
    for i in range(n_classes): 
        class_points = np.random.multivariate_normal(means[i], covs[i], n_points) 
        points.append(class_points) 
        labels.append(np.full(n_points, i)) 
     
    X = np.vstack(points) 
    y = np.hstack(labels) 
     
    return X, y