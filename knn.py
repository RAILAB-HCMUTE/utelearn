import numpy as np

# Function to compute Euclidean distance between two points 
def euclidean_distance(x1, x2): 
    """Computes the Euclidean distance between two points."""
    return np.sqrt(np.sum((x1 - x2)**2)) 
 
def knn_regression(X_train, y_train, X_test, k=10): 
    """
    Performs KNN regression to predict continuous target values.
    """
    predictions = [] 
     
    # Loop over each test point
    for test_point in X_test: 
        # Calculate distances from the test point to all training points
        distances = [euclidean_distance(test_point, train_point) for train_point in X_train] 
         
        # Get the indices of the K nearest neighbors
        nearest_neighbors_indices = np.argsort(distances)[:k] 
         
        # Get the target values of the K nearest neighbors
        nearest_neighbors_target_values = y_train[nearest_neighbors_indices] 
         
        # The prediction is the average of the target values of the nearest neighbors
        prediction = np.mean(nearest_neighbors_target_values) 
        predictions.append(prediction) 
     
    return np.array(predictions) 

def knn_classify(X_train, y_train, X_test, k=5, n_classes=None): 
    """
    Performs KNN classification/clustering using majority voting.
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    
    if n_classes is None:
        n_classes = len(np.unique(y_train))
        
    N_total = len(X_train)
    n_test_points = len(X_test)
    predictions = []
    
    for i in range(n_test_points): 
        point_considered = X_test[i] 
        distance = np.zeros((N_total, 1)) 

        # Compute all the distances
        for j in range(N_total): 
            distance[j] = euclidean_distance(point_considered, X_train[j])
         
        index = np.zeros(k, dtype=int) 
        class_KNN = np.zeros(k) 
        already_picked = np.zeros((N_total, 1)) 
     
        for step in range(k): 
            min_val = float('inf')
            min_idx = -1
            for j in range(N_total): 
                if ((distance[j] < min_val) and (already_picked[j] == 0)): 
                    min_val = distance[j]
                    min_idx = j
            
            already_picked[min_idx] = 1 # Mark to eliminate the nearest neighbor already validated
            index[step] = min_idx 
            class_KNN[step] = y_train[min_idx] 
     
        # Determine the class that the point belongs to using majority vote
        point_class_NNN = np.zeros(n_classes) 
        for t in range(n_classes): 
            count = 0
            for step in range(k): 
                if (class_KNN[step] == t): 
                    count += 1 
            point_class_NNN[t] = count 
         
        class_of_point_considered = np.argmax(point_class_NNN) 
        predictions.append(class_of_point_considered) 
        
    return np.array(predictions)