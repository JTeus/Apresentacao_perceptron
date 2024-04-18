# Author: Marcel Rodrigues de Barros
# PCS3438 - Inteligência Artificial - 2023/2

import numpy as np

def simple_perceptron(X,y ,iter_limit=10):

    weights = np.zeros(X.shape[1])
    bias = 0
    error_count = 1
    error_by_example = {i:0 for i in range(1,X.shape[0]+1)}
    while error_count > 0 and iter_limit > 0:
        error_count = 0
        for i in range(X.shape[0]):
            classification = np.dot(X[i], weights) + bias
            agreement = y[i] * classification
            correct = (agreement > 0)
            
            if not correct:
                weights += y[i] * X[i]
                bias += y[i]
                error_count += 1
                print(f"Data point {i+1} incorrectly classified - Theta: {weights}, bias: {bias}")
                error_by_example[i+1] += 1

        iter_limit -= 1
    
    return weights, bias, error_by_example

def main():
    X = np.array([[-1, -0.5], [-1, 1], [0, 0], [0, 1]])  
    y = np.array([-1, -1, 1,1])  

    weights, bias,error_by_example = simple_perceptron(X, y)
    print(f"Final weights: Bias = [{bias}] - Weights = {weights}")
    print(f"Error by example: {error_by_example}")


if __name__ == "__main__":
    main()