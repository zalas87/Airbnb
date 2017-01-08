# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 20:06:04 2016

@author: gonzalo
"""

import numpy as np

def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]

    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            gain = 2**r-1 #just use with binary 
            discount = np.log2(np.arange(2, r.size + 2))
            #return np.sum(r / np.log2(np.arange(2, r.size + 2))) -> the same method
            return np.sum(gain/discount) 
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max



def score_ndcg5(preds, ground_truth, k=5):
    """
    preds : ndarray, shape = [n_samples, n_classes]
        Predicted probabilities.
    ground_truth : ndarray, shape = [n_samples]
        Ground_truth (true labels represended as integers)..
    """
    assert(preds.shape[0]==ground_truth.shape[0])
    
    scores = []
    
    rows = preds.shape[0]
    r = np.zeros((rows,preds.shape[1]))
    
    preds_sorted = np.argsort(preds)[:,::-1]  
    ground_truth.shape = rows,1 #Make ground as unidimensional label
    r[preds_sorted==ground_truth] = 1

    for y_score in r:
        score = ndcg_at_k(y_score,k=5,method=1)
        scores.append(score)
    
    return np.mean(scores)
    
    
#preds = np.array([[0.34,0.22,0.,0.44],[0,0.8,0.05, 0.15],[0,0.3,0.34,0.36]])
#truth = np.array([1,1,2])
#print('predictions: \n', preds)
#print('\n\n truth: \n', truth)
#print('\n\n scores: \n', score_ndcg5(preds, truth))