import numpy as np

def center_kernel(K):
    n = K.shape[0]
    M = np.eye(n)-np.ones((n, n))/n
    return np.dot(M, np.dot(K, M))

def normalize_kernel(K):
    D = np.diag(1/np.sqrt(np.diag(K)))
    return np.dot(D, np.dot(K, D))

def find_rankings(pred, y_test, cand, ranks, train_ix):
    for i in range(len(ranks)):
        # The biggest (pearson) correlation coef should be between the predicted y and true y
        biggest=np.corrcoef(pred[i,:], y_test[i,:])[0,1]
        # Go through all candidates to see if some give higher correlations 
        for j in range(cand.shape[0]):
            # Increase the rank by one if a candidate with a bigger correlation is found
            if np.corrcoef(pred[i,:], cand[j,np.ix_(train_ix)])[0,1] > biggest:
                ranks[i]=ranks[i]+1
    return ranks
    
def tanimoto(X, Y):

    tmp = np.dot(X, Y.T)
    ans = tmp / ((X*X).sum(-1)[:, np.newaxis] + (Y*Y).sum(-1)[np.newaxis, :] - tmp)
    ans[np.isnan(ans)] = 0  
    return ans
    
    
def tancoef(x,y):
    r=np.dot(x,y)/(len(np.where(x==1)[0])+len(np.where(y==1)[0])-np.dot(x,y))
    return r
