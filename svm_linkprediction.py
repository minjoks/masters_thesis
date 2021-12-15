import numpy as np
import sys

"""
Link prediction framework using SVM (a modified version of https://github.com/aatapa/RLScore/blob/master/rlscore/learner/kron_svm.py)

"""


from rlscore.utilities import sampled_kronecker_products
from rlscore.predictor import KernelPairwisePredictor
from rlscore.predictor import PairwisePredictorInterface
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import qmr
from rlscore.learner.kron_svm import KronSVM


if __name__=="__main__":
    args=sys.argv[1:]
    path=args[0]
    
    #link_list = np.load("/mibigdata/link_list.npy")
    #fold = np.load("/mibigdata/fold_stratified.npy")
    bgc_kernel = np.load(path+"/bgc_kernel_svm.npy")
    molecule_kernel = np.load(path+"/mol_kernel_svm.npy")
    Y = np.load(path+"/Y_svm.npy")
    Y[Y==0]=-1
    
                
    n_bgc, n_mol = Y.shape
    y_vec = Y.ravel(order = 'C')
    ids = np.arange(n_bgc*n_mol)
    bgc_ids, mol_ids = np.unravel_index(ids, (n_bgc,n_mol), order = 'C') 
    
    for f in range(0,10):
    
        folds=np.load(path+"/svmfolds/folds_"+str(f)+".npy")
           
        
        test_ids = np.array(np.where(folds==f)).squeeze()
        train_ids = np.array(np.where(folds!=f)).squeeze()
        
        y_test = y_vec[test_ids]
        bgc_ids_test = bgc_ids[test_ids]
        mol_ids_test = mol_ids[test_ids]
        
        y_train = y_vec[train_ids]
        bgc_ids_train = bgc_ids[train_ids]
        mol_ids_train = mol_ids[train_ids]
        
        
        class DualCallback(object):
            def __init__(self):
                self.iter = 0
                self.atol = None
            def callback(self, learner):
                K1 = learner.resource_pool["K1"]
                K2 = learner.resource_pool["K2"]
                rowind = learner.label_row_inds
                colind = learner.label_col_inds
                self.iter += 1 
            def finished(self, learner):
                pass
            
        regparam = 0.01
        params = {}
        params["K1"] = bgc_kernel
        params["K2"] = molecule_kernel
        params["Y"] = y_train
        params["label_row_inds"] = bgc_ids_train 
        params["label_col_inds"] = mol_ids_train 
        params["maxiter"] = 300
        params["inneriter"] = 100
        params["regparam"] = regparam
        params["callback"] = DualCallback()
        
        
        learner = KronSVM(**params)
        
        P_dual = learner.predictor.predict(bgc_kernel, molecule_kernel, bgc_ids_test, mol_ids_test)
        
        preds = np.zeros((len(P_dual), 1))
        for i in range(len(P_dual)):
            if P_dual[i] < 0:
                preds[i] = 0 
            else:
                preds[i] = 1
    
        for i in range(len(y_test)):
            if y_test[i] == -1:
                y_test[i] = 0
    
        np.save(path+"/svm_preds_"+str(f)+".npy", preds)
        np.save(path+"/svm_ytest_"+str(f)+".npy", y_test)
        np.save(path+"/Pdual_final_"+str(f)+".npy", P_dual)
        
        

