#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# This is the basic setup for output prediction (multi-output regression) with a candidate list
#
# Input arguments: 
# S/B/M : S=stratifield fold setting, 
#         B=mbfold has all the same BGCs in the same fold, used in the molecule->Bgc prediction as a "relaxed setting"
#         M=bmfold has all the same molecules in the same fold, used in the Bgc->molecule prediction as a "relaxed setting"
# MtoB/BtoM : Prediction direction, MtoB=Molecule->Bgc, BtoM=Bgc->Molecule, denoted as X->y
# mlp/IOKR/CCA/Homals/MMR : mlp=multilayer perceptron
#                           IOKR=Input output kernel regression
#                           CCA=canonical correlation analysis
#                           MMR=maximum margin regression
# 
# Output: rankings 
"""

import sys
import numpy as np
from sklearn.cross_decomposition import CCA
from models import homals, mlp, get_iokr_preimage, get_iokr_inverse, cls_mmr_solver
from helpers import center_kernel, normalize_kernel, find_rankings

if __name__=="__main__":
    args=sys.argv[1:]
    path=args[0]
    
    # Fold is chosen
    if args[1]=="S":
        fold=np.load(path+"/fold_stratified.npy")
    elif args[1]=="B":
        fold_b=np.load(path+"/mbfold.npy")
    elif args[1]=="M":
        fold_m=np.load(path+"/bmfold.npy")
    else:
        print("Fold is not defined")
    
    # Tanimoto kernels are calculated based on pfams (protein families) and fingerprints---see pfams.npy and fps.npy for original files
    molecule_kernel=np.load(path+"/mol_kernel_tanimoto.npy")
    bgc_kernel=np.load(path+"/bgc_kernel_tanimoto.npy")
    # String kernel is centralized and normalized as it is 
    string_kernel=np.load(path+"/string_9_2.npy")
    
    # Center and normalize kernels
    bgc_kernel = center_kernel(bgc_kernel)
    bgc_kernel = normalize_kernel(bgc_kernel)
    
    # The string kernel is useful for some prediction scenarios, so it's combined with tanimoto 
    bgc_kernel = 0.5*(bgc_kernel + string_kernel)
    
    molecule_kernel = center_kernel(molecule_kernel)
    molecule_kernel = normalize_kernel(molecule_kernel)
    
    if args[2]=="MtoB":
        y=bgc_kernel
        X=molecule_kernel
    elif args[2]=="BtoM":
        y=molecule_kernel
        X=bgc_kernel
    
    # Cand is the (unique) candidate list of y
    cand=np.unique(y, axis = 0)
    accuracies=np.zeros((4,10))
    sds=np.zeros((4,10))
    
    for f in range(10):
        train_ix=np.where(fold!=f)[0]
        test_ix=np.where(fold==f)[0]
        
        n_inputs=X[np.ix_(train_ix, train_ix)].shape[1]
        n_outputs=y[np.ix_(train_ix, train_ix)].shape[1]
        X_train=X[np.ix_(train_ix, train_ix)]
        Y_train=y[np.ix_(train_ix, train_ix)]
        X_test=X[np.ix_(test_ix, train_ix)]
        y_test=y[np.ix_(test_ix, train_ix)]
        ranks=np.zeros((len(test_ix), 1))
        
        if args[3]=="MLP":
            model=mlp(n_inputs, n_outputs)
            model.fit(X_train, Y_train, verbose=0, epochs=200)
            pred=model.predict(X_test)
            ranks=find_rankings(pred, y_test, cand, ranks, train_ix)
            
        elif args[3]=="IOKR":
            # Slightly different results with different values of lmbda
            lmbda=0.01
            iokr_inverse=get_iokr_inverse(X_train, lmbda)
            for j in range(len(test_ix)):
                r=0
                score=get_iokr_preimage(X_test[j, :], y[:,train_ix], iokr_inverse)
                correct=score[test_ix[j]]
                sort=np.sort(score)[::-1]
                for k in range(len(score)):
                    if sort[k]!=correct:
                        r=r+1
                    if sort[k]==correct:
                        ranks[j]=r
                        break 
                    
        # CCA takes quite a lot of RAM...
        elif args[3]=="CCA":
            cca=CCA(n_components=140)
            cca.fit(X_train, Y_train)
            pred=cca.predict(X_test)
            ranks=find_rankings(pred,y_test,cand,ranks,train_ix)
                
        elif args[3]=="Homals":
            ndim = 150
            inonlinpca=0
            ndegree=0
            ldecomp=[]
            fps=np.load(path+"/mol_fps.npy")
            pfams=np.load(path+"/bgc_pfam.npy")
            if args[1]=="MtoB":
                x_train=fps[train_ix,:]
                x_test=fps[test_ix,:]
                y_train=pfams[train_ix,:]
            else:
                y_train=fps[train_ix,:]
                x_train=pfams[train_ix,:]
                x_test=pfams[test_ix,:]
            ldata=[X_train, x_train, Y_train, y_train]
            ranks=homals(ldata, ndim, 200, ldecomp, ndegree, test_ix, train_ix, cand, X_test, x_test, y_test)
                
        elif args[3]=="MMR":
            solver=cls_mmr_solver()
            
            # Slightly different results with different values of param
            param=1
            alpha=solver.mmr_solver(X_train, Y_train, param)
            dist=[]
            for j in range(len(test_ix)):
                for i in range(y.shape[0]):
                    d=np.dot(y[i,train_ix]*X_test[j,:], alpha)
                    dist.append(d)
                dist=np.asarray(dist)
                # id_max=np.where(dist==maxv)[0][0] 
                # Real max is the true y
                real_max=dist[test_ix[j]]
                sorted=np.sort(dist)
                sorted=np.unique(sorted)
                for i in range(len(sorted)):
                    if sorted[::-1][i]>real_max:
                        ranks[j]+=1
                dist=[]
            
            
        accuracies[0,f]=len(np.where(ranks<1)[0])/len(test_ix)*100
        sds[0,f]=len(np.where(ranks<1)[0])/len(test_ix)*100
        accuracies[1,f]=len(np.where(ranks<5)[0])/len(test_ix)*100
        sds[1,f]=len(np.where(ranks<5)[0])/len(test_ix)*100
        accuracies[2,f]=len(np.where(ranks<10)[0])/len(test_ix)*100
        sds[2,f]=len(np.where(ranks<10)[0])/len(test_ix)*100
        accuracies[3,f]=len(np.where(ranks<20)[0])/len(test_ix)*100
        sds[3,f]=len(np.where(ranks<20)[0])/len(test_ix)*100
        
    print("\n")    
    print("Rank less than 1: "+str(round(np.average(accuracies[0,:]),2))+"%, with sd "+str(round(np.std(sds[0,:]),2))+"%")
    print("Rank less than 5: "+str(round(np.average(accuracies[1,:]),2))+"%, with sd "+str(round(np.std(sds[1,:]),2))+"%")
    print("Rank less than 10: "+str(round(np.average(accuracies[2,:]),2))+"%, with sd "+str(round(np.std(sds[2,:]),2))+"%")
    print("Rank less than 20: "+str(round(np.average(accuracies[3,:]),2))+"%, with sd "+str(round(np.std(sds[3,:]),2))+"%")
    
