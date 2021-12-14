#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"Gold standard dataset" is analyzed using IOKR
Direction of prediction is from Molecules->BGCs

Input arguments:  path of data (PRISM data=mol_fps, bgc_pfam, bioclasses, fold)
                  y, if you want plots in the master's thesis (data used=Tanimoto coefficients&bioclasses of PRISM predictions) (optional)
    

Output: Ranks of PRISM4 data using IOKR, 
        Plots comparing IOKR and PRISM (optional)


"""
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from models import get_iokr_preimage, get_iokr_inverse
import numpy as np
from helpers import center_kernel, normalize_kernel, tanimoto

if __name__=="__main__":
    
    
    args=sys.argv[1:]
    path=args[0]
    mol_fps=np.load(path+"/prismdata/mol_fps.npy")
    bgc_pfam=np.load(path+"/prismdata/bgc_pfam.npy")
    final_bio_class=np.load(path+"/prismdata/bio_class_of_bgcs.npy")
    fold=np.load(path+"/prismdata/fold.npy")

    # The BGCs are classified into several different biosynthetic classes....
    coefs = []
    nrp_coefs = []
    pkd1_coefs = []
    pkd2_coefs = []
    ripp_coefs = []
    aminocoumarin_coefs = []
    aminoglycoside_coefs = []
    antimetabolite_coefs = []
    betalactam_coefs = []
    bisindole_coefs = []
    cyclodipeptide_coefs = []
    isonitrilealkaloid_coefs = []
    lincoside_coefs = []
    nucleoside_coefs = []
    other_coefs = []
    phosphonate_coefs = []
    bgc_kernel=tanimoto(bgc_pfam, bgc_pfam)
    mol_kernel=tanimoto(mol_fps,mol_fps)
    mol_kernel=center_kernel(mol_kernel)
    mol_kernel=normalize_kernel(mol_kernel)
    bgc_kernel=center_kernel(bgc_kernel)
    bgc_kernel=normalize_kernel(bgc_kernel)
    y=mol_kernel
    accuracies=np.zeros((4,10))
    sds=np.zeros((4,10))
    for f in range(10):
        test = np.where(fold == f)[0]
        train = np.where(fold != f)[0]
        X_train=bgc_kernel[np.ix_(train, train)]
        Y_train=mol_kernel[np.ix_(train, train)]
        X_test=bgc_kernel[np.ix_(test, train)]
        y_test=mol_kernel[np.ix_(test, train)]
        ranks=np.zeros((len(test), 1))
        lmbda=0.01
        iokr_inverse=get_iokr_inverse(X_train, lmbda)
        for j in range(len(test)):
            r=0
            score=get_iokr_preimage(X_test[j, :], y[:,train], iokr_inverse)
            correct=score[test[j]]
            maxs=np.max(score)
            max_s=np.where(score==maxs)[0][0]
            correct_fp=mol_fps[test[j],:]
            sort=np.sort(score)[::-1]
            for k in range(len(score)):
                if sort[k]!=correct:
                    r=r+1
                if sort[k]==correct:
                    ranks[j]=r
                    break 
            ranked_fps=mol_fps[max_s,:]
            correct_fp=np.reshape(correct_fp, (1024,))
            if "RiPP" in final_bio_class[test[j]]:
              ripp_coefs.append(np.dot(correct_fp, ranked_fps)/(len(np.where(correct_fp == 1)[0])+len(np.where(ranked_fps== 1)[0])-np.dot(correct_fp, ranked_fps)))
    
            if "nonribosomal peptide" in final_bio_class[test[j]]:
              nrp_coefs.append(np.dot(correct_fp, ranked_fps)/(len(np.where(correct_fp == 1)[0])+len(np.where(ranked_fps == 1)[0])-np.dot(correct_fp, ranked_fps)))
    
            if "type 1 polyketide" in final_bio_class[test[j]]:
              pkd1_coefs.append(np.dot(correct_fp, ranked_fps)/(len(np.where(correct_fp == 1)[0])+len(np.where(ranked_fps== 1)[0])-np.dot(correct_fp, ranked_fps)))
    
            if "type 2 polyketide" in final_bio_class[test[j]]:
              pkd2_coefs.append(np.dot(correct_fp, ranked_fps)/(len(np.where(correct_fp == 1)[0])+len(np.where(ranked_fps== 1)[0])-np.dot(correct_fp, ranked_fps)))
    
            if "aminocoumarin" in final_bio_class[test[j]]:
              aminocoumarin_coefs.append(np.dot(correct_fp, ranked_fps)/(len(np.where(correct_fp == 1)[0])+len(np.where(ranked_fps == 1)[0])-np.dot(correct_fp, ranked_fps)))
    
            if "aminoglycoside" in final_bio_class[test[j]]:
              aminoglycoside_coefs.append(np.dot(correct_fp, ranked_fps)/(len(np.where(correct_fp == 1)[0])+len(np.where(ranked_fps == 1)[0])-np.dot(correct_fp, ranked_fps)))
    
            if "antimetabolite" in final_bio_class[test[j]]:
              antimetabolite_coefs.append(np.dot(correct_fp, ranked_fps)/(len(np.where(correct_fp == 1)[0])+len(np.where(ranked_fps == 1)[0])-np.dot(correct_fp, ranked_fps)))
    
            if "beta-lactam" in final_bio_class[test[j]]:
              betalactam_coefs.append(np.dot(correct_fp, ranked_fps)/(len(np.where(correct_fp == 1)[0])+len(np.where(ranked_fps == 1)[0])-np.dot(correct_fp, ranked_fps)))
    
            if "cyclodipeptide" in final_bio_class[test[j]]:
              cyclodipeptide_coefs.append(np.dot(correct_fp, ranked_fps)/(len(np.where(correct_fp == 1)[0])+len(np.where(ranked_fps == 1)[0])-np.dot(correct_fp, ranked_fps)))
    
            if "isonitrile alkaloid" in final_bio_class[test[j]]:
              isonitrilealkaloid_coefs.append(np.dot(correct_fp, ranked_fps)/(len(np.where(correct_fp == 1)[0])+len(np.where(ranked_fps == 1)[0])-np.dot(correct_fp, ranked_fps)))
    
            if "lincoside" in final_bio_class[test[j]]:
              lincoside_coefs.append(np.dot(correct_fp, ranked_fps)/(len(np.where(correct_fp == 1)[0])+len(np.where(ranked_fps == 1)[0])-np.dot(correct_fp, ranked_fps)))
    
            if "nucleoside" in final_bio_class[test[j]]:
              nucleoside_coefs.append(np.dot(correct_fp, ranked_fps)/(len(np.where(correct_fp == 1)[0])+len(np.where(ranked_fps == 1)[0])-np.dot(correct_fp, ranked_fps)))
    
            if "other" in final_bio_class[test[j]]:
              other_coefs.append(np.dot(correct_fp, ranked_fps)/(len(np.where(correct_fp == 1)[0])+len(np.where(ranked_fps == 1)[0])-np.dot(correct_fp, ranked_fps)))
    
            if "phosphonate" in final_bio_class[test[j]]:
              phosphonate_coefs.append(np.dot(correct_fp, ranked_fps)/(len(np.where(correct_fp == 1)[0])+len(np.where(ranked_fps == 1)[0])-np.dot(correct_fp, ranked_fps)))
    
            if "bisindole" in final_bio_class[test[j]]:
              bisindole_coefs.append(np.dot(correct_fp, ranked_fps)/(len(np.where(correct_fp == 1)[0])+len(np.where(ranked_fps == 1)[0])-np.dot(correct_fp, ranked_fps)))
            #if ranks[j] < 5:
            #  for h in range(5):
                  
            #    closest_five[:,r+h] = ranked_fps[:,h]
            #  r = r + 5
            coefs.append(np.dot(correct_fp, ranked_fps)/(len(np.where(correct_fp == 1)[0])+len(np.where(ranked_fps == 1)[0])-np.dot(correct_fp, ranked_fps)))
                
        accuracies[0,f]=len(np.where(ranks<1)[0])/len(test)*100
        sds[0,f]=len(np.where(ranks<1)[0])/len(test)*100
        accuracies[1,f]=len(np.where(ranks<5)[0])/len(test)*100
        sds[1,f]=len(np.where(ranks<5)[0])/len(test)*100
        accuracies[2,f]=len(np.where(ranks<10)[0])/len(test)*100
        sds[2,f]=len(np.where(ranks<10)[0])/len(test)*100
        accuracies[3,f]=len(np.where(ranks<20)[0])/len(test)*100
        sds[3,f]=len(np.where(ranks<20)[0])/len(test)*100
        
    print("\n")    
    print("Rank less than 1: "+str(round(np.average(accuracies[0,:]),2))+"%, with sd "+str(round(np.std(sds[0,:]),2))+"%")
    print("Rank less than 5: "+str(round(np.average(accuracies[1,:]),2))+"%, with sd "+str(round(np.std(sds[1,:]),2))+"%")
    print("Rank less than 10: "+str(round(np.average(accuracies[2,:]),2))+"%, with sd "+str(round(np.std(sds[2,:]),2))+"%")
    print("Rank less than 20: "+str(round(np.average(accuracies[3,:]),2))+"%, with sd "+str(round(np.std(sds[3,:]),2))+"%")
    
    if args[1]=="y":
        # Tanimoto coefficient between random pairs
        other = []
        Y = mol_fps.T
        for i in range(0,982):
          for j in range(982):
            if i < j :
              dist = np.dot(Y[:,i], Y[:,j])/(len(np.where(Y[:,i] == 1)[0])+len(np.where(Y[:,j] == 1)[0])-np.dot(Y[:,i], Y[:,j]))
              other.append(dist)
              
        ripp_coef = pd.DataFrame(ripp_coefs)
        nrp_coef = pd.DataFrame(nrp_coefs)
        pkd1_coef = pd.DataFrame(pkd1_coefs)
        pkd2_coef = pd.DataFrame(pkd2_coefs)
        aminocoumarin_coef = pd.DataFrame(aminocoumarin_coefs)
        aminoglycoside_coef = pd.DataFrame(aminoglycoside_coefs)
        antimetabolite_coef = pd.DataFrame(antimetabolite_coefs)
        betalactam_coef = pd.DataFrame(betalactam_coefs)
        bisindole_coef = pd.DataFrame(bisindole_coefs)
        cyclodipeptide_coef = pd.DataFrame(cyclodipeptide_coefs)
        isonitrilealkaloid_coef = pd.DataFrame(isonitrilealkaloid_coefs)
        lincoside_coef = pd.DataFrame(lincoside_coefs)
        nucleoside_coef = pd.DataFrame(nucleoside_coefs)
        other_coef = pd.DataFrame(other_coefs)
        phosphonate_coef = pd.DataFrame(phosphonate_coefs)
        other = pd.DataFrame(other)
        coefff = pd.DataFrame(coefs)
        
        frames = [coefff, other, aminocoumarin_coef, aminoglycoside_coef, antimetabolite_coef, betalactam_coef, bisindole_coef, cyclodipeptide_coef, isonitrilealkaloid_coef, lincoside_coef, nrp_coef, nucleoside_coef, other_coef, phosphonate_coef, ripp_coef, pkd1_coef, pkd2_coef]
        ax = sns.boxplot(data=frames, palette="Set3", showfliers=False, whis=1.5, width= 0.7)
        labels_list = ["all","random pairs", "aminocoumarin", "aminoglycoside", "antimetabolite", "beta-lactam", "bisindole", "cyclopeptide", "isonitrile alkaloid", "lincoside", "NRP", "nucleoside", "other", "phosphonate", "RiPP", "type 1 polyketide", "type 2 polyketide"]
        ax.set_xticklabels(labels_list)
        plt.xticks(rotation=90)
        ax.set_title('With IOKR, Tanimoto coefficients between the true and highest ranked molecules')
        plt.ylabel("Tanimoto coefficient")
        plt.show()
        
        # There are three BGCs used in IOKR predictions that are not found in this file (PRISM predictions)
        # Does not mean a lot
        tansp=np.loadtxt(path+"/prismdata/prismtans.csv", dtype = str, delimiter = "," )
        
        
          
        # PRISM bioclasses
        new_cls_p = tansp[:,1]
        #PRISM tanimoto coefficients
        new_tans = tansp[:,3]
        
        other = pd.DataFrame(other)
        coefff = pd.DataFrame(coefs)
        paper = pd.DataFrame(new_tans)
        frames = [coefff, paper, other]
        ax = sns.boxplot(data=frames, palette="Set1", showfliers=False, whis=1.5, width= 0.7)
        labels_list = ["IOKR","PRISM 4","random pairs"]
        ax.set_xticklabels(labels_list)
        plt.xticks(rotation=0)
        ax.set_title('IOKR vs PRISM 4')
        plt.ylabel("Tanimoto coefficient")
        plt.show()
        
        
        ripp_p = []
        nrp_p = []
        pkd1_p = []
        pkd2_p = []
        aminocoumarin_p = []
        aminoglyc_p = []
        antim_p = []
        b_lactam = []
        cyclo_p = []
        ison_p = []
        lincoside_p = []
        phosphonate_p = []
        bisindole_p = []
        nucleoside_p = []
        other_p = []
        for j in range(len(new_tans)):
          if "RiPP" in new_cls_p[j]: 
            ripp_p.append(new_tans[j])
          if "nonribosomal peptide" in new_cls_p[j]:
            nrp_p.append(new_tans[j])
          if "type 1 polyketide" in new_cls_p[j]:
            pkd1_p.append(new_tans[j])
          if "type 2 polyketide" in new_cls_p[j]:
            pkd2_p.append(new_tans[j])
          if "aminocoumarin" in new_cls_p[j]:
            aminocoumarin_p.append(new_tans[i])
          if "aminoglycoside" in new_cls_p[j]:
            aminoglyc_p.append(new_tans[j])
          if "antimetabolite" in new_cls_p[j]:
            antim_p.append(new_tans[j])
          if "beta-lactam" in new_cls_p[j]:
            b_lactam.append(new_tans[j])
          if "cyclodipeptide" in new_cls_p[j]:
            cyclo_p.append(new_tans[j])
          if "isonitrile alkaloid" in new_cls_p[j]:
            ison_p.append(new_tans[j])
          if "lincoside" in new_cls_p[j]:
            lincoside_p.append(new_tans[j])
          if "nucleoside" in new_cls_p[j]:
            nucleoside_p.append(new_tans[j])
          if "other" in new_cls_p[j]:
            other_p.append(new_tans[j])
          if "phosphonate" in new_cls_p[j]:
            phosphonate_p.append(new_tans[j])
          if "bisindole" in new_cls_p[j]:
            bisindole_p.append(new_tans[j])
        
        ripp_coef = pd.DataFrame(ripp_coefs)
        nrp_coef = pd.DataFrame(nrp_coefs)
        pkd1_coef = pd.DataFrame(pkd1_coefs)
        pkd2_coef = pd.DataFrame(pkd2_coefs)
        aminocoumarin_coef = pd.DataFrame(aminocoumarin_coefs)
        aminoglycoside_coef = pd.DataFrame(aminoglycoside_coefs)
        antimetabolite_coef = pd.DataFrame(antimetabolite_coefs)
        betalactam_coef = pd.DataFrame(betalactam_coefs)
        bisindole_coef = pd.DataFrame(bisindole_coefs)
        cyclodipeptide_coef = pd.DataFrame(cyclodipeptide_coefs)
        isonitrilealkaloid_coef = pd.DataFrame(isonitrilealkaloid_coefs)
        lincoside_coef = pd.DataFrame(lincoside_coefs)
        nucleoside_coef = pd.DataFrame(nucleoside_coefs)
        other_coef = pd.DataFrame(other_coefs)
        phosphonate_coef = pd.DataFrame(phosphonate_coefs)
        other = pd.DataFrame(other)
        coefff = pd.DataFrame(coefs)
        
        ripp_p = pd.DataFrame(ripp_p)
        nrp_p = pd.DataFrame(nrp_p)
        pkd1_p = pd.DataFrame(pkd1_p)
        pkd2_p = pd.DataFrame(pkd2_p)
        aminocoumarin_p = pd.DataFrame(aminocoumarin_p)
        aminoglyc_p = pd.DataFrame(aminoglyc_p)
        antim_p = pd.DataFrame(antim_p)
        b_lactam = pd.DataFrame(b_lactam)
        cyclo_p = pd.DataFrame(cyclo_p)
        ison_p = pd.DataFrame(ison_p)
        lincoside_p = pd.DataFrame(lincoside_p)
        phosphonate_p = pd.DataFrame(phosphonate_p)
        bisindole_p = pd.DataFrame(bisindole_p)
        nucleoside_p = pd.DataFrame(nucleoside_p)
        other_p = pd.DataFrame(other_p)
        
        # IOKR
        aminocoumarin_coef = pd.DataFrame(aminocoumarin_coefs)
        aminoglycoside_coef = pd.DataFrame(aminoglycoside_coefs)
        antimetabolite_coef = pd.DataFrame(antimetabolite_coefs)
        betalactam_coef = pd.DataFrame(betalactam_coefs)
        bisindole_coef = pd.DataFrame(bisindole_coefs)
        cyclodipeptide_coef = pd.DataFrame(cyclodipeptide_coefs)
        isonitrilealkaloid_coef = pd.DataFrame(isonitrilealkaloid_coefs)
        lincoside_coef = pd.DataFrame(lincoside_coefs)
        nucleoside_coef = pd.DataFrame(nucleoside_coefs)
        other_coef = pd.DataFrame(other_coefs)
        phosphonate_coef = pd.DataFrame(phosphonate_coefs)
        other = pd.DataFrame(other)
        coefff = pd.DataFrame(coefs)
        
        # PRISM
        aminocoumarin_p = pd.DataFrame(aminocoumarin_p)
        aminoglyc_p = pd.DataFrame(aminoglyc_p)
        antim_p = pd.DataFrame(antim_p)
        b_lactam = pd.DataFrame(b_lactam)
        cyclo_p = pd.DataFrame(cyclo_p)
        ison_p = pd.DataFrame(ison_p)
        lincoside_p = pd.DataFrame(lincoside_p)
        phosphonate_p = pd.DataFrame(phosphonate_p)
        bisindole_p = pd.DataFrame(bisindole_p)
        nucleoside_p = pd.DataFrame(nucleoside_p)
        other_p = pd.DataFrame(other_p)
        
        ripp_coef["method"] = "IOKR"
        ripp_p["method"] = "PRISM"
        ripps = [ripp_coef, ripp_p]
        ripp = pd.concat(ripps)
        ripp.columns = ["Tanimoto coefficient", "method"]
        ripp["bioclass"] = "RiPP"
        ripp = ripp.explode('Tanimoto coefficient')
        ripp['Tanimoto coefficient'] = ripp['Tanimoto coefficient'].astype('float')
        
        nrp_coef["method"] = "IOKR"
        nrp_p["method"] = "PRISM"
        nrps = [nrp_coef, nrp_p]
        nrp = pd.concat(nrps)
        nrp.columns = ["Tanimoto coefficient", "method"]
        nrp["bioclass"] = "NRP"
        nrp = nrp.explode('Tanimoto coefficient')
        nrp['Tanimoto coefficient'] = nrp['Tanimoto coefficient'].astype('float')
        
        pkd1_coef["method"] = "IOKR"
        pkd1_p["method"] = "PRISM"
        pkds = [pkd1_coef, pkd1_p]
        pkd = pd.concat(pkds)
        pkd.columns = ["Tanimoto coefficient", "method"]
        pkd["bioclass"] = "Polyketide 1"
        pkd = pkd.explode('Tanimoto coefficient')
        pkd['Tanimoto coefficient'] = pkd['Tanimoto coefficient'].astype('float')
        
        pkd2_coef["method"] = "IOKR"
        pkd2_p["method"] = "PRISM"
        pkds2 = [pkd2_coef, pkd2_p]
        pkd2 = pd.concat(pkds2)
        pkd2.columns = ["Tanimoto coefficient", "method"]
        pkd2["bioclass"] = "Polyketide 2"
        pkd2 = pkd2.explode('Tanimoto coefficient')
        pkd2['Tanimoto coefficient'] = pkd2['Tanimoto coefficient'].astype('float')
        
        frames = [ripp, nrp, pkd, pkd2]
        data = pd.concat(frames)
        ax = sns.boxplot(x="bioclass", y="Tanimoto coefficient", hue="method",
                         data=data, palette="Set1", showfliers=False, whis=1.5, width= 0.7)
        ax.set_title('Prediction comparison between bioclasses and methods')
        plt.show()
        
    
    
    
