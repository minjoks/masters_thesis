An overall illustration of the topic; relationships between data and the prediction task. 

![alt txt](https://github.com/minjoks/masters_thesis/blob/main/dataview.png?raw=true)




**mibigdata**  
Includes data for 1869 unique BGC-metabolite links, gathered from MIBiG [1](https://mibig.secondarymetabolites.org/download). There are 1292 unique BGCs in the links. The same row in bgc_pfams and mol_fps files corresponds to a link.


* bgc_pfam.npy -- pfam vectors for BGCs, size=[1869,2377] 
* pfam_features.npy -- the protein family names used as features for BGCs, size=[2377]
* * bgcnames.txt -- names of the BGCs in the links (and bgc_pfam.npy), there are several names for one BGC "line" as the pfam representation of these BGCs was the same 
* string_9_2.npy -- string kernel with parameters g=9 (g-kmer length) and m=2 (n.o. missing) [2](https://github.com/QData/FastSK)  
* mol_fps.npy -- fingerprint vectors for metabolites, size=[1869,2457]. Fingerprint vectors were originally 6191 in length, zero columns deleted. Calculated using chemistry development kit; klekota roth (4860)+substructure (307)+standard fingerprint of (1024). 
* fold_stratified.npy -- stratified 10 fold CV for the predictions (most practical)
* bmfold.npy -- 10 fold CV, has all the same metabolites in the same fold (impractical), used for Bgc->Mol prediction
* mbfold.npy --  10 fold CV, has all the same BGCss in the same fold (impractical), used for Mol->Bgc prediction
* molbgcsmiles.csv -- csv-file includig the names of the BGCs, metabolites, SMILES, bioclass. Includes duplicate links. 
* link_list.npy -- link list of size=[1869,2], the first colum corresponds to a unique BGC and the second to a unique molecule (can be obtained from bgc_pfam and mol_fps)
* bgc_kernel_tanimoto.npy -- pre-calculated tanimoto kernel for BGCs 
* mol_kernel_tanimoto.npy -- pre-calculated tanimoto kernel for molecules
* Y_svm.npy -- matrix of links for SVM implementation

**prismdata**  
Includes data gathered and cleaned from [3](https://zenodo.org/record/3985982#.YbjNSJFByV4). The naming inconsistensies of the original files in [3](https://zenodo.org/record/3985982#.YbjNSJFByV4) is a bad situation, so some BGCs were dropped, leaving with 1151 BGCs.  

* bgc_pfam.npy -- pfam vectors for BGCs
* mol_fps.npy -- ecfp6 fingerprint vectors for natural products
* bio_class_of_bgcs.npy -- biosynthetic classes of the BGCs
* fold.npy -- 10 fold CV 
* molbgc_names.npy -- names of the BGCs/natural products in order
* prismtans.csv -- median tanimoto coefficients of the PRISM 4 predictions  

**running the experiments**  

output_prediction and prism print top-1, top-5, top-10 and top-20 accuracies.  

* output_prediction.py can be run with 
```console
you@you:~$ python output_prediction.py path/to/mibigdata {S|B|M} {MtoB|BtoM} {IOKR|MLP|CCA|Homals|MMR} 
```

where the first argument is the datapath, second indicates the type of fold (S recommended), third the prediction direction and fourth the chosen method. 

* prism.py can be run with 
```console
you@you:~$ python prism.py path/to/prismdata [y]
```
where the first argument is the datapath, and second can be chosen as 'y' for plots.  


svm_linkprediction.py requires rlscore  [4](https://github.com/aatapa/RLScore) and saves the predictions to a file instead of printing anything
* svm_linkprediction.py can be run with 
```console
you@you:~$ python svm_linkprediction.py path/to/mibigdata
```

**dependencies**  
The following packages were used  

numpy 1.21.4  
keras 2.7.0 (with tensorflow backend)  
sklearn 1.10.1  
pandas 1.1.3  
seaborn 0.23.2  
matplotlib 3.3.2  
rlscore


