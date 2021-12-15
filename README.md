An overall illustration of the topic; relationships between data and the prediction task. 

![alt txt](https://github.com/minjoks/masters_thesis/blob/main/dataview.png?raw=true)




**mibigdata**  
Includes data for 1869 unique BGC-metabolite links, gathered from MIBiG [1](https://mibig.secondarymetabolites.org/download). There are 1292 unique BGCs in the links. The same row in bgc_pfams and mol_fps files corresponds to a link.


* bgcnames.txt -- names of the BGCs in the links (and bgc_pfam.npy), there are several names for one BGC as the pfam representation of these BGCs was the same 
* bgc_pfam.npy -- pfam vectors for BGCs, size=[1869,2377] 
* pfam_features.npy -- the protein family names used as features for BGCs, size=[2377]
* string_9_2.npy -- string kernel with parameters g=9 (g-kmer length) and m=2 (n.o. missing) [2](https://github.com/QData/FastSK)  
* mol_fps.npy -- fingerprint vectors for metabolites, size=[1869,2457]
* fold_stratified.npy -- stratified 10 fold CV for the predictions (most practical)
* bmfold.npy -- 10 fold CV, has all the same BGCs in the same fold (impractical)
* mbfold.npy --  10 fold CV, has all the same metabolites in the same fold (impractical)
* molbgcsmiles.csv -- csv-file includig the names of the BGCs, metabolites, SMILES, bioclass. Includes duplicate links. 
* link_list.npy -- link list of size=[1869,2], the first colum corresponds to a unique BGC and the second to a unique molecule

**prismdata**  
Includes data gathered and cleaned from [3](https://zenodo.org/record/3985982#.YbjNSJFByV4). The naming inconsistensies of the original files in [3](https://zenodo.org/record/3985982#.YbjNSJFByV4) is a s**show, so some BGCs were dropped, leaving with 1151 BGCs.  

* bgc_pfam.npy -- pfam vectors for BGCs
* mol_fps.npy -- ecfp6 fingerprint vectors for natural products
* bio_class_of_bgcs.npy -- biosynthetic classes of the BGCs
* fold.npy -- 10 fold CV 
* molbgc_names.npy -- names of the BGCs/natural products in order
* prismtans.csv -- median tanimoto coefficients of the PRISM 4 predictions  

**running the experiments**  

All of the experiments print top-1, top-5, top-10 and top-20 accuracies.  

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

**dependencies**  
The following packages were used  

numpy 1.21.4  
keras 2.7.0 (with tensorflow backend)  
sklearn 1.10.1  
pandas 1.1.3  
seaborn 0.23.2  
matplotlib 3.3.2  



