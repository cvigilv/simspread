# README - Datasets

The `data` folder contains all the information ocncerning the datasets used in the study (i.e., drug-target interactions, drug-substructure annotations and drug-drug chemical similarity).
The directory structure has the following convention:

- The `DT` folder contains the drug-target interaction matrices for each dataset
- The `DS` folder contains the drug-substructure interaction matrices for each combination of dataset and molecular descriptor
- The `DD` folder contains the drug-drug similarity matrices for each combination of dataset and molecular descriptor/similarity measure
- The `SMILES` folder contains the SMILES for the drugs present in each dataset
- The `LUT` folder contains the look-up tables for drugs and targets of each dataset

The folder `data/yamanishi2008` contains all the information of the 4 gold-standard datasets proposed by Yamanishi, et al (2008): _Enzyme_ (denoted as `e` throughout the files), _Ion Channel_ (denoted as `ic` throughout the files), _GPCR_ (denoted as `gpcr` throughout the files) and  _Nuclear Receptor_ (denoted as `nr` throughout the files).
The folder `data/wu2017` contains all the information for the _Global_ dataset proposed by Wu, et al (2017).
The folder `data/chembl` contains all the information for the ChEMBL Clinical Candidates & Drugs dataset used in the time-split validation.
