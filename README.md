MIMIC
DC-AKI is a dual-channel CNN and GRU-based model for predicting AKI patients in the ICU
=====
## Introduction
DC-AKI employs dual-channel mechanism to extract features of temporal variables of EHR data at different granularity, comprehensively and accurately obtains effective information, and finally continuously predicting AKI risks for the patients. The CNN channel of the model calculates local interactions of the variables through CNN. While the GRU channel uses the GRU network and attention mechanism to obtain global interactions information among the variables. Finally, the model uses full connection layer to combine the representation vectors of the two channels, and uses Sigmoid
classifier for prediction.

## Requirements
pandas==1.1.5  
numpy==1.18.5  
matplotlib==3.3.0  
tensorflow==1.15.0  
sklearn==0.21.3 

## Training
1. Intsall [MIMIC-IV(Windows)](https://mimic.mit.edu/docs/iv/)  
2. [Generating the concepts in PostgreSQL (Windows)](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/concepts_postgres)
3. Run the scripts in the [data_preprocessing/SQL] folder sequentially
    * `inclusion_0721.sql`
    * `icustay_deatil.sql`
    * `labs_icu.sql`
    * `vitals_icu.sql`
    * `rrt_admission.sql` 
    * `sedative_0721.sql`
4. Run the scripts in the [data_preprocessing] folder sequentially  
    * `get_data_from_postgres_1.py`
    * `data_preprocessing.py`
    * `get_data_from_postgres_2.py`
5. Run the scripts in the [model] folder sequentially
    * `DC_AKI.py`
    * `LSTM_RNN_GRU.py`
eICU(https://github.com/BioMedBigDataCenter/DC-AKI/tree/main/eICU)

## Result
The ROC AUC, PR AUC, and F1's of the output models were compared as discriminatory criteria.