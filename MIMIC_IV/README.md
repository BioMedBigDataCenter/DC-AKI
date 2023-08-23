MIMIC
=====
## Data
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

## Model
1. Run the scripts in the [model] folder sequentially
    * `DC_AKI.py`
    * `LSTM_RNN_GRU.py`
