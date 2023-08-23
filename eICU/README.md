eICU
=====
## Data
1. Intsall [eICU](https://eicu-crd.mit.edu/gettingstarted/access/)  
2. [Generating the concepts in PostgreSQL (Windows)](https://kgithub.com/MIT-LCP/eicu-code)
3. Run the scripts in the [data_preprocessing] folder sequentially（） 
    * `aki_atage.ipynb`
    * `get_data_from_postgres_1.ipynb`
    * `data_preprocessing.ipynb`
    * `get_data_from_postgres_2.ipynb`

## Model
1. Run the scripts in the [model] folder sequentially
    * `DC_AKI.ipynb`（A dual-channel model based on CNN and GRU）
    * `LSTM_RNN_GRU.ipynb`