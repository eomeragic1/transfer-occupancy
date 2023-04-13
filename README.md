# Analysing Transfer Learning with Different Feature Sets for Occupancy Detection

Repository is organized as follows:
  - *data_preprocessing* folder contains Tableau dashboards containing visualizations about the datasets, and also contains Jupyter notebooks for preprocessing of the datasets
  - *fig* folder contains the figures
  - *model_checkpoints* contains source models for the three datasets, also serves as a folder which keeps temporary model files during training
  - *results* contains .csv files of the hyperparameter tuning and model trainind and evaluation
  - *src* contains all the source code 
 
First start by installing the necessary packages:

`$ pip install -r requirements.txt`

Download the data and put it in the folder above the repository base folder:

  - *data*
    - *ROBOD*
      - *combined_Room1.csv*
      - *combined_Room2.csv*
      - *...*
    - *ECO*
      - *Residency 01*
        - *01_sm_csv*
      - *...*
    - *HPDMobile*
      - *Household 01*
        - *H1_AUDIO*
        - *H1_ENVIRONMENTAL*
        - *H1_GROUNDTRUTH*
      - *...*
  - *transfer-occupancy*
    - *data_preprocessing*
    - *fig*
    - *...*
    
Run the data preprocessing pipelines, it will generate *combined_cleaned.csv* files in each of the datasets' folders. Now you can run *main.py* or *transfer_pipeline.py* for model training and testing.
