import pickle

from workflows import Workflow
import pandas as pd


# Load the dataset from a CSV file
df = pd.read_csv('studd/data/insects.csv')

# Rename the columns of the dataframe to be sequential integers, 
# with the last column named "target".
column = []
for i in range(df.shape[1] - 1):
    column.append(i)
column.append("target")
df.columns = column

delta = 0.8

results = dict()
for i in range(1):
    
    y = df.target.values
    X = df.drop(['target'], axis=1)
   

    n_train_obs = 500
    W = n_train_obs
    
    predictions, detections, train_size, training_info, results_comp = \
        Workflow(X=X, y=y,delta=delta,window_size=W)
    
    ds_results = \
        dict(predictions=predictions,
             detections=detections,
             n_updates=train_size,
             data_size=len(y),
             training_info=training_info,
             results_comp=results_comp)
    
    results[f"experiment_{i}"] = ds_results
    
    with open('studd/data/studd_experiments.pkl', 'wb') as fp:
        pickle.dump(results, fp)
