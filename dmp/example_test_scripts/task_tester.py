from dmp.task.growth_test.growth_test_task import GrowthTestTask
from dmp.worker import Worker
import pandas as pd
import tensorflow as tf
import pickle

def main():
    task = GrowthTestTask(seed=0,batch='growth_test_1',
                          dataset='201_pol',
                          input_activation='relu',
                          activation='relu',
                          optimizer={'class_name': 'adam', 'config': {'learning_rate': 0.001}},
                          shape='rectangle',
                          size=256,
                          depth=3,
                          test_split=0.3,
                          test_split_method='shuffled_train_val_test_split',
                          run_config= {'shuffle': True,'epochs': 50,'batch_size': 256,'verbose': 0,},
                          label_noise=0.0,
                          kernel_regularizer=None,
                          bias_regularizer=None,
                          activity_regularizer=None,
                          early_stopping=None,
                          save_every_epochs=None,
                          growth_trigger='EarlyStopping',
                          growth_trigger_params={'patience':10},
                          growth_method='grow_network',
                          growth_method_params=None,
                          growth_scale=2.0,
                          max_size=1224,
                          val_split=0.1,)
    
    # print(task.max_size,task.asdf,task.val_split)
    worker = Worker(None,None,tf.distribute.get_strategy(),{})
    
    result = task(worker)
    # print(result)
    # pd.DataFrame(result).to_csv('growth_test_dataframe.csv')
    with open('growth_test_results.pkl','wb') as f:
        pickle.dump(result,f)

if __name__ == '__main__':
    main() 