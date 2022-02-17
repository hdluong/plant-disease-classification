from ast import Param
from sklearn import metrics
import tensorflow as tf
import json
import os
import random

PATH = 'experiments\learning_rate'

def save_dict_to_json(d, json_path):
    with open(json_path, 'w') as f:
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)

def evaluate(dir, metrics):
    last_json_path = os.path.join(dir, "metrics_eval_best_weights.json")
    save_dict_to_json(metrics, last_json_path)

def model_fn():
    model = dict()
    model['accuracy'] = random.random()
    model['loss'] = random.random()
    return model

if __name__ == '__main__':
    learning_rates = [1e-4, 1e-3, 1e-2]
    for learning_rate in learning_rates:
        job_name = "learning_rate_{}".format(learning_rate)

        model_dir = os.path.join(PATH, job_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
        m = model_fn()
        evaluate(model_dir, m)