from tabulate import tabulate
import os
import json

def metrics_to_table(metrics):
    headers = metrics[list(metrics.keys())[0]].keys()
    table = [[subdir] + [values[h] for h in headers] for subdir, values in metrics.items()]
    res = tabulate(table, headers, tablefmt='pipe')

    return res

def aggregate_metrics(parent_dir, metrics):
    metrics_file = os.path.join(parent_dir, 'metrics_eval_best_weights.json')
    if os.path.isfile(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics[parent_dir] = json.load(f)

    for subdir in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, subdir)):
            continue
        else:
            aggregate_metrics(os.path.join(parent_dir, subdir), metrics)

metrics = dict()
aggregate_metrics("experiments\learning_rate", metrics)
table = metrics_to_table(metrics)

print(table)