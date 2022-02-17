"""Peform hyperparemeters search"""

import argparse
import os
from subprocess import check_call
import sys

from utils.set_params import Params


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('-pd', '--parent_dir', default='experiments/learning_rate', required=True,
                    help="Directory containing params.json")
parser.add_argument('-d', '--data_dir', default='datasets/train', required=True,
                    help="Directory containing the dataset")
parser.add_argument('-l', '--label_encode', required=True,
	help="path to output label one-hot encoder decoder")


def launch_training_job(parent_dir, data_dir, label_encode_dir, job_name, params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name

    Args:
        parent_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "{python} train_simplecnn.py --model {model_dir} --dataset {data_dir} --label_encode {label_encode_dir}".format(python=PYTHON,
            model_dir=model_dir, data_dir=data_dir, label_encode_dir=label_encode_dir)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Perform hypersearch over one parameter
    learning_rates = [1e-4, 1e-3, 1e-2]

    for learning_rate in learning_rates:
        # Modify the relevant parameter in params
        params.learning_rate = learning_rate

        # Launch job (name has to be unique)
        job_name = "learning_rate_{}".format(learning_rate)
        launch_training_job(args.parent_dir, args.data_dir, args.label_encode, job_name, params)
