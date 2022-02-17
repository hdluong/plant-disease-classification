import json
import tensorflow as tf

def evaluate_sess(sess, model_spec, num_steps, writer=None, params=None):

    update_metrics = model_spec['update_metrics']
    eval_metrics = model_spec['metrics']
    global_step = tf.train.get_global_step()

    sess.run(model_spec['iterator_init_op'])
    sess.run(model_spec['metrics_init_op'])

    for _ in range(num_steps):
        sess.run(update_metrics)

    metrics_values = {k: v[0] for k, v in eval_metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())

    if writer is not None:
        global_step_val = sess.run(global_step)
        for tag, val in metrics_val.items():
            summ = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
            writer.add_summary(summ, global_step_val)

    return metrics_val

def save_dict_to_json(d, json_path):
    with open(json_path, 'w') as f:
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)

