from multiprocessing import Pool

from mytrain.tools import config
from mytrain.trainer import wrap
from itertools import product

if __name__ == '__main__':
    gpus = [0, 3, 1, 2] * 7 + [0, 3] * 13
    # gpus = [0, 3, 2] * 7 + [0, 3] * 13
    process_pool = Pool(len(gpus))
    exp_config = config.load_specific_config("config.yaml")

    grid = {
        "sample_ig/alpha": [0.0, 0.2, 0.8, 1.0],
        "sample_ig/beta": [0.0, 0.2, 0.4, 1.0, 1.5],
        "sample_ig/threshold": [0, 0.40],
        "optimizer/weight_decay": [4.8],  # learning_rate
        "optimizer/lr": [0.09],  # learning_rate
    }

    repeat = 1
    exp_config['log_folder'] = 'xg'

    # overwrite the _key_
    drop_keys = [
        k
        for k in exp_config['_key_'].keys()
        if k not in grid
    ]
    for k in drop_keys:
        del exp_config['_key_'][k]
    for k in grid.keys():
        if k not in exp_config['_key_']:
            exp_config['_key_'][k] = None

    task = 0
    exp_config['grid_spec/total'] = repeat * len(list(product(*list(grid.values()))))
    for r in range(repeat):
        for i, setting in enumerate(product(*list(grid.values()))):
            print(setting)
            for idx, k in enumerate(grid.keys()):
                exp_config[k] = setting[idx]
            exp_config['cuda'] = str(gpus[task % len(gpus)])
            task += 1
            exp_config['grid_spec/current'] = task
            process_pool.apply_async(wrap, args=(exp_config.clone(),))
        exp_config.random_again()

    process_pool.close()
    process_pool.join()
