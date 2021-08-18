import os
import random
import sys
import traceback
import torch
from mytrain.tools import load_specific_config
from mytrain.tools.resolvers import model_resolver, dataset_resolver, optimizer_resolver
from mytrain.tools import StdoutTee, StderrTee
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from setproctitle import setproctitle
import numpy as np

__device = 'cpu'


def wrap(config):
    pid = os.getpid()
    config['pid'] = pid
    print(f"pid is {pid}")
    grid_spec = ""
    if "grid_spec" in config:
        total = config.get_or_default("grid_spec/total", -1)
        current = config.get_or_default("grid_spec/current", -1)
        print(f"grid spec: {current:02}/{total:02} on cuda:{config['cuda']}")
        grid_spec = f"{current:02}/{total:02}/{config['cuda']}#"

    if 'writer_path' not in config:
        folder = config['log_tag']
        if config["git/state"] == "Good":
            folder += '-%s' % (config['git']['hexsha'][:5])

        config['writer_path'] = os.path.join(config['log_folder'],
                                             folder,
                                             config.postfix()
                                             )
    if not os.path.exists(config['writer_path']):
        os.makedirs(config['writer_path'])

    setproctitle(grid_spec + config['writer_path'])

    if 'logfile' not in config or config['logfile']:
        logfile_std = os.path.join(config['writer_path'], "std.log")
        logfile_err = os.path.join(config['writer_path'], "err.log")
        with StdoutTee(logfile_std, buff=1), StderrTee(logfile_err, buff=1):
            try:
                main_run(config)
            except Exception as e:
                sys.stderr.write(str(type(e)) + ": " + str(e) + "\n")
                print(traceback.format_exc())
    else:
        main_run(config)
    return


def main_run(config):
    # set random seed
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = config['cuda']
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])

    summary = SummaryWriter(config['writer_path'])
    summary.add_text('config', config.__str__())
    print(f"output to {config['writer_path']}")

    print("dataset constructing...")
    dataset_cls = dataset_resolver.lookup(config['dataset/source'])
    dataset = dataset_cls(config)
    dataloader = DataLoader(
        dataset=dataset,
        **config['DataLoader']
    )
    print("dataset construct over.")

    # 模型定义
    model_cls = model_resolver.lookup(config['model'])
    # model = model_cls(**config.get_or_default("model_args", default={}))
    model = model_cls(user_num=dataset.num_user, item_num=dataset.num_item, config=config)
    print("loading model and assign GPU memory...")
    model = model.cuda()
    print("loaded over.")

    # 优化器
    optimizer_cls = optimizer_resolver.lookup(config['optimizer/class'])
    optimizer = optimizer_cls(model.parameters(), **config['optimizer/kwargs'])

    evaluator = Evaluator(config, summary, dataset)

    epoch_loop = range(config['epochs'])
    if config.get_or_default("train/epoch_tqdm", False):
        epoch_loop = tqdm(epoch_loop,
                          desc="train",
                          bar_format="{desc}{percentage:3.0f}%|{bar:10}{r_bar}",
                          )
    for epoch in epoch_loop:
        # evaluator.analysis(model, dataset, epoch)
        # 数据记录和精度验证
        # if (epoch + 1) % config['evaluator_time'] == 0:
        if epoch % config['evaluator_time'] == 0:
            evaluator.evaluate(model, epoch)
            if config.get_or_default("sample_ig/enable", False):
                evaluator.record_ig(dataset, epoch)
            if evaluator.should_stop():
                print("early stop...")
                break

        # 我们 propose 的模型训练
        epoch_loss = []
        loader = dataloader
        if config.get_or_default("train/batch_tqdm", True):
            loader = tqdm(loader,
                          desc=f'train  \tepoch: {epoch}/{config["epochs"]}',
                          bar_format="{desc}{percentage:3.0f}%|{bar:10}{r_bar}",
                          )
        for packs in loader:
            optimizer.zero_grad()
            model.train()
            user_raw, positive, negative = [p.cuda() for p in packs]
            batch_size = user_raw.shape[0]
            user = user_raw.unsqueeze(dim=1)
            weight = 1
            # if config.get_or_default("train/softw_enable", False):
            #     weight = softw[user_raw]

            loss = dist = distance(config, model, user, positive, negative, weight)

            # if config.get_or_default("train/softw_enable", False):
            #     loss = dist + torch.exp(-softw[user_raw])

            loss.sum().backward()
            optimizer.step()
            epoch_loss.append(loss.mean().item())

            if config.get_or_default("sample_ig/enable", False):
                un_loss = dist.cpu().detach().mean(dim=2)

                if config.get_or_default("sample_ig/post_un_loss", False):
                    with torch.no_grad():
                        model.eval()
                        un_loss = distance(config, model, user, positive, negative, weight,
                                           sigma=True).cpu().detach().mean(dim=2)

                # assert un_loss.shape == torch.Size([batch_size, config['sample_top_size']])
                dataset.update_un(user_raw.cpu().detach(), negative.cpu().detach(), un_loss)

        summary.add_scalar('Epoch/Loss', np.mean(epoch_loss), global_step=epoch)

    summary.close()


if __name__ == '__main__':
    cfg = load_specific_config("config.yaml")
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            print("additional argument: " + arg)
            if "=" in arg and len(arg.split("=")) == 2:
                k, v = arg.strip().split("=")
                if v.lower() in ['false', 'no', 'N', 'n']:
                    v = False
                elif v.lower() in ['true', 'yes', 'Y', 'y']:
                    v = True
                elif v.isdigit():
                    v = int(v)
                else:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
                cfg[k] = v
                continue
            print("arg warning : " + arg)
            exit(0)
    wrap(cfg)
