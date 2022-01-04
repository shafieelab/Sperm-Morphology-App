import argparse
import copy
import csv
import os
import os.path as osp
import statistics
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import helper_utils.network as network
# import helper_utils.pre_process as prep
import helper_utils.pre_process_old as prep
import yaml

from torch.utils.data import DataLoader
import helper_utils.lr_schedule as lr_schedule
from helper_utils.data_list_m import ImageList

import argparse

from helper_utils.tools import testing_sperm_slides, validation_loss, calc_transfer_loss, Entropy, print_msg


def data_setup(config):
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params_source'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params_target'])
    prep_dict["test"] = prep.image_test(**config["prep"]['params_target'])
    prep_dict["valid_source"] = prep.image_test(**config["prep"]['params_source'])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    test_bs = data_config["test"]["batch_size"]

    dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(),
                              transform=prep_dict["test"], labelled=data_config["test"]["labelled"])
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                      shuffle=False, num_workers=4)

    return dset_loaders


def network_setup(config):
    class_num = config["network"]["params"]["class_num"]

    ## set base network
    net_config = config["network"]
    if config['dataset'] == 'malaria' and config['trained_model_path']:
        base_network = torch.load(config['trained_model_path'], map_location=torch.device('cpu'))[0]

        layers = [name.replace('.weight', '').replace('.bias', '') for name, _ in base_network.named_parameters()]
        layers_names = OrderedDict.fromkeys(layers)
        layers_freeze = list(layers_names)[len(list(layers_names)) - config['no_of_layers_freeze']:]

        for name, param in base_network.named_parameters():
            if not name.replace('.weight', '').replace('.bias', '') in layers_freeze:
                param.requires_grad = False


    else:
        base_network = net_config["name"](**net_config["params"])
        base_network = base_network.cuda()
    ad_net = network.AdversarialNetwork(base_network.output_num() * class_num, 1024)
    ad_net = ad_net.cuda()
    parameter_list = base_network.get_parameters() + ad_net.get_parameters()

    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i in gpus])
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])

    return base_network, ad_net, schedule_param, lr_scheduler, optimizer


def test(config, dset_loaders, model_path_for_testing=None):
    if model_path_for_testing:
        model = torch.load(model_path_for_testing,map_location=torch.device('cpu'))
    else:

        model = torch.load(osp.join(config["model_path"], "best_model.pth.tar"))

    # val_info = validation_loss(dset_loaders, model, dset=config['dataset'],
    #                            num_classes=config["network"]["params"]["class_num"],
    #                            logs_path=config['logs_path'], is_training=config['is_training'])

    # if config["network"]["params"]["class_num"] == 5 and 'embryo' in config["dataset"]:
    #     print("Final Model ", "| Val loss: ", val_info['val_loss'], "| Val Accuracy: ",
    #           val_info['val_accuracy'], "| 2 Class Val Accuracy: ", val_info['val_acc_2_class'])
    # else:
    #     print("Final Model ", "| Val loss: ", val_info['val_loss'], "| Val Accuracy: ",
    #           val_info['val_accuracy'])

    if config["dataset"] == "sperm" and not config["data"]["test"]["labelled"]:
        output_csv_path = testing_sperm_slides(dset_loaders, model, config['logs_path'], None,
                             config["network"]["params"]["class_num"])

        return output_csv_path
    else:
        test_info = validation_loss(dset_loaders, model, dset=config['dataset'], data_name='test',
                                    num_classes=config["network"]["params"]["class_num"],
                                    logs_path=config['logs_path'], is_training=config['is_training'])

        if config["network"]["params"]["class_num"] == 5 and 'embryo' in config["dataset"]:
            print("Final Model ", "| Test loss: ", test_info['val_loss'], "| Test Accuracy: ",
                  test_info['val_accuracy'], "| 2 Class Test Accuracy: ", test_info['val_acc_2_class'])
        else:
            print("Final Model ", "| Test loss: ", test_info['val_loss'], "| Test Accuracy: ",
                  test_info['val_accuracy'])


def parge_args(project_root,run_id,img_paths_file):
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, choices=['train', 'test'])

    parser.add_argument('--seed', type=int)
    parser.add_argument('--dset', type=str, help="The dataset or source dataset used")
    parser.add_argument('--num', type=str)
    parser.add_argument('--gpu_id', type=str, nargs='?', default='2', help="device id to run")

    parser.add_argument('--lr', type=float)
    parser.add_argument('--arch', type=str)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--power', type=float)
    parser.add_argument('--momentum', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--nesterov', type=float)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--batch_size_test', type=int)
    parser.add_argument('--use_bottleneck', type=bool)
    parser.add_argument('--bottleneck_dim', type=int)

    parser.add_argument('--new_cls', type=bool)
    parser.add_argument('--no_of_classes', type=int)
    parser.add_argument('--image_size', type=int)
    parser.add_argument('--crop_size', type=int)

    parser.add_argument('--num_iterations', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--test_interval', type=int)
    parser.add_argument('--snapshot_interval', type=int)

    parser.add_argument('--trained_model_path', type=str)
    parser.add_argument('--no_of_layers_freeze', type=int)

    parser.add_argument('--s_dset', type=str)
    parser.add_argument('--t_dset', type=str)

    parser.add_argument('--test_dset_txt', type=str)
    parser.add_argument('--s_dset_txt', type=str)
    parser.add_argument('--sv_dset_txt', type=str)
    parser.add_argument('--t_dset_txt', type=str)
    parser.add_argument('--target_labelled', type=str, default='True', choices=['True', 'False', 'true', 'false'], )
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--sperm_patient_data_clinicians_annotations', type=str)

    parser.set_defaults(
        mode="test",
        num="1",
        seed=0,
        gpu_id="1",
        dset="sperm",
        # t_dset_txt='data/demo_filtered_dset.txt',
        t_dset_txt=img_paths_file,

        s_dset="a_sd4",
        t_dset="a_sd1_f",

        lr=0.0002,
        # arch="ResNet50",
        arch="Xception",
        gamma=0.0001,
        power=0.75,
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=True,
        optimizer="SGD",
        batch_size=64,
        batch_size_test=512,
        use_bottleneck=False,
        bottleneck_dim=256,
        new_cls=True,
        no_of_classes=2,
        image_size=170,
        crop_size=170,
        # trained_model_path= "../models/best_sd4_human_model.pth.tar",
        # trained_model_path= "../models/sperm/ResNet50/1_train_h_sd4_to_a_sd3/best_model.pth.tar",
        # trained_model_path= "../models/sperm/Xception/2_train_h_sd4_to_a_sd1_f/best_model.pth.tar",

        trained_model_path="/var/www/html/Sperm_Morphology_App/core/best_model.pth.tar",
        # trained_model_path= None,
        no_of_layers_freeze=13,

        num_iterations=10000,
        patience=2000,
        test_interval=2000,
        snapshot_interval=10000,
        target_labelled="False",
        output_dir= project_root+run_id+"/",
        # sperm_patient_data_clinicians_annotations=''
    )


    args = parser.parse_args()
    return args


def set_deterministic_settings(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_md_nets(project_root, run_id,img_paths_file):
    ####################################
    # Default Project Folders#
    ####################################

    # project_root = "../../"
    # data_root = project_root + "data/"
    # models_root = project_root + "models/"

    now = datetime.now()
    timestamp = now.strftime("%d/%m/%Y %H:%M:%S")
    timestamp = timestamp.replace("/", "_").replace(" ", "_").replace(":", "_").replace(".", "_")

    args = parge_args(project_root, run_id,img_paths_file)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    set_deterministic_settings(seed=args.seed)

    dataset = args.dset

    log_output_dir_root = args.output_dir + '/logs/' + dataset + '/'
    config = {}
    no_of_classes = args.no_of_classes

    trial_number = args.num + "_" + args.mode + "_" + args.s_dset + "_to_" + args.t_dset

    ####################################
    # Dataset Locations Setup #
    ####################################

    test_input = {'path': args.t_dset_txt, 'labelled': args.target_labelled.lower() == 'true'}



    model_path_for_testing = args.trained_model_path

    config['timestamp'] = timestamp
    config['trial_number'] = trial_number
    config["gpu"] = args.gpu_id
    config["num_iterations"] = args.num_iterations
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["patience"] = args.patience

    config["num_iterations"] = 0
    best_itr = "testing"
    print("Testing:")
    config["best_itr"] = "testing"


    log_output_path = log_output_dir_root + args.arch + '/' + trial_number + '/'

    config["logs_path"] = log_output_path
    if not os.path.exists(config["logs_path"]):
        os.makedirs(config["logs_path"])



    # config["out_file"] = open(osp.join(config["logs_path"], "log.txt"), "w")
    resize_size = args.image_size

    config["prep"] = {'params_source': {"resize_size": resize_size, "crop_size": args.crop_size, "dset": dataset},
                      'params_target': {"resize_size": resize_size, "crop_size": args.crop_size, "dset": dataset}}

    config["loss"] = {"trade_off": 1.0}
    config["trained_model_path"] = args.trained_model_path
    config['no_of_layers_freeze'] = args.no_of_layers_freeze

    if "Xception" in args.arch:
        config["network"] = \
            {"name": network.XceptionFc,
             "params":
                 {
                     "use_bottleneck": args.use_bottleneck,
                     "bottleneck_dim": args.bottleneck_dim,
                     "new_cls": args.new_cls}}
    elif "ResNet50" in args.arch:
        config["network"] = {"name": network.ResNetFc,
                             "params":
                                 {"resnet_name": args.arch,
                                  "use_bottleneck": args.use_bottleneck,
                                  "bottleneck_dim": args.bottleneck_dim,
                                  "new_cls": args.new_cls}}

    elif "Inception" in args.arch:
        config["network"] = {"name": network.Inception3Fc,
                             "params":
                                 {"use_bottleneck": args.use_bottleneck,
                                  "bottleneck_dim": args.bottleneck_dim,
                                  "new_cls": args.new_cls}}

    if args.optimizer == "SGD":

        config["optimizer"] = {"type": optim.SGD, "optim_params": {'lr': args.lr, "momentum": args.momentum,
                                                                   "weight_decay": args.weight_decay,
                                                                   "nesterov": args.nesterov},
                               "lr_type": "inv",
                               "lr_param": {"lr": args.lr, "gamma": args.gamma, "power": args.power}}

    elif args.optimizer == "Adam":
        config["optimizer"] = {"type": optim.Adam, "optim_params": {'lr': args.lr,
                                                                    "weight_decay": args.weight_decay},
                               "lr_type": "inv",
                               "lr_param": {"lr": args.lr, "gamma": args.gamma, "power": args.power}}

    config["dataset"] = dataset
    config["data"] = {"test": {"list_path": test_input['path'], "batch_size": args.batch_size_test,
                               "labelled": test_input['labelled']}}
    config["optimizer"]["lr_param"]["lr"] = args.lr
    config["network"]["params"]["class_num"] = no_of_classes

    # config["out_file"].write(str(config))
    # config["out_file"].flush()
    # print("source_path", source_input)
    # print("target_path", target_input)
    # print('GPU', os.environ["CUDA_VISIBLE_DEVICES"], config["gpu"])




    ####################################
    # Dump arguments #
    ####################################
    with open(config["logs_path"] + "args.yml", "w") as f:
        yaml.dump(args, f)

    dset_loaders = data_setup(config)


    print()
    print("=" * 50)
    print(" " * 15, "Testing Started")
    print("=" * 50)
    print()

    output_csv_path = test(config, dset_loaders, model_path_for_testing=model_path_for_testing)

    return output_csv_path


if __name__ == "__main__":
    run_md_nets(project_root="../data/", run_id="1.2",img_paths_file="")
