"""Train a cost model with a dataset."""

import argparse
import logging
import pickle
import random
import os
import ast
import torch
import numpy as np
import subprocess

import tvm
from tvm.auto_scheduler.utils import to_str_round
from tvm.auto_scheduler.cost_model import RandomModelInternal

from common import load_and_register_tasks, str2bool

from tvm.auto_scheduler.dataset import Dataset, LearningTask
from tvm.auto_scheduler.cost_model.xgb_model import XGBModelInternal
from tvm.auto_scheduler.cost_model.mlp_model import MLPModelInternal
from tvm.auto_scheduler.cost_model.lgbm_model import LGBModelInternal
from tvm.auto_scheduler.cost_model.tabnet_model import TabNetModelInternal
from tvm.auto_scheduler.cost_model.metric import (
    metric_rmse,
    metric_r_squared,
    metric_pairwise_comp_accuracy,
    metric_top_k_recall,
    metric_peak_score,
    metric_mape,
    random_mix,
)
from ast_parser import convert_python_to_ast, graph_from_tree_sitter
from pathlib import Path
import transformers, datasets
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
# from torch.util.data import Dataset, DataLoader


def evaluate_model(model, test_set):
    # make prediction
    prediction = model.predict(test_set)

    # compute weighted average of metrics over all tasks
    tasks = list(test_set.tasks())
    weights = [len(test_set.throughputs[t]) for t in tasks]
    print("Test set sizes:", weights)

    rmse_list = []
    r_sqaured_list = []
    pair_acc_list = []
    mape_list = []
    peak_score1_list = []
    peak_score5_list = []


    for task in tasks:
        preds = prediction[task]
        labels = test_set.throughputs[task]

        rmse_list.append(np.square(metric_rmse(preds, labels)))
        r_sqaured_list.append(metric_r_squared(preds, labels))
        pair_acc_list.append(metric_pairwise_comp_accuracy(preds, labels))
        mape_list.append(metric_mape(preds, labels))
        peak_score1_list.append(metric_peak_score(preds, labels, 1))
        peak_score5_list.append(metric_peak_score(preds, labels, 5))

    rmse = np.sqrt(np.average(rmse_list, weights=weights))
    r_sqaured = np.average(r_sqaured_list, weights=weights)
    pair_acc = np.average(pair_acc_list, weights=weights)
    mape = np.average(mape_list, weights=weights)
    peak_score1 = np.average(peak_score1_list, weights=weights)
    peak_score5 = np.average(peak_score5_list, weights=weights)

    eval_res = {
        "RMSE": rmse,
        "R^2": r_sqaured,
        "pairwise comparision accuracy": pair_acc,
        "mape": mape,
        "average peak score@1": peak_score1,
        "average peak score@5": peak_score5,
    }
    return eval_res


def make_model(name, use_gpu=False):
    """Make model according to a name"""
    if name == "xgb":
        return XGBModelInternal(use_gpu=use_gpu)
    elif name == "mlp":
        return MLPModelInternal()
    elif name == 'lgbm':
        return LGBModelInternal(use_gpu=use_gpu)
    elif name == 'tab':
        return TabNetModelInternal(use_gpu=use_gpu)
    elif name == "random":
        return RandomModelInternal()
    else:
        raise ValueError("Invalid model: " + name)
 

def train_zero_shot(dataset, train_ratio, model_names, split_scheme, use_gpu):
    # Split dataset
    # print(dataset.py_codes)
    if split_scheme == "within_task":
        train_set, test_set = dataset.random_split_within_task(train_ratio)
    elif split_scheme == "by_task":
        train_set, test_set = dataset.random_split_by_task(train_ratio)
    elif split_scheme == "by_target":
        train_set, test_set = dataset.random_split_by_target(train_ratio)
    else:
        raise ValueError("Invalid split scheme: " + split_scheme)

    print("Train set: %d. Task 0 = %s" % (len(train_set), train_set.tasks()[0]))
    # quit()
    if len(test_set) == 0:
        test_set = train_set
    print("Test set:  %d. Task 0 = %s" % (len(test_set), test_set.tasks()[0]))

    # Make models
    names = model_names.split("@")
    models = []
    for name in names:
        models.append(make_model(name, use_gpu))

    print("Model", models)
    eval_results = []
    for name, model in zip(names, models):
        # Train the model
        filename = name + ".pkl"
        model.fit_base(train_set, valid_set=test_set)
        print("Save model to %s" % filename)
        model.save(filename)

        # Evaluate the model
        eval_res = evaluate_model(model, test_set)
        print(name, to_str_round(eval_res))
        eval_results.append(eval_res)

    # Print evaluation results
    for i in range(len(models)):
        print("-" * 60)
        print("Model: %s" % names[i])
        for key, val in eval_results[i].items():
            print("%s: %.4f" % (key, val))

def train_BERT_tokenizer(paths):
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=True
    )

    tokenizer.train(
        files=paths,
        vocab_size=30_000,
        min_frequency=5,
        limit_alphabet=1000,
        wordpieces_prefix='##',
        special_tokens=['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
    )
    os.mkdir('./tvm_bert')
    tokenizer.save_model('./tvm_bert', 'tvm_bert_it')

def setup_AST_graphs(dataset, train_ratio, split_scheme):
    if split_scheme == "within_task":
        train_set, test_set = dataset.random_split_within_task(train_ratio)
    elif split_scheme == "by_task":
        train_set, test_set = dataset.random_split_by_task(train_ratio)
    elif split_scheme == "by_target":
        train_set, test_set = dataset.random_split_by_target(train_ratio)
    else:
        raise ValueError("Invalid split scheme: " + split_scheme)
    
    if len(test_set) == 0:
        test_set = train_set

    pickle_dict = {}
    idx = 0
    if not os.path.exists(os.path.join(os.getcwd(), "tokenizer_data")):
        os.mkdir(os.path.join(os.getcwd(), "tokenizer_data"))
        for task in train_set.features:
            for pc in train_set.py_codes[task]:
                fname = "tokenizer_data/file_" + str(idx) + ".txt"
                f = open(fname, "w")
                f.write(pc)
                f.close()
                print(fname)
                idx += 1
    else:
        from pathlib import Path
        if not os.path.exists(os.path.join(os.getcwd(), "tvm_bert")):
            paths = [str(x) for x in Path('./tokenizer_data').glob('**/*.txt')]
            train_BERT_tokenizer(paths)
        # else:
        #     tokenizer = BertTokenizer.from_pretrained('./tvm_bert/tvm_bert_it-vocab.txt', local_files_only=True)
        #     print(tokenizer.encode("hagu kahbi?"))
    if not os.path.exists(os.path.join(os.getcwd(), "output_dir")):
        os.mkdir(os.path.join(os.getcwd(), "output_dir"))
    idx = 0
    for task in train_set.features:
        hagu = False
        code_features = []
        source_nodes = []
        sink_nodes = []
        fname = "output_dir/" + str(idx) + ".pkl"

        if os.path.exists(fname):
            print("exists", idx)
            idx += 1
        else:
            for pc in train_set.py_codes[task]:
                code_features_, source_nodes_, sink_nodes_ = graph_from_tree_sitter(pc)
                code_features.append(code_features_)
                source_nodes.append(source_nodes_)
                sink_nodes.append(sink_nodes_)
                
            op_dict = {}
            # op_dict['features'] = train_set.features[task]
            op_dict['code_features'] = code_features
            op_dict['sources'] = source_nodes
            op_dict['sinks'] = sink_nodes
            op_dict['throughputs'] = train_set.throughputs[task]
            op_dict['min_latency'] = train_set.min_latency[task]

        # pickle_dict[task] = op_dict
        #fname = "output_dir/" + str(idx) + ".pkl"
            pickle.dump(op_dict, open(fname, "wb"))
            idx += 1
        # f = open(fname, "w")
        # f.write(pc)
        # f.close()
    
    #out_file = "train_" + split_scheme + ".pkl"
    #pickle.dump(pickle_dict, open(out_file, "wb"))

            # quit()
            # try:
            #     graph_from_tree_sitter(pc)
            #     # index, edge_index, types, features, edge_types, edge_in_out_indexs_s, edge_in_out_indexs_t, edge_in_out_head_tail = convert_python_to_ast(pc)
            #     # print(edge_in_out_head_tail, len(edge_in_out_head_tail))
            #     quit()
            # except:
            #     print("An error occured")
                # pc_lines = pc.split("\n")
                # for line_idx, line in enumerate(pc_lines):
                #     if line.startswith(" ="):
                #         pc_lines[line_idx] = "temp" + line
                # mod_pc = '\n'.join(pc_lines)
                # index, edge_index, types, features, edge_types, edge_in_out_indexs_s, edge_in_out_indexs_t, edge_in_out_head_tail = convert_python_to_ast(mod_pc)
                # reindent_cmd = "python3 reindent.py " + pc
                # # print("inside try")
                # try:
                #     result = subprocess.check_output(reindent_cmd, shell=True)
                # except subprocess.CalledProcessError as e:
                #     print(e.output)
                #     print(e.returncode)
                # output = result.decode('utf-8')
                # temp_ast = ast.parse(output)
                # print(temp_ast)
                # print(err)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", type=str, default=["dataset.pkl"])
    parser.add_argument("--models", type=str, default="xgb")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split-scheme",
        type=str,
        choices=["by_task", "within_task", "by_target"],
        default="within_task",
    )
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--use-gpu", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Whether to use GPU for xgb.")
    args = parser.parse_args()
    print("Arguments: %s" % str(args))

    # Setup random seed and logging
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    logging.basicConfig()
    logging.getLogger("auto_scheduler").setLevel(logging.DEBUG)

    print("Load all tasks...")
    load_and_register_tasks()

    print("Load dataset...")
    dataset = pickle.load(open(args.dataset[0], "rb"))
    # for i in range(1, len(args.dataset)):
    #     tmp_dataset = pickle.load(open(args.dataset[i], "rb"))
    #     dataset.update_from_dataset(tmp_dataset)

    if args.models == "gnn":
        setup_AST_graphs(dataset, args.train_ratio, args.split_scheme)
    quit()

    train_zero_shot(dataset, args.train_ratio, args.models, args.split_scheme, args.use_gpu)

