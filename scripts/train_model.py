"""Train a cost model with a dataset."""

import argparse, logging, pickle, random, os, ast, torch, time, copy
import numpy as np
import subprocess
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import tvm
from tvm.auto_scheduler.utils import to_str_round
from tvm.auto_scheduler.cost_model import RandomModelInternal

from common import load_and_register_tasks, str2bool

from tvm.auto_scheduler.dataset import Dataset, LearningTask
from tvm.auto_scheduler.cost_model.xgb_model import XGBModelInternal
from tvm.auto_scheduler.cost_model.mlp_model import MLPModelInternal
from tvm.auto_scheduler.cost_model.lgbm_model import LGBModelInternal
from tvm.auto_scheduler.cost_model.tabnet_model import TabNetModelInternal
from tvm.auto_scheduler.cost_model.gnn_model import GNNModelInternal
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

def evaluate_GNN_model(model):
    # make prediction
    tasks = []
    weights = []
    rmse_list = []
    r_sqaured_list = []
    pair_acc_list = []
    mape_list = []
    peak_score1_list = []
    peak_score5_list = []

    for f in os.listdir("test_output_dir"):
        with open("test_output_dir/" + f, "rb") as pcontents:
            data = pickle.load(pcontents)
            start_time = time.time()
            weights.append(len(data['throughputs']))
            test_list = []
            for idx, val in enumerate(data['code_features']):
                code_features_ = copy.deepcopy(data['code_features'][idx])
                throughputs_ = copy.deepcopy(data['throughputs'][idx])
                sources_ = copy.deepcopy(data['sources'][idx])
                sinks_= copy.deepcopy(data['sinks'][idx])
                min_latency = copy.deepcopy(data['min_latency'])

                inp_data = process_asts(code_features_, throughputs_, sources_, sinks_)
                # inp_data = Data(x=node_feats_, edge_index=edge_indices_, y=torch.tensor(throughputs_, dtype=torch.float))
                test_list.append(inp_data)
            end_time = time.time()
            print("Time", end_time-start_time)
            preds, labels = model.predict(test_list)
            # print(preds)
            # print(labels)
            rmse_list.append(np.square(metric_rmse(preds, labels)))
            r_sqaured_list.append(metric_r_squared(preds, labels))
            pair_acc_list.append(metric_pairwise_comp_accuracy(preds, labels))
            mape_list.append(metric_mape(preds, labels))
            peak_score1_list.append(metric_peak_score(preds, labels, 1))
            peak_score5_list.append(metric_peak_score(preds, labels, 5))
    print("Test set sizes:", weights)

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
    elif name == "gnn":
        return GNNModelInternal(loss_type='rmse')
    else:
        raise ValueError("Invalid model: " + name)
 

def train_zero_shot(dataset, train_ratio, model_names, split_scheme, use_gpu):
    # Split dataset
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

# class Net(torch.nn.Module):
#   def __init__(self):
#     super(Net, self).__init__()
#     # self.embeddings = nn.Embedding(vocab_size + 1, embedding_dim)
#     self.conv1 = GCNConv(16, 64)
#     self.conv2 = GCNConv(64, 32)
#     self.conv3 = GCNConv(32, 12)
#     self.conv4 = GCNConv(12, 6)
#     self.output = torch.nn.Linear(6, 1)
#     self.dropout1 = torch.nn.Dropout(0.1)

#     # nn.init.xavier_uniform_(self.embeddings.weight)
#     torch.nn.init.xavier_uniform_(self.output.weight)
#     torch.nn.init.zeros_(self.output.bias)

#   def forward(self, data):
#     x, edge_index, batch = data.x, data.edge_index, data.batch
#     g = self.conv1(x, edge_index)
#     g = torch.nn.functional.relu(g)
#     g = self.conv2(g, edge_index)
#     g = torch.nn.functional.relu(g)
#     g = self.conv3(g, edge_index)
#     g = torch.nn.functional.relu(g)
#     g = self.conv4(g, edge_index)
#     g = gmp(g, batch)
#     z = self.output(g)
#     z = torch.nn.functional.relu(z)
#     return z

# model = Net().to('cpu')
# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# num_epoch = 5
# batch_size = 32
# list_epoch_lost = []
# model.train()

def process_asts(code_features_, throughputs_, sources_, sinks_):
    node_feats_ = []
    start_idx = 0
    node_idx_map = {}
    for node in code_features_:
        if len(node['text']) < 16:
            num_to_append = 16 - len(node['text'])
            temp_feat = node['text']
            temp_feat.extend([0]*num_to_append)
            # node['text'].extend([0]*num_to_append)
            node_feats_.append(temp_feat)
        else:
            temp_feat = node['text']
            node_feats_.append(temp_feat[:16]) 
        # node_feats_.append(node['text'])
    # updated_sources_ = []
    for i, source in enumerate(sources_):
        if source not in node_idx_map:
            node_idx_map[source] = start_idx
            sources_[i] = start_idx
            start_idx += 1
        else:
            sources_[i] = node_idx_map[source]
    
    # print(sources_)
    for i, sink in enumerate(sinks_):
        if sink in node_idx_map:
            sinks_[i] = node_idx_map[sink]
        else:
            node_idx_map[sink] = start_idx
            sinks_[i] = start_idx
            start_idx += 1

    node_feats_ = torch.tensor(node_feats_, dtype=torch.long)
    edge_indices_ = torch.tensor([sources_, sinks_], dtype=torch.long)
    inp_data = Data(x=node_feats_, edge_index=edge_indices_, y=torch.tensor(throughputs_, dtype=torch.float))
    return inp_data

def train_zero_shot_GNN(use_gpu=False):
    train_list = []
    if not os.path.exists("train_loader.pt"):
        for f in os.listdir("train_output_dir"):
            with open("train_output_dir/" + f, "rb") as pcontents:
                data = pickle.load(pcontents)
                start_time = time.time()
                
                for idx, val in enumerate(data['code_features']):
                    code_features_ = copy.deepcopy(data['code_features'][idx])
                    throughputs_ = copy.deepcopy(data['throughputs'][idx])
                    sources_ = copy.deepcopy(data['sources'][idx])
                    sinks_= copy.deepcopy(data['sinks'][idx])
                    min_latency = copy.deepcopy(data['min_latency'])

                    inp_data = process_asts(code_features_, throughputs_, sources_, sinks_)
                    train_list.append(inp_data)
                end_time = time.time()
                print("Time", end_time-start_time)
            # print(len(data['throughputs']), len(data['code_features']), len(data['sources']), len(data['sinks']), data['min_latency'])
        # pcontents = pickle.load(open("output_dir_100/" + f, "rb"))
    
    # test_list = []
    # for f in os.listdir("test_output_dir"):
    #     with open("test_output_dir/" + f, "rb") as pcontents:
    #         data = pickle.load(pcontents)
    #         start_time = time.time()
            
    #         for idx, val in enumerate(data['code_features']):
    #             code_features_ = copy.deepcopy(data['code_features'][idx])
    #             throughputs_ = copy.deepcopy(data['throughputs'][idx])
    #             sources_ = copy.deepcopy(data['sources'][idx])
    #             sinks_= copy.deepcopy(data['sinks'][idx])
    #             min_latency = copy.deepcopy(data['min_latency'])

    #             inp_data = process_asts(code_features_, throughputs_, sources_, sinks_)
    #             # inp_data = Data(x=node_feats_, edge_index=edge_indices_, y=torch.tensor(throughputs_, dtype=torch.float))
    #             test_list.append(inp_data)
    #         end_time = time.time()
    #         print("Time", end_time-start_time)

    model = make_model("gnn", use_gpu)
    filename = "gnn.pkl"
    model.fit_base(train_list)
    print("Save model to %s" % filename)
    model.save(filename)

    eval_res = evaluate_GNN_model(model)
    for key, val in eval_res.items():
        print("%s: %.4f" % (key, val))
    # print('gnn', to_str_round(eval_res))
    # eval_results.append(eval_res)

    # Print evaluation results
    # for i in range(len(models)):
    #     print("-" * 60)
    #     print("Model: %s" % names[i])
    #     for key, val in eval_results[i].items():
    #         print("%s: %.4f" % (key, val))
    # ****************** keep code below for reference
    # loader = DataLoader(data_list, batch_size=32)
    # print(loader)
    # for epoch in range(num_epoch):
    #     print(f'Epoch: {epoch}')
    #     epoch_loss = 0
    #     correct = 0
    #     for batch in loader:
    #         optimizer.zero_grad()
    #         batch.batch.to('cpu')
    #         batch.edge_index.to('cpu')
    #         batch.x.to('cpu')
    #         batch.y = batch.y.reshape(-1,1)
    #         batch.y.to('cpu')
    #         y_pred = model(batch.to('cpu'))
    #         loss = criterion(y_pred, batch['y'].type(torch.FloatTensor).to('cpu'))
    #         loss.backward()
    #         optimizer.step()
    #         epoch_loss += loss.item()
    #     print('epoch loss ', epoch_loss)
    #     list_epoch_lost.append(epoch_loss)

    # Make models
    # names = model_names.split("@")
    # models = []
    # for name in names:
    #     models.append(make_model(name, use_gpu))

    # print("Model", models)
    # eval_results = []
    # for name, model in zip(names, models):
    #     # Train the model
    #     filename = name + ".pkl"
    #     model.fit_base(train_set, valid_set=test_set)
    #     print("Save model to %s" % filename)
    #     model.save(filename)

    #     # Evaluate the model
    #     eval_res = evaluate_model(model, test_set)
    #     print(name, to_str_round(eval_res))
    #     eval_results.append(eval_res)

    # # Print evaluation results
    # for i in range(len(models)):
    #     print("-" * 60)
    #     print("Model: %s" % names[i])
    #     for key, val in eval_results[i].items():
    #         print("%s: %.4f" % (key, val))

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
    if not os.path.exists(os.path.join(os.getcwd(), "train_output_dir")):
        os.mkdir(os.path.join(os.getcwd(), "train_output_dir"))
    if not os.path.exists(os.path.join(os.getcwd(), "test_output_dir")):
        os.mkdir(os.path.join(os.getcwd(), "test_output_dir"))
    idx = 0
    for task in train_set.features:
        hagu = False
        code_features = []
        source_nodes = []
        sink_nodes = []
        fname = "train_output_dir/" + str(idx) + ".pkl"

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

    idx = 0
    for task in test_set.features:
        hagu = False
        code_features = []
        source_nodes = []
        sink_nodes = []
        fname = "test_output_dir/" + str(idx) + ".pkl"

        if os.path.exists(fname):
            print("exists", idx)
            idx += 1
        else:
            for pc in test_set.py_codes[task]:
                code_features_, source_nodes_, sink_nodes_ = graph_from_tree_sitter(pc)
                code_features.append(code_features_)
                source_nodes.append(source_nodes_)
                sink_nodes.append(sink_nodes_)
                
            op_dict = {}
            # op_dict['features'] = train_set.features[task]
            op_dict['code_features'] = code_features
            op_dict['sources'] = source_nodes
            op_dict['sinks'] = sink_nodes
            op_dict['throughputs'] = test_set.throughputs[task]
            op_dict['min_latency'] = test_set.min_latency[task]

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
    # evaluate_GNN_model()

    print("Load all tasks...")
    load_and_register_tasks()

    print("Load dataset...")
    dataset = pickle.load(open(args.dataset[0], "rb"))
    if os.path.getsize(args.dataset[0]) > 0:
        with open(args.dataset[0], 'rb') as my_file:
            unpickler = pickle.Unpickler(my_file)
            dataset = unpickler.load()
            # print(len(dataset))
    else:
        print('The file is empty')
        quit()
    for i in range(1, len(args.dataset)):
        tmp_dataset = pickle.load(open(args.dataset[i], "rb"))
        dataset.update_from_dataset(tmp_dataset)

    if args.models == "gnn":
        if not os.path.exists("train_output_dir"):
            setup_AST_graphs(dataset, args.train_ratio, args.split_scheme)
        train_zero_shot_GNN()
    else:
        train_zero_shot(dataset, args.train_ratio, args.models, args.split_scheme, args.use_gpu)

