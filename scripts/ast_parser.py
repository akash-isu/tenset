# import ast
# from pprint import pprint

# code = """
# tensor_ax0, tensor_ax1, tensor_ax2, tensor_ax3, tensor_rv0, tensor_rv1 = tuple(tensor.op.axis) + tuple(tensor.op.reduce_axis)
# tensor_ax0, tensor_ax1, tensor_ax2, tensor_ax3 = tuple(tensor.op.axis) + tuple(tensor.op.reduce_axis)
# """

# tree=ast.parse(code)
# pprint(ast.dump(tree))
#!/usr/bin/python3

# cmds to generate ast in json format
# python3 ast_parser.py "import ast"
# python3 ast_parser.py --file <name of python file>

import ast
import graphviz as gv
import subprocess
import numbers
import re
from uuid import uuid4 as uuid
import optparse
import sys, json
import pydotplus
import networkx
from tree_sitter import Language, Parser
import transformers, datasets
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer

def main(args):
    parser = optparse.OptionParser(usage="astvisualizer.py [options] [string]")
    parser.add_option("-f", "--file", action="store",
                      help="Read a code snippet from the specified file")
    parser.add_option("-l", "--label", action="store",
                      help="The label for the visualization")


    options, args = parser.parse_args(args)
    if options.file:
        with open(options.file) as instream:
            code = instream.read()
        label = options.file
    elif len(args) == 2:
        code = args[1]
        label = "<code read from command line parameter>"
    else:
        print("Expecting Python code on stdin...")
        code = sys.stdin.read()
        label = "<code read from stdin>"
    if options.label:
        label = options.label

    print(code)
    code = """
tensor_ax0, tensor_ax1, tensor_ax2, tensor_ax3, tensor_rv0, tensor_rv1 = tuple(tensor.op.axis) + tuple(tensor.op.reduce_axis)
tensor_ax0, tensor_ax1, tensor_ax2, tensor_ax3 = tuple(tensor.op.axis) + tuple(tensor.op.reduce_axis)
    """

    code_ast = ast.parse(code)
    print("hagu", label)
    transformed_ast = transform_ast(code_ast)
    # print(transformed_ast)

    renderer = GraphRenderer()
    renderer.render(transformed_ast, label=label)

def generate_python_ast(code):
    code_ast = ast.parse(code)
    print("hagu")
    transformed_ast = transform_ast(code_ast)
    # print(transformed_ast)

    renderer = GraphRenderer()
    renderer.render(transformed_ast)


def transform_ast(code_ast):
    if isinstance(code_ast, ast.AST):
        node = {to_camelcase(k): transform_ast(getattr(code_ast, k)) for k in code_ast._fields}
        node['node_type'] = to_camelcase(code_ast.__class__.__name__)
        return node
    elif isinstance(code_ast, list):
        return [transform_ast(el) for el in code_ast]
    else:
        return code_ast

def pre_walk_tree(node, index, edge_index):
    types = []
    features = []
    edge_types = []
    edge_in_out_indexs_s, edge_in_out_indexs_t = [], []
    edge_in_out_head_tail = []

    child_index = index + 1
    types.append("ast")

    features.append(str(type(node)))

    for field_name, field in ast.iter_fields(node):
        if isinstance(field, ast.AST):
            edge_types.append(field_name)
            edge_in_out_indexs_s.extend([edge_index, edge_index])
            edge_in_out_indexs_t.extend([index, child_index])
            edge_in_out_head_tail.extend([0, 1])
            child_edge_index = edge_index + 1
            child_index, child_edge_index, child_types, child_features, child_edge_types, child_edge_in_out_indexs_s, child_edge_in_out_indexs_t, child_edge_in_out_head_tail = pre_walk_tree(
                field, child_index, child_edge_index)
            types.extend(child_types)
            features.extend(child_features)
            edge_types.extend(child_edge_types)
            edge_in_out_indexs_s.extend(child_edge_in_out_indexs_s)
            edge_in_out_indexs_t.extend(child_edge_in_out_indexs_t)
            edge_in_out_head_tail.extend(child_edge_in_out_head_tail)
            edge_index = child_edge_index
        elif isinstance(field, list) and field and isinstance(field[0], ast.AST):
            edge_types.append(field_name)
            edge_in_out_indexs_s.append(edge_index)
            edge_in_out_indexs_t.append(index)
            edge_in_out_head_tail.append(0)
            child_edge_index = edge_index + 1
            for item in field:
                edge_in_out_indexs_s.append(edge_index)
                edge_in_out_indexs_t.append(child_index)
                edge_in_out_head_tail.append(1)
                child_index, child_edge_index, child_types, child_features, child_edge_types, child_edge_in_out_indexs_s, child_edge_in_out_indexs_t, child_edge_in_out_head_tail = pre_walk_tree(
                    item, child_index, child_edge_index)
                types.extend(child_types)
                features.extend(child_features)
                edge_types.extend(child_edge_types)
                edge_in_out_indexs_s.extend(child_edge_in_out_indexs_s)
                edge_in_out_indexs_t.extend(child_edge_in_out_indexs_t)
                edge_in_out_head_tail.extend(child_edge_in_out_head_tail)
            edge_index = child_edge_index
        elif isinstance(field, list) and field:
            edge_types.append(field_name)
            edge_in_out_indexs_s.append(edge_index)
            edge_in_out_indexs_t.append(index)
            edge_in_out_head_tail.append(0)
            for item in field:
                types.append("ident")
                features.append(str(item))
                edge_in_out_indexs_s.append(edge_index)
                edge_in_out_indexs_t.append(child_index)
                edge_in_out_head_tail.append(1)
                child_index += 1
            edge_index += 1
        elif field:
            edge_types.append(field_name)
            edge_in_out_indexs_s.append(edge_index)
            edge_in_out_indexs_t.append(index)
            edge_in_out_head_tail.append(0)
            types.append("ident")
            features.append(str(field))
            edge_in_out_indexs_s.append(edge_index)
            edge_in_out_indexs_t.append(child_index)
            edge_in_out_head_tail.append(1)
            child_index += 1
            edge_index += 1

    return child_index, edge_index, types, features, edge_types, edge_in_out_indexs_s, edge_in_out_indexs_t, edge_in_out_head_tail

def convert_python_to_ast(code):
    root = ast.parse(code)
    return pre_walk_tree(root, 0, 0)
    # print(index)
    # print(edge_index)
    # print(len(types))
    # print(len(features))
    # print(len(edge_types))
    # print(len(edge_in_out_indexs_s))
    # print(len(edge_in_out_indexs_t))
    # print(len(edge_in_out_head_tail))


def generate_tree_sitter_edges(node, node_features, source, sink):
    tokenizer = BertTokenizer.from_pretrained('./tvm_bert/tvm_bert_it-vocab.txt', local_files_only=True)
    if len(node.children) > 0:
        for child in node.children:
            source.append(node.id)
            sink.append(child.id)
            node_dict = {}
            node_dict['id'] = child.id
            node_dict['text'] = tokenizer.encode(child.text.decode())
            node_features.append(node_dict)

            generate_tree_sitter_edges(child, node_features, source, sink)
    else:
        source.append(node.id)
        sink.append(99999) #leaf nodes
        
        node_dict = {}
        node_dict['id'] = node.id
        node_dict['text'] = tokenizer.encode(node.text.decode())
        node_features.append(node_dict)

def graph_from_tree_sitter(code):
    PY_LANGUAGE = Language("tree-sitter/build/my-languages.so", "python")
    parser = Parser()
    parser.set_language(PY_LANGUAGE)
    tree = parser.parse(bytes(code, 'utf-8'))
    # assert tree.root_node.type == 'module'
    # print(tree.root_node.id)
    source = []
    sink = []
    node_features = []
    # print(tree.root_node.type)
    generate_tree_sitter_edges(tree.root_node, node_features, source, sink)
    # print("done", len(node_features), len(source), len(sink))
    return node_features, source, sink
    # for child in tree.root_node.children:
    #     for grandchild in child.children:
    #         print(len(grandchild.children))
    # print("done", len(source), len(sink))
    # for children in tree.root_node.children:
    #     if len(children.children) == 0:
    #         print("podu")
    # print(tree.root_node.type)
    # print(tree.root_node.text)


def to_camelcase(string):
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', string).lower()


class GraphRenderer:
    """
    this class is capable of rendering data structures consisting of
    dicts and lists as a graph using graphviz
    """

    graphattrs = {
        'labelloc': 't',
        'fontcolor': 'white',
        'bgcolor': '#333333',
        'margin': '0',
    }

    nodeattrs = {
        'color': 'white',
        'fontcolor': 'white',
        'style': 'filled',
        'fillcolor': '#006699',
    }

    edgeattrs = {
        'color': 'white',
        'fontcolor': 'white',
    }

    _graph = None
    _rendered_nodes = None


    @staticmethod
    def _escape_dot_label(str):
        return str.replace("\\", "\\\\").replace("|", "\\|").replace("<", "\\<").replace(">", "\\>")


    def _render_node(self, node):
        if isinstance(node, (str, numbers.Number)) or node is None:
            node_id = uuid()
        else:
            node_id = id(node)
        node_id = str(node_id)

        if node_id not in self._rendered_nodes:
            self._rendered_nodes.add(node_id)
            if isinstance(node, dict):
                self._render_dict(node, node_id)
            elif isinstance(node, list):
                self._render_list(node, node_id)
            else:
                self._graph.node(node_id, label=self._escape_dot_label(str(node)))

        return node_id


    def _render_dict(self, node, node_id):
        self._graph.node(node_id, label=node.get("node_type", "[dict]"))
        for key, value in node.items():
            if key == "node_type":
                continue
            child_node_id = self._render_node(value)
            self._graph.edge(node_id, child_node_id, label=self._escape_dot_label(key))


    def _render_list(self, node, node_id):
        self._graph.node(node_id, label="[list]")
        for idx, value in enumerate(node):
            child_node_id = self._render_node(value)
            self._graph.edge(node_id, child_node_id, label=self._escape_dot_label(str(idx)))


    def render(self, data, *, label=None):
        # create the graph
        graphattrs = self.graphattrs.copy()
        if label is not None:
            graphattrs['label'] = self._escape_dot_label(label)
        graph = gv.Digraph(graph_attr = graphattrs, node_attr = self.nodeattrs, edge_attr = self.edgeattrs)

        # recursively draw all the nodes and edges
        self._graph = graph
        self._rendered_nodes = set()
        self._render_node(data)
        self._graph = None
        self._rendered_nodes = None

        dotplus = pydotplus.graph_from_dot_data(graph.source)
        nx_graph = networkx.nx_pydot.from_pydot(dotplus)
        data = networkx.readwrite.json_graph.tree_data(nx_graph,root=list(nx_graph)[0])
        print(data)

        # json.dump(data,open("pocha.json",'w'))

        # display the graph
        # graph.format = "pdf"
        # graph.view()
        # subprocess.Popen(['xdg-open', "test.pdf"])

if __name__ == '__main__':
    main(sys.argv)