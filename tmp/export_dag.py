# 绘制dot->svg/png
import graphviz
# tvm 
import tvm
from tvm import relax
from tvm.ir.module import IRModule

from tvm import relax as rx, tir

from tools import *

# data sturct
from typing import Dict, List, Optional, Union, Any, Callable

def single_node(mod: IRModule, graph: graphviz.Digraph, node: rx.Call):
    node_label = node.op.astext()
    graph.node(node.op.astext(), node_label)

def html_table(*content, **kwargs):
    kwargs_pairs = [f'{k}="{v}"' for k, v in kwargs.items()]
    return f'<table {" ".join(kwargs_pairs)}>' + "\n".join(content) + "</table>"

def html_tr(*content, **kwargs):
    kwargs_pairs = [f'{k}="{v}"' for k, v in kwargs.items()]
    return f'<tr {" ".join(kwargs_pairs)}>' + "\n".join(content) + "</tr>"

def html_td(content, **kwargs):
    kwargs_pairs = [f'{k}="{v}"' for k, v in kwargs.items()]
    return f'<td {" ".join(kwargs_pairs)}>' + str(content) + "</td>"

def node_label_html(func: rx.Function, binding: rx.Binding):
    var, node = binding.var, binding.value
    name = node.op.name
    head = name + "(" + ", ".join([str(arg) for arg in node.args]) + ")"
    # for call tir instrution
    if isinstance(node, rx.Call) and node.op.name == "relax.call_tir":
        head = "tir."+node.args[0].name_hint + "(" + ", ".join([str(arg) for arg in node.args[1]]) + ")"
    cols = [html_td(str(var)), html_td(str(node.struct_info))]
    
    head_kwargs = dict(colspan=len(cols))
    html = html_table(
        html_tr(html_td(head, **head_kwargs)), # 第一行: 算子名称
        html_tr("".join(cols)), # 第二行：输出tensor的var, 形状
        border=0,
        cellborder=1,
        cellspacing=0,
    )
    return f"<{html}>"

# 函数参数 -> html
def var_label_html(func: rx.Function, var: rx.Var):
    node_label = ','.join([var.name_hint, str(var.struct_info)])
    return node_label

def model_graph(mod: IRModule, func_name: str, *args, **kwargs) -> graphviz.Digraph:
    # 定义Graphvize有向图
    graph = graphviz.Digraph(func_name, format="svg", node_attr={"shape": "plaintext"})
    
    # 选择function
    func = mod[func_name]
    # 构造函数输入
    for arg in func.params:
        node_label = var_label_html(func, arg)
        node_kwargs = dict(shape="plaintext")
        print(node_label)
        graph.node(arg.name_hint, node_label, **node_kwargs)
    # 构造函数输出
    if isinstance(func.body.body, rx.Var):
        node_label = var_label_html(func, func.body.body)
        node_kwargs = dict(shape="plaintext")
        print(node_label)
        graph.node(func.body.body.name_hint, node_label, **node_kwargs)
    # 遍历function body
    for func_block in func.body.blocks:
        for binding in func_block.bindings:
            var, node = binding.var, binding.value

            if isinstance(node, rx.Call):
                node_name = var.name_hint
                node_label = node_label_html(func, binding)
                node_kwargs = dict(shape="plaintext")
                graph.node(node_name, node_label, **node_kwargs)
                
                args_list = node.args 
                if node.op.name == "relax.call_tir" or node.op.name == "relax.call_dps_packed":
                    args_list = args_list[1]
                for arg in args_list:
                    # 构建args var -> var
                    if isinstance(arg, rx.expr.Var):
                        graph.edge(arg.name_hint, node_name)
            elif isinstance(node, rx.DataflowVar):
                graph.edge(node.name_hint, var.name_hint)
            
    return graph

def test():
    mod = test_masked_softmax()
    graph = model_graph(mod, "masked_softmax")