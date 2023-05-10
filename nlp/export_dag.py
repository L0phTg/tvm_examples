# 绘制dot->svg/png
import graphviz
# tvm 
import tvm
from tvm import relax
from tvm.ir.module import IRModule

from tvm import relax as rx, tir

from transformer import *

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
    name = None 
    if isinstance(node.op, rx.GlobalVar):
        name = node.op.name_hint
    else:
        name = node.op.name
    head = name + "(" + ", ".join([str(arg) for arg in node.args if not isinstance(arg, rx.Constant)]) + ")"
    cols = [html_td(str(var)), html_td(str(node.struct_info))]

    head_kwargs = dict(colspan=len(cols))    
    # for call tir instrution
    if isinstance(node.op, tvm.ir.Op) and node.op.name == "relax.call_tir":
        head = "tir."+node.args[0].name_hint + \
                "(" + ", ".join(
                    [str(arg) for arg in node.args[1] if not isinstance(arg, rx.Constant)]
                ) + ")"
        head_kwargs["bgcolor"] = "lightgray"
    # for user def global func
    elif isinstance(node.op, rx.GlobalVar):
        head_kwargs["bgcolor"] = "lightgray"
    
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
    node_label = ':'.join([var.name_hint, str(var.struct_info)])
    return node_label

def tuple_getitem_label_html(var: rx.Var, getit: rx.TupleGetItem):
    label = var.name_hint + " = " + str(getit)
    html = html_table(
        html_tr(html_td(label)),
        border=0,
        cellborder=1,
        cellspacing=0,
    )
    return f"<{html}>"

def model_func_inner(graph: graphviz.Digraph,
                     src_node, out_node,
                     mod, func_name, recursive_num):
    return 

def model_func(graph: graphviz.Digraph, 
                src_node, 
                mod: IRModule, func_name, recursive_num):
    # 选择function
    func = mod[func_name]
    # 构造函数输入
    for arg in func.params:
        node_name = func_name+"."+arg.name_hint
        node_label = var_label_html(func, arg)
        node_kwargs = dict(shape="plaintext")
        # print("input params: ", node_name)
        graph.node(node_name, node_label, **node_kwargs)
    graph.edge(src_node, func_name+"."+func.params[0].name_hint,
                **dict(style="dashed", color="lightgrey"))
    # 构造函数输出
    if isinstance(func.body.body, rx.Var):
        node_name = func_name+"."+func.body.body.name_hint
        node_label = var_label_html(func, func.body.body)
        node_kwargs = dict(shape="plaintext")
        # print("func output: ", node_name)
        graph.node(node_name, node_label, **node_kwargs)
    # 遍历function body
    for func_block in func.body.blocks:
        for binding in func_block.bindings:
            var, node = binding.var, binding.value
            node_name = func_name+"."+var.name_hint
            if isinstance(node, rx.Call):
                node_label = node_label_html(func, binding)
                node_kwargs = dict(shape="plaintext")
                # print("func call: ", node_name)
                graph.node(node_name, node_label, **node_kwargs)
                
                args_list = node.args
                if isinstance(node.op, tvm.ir.Op):
                    if node.op.name == "relax.call_tir" or node.op.name == "relax.call_dps_packed":
                        args_list = args_list[1]
                for arg in args_list:
                    # 构建args var -> var
                    if isinstance(arg, rx.expr.Var):
                        graph.edge(func_name+"."+arg.name_hint, node_name)

                # func impl
                call_func_name = None
                if isinstance(node.op, rx.GlobalVar):
                    call_func_name = node.op.name_hint
                else:
                    call_func_name = node.op.name
                if recursive_num > 1 and \
                        call_func_name in [v.name_hint for v in mod.get_global_vars()]:
                    model_func(graph, node_name, mod, call_func_name, recursive_num=recursive_num-1)
            elif isinstance(node, rx.DataflowVar):
                graph.edge(func_name+"."+node.name_hint, node_name)
            elif isinstance(node, rx.TupleGetItem): # case: lv1 = lv0[0]
                node_label = tuple_getitem_label_html(var, node)
                graph.node(node_name, node_label)
                graph.edge(func_name+"."+node.tuple_value.name_hint, node_name)
    return 

def model_graph(mod: IRModule, func_name: str, recursive_num=1, *args, **kwargs) -> graphviz.Digraph:
    # 定义Graphvize有向图
    graph = graphviz.Digraph(func_name, format="svg", node_attr={"shape": "plaintext"})
    graph.node(func_name, func_name)
    model_func(graph, func_name, mod, func_name, recursive_num)
    return graph

def test_model_graph():
    masked_softmax_mod = test_masked_softmax()
    masked_softmax_graph = model_graph(masked_softmax_mod, "masked_softmax")

    positivewise_ffn_mod = positivewise_ffn()
    positivewise_ffn_graph = model_graph(positivewise_ffn_mod, "positivewise_ffn", recursive_num=1)

    addnorm_mod = addnorm((10, 20, 30))
    addnorm_graph = model_graph(addnorm_mod, "addnorm", recursive_num=1)

    transpose_qkv_mod = transpose_qkv(num_heads=10)
    transpose_qkv_graph = model_graph(transpose_qkv_mod, "transpose_qkv", recursive_num=1)

    transpose_output_mod = transpose_output(num_heads=10)
    transpose_output_graph = model_graph(transpose_output_mod, "transpose_output", recursive_num=1)