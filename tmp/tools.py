# This is needed for deferring annotation parsing in TVMScript
from __future__ import annotations
import numpy as np
import tvm
from tvm import relax
from tvm import relay
from tvm.ir.module import IRModule
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R

from tvm import te
from tvm import topi

from tvm import relax as rx, tir

import torch

import math

# 支持二维矩阵乘/三维矩阵乘
def te_linear(X: te.Tensor, W: te.Tensor, B: te.Tensor) -> te.Tensor:
    if len(X.shape) == 2:
        Y = topi.nn.matmul(X, W)       # transpose_b=False
    elif len(X.shape) == 3:
        Y = topi.nn.batch_matmul(X, W) # transpose_b=True
    res = topi.add(Y, B)
    return res

def te_linear_withparams(X: te.Tensor, params: dict) -> te.Tensor:
    if len(X.shape) == 2:
        Y = topi.nn.matmul(X, params["w"])
    elif len(X.shape) == 3:
        Y = topi.nn.batch_matmul(X, params["w"])
    res = topi.add(Y, params["b"])
    return res

def rx_linear():
    return

def te_layernorm(X: te.Tensor, gamma: te.Tensor, beta: te.Tensor):
    out = topi.nn.layer_norm(X, gamma, beta, axis=[-1])
    return out

def masked_softmax(x_shape: R.Shape, valid_shape: R.Shape):
    bb = rx.BlockBuilder()

    # construct params and x 
    x = rx.Var("x", x_shape, rx.DynTensorType(len(x_shape), "float32"))
    valid_lens = rx.Var("valid_lens", valid_shape, rx.DynTensorType(len(valid_shape), "int64"))

    # relax function def 
    fn_inputs = [x, valid_lens]
    fn_output = None
    with bb.function("masked_softmax"):
        with bb.dataflow():
            lv0 = bb.emit(rx.op.reshape(x, (-1, x_shape[2])))
            if (len(valid_shape) == 2):
                valid_lens_reshape = bb.emit(rx.op.reshape(valid_lens, (-1)))
            lv1 = bb.emit_te(topi.sequence_mask, lv0, valid_lens_reshape, -1e6, 1)
            output = bb.emit(rx.nn.softmax(R.reshape(lv1, x_shape), axis=-1))
            fn_output = bb.emit_output(output)
        bb.emit_func_output(fn_output, fn_inputs)
    return bb.get()

def positivewise_ffn(batch_num = 10, ffn_input = 20, ffn_hidden = 30, ffn_output = 40):
    bb = rx.BlockBuilder()

    # construct x and params 
    x = rx.Var("x", [batch_num, ffn_input], rx.DynTensorType(2, "float32"))
    w0 = rx.Var("w0", [ffn_input, ffn_hidden], rx.DynTensorType(2, "float32"))
    b0 = rx.Var("b0", [ffn_hidden, ], rx.DynTensorType(1, "float32"))

    w1 = rx.Var("w1", [ffn_hidden, ffn_output], rx.DynTensorType(2, "float32"))
    b1 = rx.Var("b1", [ffn_output, ], rx.DynTensorType(1, "float32"))

    # relax function def 
    fn_inputs = [x, w0, b0, w1, b1]
    fn_output = None
    with bb.function("positivewise_ffn"):
        with bb.dataflow():
            lv0 = bb.emit_te(te_linear, x, w0, b0)
            lv1 = bb.emit_te(topi.nn.relu, lv0)
            output = bb.emit_te(te_linear, lv1, w1, b1)
            fn_output = bb.emit_output(output)
        bb.emit_func_output(fn_output, fn_inputs)
    return bb.get()

def addnorm(x_shape):
    bb = rx.BlockBuilder()

    # construct x and params 
    x = rx.Var("x", x_shape, rx.DynTensorType(len(x_shape), "float32"))
    gamma = rx.Var("gamma", [x_shape[-1]], rx.DynTensorType(1, "float32"))
    beta = rx.Var("beta", [x_shape[-1]], rx.DynTensorType(1, "float32"))

    # relax function def 
    fn_inputs = [x, gamma, beta]
    fn_output = None
    with bb.function("addnorm"):
        with bb.dataflow():
            lv0 = bb.emit(R.TupleGetItem(rx.nn.dropout(x), 1))
            lv1 = bb.emit_te(topi.add, lv0, x)
            output = bb.emit_te(te_layernorm, lv1, gamma, beta)
            fn_output = bb.emit_output(output)
        bb.emit_func_output(fn_output, fn_inputs)
    return bb.get()

def transpose_qkv(batch_size, qkv_nums, num_hiddens=768, num_heads=2):
    bb = rx.BlockBuilder()

    # construct params and x 
    type_anno = rx.DynTensorType(3, "float32")
    x = rx.Var("x", [batch_size, qkv_nums, num_hiddens], type_anno)

    # relax function def 
    fn_inputs = [x]
    fn_output = None
    with bb.function("transpose_qkv"):
        with bb.dataflow():
            lv0 = bb.emit(rx.op.reshape(x, (batch_size, qkv_nums, num_heads, -1)))
            lv1 = bb.emit(rx.op.transpose(lv0, (0, 2, 1, 3)))
            output = bb.emit(rx.op.reshape(lv1, (-1, lv1.shape[2], lv1.shape[3])))
            fn_output = bb.emit_output(output)
        bb.emit_func_output(fn_output, fn_inputs)
    return bb.get()

def transpose_output(x_shape: R.Shape, num_heads=2):
    bb = rx.BlockBuilder()
    
    # construct params and x 
    type_anno = rx.DynTensorType(3, "float32")    
    x = rx.Var("x", x_shape, type_anno)

    # relax function def
    fn_inputs = [x]
    fn_output = None
    with bb.function("transpose_output"):
        with bb.dataflow():
            lv0 = bb.emit(rx.op.reshape(x, (-1, num_heads, x.shape[1], x.shape[2])))
            lv1 = bb.emit(rx.op.transpose(lv0, (0, 2, 1, 3)))
            output = bb.emit(rx.op.reshape(lv1, (lv1.shape[0], lv1.shape[1], -1)))
            fn_output = bb.emit_output(output)
        bb.emit_func_output(fn_output, fn_inputs)
    return bb.get()


def dotproduct_attention(
        batch_size, query_size, kv_size,
        dim, value_dim
    ):  
    # 
    # output: 
    bb = rx.BlockBuilder()

    # add functions
    mask_softmax_mod = masked_softmax(x_shape=[batch_size, query_size, kv_size], valid_shape=[batch_size, query_size])
    seq_mask = bb.add_func(mask_softmax_mod["sequence_mask"], "sequence_mask")
    rx_masked_softmax = bb.add_func(mask_softmax_mod["masked_softmax"], "masked_softmax")

    # construct params and x 
    queries = rx.Var("queries", [batch_size, query_size, dim], rx.DynTensorType(3, "float32"))
    keys = rx.Var("keys", [batch_size, kv_size, dim], rx.DynTensorType(3, "float32"))
    values = rx.Var("values", [batch_size, kv_size, value_dim], rx.DynTensorType(3, "float32"))
    valid_lens = rx.Var("valid_lens", [batch_size, query_size], rx.DynTensorType(2, "int64"))
    
    # relax function def
    fn_inputs = [queries, keys, values, valid_lens]
    fn_output = None

    with bb.function("dotproduct_attention"):
        with bb.dataflow():
            # lv0: batch_size, query_size, kv_size
            lv0 = bb.emit_te(topi.nn.batch_matmul, queries, keys)
            scores = bb.emit(R.divide(lv0, rx.const(math.sqrt(dim))))
            # attention_weights: batch_size, query_size, kv_size
            attention_weights = bb.emit(rx_masked_softmax(scores, valid_lens))
            w_dp = bb.emit(R.TupleGetItem(R.nn.dropout(attention_weights), 1))
            # output: batch_size, query_size, value_dim
            output = bb.emit_te(topi.nn.batch_matmul, w_dp, values, transpose_b=False)
            fn_output = bb.emit_output(output)
        bb.emit_func_output(fn_output, fn_inputs)
    return bb.get()

def multihead_attention(
            batch_size, query_size, kv_size, 
            dim = 100, value_dim = 200,
            num_hiddens = 300, num_heads = 2
    ):
    bb = rx.BlockBuilder()
    # 权重
    params = {
        "linear_wq" : {
            "w": rx.Var("q_w", (batch_size, num_hiddens, dim), R.Tensor),
            "b": rx.Var("q_b", (query_size, num_hiddens), R.Tensor),
        },
        "linear_wk" : {
            "w": rx.Var("k_w", (batch_size, num_hiddens, dim), R.Tensor),
            "b": rx.Var("k_b", (kv_size, num_hiddens), R.Tensor)
        },
        "linear_wv" : {
            "w": rx.Var("v_w", (batch_size, num_hiddens, value_dim), R.Tensor),
            "b": rx.Var("v_b", (kv_size, num_hiddens), R.Tensor)
        },
        "linear_wo" : {
            "w": rx.Var("o_w", (batch_size, num_hiddens, num_hiddens), R.Tensor),
            "b": rx.Var("o_b", (query_size, num_hiddens), R.Tensor)
        }
    }
    # add functions
    transpose_q_mod = transpose_qkv(batch_size, query_size, num_hiddens, num_heads)
    transpose_kv_mod = transpose_qkv(batch_size, kv_size, num_hiddens, num_heads)
    transpose_output_mod = transpose_output(x_shape=(batch_size*num_heads, query_size, int(num_hiddens/num_heads)), num_heads=num_heads)
    dot_attention_mod = dotproduct_attention(batch_size*num_heads, query_size, kv_size, int(num_hiddens/num_heads), int(num_hiddens/num_heads))
    
    transpose_q_var = bb.add_func(transpose_q_mod["transpose_qkv"], "transpose_q")
    transpose_kv_var = bb.add_func(transpose_kv_mod["transpose_qkv"], "transpose_kv")
    transpose_output_var = bb.add_func(transpose_output_mod["transpose_output"], "transpose_output")
    for func_name in dot_attention_mod.global_var_map_:
        bb.add_func(dot_attention_mod[func_name], func_name)
    dot_attention_var = bb.get().get_global_var("dotproduct_attention")

    # construct x and params
    queries = rx.Var("queries", [batch_size, query_size, dim], rx.DynTensorType(3, "float32"))
    keys = rx.Var("keys", [batch_size, kv_size, dim], rx.DynTensorType(3, "float32"))
    values = rx.Var("values", [batch_size, kv_size, value_dim], rx.DynTensorType(3, "float32"))
    valid_lens = rx.Var("valid_lens", [batch_size, query_size], rx.DynTensorType(2, "int64"))
 
    # relax function def
    ext_params = [var for k in params.keys() for var in params[k].values()]
    fn_inputs = [queries, keys, values, valid_lens, *ext_params]
    fn_output = None

    with bb.function("multihead_attention"):
        with bb.dataflow(): # output: batch_size * num_heads, query_size, num_hidden/num_heads
            W_q = bb.emit_te(te_linear_withparams, queries, params["linear_wq"])
            quries_t = bb.emit(transpose_q_var(W_q))
            flow1_out = bb.emit_output(quries_t)
        with bb.dataflow(): # output: batch_size * num_heads, kv_size, num_hidden/num_heads
            W_k = bb.emit_te(te_linear_withparams, keys, params["linear_wk"])
            keys_t = bb.emit(transpose_kv_var(W_k))
            flow2_out = bb.emit_output(keys_t)
        with bb.dataflow(): # output: batch_size * num_heads, kv_size, num_hidden/num_heads
            W_v = bb.emit_te(te_linear_withparams, values, params["linear_wv"])
            values_t = bb.emit(transpose_kv_var(W_v))
            flow3_out = bb.emit_output(values_t)
        with bb.dataflow():
            # output: batch_size * num_heads, query_size, num_hidden/num_heads
            output = bb.emit(dot_attention_var(flow1_out, flow2_out, flow3_out, valid_lens))
            # output_concat: batch_size, query_size, num_hidden
            output_concat = bb.emit(transpose_output_var(output))
            # batch_size, query_size, num_hidden
            output_concat_wo = bb.emit_te(te_linear_withparams, output_concat, params["linear_wo"])
            fn_output = bb.emit_output(output_concat_wo)
        bb.emit_func_output(fn_output, fn_inputs)
    return bb.get()

def test_te_linear():
    # 2 维
    X = te.placeholder((10, 20), dtype="float32")
    W = te.placeholder((20, 30), dtype="float32")
    B = te.placeholder((30, ), dtype="float32")

    Y = te_linear(X, W, B)
    linear_mod = te.create_prim_func([X, W, B, Y])

    ### use params
    params = {
        "w": W, "b": B
    }
    Y_1 = te_linear(X, params["w"], params["b"])
    Y_2 = te_linear_withparams(X, params=params)

def test_rx_call_te_linear():
    # 2 维
    bb = rx.BlockBuilder()

    X = rx.Var("x", (10, 20), rx.DynTensorType(2, "float32"))
    W = rx.Var("w", (20, 30), rx.DynTensorType(2, "float32"))
    B = rx.Var("b", (30, ), rx.DynTensorType(1, "float32"))

    params = {
        "w": W, "b": B
    }

    fn_inputs = [X, params["w"], params["b"]]
    fn_output = None
    with bb.function("test_te_linear_withparams"):
        with bb.dataflow():
            output = bb.emit_te(te_linear_withparams, X, params)
            fn_output = bb.emit_output(output)
        bb.emit_func_output(fn_output, fn_inputs)
    bb.get()

def test_masked_softmax():
    # valid shape is 2D:
    masked_softmax((10, 100, 50), (10, 100))

def test_positivewise_ffn():
    batch_num, ffn_input, ffn_hidden, ffn_output=10, 20, 30, 40
    ffn_params = {
        "w0": tvm.nd.array(torch.rand(ffn_input, ffn_hidden).numpy()),
        "b0": tvm.nd.array(torch.rand(ffn_hidden).numpy()),
        "z0": tvm.nd.array(torch.rand(batch_num, ffn_hidden).numpy()),
        "w1": tvm.nd.array(torch.rand(ffn_hidden, ffn_output).numpy()),
        "b1": tvm.nd.array(torch.rand(ffn_output).numpy()),
        "z1": tvm.nd.array(torch.rand(batch_num, ffn_output).numpy()),
    }

    ffn_mod = positivewise_ffn(batch_num=10, ffn_input=20, ffn_hidden=30, ffn_output=40)
    ffn_mod_with_params = relax.transform.BindParams("positivewise_ffn", ffn_params)(ffn_mod)

def test_addnorm():
    return

def test_transpose_qkv():
    return

def test_transpose_output():
    return

def test_dotproduct_attention():
    dotproduct_attention(
            batch_size=10, query_size=50, kv_size=60,
            dim=100, value_dim=300
        )

def test_multihead_attention():
    # 超参
    batch_size, query_size, kv_size = 10, 40, 50
    dim, value_dim = 100, 200
    num_hiddens, num_heads = 300, 2

    multihead_attention_mod = multihead_attention(
        batch_size, query_size, kv_size,
        dim, value_dim,
        num_hiddens=num_hiddens, num_heads=num_heads
    )
    # 参数绑定
    
    return multihead_attention_mod

def main():
    test_te_linear()
    test_rx_call_te_linear()

    test_positivewise_ffn()
    test_addnorm()

    test_transpose_qkv()
    test_transpose_output()

    test_dotproduct_attention()