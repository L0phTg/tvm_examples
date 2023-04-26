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

def te_linear(X: te.Tensor, W: te.Tensor, B: te.Tensor, Z: te.Tensor) -> te.Tensor:
    Y = topi.nn.matmul(X, W, bias=B)
    res = topi.add(Y, Z)
    return res

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
            lv1 = bb.emit_te(topi.sequence_mask, lv0, valid_lens, -1e6, 1)
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
    z0 = rx.Var("z0", [batch_num, ffn_hidden], rx.DynTensorType(2, "float32"))

    w1 = rx.Var("w1", [ffn_hidden, ffn_output], rx.DynTensorType(2, "float32"))
    b1 = rx.Var("b1", [ffn_output, ], rx.DynTensorType(1, "float32"))
    z1 = rx.Var("z1", [batch_num, ffn_output], rx.DynTensorType(2, "float32"))

    # relax function def 
    fn_inputs = [x, w0, b0, z0, w1, b1, z1]
    fn_output = None
    with bb.function("positivewise_ffn"):
        with bb.dataflow():
            lv0 = bb.emit_te(te_linear, x, w0, b0, z0)
            lv1 = bb.emit_te(topi.nn.relu, lv0)
            output = bb.emit_te(te_linear, lv1, w1, b1, z1)
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

def transpose_qkv(batch_size, query_nums, num_hiddens=768, num_heads=2):
    bb = rx.BlockBuilder()

    # construct params and x 
    type_anno = rx.DynTensorType(3, "float32")
    x = rx.Var("x", [batch_size, query_nums, num_hiddens], type_anno)

    # relax function def 
    fn_inputs = [x]
    fn_output = None
    with bb.function("transpose_qkv"):
        with bb.dataflow():
            lv0 = bb.emit(rx.op.reshape(x, (batch_size, query_nums, num_heads, -1)))
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

def dotproduct_attention(queries_shape: R.Shape, keys_shape: R.Shape, 
            values_shape: R.Shape, valid_lens_shape: R.Shape):
    bb = rx.BlockBuilder()

    # add functions
    mask_softmax_mod = masked_softmax(x_shape=(10, 20, 40), valid_shape=valid_lens_shape)
    seq_mask = bb.add_func(mask_softmax_mod["sequence_mask"], "sequence_mask")
    rx_masked_softmax = bb.add_func(mask_softmax_mod["masked_softmax"], "masked_softmax")

    # construct params and x 
    queries = rx.Var("queries", queries_shape, rx.DynTensorType(len(queries_shape), "float32"))
    keys = rx.Var("keys", keys_shape, rx.DynTensorType(len(keys_shape), "float32"))
    values = rx.Var("values", values_shape, rx.DynTensorType(len(values_shape), "float32"))
    valid_lens = rx.Var("valid_lens", valid_lens_shape, rx.DynTensorType(len(valid_lens_shape), "int64"))
    
    d = queries.shape[2]
    # relax function def
    fn_inputs = [queries, keys, values, valid_lens]
    fn_output = None

    with bb.function("dotproduct_attention"):
        with bb.dataflow():
            lv0 = bb.emit_te(topi.nn.batch_matmul, queries, keys)
            scores = bb.emit(R.divide(lv0, rx.const(math.sqrt(int(d)))))
            attention_weights = bb.emit(rx_masked_softmax(scores, valid_lens))
            w_dp = bb.emit(R.TupleGetItem(R.nn.dropout(attention_weights), 1))
            output = bb.emit_te(topi.nn.batch_matmul, w_dp, values, transpose_b=False)
            fn_output = bb.emit_output(output)
        bb.emit_func_output(fn_output, fn_inputs)
    return bb.get()

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
    dotproduct_attention(queries_shape=(10, 20, 30), 
            keys_shape=(10, 40, 30),
            values_shape=(10, 40, 100), 
            valid_lens_shape=(40, )).show()