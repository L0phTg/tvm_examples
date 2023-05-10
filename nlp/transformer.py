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
from tvm import tir
from tvm import relax as rx

import torch

import math

from transformer_static import st_masked_softmax

"""
    relax functions:
        - linear
        - masked_softmax
        - transpose_qkv
        - transpose_output

    un relax functions:
        - layernorm
        - batch_matmul
        - dot_attention
"""

def gen_linear_params(in_features, out_features):
    return {
        "w": np.random.rand(in_features, out_features).astype("float32"),
        "b": np.random.rand(out_features).astype("float32"),
    }

def te_layernorm(X: te.Tensor, gamma: te.Tensor, beta: te.Tensor):
    out = topi.nn.layer_norm(X, gamma, beta, axis=[-1])
    return out

# for 2d, 3d input ndim
def linear(in_features, out_features, W=None, B=None, ndim=2) -> rx.Expr:
    bb = rx.BlockBuilder()

    m, n = tir.Var("m", "int64"), tir.Var("n", "int64")
    X = rx.Var("x", R.Tensor([m, in_features], "float32"))
    if ndim == 3:
        X = rx.Var("x", R.Tensor([m, n, in_features], "float32"))
    W_placeholder = rx.Var("w", R.Tensor([in_features, out_features], "float32"))
    B_placeholder = rx.Var("b", R.Tensor([out_features], "float32"))

    fn_inputs = [X, W_placeholder, B_placeholder]
    fn_output = None
    with bb.function("linear"):
        with bb.dataflow():
            lv0 = relax.op.matmul(X, W_placeholder)
            gv0 = relax.op.add(lv0, B_placeholder)
            fn_output = bb.emit_output(gv0)
        bb.emit_func_output(fn_output, fn_inputs)

    linear_params = {
        "w": W if W is not None else np.random.rand(in_features, out_features).astype(np.float32),
        "b": B if B is not None else np.random.rand(out_features).astype(np.float32),
    }
    mod = relax.transform.BindParams("linear", linear_params)(bb.get())
    return mod

def masked_softmax():
    bb = rx.BlockBuilder()

    # construct params and x 
    m, n = tir.Var("m", "int64"), tir.Var("n", "int64")
    k = tir.Var("k", "int64")

    x = rx.Var("x", R.Tensor([m, n, k], "float32"))
    valid_lens = rx.Var("valid_lens", R.Tensor([m, n], "int64"))

    # relax function def 
    fn_inputs = [x, valid_lens]
    fn_output = None
    with bb.function("masked_softmax"):
        with bb.dataflow():
            lv0 = bb.emit(rx.op.reshape(x, (-1, k)))
            valid_lens_reshape = bb.emit(rx.op.reshape(valid_lens, (-1, )))
            lv1 = bb.emit_te(topi.sequence_mask, lv0, valid_lens_reshape, -1e6, 1)
            output = bb.emit(rx.op.nn.softmax(R.reshape(lv1, [m,n,k]), axis=-1))
            fn_output = bb.emit_output(output)
        bb.emit_func_output(fn_output, fn_inputs)
    return bb.get()

def positivewise_ffn(ffn_input = 20, ffn_hidden = 30, ffn_output = 40, params: dict=None):
    cur_func_name = "positivewise_ffn"
    bb = rx.BlockBuilder()
    # add linear func call
    if params is None:
        params = {
            "linear0": gen_linear_params(ffn_input, ffn_hidden),
            "linear1": gen_linear_params(ffn_hidden, ffn_output)
        }
    linear1_mod = linear(ffn_input, ffn_hidden, params["linear0"]["w"], params["linear0"]["b"], ndim=3)
    linear2_mod = linear(ffn_hidden, ffn_output, params["linear1"]["w"], params["linear1"]["b"], ndim=3)
    rx_linear0 = bb.add_func(linear1_mod["linear"], cur_func_name+"__linear0")
    rx_linear1 = bb.add_func(linear2_mod["linear"], cur_func_name+"__linear1")
    # construct x and params 
    batch_num, seq_num = tir.Var("m", "int64"), tir.Var("n", "int64")
    x = rx.Var("x", R.Tensor([batch_num, seq_num, ffn_input], "float32"))

    # relax function def 
    fn_inputs = [x]
    fn_output = None
    with bb.function(cur_func_name):
        with bb.dataflow():
            lv0 = bb.emit(rx_linear0(x))
            lv1 = bb.emit(rx.op.nn.relu(lv0))
            output = bb.emit(rx_linear1(lv1))
            fn_output = bb.emit_output(output)
        bb.emit_func_output(fn_output, fn_inputs)
    return bb.get()

def addnorm(x_shape):
    bb = rx.BlockBuilder()

    # construct x and params 
    x = rx.Var("x", rx.TensorStructInfo(x_shape, "float32"))
    gamma = rx.Var("gamma", rx.TensorStructInfo([x_shape[-1]], "float32"))
    beta = rx.Var("beta", rx.TensorStructInfo([x_shape[-1]], "float32"))

    # relax function def 
    fn_inputs = [x, gamma, beta]
    fn_output = None
    with bb.function("addnorm"):
        with bb.dataflow():
            # lv0,lv1: batch_size, seq_num, dim
            lv0 = bb.emit(rx.op.nn.dropout(x))
            lv1 = bb.emit(R.TupleGetItem(lv0, 1))
            lv2 = bb.emit(rx.op.add(lv1, x))
            output = bb.emit_te(te_layernorm, lv2, gamma, beta)
            fn_output = bb.emit_output(output)
        bb.emit_func_output(fn_output, fn_inputs)
    return bb.get()

def transpose_qkv(num_heads=2):
    bb = rx.BlockBuilder()

    # construct params and x 
    batch_size = tir.Var("batch_size", "int64")
    qkv_nums = tir.Var("qkv_nums", "int64")
    num_hiddens = tir.Var("num_hiddens", "int64")
    x = rx.Var("x", R.Tensor([batch_size, qkv_nums, num_hiddens], "float32"))

    # relax function def 
    fn_inputs = [x]
    fn_output = None
    with bb.function("transpose_qkv"):
        with bb.dataflow():
            lv0 = bb.emit(rx.op.reshape(x, (batch_size, qkv_nums, num_heads, -1)))
            lv1 = bb.emit(rx.op.permute_dims(lv0, (0, 2, 1, 3)))
            output = bb.emit(rx.op.reshape(lv1, (-1, lv1.struct_info.shape[2], lv1.struct_info.shape[3])))
            fn_output = bb.emit_output(output)
        bb.emit_func_output(fn_output, fn_inputs)
    return bb.get()

def transpose_output(num_heads=2):
    bb = rx.BlockBuilder()
    
    # construct params and x 
    batch_t_heads = tir.Var("batch_t_heads", "int64")
    qkv_nums = tir.Var("qkv_nums", "int64")
    num_hiddens = tir.Var("num_hiddens", "int64")
    x = rx.Var("x", R.Tensor([batch_t_heads, qkv_nums, num_hiddens], "float32"))

    # relax function def
    fn_inputs = [x]
    fn_output = None
    with bb.function("transpose_output"):
        with bb.dataflow():
            lv0 = bb.emit(rx.op.reshape(x, (-1, num_heads, qkv_nums, num_hiddens)))
            lv1 = bb.emit(rx.op.permute_dims(lv0, (0, 2, 1, 3)))
            output = bb.emit(rx.op.reshape(lv1, (lv1.struct_info.shape[0], lv1.struct_info.shape[1], -1)))
            fn_output = bb.emit_output(output)
        bb.emit_func_output(fn_output, fn_inputs)
    return bb.get()

def dotproduct_attention(
        batch_size, query_size, kv_size,
        dim, value_dim
    ) -> IRModule:  
    # 
    # output: 
    bb = rx.BlockBuilder()

    # add functions
    mask_softmax_mod = st_masked_softmax(x_shape=[batch_size, query_size, kv_size], valid_shape=[batch_size, query_size])
    seq_mask = bb.add_func(mask_softmax_mod["sequence_mask"], "sequence_mask")
    rx_masked_softmax = bb.add_func(mask_softmax_mod["masked_softmax"], "masked_softmax")

    # construct params and x 
    queries = rx.Var("queries", rx.TensorStructInfo([batch_size, query_size, dim], "float32"))
    keys = rx.Var("keys", rx.TensorStructInfo([batch_size, kv_size, dim], "float32"))
    values = rx.Var("values", rx.TensorStructInfo([batch_size, kv_size, value_dim], "float32"))
    valid_lens = rx.Var("valid_lens", rx.TensorStructInfo([batch_size, query_size], "int64"))
    
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
            query_size, kv_size, 
            dim = 100, value_dim = 200,
            num_hiddens = 300, num_heads = 2, params: dict=None
    ):
    cur_func_name = "multihead_attention"

    bb = rx.BlockBuilder()

    if params is None:
        params = {
            "linear0": gen_linear_params(query_size, num_hiddens),
            "linear1": gen_linear_params(kv_size, num_hiddens),
            "linear2": gen_linear_params(kv_size, num_hiddens),
            "linear3": gen_linear_params(num_hiddens, num_hiddens),
        }
    linear1_mod = linear(query_size, num_hiddens, params["linear0"]["w"], params["linear0"]["b"], ndim=3)
    linear2_mod = linear(kv_size, num_hiddens, params["linear1"]["w"], params["linear1"]["b"], ndim=3)
    linear3_mod = linear(kv_size, num_hiddens, params["linear2"]["w"], params["linear2"]["b"], ndim=3)
    linear4_mod = linear(num_hiddens, num_hiddens, params["linear3"]["w"], params["linear3"]["b"], ndim=3)
    rx_linear0 = bb.add_func(linear1_mod["linear"], cur_func_name+"__Wq")
    rx_linear1 = bb.add_func(linear2_mod["linear"], cur_func_name+"__Wk")
    rx_linear2 = bb.add_func(linear3_mod["linear"], cur_func_name+"__Wv")
    rx_linear3 = bb.add_func(linear4_mod["linear"], cur_func_name+"__Wo")
    # add functions
    transpose_qkv_mod = transpose_qkv(num_heads)
    transpose_output_mod = transpose_output(num_heads=num_heads)
    dot_attention_mod = dotproduct_attention(batch_size*num_heads, query_size, kv_size, int(num_hiddens/num_heads), int(num_hiddens/num_heads))
    
    transpose_qkv_var = bb.add_func(transpose_qkv_mod["transpose_qkv"], cur_func_name+"__transpose_qkv")
    transpose_output_var = bb.add_func(transpose_output_mod["transpose_output"], cur_func_name+"__transpose_output")
    for func_name in dot_attention_mod.global_var_map_:
        bb.add_func(dot_attention_mod[func_name], cur_func_name+"__"+func_name)
    dot_attention_var = bb.get().get_global_var(cur_func_name+"__dotproduct_attention")

    # construct x and params
    batch_size = tir.Var("batch_size", "int64")
    
    queries = rx.Var("queries", R.Tensor([batch_size, query_size, dim], "float32"))
    keys = rx.Var("keys", R.Tensor([batch_size, kv_size, dim], "float32"))
    values = rx.Var("values", R.Tensor([batch_size, kv_size, value_dim], "float32"))
    valid_lens = rx.Var("valid_lens", R.Tensor([batch_size*num_heads, query_size], "int64"))
 
    # relax function def
    ext_params = [var for k in params.keys() for var in params[k].values()]
    fn_inputs = [queries, keys, values, valid_lens, *ext_params]
    fn_output = None

    with bb.function("multihead_attention"):
        with bb.dataflow(): # output: batch_size * num_heads, query_size, num_hidden/num_heads
            W_q = bb.emit(rx_linear0(queries))
            quries_t = bb.emit(transpose_qkv_var(W_q))
            flow1_out = bb.emit_output(quries_t)
        with bb.dataflow(): # output: batch_size * num_heads, kv_size, num_hidden/num_heads
            W_k = bb.emit(rx_linear1(keys))
            keys_t = bb.emit(transpose_qkv_var(W_k))
            flow2_out = bb.emit_output(keys_t)
        with bb.dataflow(): # output: batch_size * num_heads, kv_size, num_hidden/num_heads
            W_v = bb.emit(rx_linear2(values))
            values_t = bb.emit(transpose_qkv_var(W_v))
            flow3_out = bb.emit_output(values_t)
        with bb.dataflow():
            # output: batch_size * num_heads, query_size, num_hidden/num_heads
            output = bb.emit(dot_attention_var(flow1_out, flow2_out, flow3_out, valid_lens))
            # output_concat: batch_size, query_size, num_hidden
            output_concat = bb.emit(transpose_output_var(output))
            # batch_size, query_size, num_hidden
            output_concat_wo = bb.emit(rx_linear3(output_concat))
            fn_output = bb.emit_output(output_concat_wo)
        bb.emit_func_output(fn_output, fn_inputs)
    return bb.get()

def test_linear():
    in_features, out_features = 20, 30
    W = tvm.nd.array(np.random.rand(in_features, out_features).astype("float32"))
    B = tvm.nd.array(np.random.rand(out_features).astype("float32"))

    mod2d = linear(in_features, out_features, ndim=2)
    mod2d_with_spc_params = linear(20, 30, W, B, ndim=2) 
    mod3d = linear(in_features, out_features, ndim=3)
    mod3d_with_spc_params = linear(20, 30, W, B, ndim=3) 
    return mod3d_with_spc_params

def test_masked_softmax():
    # valid shape is 2D:
    mod = masked_softmax()
    return mod

def test_positivewise_ffn():
    batch_num, seq_num, ffn_input, ffn_hidden, ffn_output=10, 100, 20, 30, 40
    ffn_params = {
        "w0": tvm.nd.array(torch.rand(batch_num, ffn_hidden, ffn_input).numpy()),
        "b0": tvm.nd.array(torch.rand(seq_num, ffn_hidden).numpy()),
        "w1": tvm.nd.array(torch.rand(batch_num, ffn_output, ffn_hidden).numpy()),
        "b1": tvm.nd.array(torch.rand(seq_num, ffn_output).numpy()),
    }

    ffn_mod = positivewise_ffn(batch_num, seq_num, ffn_input, ffn_hidden, ffn_output)
    ffn_mod_with_params = relax.transform.BindParams("positivewise_ffn", ffn_params)(ffn_mod)

    return ffn_mod_with_params

def test_addnorm():
    return

def test_transpose_qkv():
    mod = transpose_qkv(num_heads=10)
    return mod

def test_transpose_output():
    mod = transpose_output(num_heads=10)
    return mod

def test_dotproduct_attention():
    mod = dotproduct_attention(
            batch_size=10, query_size=50, kv_size=60,
            dim=100, value_dim=300
        )
    return mod

def test_multihead_attention():
    # 超参
    batch_size, query_size, kv_size = 10, 40, 50
    dim, value_dim = 100, 200
    num_hiddens, num_heads = 300, 2

    multihead_attention_mod = multihead_attention(
        query_size, kv_size,
        dim, value_dim,
        num_hiddens=num_hiddens, num_heads=num_heads
    )
    # 参数绑定
    
    return multihead_attention_mod

def main():

    test_linear()

    test_positivewise_ffn()
    test_addnorm()

    test_transpose_qkv()
    test_transpose_output()

    test_dotproduct_attention()

    test_multihead_attention()