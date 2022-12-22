
本项目包含一些利用`tvm`进行优化算子或者模型的实例：

项目目录：
- `ops`：包含一些基本算子的优化笔记，如`matmul`优化；
- `cnn`：包含利用tvm对`LeNet`模型（包含基本模型`CNN`、`池化层`、`线性层`）进行端到端优化的例子；
- `resnet`**【未完成】**：包含利用tvm对ResNet模型（包含基本模型`batchnorm`）进行端到端优化的例子；
- `......`

# 环境安装
`tvm-cpu`版本安装
```bash
$ python3 -m  pip install mlc-ai-nightly -f https://mlc.ai/wheels
```
`tvm-cuda`版本安装：
```bash
$ python3 -m pip install mlc-ai-nightly-cu110 -f https://mlc.ai/wheels
```

# Ops

1. matmul：
    - cpu优化版本：参考`ops/mm_optimize.ipynb`

# LeNet
代码介绍：

cpu优化版本：`cnn/notebook/train.ipynb`，主要内容包含：

- 基线版本，参考[从PyTorch导入模型](https://mlc.ai/zh/chapter_integration/index.html#pytorch)：
    1. 对`LeNet`进行端到端的转化，转化为tvm中的IRModule（利用`tvm.relax`中的`blockbuilder`来构建`IRModule`，实现函数`relax_wrapper.from_fx`）；

    2. 由于此时得到的IRModule中包含一些`relax.op`函数，需要将其转化为`tir function`来能构建运行，所以需要对降级，这里利用了`transform.LowerWithRelayOpStrategyPass`来实现；

- 算子优化版本，对基线最终版本的`IRModule`中的算子进行优化。这里通过编写pass实现对relax main函数的更改和算子的优化，主要需要做两件事：
    1. 找到待优化的算子函数，对其进行优化，生成新的优化版本的函数；

    2. 更改对待优化算子的调用，将其指向新的优化版本的函数；


- 算子融合版本，对基线第1步的IRModule版本进行算子融合，这里将`dense`和`add`做了合并，参考：[融合Linear和ReLU算子](https://mlc.ai/zh/chapter_graph_optimization/index.html#linear-relu)：
    1. 找到`add`函数的调用位置，通过判断`add`函数的第一个参数是不是指向了`dense`，来找到`dense=>add`的调用对；
    2. 利用`bb.BlockBuilder`新建一个`relax`函数，该函数包含了对`dense=>add`的调用；
    3. 使用新创建的函数替换原来add函数的绑定value.
    4. 利用`transform.LowerWithRelayOpStrategyPass`对`relax.op`的函数调用进行降级，转换为对`tir`函数的调用；
    5. 【************】利用Pass：`relax.transform.FuseTIR()(ModelTIR)`对生成的`fused_dense_add#`函数进行Fuse，将其转换为一个`tir`函数；（否则无法利用`relax.vm.build(ModelFinal, target="llvm")`对其构建）

需要注意的地方：



# ResNet





参考资料：
- https://mlc.ai/zh/index.html
- https://tvm.hyper.ai/docs
- https://zh-v2.d2l.ai/