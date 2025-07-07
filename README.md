# DeepFUSE

## 说明
DeepFUSE 是一个基于DeepEP和DeepGEMM的计算通信overlap工具，用于加速分布式推理。

## 使用方式

### Tips
需要先查看`deep_ep/csrc/CMakeLists.txt`和`deep_gemm/include/CMakeLists.txt`和`fuse_kernel/CMakeLists.txt`下`Torch_DIR`变量，改成自己环境对应的torch路径。

### 1. 编译DeepEP

```shell
mkdir -p deep_ep/csrc/build
cd deep_ep/csrc/build
cmake ..
make
```

### 2. 编译DeepGEMM

#### a.代码生成
在`deep_gemm/gemm_codegen.py`中修改需要使用的参数范围，并运行`python gemm_codegen.py`生成代码。
```python
if __name__ == "__main__":
    generate_kernel_template(
        [32, 64, 128, 256, 512], # num_tokens_l
        [4096], # hidden_l
        [3072], # intermediate_l
        [4, 8, 16], # num_groups_l
        range(6, 66, 4) # sms_l
    )
```
该函数会生成两部分代码：
- `deep_gemm/kernels/*.cu`：使用不同模板参数生成的DeepGEMM kernels
- `fuse_kernel/src/gemm_gen.hpp`：主函数调用DeepGEMM的头文件

#### b.编译生成静态链接
这部分会比较耗时
```shell
mkdir -p deep_gemm/kernels/build
cd deep_gemm/kernels/build
cmake ..
make
```

### 3. 编译DeepFUSE
```shell
mkdir -p fuse_kernel/build
cd fuse_kernel/build
cmake ..
make
```

#### a. C++ 方式运行
```shell
cd fuse_kernel
bash begin.sh
```

#### b. Python 方式运行
##### 编译安装

```shell
source /mnt/data/nas/zhihao/zhl-sglang/bin/activate # your active env
export LD_LIBRARY_PATH=/mnt/data/nas/zhihao/zhl-sglang/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH # your torch lib path
python setup.py bdist_whel
```
```
pip install dist/*.whl
```

##### 在主函数中运行
```python
import deep_fuse_cpp
```
