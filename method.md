
## Neuron FR ==> Neuropil FR

计算果蝇**neuropil**（神经节/神经纤维层）的整体发放率，可以基于单个神经元的发放率进行加权平均。

按体积或突触数加权的发放率.  如果 neuropil 内不同区域的神经元密度不同，可以基于突触数或体积进行加权：

$r_{\text{neuropil}} = \frac{\sum_i r_i n_i}{\sum_i n_i}$

其中：

- $n_i$ 是与第 $i$ 个神经元相关的突触数量或该神经元所在区域的体积。

如果 neuropil 的某个子区具有较高的神经元密度，则该区域的神经元对整体发放率的贡献更大。


## Neuropil FR ==> Neuron FR

根据 **neuropil** 的平均发放率 $r_{\text{neuropil}}$，推算每个下游相连神经元接收到的输入发放率，可以使用突触加权平均的方法。这个问题可以建模为**突触传递和加权求和**，考虑以下几个关键因素：

近似为整体 neuropil 贡献. 

如果已知 **neuropil 的整体发放率** $r_{\text{neuropil}}$ ，可以用突触加权平均的方式估算：

$r_j^{\text{in}} = C_j r_{\text{neuropil}}$

其中：

- $C_j$ 是下游神经元 $j$ 从 neuropil 接收的**有效突触数**（可以用总突触数或有效输入突触数归一化）。
- $r_{\text{neuropil}}$ 是 neuropil 的平均发放率。

在这种情况下，假设 neuropil 内神经元的发放是均匀的，这种方法可用于大规模估计。