### 融合模型中的 conv + bn 层, 加速模型推理速度

推导过程:

![BN2](https://github.com/youngx123/pic/blob/main/BN2.png?raw=true)

![BN3](https://github.com/youngx123/pic/blob/main/BN3.png?raw=true)

根据以上公式， 可以将其表示为矩阵形式为：

![BN4](https://github.com/youngx123/pic/blob/main/BN4.jpg?raw=true)

进而可以表示为：

![BN5](https://github.com/youngx123/pic/blob/main/BN5.png?raw=true)

nn.BatchNorm2d中参数和超参数表示：

![nn.BatchNorm参数](https://github.com/youngx123/pic/blob/main/bn_weight.png?raw=true)

基础模块
```python
class Conv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.relu(self.conv(x))
```

融合conv + bn 层
```python
for m in model.modules():
    if isinstance(m, Conv) and hasattr(m, "bn"):
        m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
        delattr(m, 'bn')            # remove batchnorm
        m.forward = m.forward_fuse  # update forward
```

### 卷积核参数重构
卷积核参数重构在推理阶段进行，主要目的还是加入模型的推理速度，因此在卷积核参数重构时，最好可以先进行 卷积层和 BN 层的参数融合

`structural_reparam.py` 为 `ACNet`网络结构，通过函数
```python
def _add_to_square_kernel(self, square_kernel, asym_kernel):
    asym_h = asym_kernel.size(2)
    asym_w = asym_kernel.size(3)
    square_h = square_kernel.size(2)
    square_w = square_kernel.size(3)
    square_kernel[:, :, square_h // 2 - asym_h // 2: square_h // 2 - asym_h // 2 + asym_h,
    square_w // 2 - asym_w // 2: square_w // 2 - asym_w // 2 + asym_w] += asym_kernel
```
`replknet.py`为 `replknet`网络结构， 实现将小卷积核和大卷积核进行参数重构
```python
def get_equivalent_kernel_bias(self):
    eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
    if hasattr(self, 'small_conv'):
        small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
        eq_b += small_b
        #   add to the central part
        eq_k += nn.functional.pad(small_k, [(self.kernel_size - self.small_kernel) // 2] * 4)
    return eq_k, eq_b
```
将不同卷积核参数进行重构，得到新的卷积权重和偏置

> https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/fusion.py

yolov5 fuse conv and bn layer

> https://github.com/ultralytics/yolov5/blob/master/models/yolo.py

ACNet
> https://github.com/DingXiaoH/ACNet/blob/master/acnet/acb.py

replknet 
> https://github.com/DingXiaoH/RepLKNet-pytorch/blob/main/replknet.py