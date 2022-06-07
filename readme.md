融合模型中的 conv + bn 层, 加速模型推理速度

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


> https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/fusion.py

yolov5 fuse conv and bn layer

> https://github.com/ultralytics/yolov5/blob/master/models/yolo.py