# resNeSt ，结合通道注意力分组和卷积分组的新网络
# resNeSt的安装 pip install git+https://github.com/zhanghang1989/ResNeSt

import resnest
from resnest.torch import resnest50

img = None
net = resnest50(pretrained=True)

output_tensor = net(img)
