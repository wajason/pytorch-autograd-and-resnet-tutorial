"""
Implements pytorch autograd and nn in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""

import torch
import torch.nn as nn
from helper import *
import torch.nn.functional as F
import torch.optim as optim

def hello():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from pytorch_autograd_and_nn.py!')



################################################################################
# Part II. Barebones PyTorch                         
################################################################################
# Before we start, we define the flatten function for your convenience.
def flatten(x, start_dim=1, end_dim=-1):
  return x.flatten(start_dim=start_dim, end_dim=end_dim)


def three_layer_convnet(x, params):
  """
  Performs the forward pass of a three-layer convolutional network with the
  architecture defined above.

  Inputs:
  - x: A PyTorch Tensor of shape (N, C, H, W) giving a minibatch of images.
       N = batch size (一次丟進網路的圖片數，例如 64) 
       C = channel 數（彩色圖像 3） 
       H, W = 高度、寬度（這裡是 32 × 32）
       params: 一個 list，存放所有網路的權重 (weights) 和偏差 (biases)。
  - params: A list of PyTorch Tensors giving the weights and biases for the
    network; should contain the following:
    - conv_w1: PyTorch Tensor of shape (channel_1, C, KH1, KW1) giving weights
      for the first convolutional layer
      第一層卷積的權重，形狀(channel_1,𝐶,𝐾𝐻1,𝐾𝑊1)，設計裡是 (32, 3, 5, 5)，
      表示有 32 個 filter，每個 filter 吃 3 個通道，大小是 5×5。
    - conv_b1: PyTorch Tensor of shape (channel_1,) giving biases for the first
      convolutional layer
      第一層卷積的偏差，形狀 (channel_1,) = (32,)。
    - conv_w2: PyTorch Tensor of shape (channel_2, channel_1, KH2, KW2) giving
      weights for the second convolutional layer
      設計裡是 (16, 32, 3, 3)，表示有 16 個 filter，每個 filter 吃 32 個通道，大小是 3×3。
    - conv_b2: PyTorch Tensor of shape (channel_2,) giving biases for the second
      convolutional layer
      第二層卷積的偏差，形狀 (channel_2,) = (16,)。
    - fc_w: PyTorch Tensor giving weights for the fully-connected layer. Can you
      figure out what the shape should be?
      這裡是 (10, 16×32×32) = (10, 16384)。
    - fc_b: PyTorch Tensor giving biases for the fully-connected layer. Can you
      figure out what the shape should be?
      fully connected 層的偏差，形狀 (num_classes,) = (10,)。
  
  Returns:
  - scores: PyTorch Tensor of shape (N, C) giving classification scores for x
  """
  conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b = params
  scores = None
  ########################################################################################
  # TODO: Implement the forward pass for the three-layer ConvNet.                        #
  # The network have the following architecture:                                         #
  #   1. Conv layer (with bias) with 32 5x5 filters, with zero-padding of 2              #
  # (N, 3, 32, 32) → (N, 32, 32, 32)，padding=2 保持 H, W 不變，(32 + 2*2 - 5)/1 + 1 = 32 #
  out = F.conv2d(x, conv_w1, bias=conv_b1, stride=1, padding=2)
  #   2. ReLU                                                                            #
  # (N, 32, 32, 32)
  out = F.relu(out)
  #   3. Conv layer (with bias) with 16 3x3 filters, with zero-padding of 1              #
  # (N, 32, 32, 32) → (N, 16, 32, 32)，padding=1 保持 H, W 不變，(32 + 2*1 - 3)/1 + 1 = 32#
  out = F.conv2d(out, conv_w2, bias=conv_b2, stride=1, padding=1)
  #   4. ReLU                                                                            #
  # (N, 16, 32, 32)
  out = F.relu(out)
  #   5. Fully-connected layer (with bias) to compute scores for 10 classes              #
  # (N, 16*32*32) → (N, 10)                                                              #
  out = flatten(out)
  scores = F.linear(out, fc_w, bias=fc_b)
  # Hint: F.linear, F.conv2d, F.relu, flatten (implemented above)                        #         
  ########################################################################################
  # Replace "pass" statement with your code
  # pass
  ########################################################################################
  #                                  END OF YOUR CODE                                    #
  ########################################################################################
  return scores


def initialize_three_layer_conv_part2(dtype=torch.float, device='cpu'):
  '''
  Initializes weights for the three_layer_convnet for part II
  Inputs:
    - dtype: A torch data type object; all computations will be performed using
        this datatype. float is faster but less accurate, so you should use
        double for numeric gradient checking.
      - device: device to use for computation. 'cpu' or 'cuda'
  '''
  # Input/Output dimenssions
  C, H, W = 3, 32, 32
  num_classes = 10

  # Hidden layer channel and kernel sizes
  channel_1 = 32
  channel_2 = 16
  kernel_size_1 = 5
  kernel_size_2 = 3

  # Initialize the weights
  conv_w1 = None
  conv_b1 = None
  conv_w2 = None
  conv_b2 = None
  fc_w = None
  fc_b = None

  #####################################################################################
  # TODO: Define and initialize the parameters of a three-layer ConvNet               #
  # using nn.init.kaiming_normal_. You should initialize your bias vectors            #
  # using the zero_weight function.                                                   #
  # You are given all the necessary variables above for initializing weights.         #
  #####################################################################################
  # Replace "pass" statement with your code
  # Conv1: (32, 3, 5, 5)
  # 初始化時先開一個 空張量（torch.empty），再用 nn.init 的方法把它填滿。
  # Kaiming initialization是一種專門為 ReLU / LeakyReLU 這類非線性激活設計的初始化方式。
  # 它的數學原理是保持每層輸入輸出 variance 大致一致，避免梯度爆炸或消失。
  # 在 Conv2d 的 weight shape 裡是 (out_channels, in_channels, kernel_H, kernel_W)
  conv_w1 = torch.empty(channel_1, C, kernel_size_1, kernel_size_1, 
                        dtype=dtype, device=device, requires_grad=True)
  # 用高斯分布(normal distribution)來初始化權重，並根據 fan-in 自動調整標準差。
  # 後面的底線 _ 表示 in-place，直接修改原來的 tensor。
  nn.init.kaiming_normal_(conv_w1)
  conv_b1 = torch.zeros(channel_1, dtype=dtype, device=device, requires_grad=True)

  # Conv2: (16, 32, 3, 3)
  conv_w2 = torch.empty(channel_2, channel_1, kernel_size_2, kernel_size_2, 
                        dtype=dtype, device=device, requires_grad=True)
  nn.init.kaiming_normal_(conv_w2, nonlinearity='relu')
  conv_b2 = torch.zeros(channel_2, dtype=dtype, device=device, requires_grad=True)

  # FC: input_dim = 16 * 32 * 32
  fc_in_dim = channel_2 * H * W
  fc_w = torch.empty(num_classes, fc_in_dim, 
                     dtype=dtype, device=device, requires_grad=True)
  nn.init.kaiming_normal_(fc_w, nonlinearity='linear')
  fc_b = torch.zeros(num_classes, dtype=dtype, device=device, requires_grad=True)

  #####################################################################################
  #                                  END OF YOUR CODE                                 #
  #####################################################################################
  return [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]




################################################################################
# Part III. PyTorch Module API                         
################################################################################

class ThreeLayerConvNet(nn.Module):
  def __init__(self, in_channel, channel_1, channel_2, num_classes):
    super().__init__()
    #####################################################################################
    # TODO: Set up the layers you need for a three-layer ConvNet with the               #
    # architecture defined below. You should initialize the weight  of the              #
    # model using Kaiming normal initialization, and zero out the bias vectors.         #
    #                                                                                   #
    # The network architecture should be the same as in Part II:                        #
    #   1. Convolutional layer with channel_1 5x5 filters with zero-padding of 2        #
    #   2. ReLU                                                                         #
    #   3. Convolutional layer with channel_2 3x3 filters with zero-padding of 1        #
    #   4. ReLU                                                                         #
    #   5. Fully-connected layer to num_classes classes                                 #
    #                                                                                   #
    # We assume that the size of the input of this network is `H = W = 32`, and         #
    # there is no pooing; this information is required when computing the number        #
    # of input channels in the last fully-connected layer.                              #
    #                                                                                   #
    # HINT: nn.Conv2d, nn.init.kaiming_normal_, nn.init.zeros_                          #
    #####################################################################################
    # Replace "pass" statement with your code
    #   1. Convolutional layer with channel_1 5x5 filters with zero-padding of 2 
    self.conv1 = nn.Conv2d(in_channel, channel_1, kernel_size=5, padding=2)
    #   3. Convolutional layer with channel_2 3x3 filters with zero-padding of 1
    self.conv2 = nn.Conv2d(channel_1, channel_2, kernel_size=3, padding=1)
    #   5. Fully-connected layer to num_classes classes 
    self.fc = nn.Linear(channel_2 * 32 * 32, num_classes)

    # 初始化
    nn.init.kaiming_normal_(self.conv1.weight)
    nn.init.zeros_(self.conv1.bias)
    nn.init.kaiming_normal_(self.conv2.weight)
    nn.init.zeros_(self.conv2.bias)
    nn.init.kaiming_normal_(self.fc.weight)
    nn.init.zeros_(self.fc.bias)
    #####################################################################################
    #                                  END OF YOUR CODE                                 #
    #####################################################################################

  def forward(self, x):
    scores = None
    #####################################################################################
    # TODO: Implement the forward function for a 3-layer ConvNet. you                   #
    # should use the layers you defined in __init__ and specify the                     #
    # connectivity of those layers in forward()                                         #
    # Hint: flatten (implemented at the start of part II)                               #
    #####################################################################################
    # Replace "pass" statement with your code
    out = F.relu(self.conv1(x))
    out = F.relu(self.conv2(out))
    out = flatten(out)
    scores = self.fc(out)
    #####################################################################################
    #                                  END OF YOUR CODE                                 #
    #####################################################################################
    return scores


def initialize_three_layer_conv_part3():
  '''
  Instantiates a ThreeLayerConvNet model and a corresponding optimizer for part III
  '''

  # Parameters for ThreeLayerConvNet
  C = 3
  num_classes = 10

  channel_1 = 32
  channel_2 = 16

  # Parameters for optimizer
  learning_rate = 3e-3
  weight_decay = 1e-4

  model = None
  optimizer = None
  #####################################################################################
  # TODO: Instantiate ThreeLayerConvNet model and a corresponding optimizer.          #
  # Use the above mentioned variables for setting the parameters.                     #
  # You should train the model using stochastic gradient descent without              #
  # momentum, with L2 weight decay of 1e-4.                                           #
  #####################################################################################
  # Replace "pass" statement with your code
  model = ThreeLayerConvNet(C, channel_1, channel_2, num_classes)
  optimizer = optim.SGD(model.parameters(), lr=3e-3, weight_decay=1e-4)
  #####################################################################################
  #                                  END OF YOUR CODE                                 #
  #####################################################################################
  return model, optimizer


################################################################################
# Part IV. PyTorch Sequential API                        
################################################################################

# Before we start, We need to wrap `flatten` function in a module in order to stack it in `nn.Sequential`.
# As of 1.3.0, PyTorch supports `nn.Flatten`, so this is not required in the latest version.
# However, let's use the following `Flatten` class for backward compatibility for now.
class Flatten(nn.Module):
  def forward(self, x):
    return flatten(x)


def initialize_three_layer_conv_part4():
  '''
  Instantiates a ThreeLayerConvNet model and a corresponding optimizer for part IV
  '''
  # Input/Output dimenssions
  C, H, W = 3, 32, 32
  num_classes = 10

  # Hidden layer channel and kernel sizes
  channel_1 = 32
  channel_2 = 16
  kernel_size_1 = 5
  pad_size_1 = 2
  kernel_size_2 = 3
  pad_size_2 = 1

  # Parameters for optimizer
  learning_rate = 1e-2
  weight_decay = 1e-4
  momentum = 0.5

  model = None
  optimizer = None
  #####################################################################################
  # TODO: Rewrite the 3-layer ConvNet with bias from Part III with Sequential API and #
  # a corresponding optimizer.                                                        #
  # You don't have to re-initialize your weight matrices and bias vectors.            #
  # Here you should use `nn.Sequential` to define a three-layer ConvNet with:         #
  #   1. Convolutional layer (with bias) with 32 5x5 filters, with zero-padding of 2  #
  #   2. ReLU                                                                         #
  #   3. Convolutional layer (with bias) with 16 3x3 filters, with zero-padding of 1  #
  #   4. ReLU                                                                         #
  #   5. Fully-connected layer (with bias) to compute scores for 10 classes           #
  #                                                                                   #
  # You should optimize your model using stochastic gradient descent with Nesterov    #
  # momentum 0.5, with L2 weight decay of 1e-4 as given in the variables above.       #
  # Hint: nn.Sequential, Flatten (implemented at the start of Part IV)                #
  #####################################################################################
  # Replace "pass" statement with your code
  model = nn.Sequential(
      #   1. Convolutional layer (with bias) with 32 5x5 filters, with zero-padding of 2 
      nn.Conv2d(C, channel_1, kernel_size=5, padding=2),
      #   2. ReLU 
      nn.ReLU(),
      #   3. Convolutional layer (with bias) with 16 3x3 filters, with zero-padding of 1
      nn.Conv2d(channel_1, channel_2, kernel_size=3, padding=1),
      #   4. ReLU 
      nn.ReLU(),
      #   5. Fully-connected layer (with bias) to compute scores for 10 classes 
      Flatten(),
      nn.Linear(channel_2 * H * W, num_classes)
  )

  optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.5, 
                        weight_decay=1e-4, nesterov=True)
  #####################################################################################
  #                                  END OF YOUR CODE                                 #
  #####################################################################################
  return model, optimizer


################################################################################
# Part V. ResNet for CIFAR-10                        
################################################################################
# PlainBlock是一個「兩層卷積的模組」，跟 ResNet 的「ResidualBlock」差別是PlainBlock 沒有 shortcut（殘差連接）。
class PlainBlock(nn.Module):
  def __init__(self, Cin, Cout, downsample=False):
    super().__init__()

    self.net = None
    #############################################################################
    # TODO: Implement PlainBlock.                                               #
    # Hint: Wrap your layers by nn.Sequential() to output a single module.      #
    #       You don't have use OrderedDict.                                     #
    # Inputs:                                                                   #
    # - Cin: number of input channels                                           #
    # - Cout: number of output channels                                         #
    # - downsample: add downsampling (a conv with stride=2) if True             #
    # Store the result in self.net.                                             #
    #############################################################################
    # Replace "pass" statement with your code
    # 如果 downsample=True，stride=2，就會把影像 長寬各除以 2。
    stride = 2 if downsample else 1
    # BN -> ReLU -> Conv -> BN -> ReLU -> Conv
    self.net = nn.Sequential(
        # 對輸入進行歸一化，穩定訓練、加速收斂。先 BN 再 ReLU，再 Conv。
        nn.BatchNorm2d(Cin),
        nn.ReLU(inplace=True),
        # 第一層卷積，kernel=3，padding=1
        nn.Conv2d(Cin, Cout, kernel_size=3, stride=stride, padding=1, bias=False),
        # 再次做標準化
        nn.BatchNorm2d(Cout),
        nn.ReLU(inplace=True),
        # 第二層卷積，kernel=3，padding=1
        nn.Conv2d(Cout, Cout, kernel_size=3, stride=1, padding=1, bias=False),
    )
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

  def forward(self, x):
    return self.net(x)


class ResidualBlock(nn.Module):
  def __init__(self, Cin, Cout, downsample=False):
    super().__init__()

    self.block = None # F
    self.shortcut = None # G
    #############################################################################
    # TODO: Implement residual block using plain block. Hint: nn.Identity()     #
    # Inputs:                                                                   #
    # - Cin: number of input channels                                           #
    # - Cout: number of output channels                                         #
    # - downsample: add downsampling (a conv with stride=2) if True             #
    # Store the main block in self.block and the shortcut in self.shortcut.     #
    #############################################################################
    # Replace "pass" statement with your code
    # main path: use PlainBlock (which already handles downsample via stride)
    self.block = PlainBlock(Cin, Cout, downsample)

    # shortcut: identity if channels match and no downsample, else 1x1 conv
    stride = 2 if downsample else 1
    # 如果輸入輸出 維度不相同，或是 downsample=True，就用 1x1 卷積來改變維度。
    if Cin != Cout or downsample:
      self.shortcut = nn.Conv2d(Cin, Cout, kernel_size=1, stride=stride, bias=False)
    # 如果輸入輸出 維度相同，就直接用 nn.Identity()
    else:
      self.shortcut = nn.Identity()
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
  
  # 殘差學習（Residual Learning）y=F(x)+G(x)
  def forward(self, x):
    return self.block(x) + self.shortcut(x)


class ResNet(nn.Module):
  def __init__(self, stage_args, Cin=3, block=ResidualBlock, num_classes=10):
    super().__init__()

    self.cnn = None
    #############################################################################
    # TODO: Implement the convolutional part of ResNet using ResNetStem,        #
    #       ResNetStage, and wrap the modules by nn.Sequential.                 #
    # Store the model in self.cnn.                                              #
    #############################################################################
    # Replace "pass" statement with your code
    layers = [ResNetStem(Cin, 8)]
    in_channels = 8

    # stage_args can be flexible: either (Cout, num_blocks) or (Cin, Cout, num_blocks)
    for stage in stage_args:
      if len(stage) >= 3:
        # accept either (Cin, Cout, num_blocks) or (something, Cout, num_blocks)
        Cout_stage = stage[1]
        num_blocks = stage[2]
      elif len(stage) == 2:
        Cout_stage, num_blocks = stage
      else:
        raise ValueError("Each stage entry in stage_args must have 2 or 3 elements")

      Cin_stage = in_channels  # use the tracked in_channels (robust to different stage_args formatting)
      downsample = (Cin_stage != Cout_stage)
      layers.append(ResNetStage(Cin_stage, Cout_stage, num_blocks, downsample=downsample, block=block))
      in_channels = Cout_stage

    self.cnn = nn.Sequential(*layers)

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    self.fc = nn.Linear(stage_args[-1][1], num_classes)

  def forward(self, x):
    scores = None
    #############################################################################
    # TODO: Implement the forward function of ResNet.                           #
    # Store the output in `scores`.                                             #
    #############################################################################
    # Replace "pass" statement with your code
    out = self.cnn(x)                      # (N, C, H, W)
    out = F.adaptive_avg_pool2d(out, (1,1))# (N, C, 1, 1)
    out = out.view(out.size(0), -1)        # (N, C)
    scores = self.fc(out)                  # (N, num_classes)
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return scores


# 設計像「瓶頸」： 輸入通道 → 壓縮 (1×1 conv) → 處理 (3×3 conv) → 擴張 (1×1 conv) → 輸出通道。
# 這樣可以減少計算量，提升效率，同時保持模型的表達能力。
class ResidualBottleneckBlock(nn.Module):
  def __init__(self, Cin, Cout, downsample=False):
    super().__init__()

    self.block = None
    self.shortcut = None
    #############################################################################
    # TODO: Implement residual bottleneck block.                                #
    # Inputs:                                                                   #
    # - Cin: number of input channels                                           #
    # - Cout: number of output channels                                         #
    # - downsample: add downsampling (a conv with stride=2) if True             #
    # Store the main block in self.block and the shortcut in self.shortcut.     #
    #############################################################################
    # Replace "pass" statement with your code
    stride = 2 if downsample else 1
    # bottleneck 中間 channel，通常是 Cout 的 1/4
    mid = max(1, Cout // 4)

    self.block = nn.Sequential(
        nn.BatchNorm2d(Cin),
        nn.ReLU(inplace=True),
        nn.Conv2d(Cin, mid, kernel_size=1, stride=1, bias=False),  # stride=1 固定
        nn.BatchNorm2d(mid),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid, mid, kernel_size=3, stride=stride, padding=1, bias=False),  # 3x3 卷積
        nn.BatchNorm2d(mid),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid, Cout, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(Cout),   # 最後一個 BatchNorm
    )

    # 如果輸入輸出維度不同，或需要下採樣，shortcut 用 1×1 Conv 調整。
    if Cin != Cout or downsample:
      self.shortcut = nn.Conv2d(Cin, Cout, kernel_size=1, stride=stride, bias=False)
    # 如果輸入輸出 維度相同，就直接用 nn.Identity()
    else:
      self.shortcut = nn.Identity()
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

  def forward(self, x):
    return self.block(x) + self.shortcut(x)

##############################################################################
# No need to implement anything here                     
##############################################################################
class ResNetStem(nn.Module):
  def __init__(self, Cin=3, Cout=8):
    super().__init__()
    layers = [
        nn.Conv2d(Cin, Cout, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
    ]
    self.net = nn.Sequential(*layers)
    
  def forward(self, x):
    return self.net(x)

class ResNetStage(nn.Module):
  def __init__(self, Cin, Cout, num_blocks, downsample=True,
               block=ResidualBlock):
    super().__init__()
    blocks = [block(Cin, Cout, downsample)]
    for _ in range(num_blocks - 1):
      blocks.append(block(Cout, Cout))
    self.net = nn.Sequential(*blocks)
  
  def forward(self, x):
    return self.net(x)