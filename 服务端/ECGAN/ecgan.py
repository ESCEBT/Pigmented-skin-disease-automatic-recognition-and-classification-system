import math
import numpy as np
import matplotlib.pyplot as plt
import pdb

# torch imports
import torch
from torch.utils.data import DataLoader,Dataset
from torch import optim,nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.autograd import Variable
from dataLoader import MyDataset
from resNet import ResNet, BasicBlock


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# DC Generator
class DCGAN_generator(nn.Module):
  """

  Attributes
  ----------
    ngpu : int
      The number of available GPU devices

  """
  def __init__(self, ngpu):
    """Init function

    Parameters
    ----------
      ngpu : int
        The number of available GPU devices

    """
    super(DCGAN_generator, self).__init__()
    self.ngpu = ngpu
    
    nz = 100 # noise dimension
    ngf = 64 # number of features map on the first layer
    nc = 3 # number of channels

    self.main = nn.Sequential(
      # input is Z, going into a convolution
      nn.ConvTranspose2d(     nz, ngf * 4, 4, 1, 0, bias=False),
      nn.BatchNorm2d(ngf * 4),
      nn.ReLU(True),
      # state size. (ngf*8) x 4 x 4
      nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf * 2),
      nn.ReLU(True),
      # state size. (ngf*4) x 8 x 8
      nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf),
      nn.ReLU(True),
      # state size. (ngf*2) x 16 x 16
      nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
      nn.Tanh()
      # state size. (nc) x 64 x 64
    )

  def forward(self, input):
    """Forward function

    Parameters
    ----------
    input : :py:class:`torch.Tensor`
    
    Returns
    -------
    :py:class:`torch.Tensor`
      the output of the generator (i.e. an image)

    """
    output = self.main(input)
    return output


# Discriminator
class DCGAN_discriminator(nn.Module):
  """ 

  Attributes
  ----------
    ngpu : int
      The number of available GPU devices

  """
  def __init__(self, ngpu):
    """Init function

    Parameters
    ----------
      ngpu : int
        The number of available GPU devices

    """
    super(DCGAN_discriminator, self).__init__()
    self.ngpu = ngpu
        
    ndf = 64
    nc = 3
       
    self.main = nn.Sequential(
      nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ndf),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ndf * 2),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*4) x 8 x 8
      nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ndf * 4),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*8) x 4 x 4
      nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
      nn.Sigmoid()
    )

  def forward(self, input):
    """Forward function

    Parameters
    ----------
    input : :py:class:`torch.Tensor`
    
    Returns
    -------
    :py:class:`torch.Tensor`
      the output of the generator (i.e. an image)

    """
    output = self.main(input)

    return output.view(-1, 1).squeeze(1)

# Data Propcessing
file = open("StandardClassifier.txt", "w")
# 读取数据
train_img_data = MyDataset("../data/cifar10/train.txt", train="train", transform=transforms.ToTensor())
test_img_data = MyDataset("../data/cifar10/test.txt", train="test", transform=transforms.ToTensor())

transform = transforms.Compose(
    [transforms.ToTensor()])

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# regular data loaders
batch_size = 64
train_dataloader = torch.utils.data.DataLoader(
    train_img_data,
    batch_size=batch_size,
    shuffle=True,
)

test_dataloader = torch.utils.data.DataLoader(
    train_img_data,
    batch_size=batch_size,
    shuffle=True,
)

# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data for plotting purposes
generatorLosses = []
discriminatorLosses = []
classifierLosses = []

#training starts

epochs = 100

# models
netG = DCGAN_generator(1)
netD = DCGAN_discriminator(1)
netC = ResNet18()

netG.to(device)
netD.to(device)
netC.to(device)

# optimizers 
optD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay = 1e-3)
optG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optC = optim.Adam(netC.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay = 1e-3)

advWeight = 0.1 # adversarial weight

loss = nn.BCELoss()
criterion = nn.CrossEntropyLoss()

file = open("ExternalClassifier.txt", "w")


# Train Loop
def train(train_dataloader):
  # file.write(text)
  for epoch in range(epochs):
    netC.train()

    running_loss = 0.0
    total_train = 0
    correct_train = 0
    for i, data in enumerate(train_dataloader, 0):
      
      dataiter = iter(train_dataloader)
      inputs, labels = dataiter.next()
      inputs, labels = inputs.to(device), labels.to(device) # label shape is [64]

      tmpBatchSize = len(labels)

      real_cpu = data[0].to(device)
      batch_size = real_cpu.size(0)

      # reshape -> 64 *1
      # labels = torch.reshape(labels, (batch_size, 1))

      # create label arrays ， [64 * 1]
      true_label = torch.ones(tmpBatchSize, 1, device=device)
      fake_label = torch.zeros(tmpBatchSize, 1, device=device)

      # 64 * 100 * 1 * 1 的随机向量
      r = torch.randn(tmpBatchSize, 100, 1, 1, device=device)
      fakeImageBatch = netG(r)

      # train discriminator on real images
      predictionsReal = netD(inputs)
      lossDiscriminator = loss(predictionsReal, true_label) #labels = 1
      lossDiscriminator.backward(retain_graph = True)

      # train discriminator on fake images
      predictionsFake = netD(fakeImageBatch)
      lossFake = loss(predictionsFake, fake_label) #labels = 0
      lossFake.backward(retain_graph= True)
      optD.step() # update discriminator parameters    

      # train generator 
      optG.zero_grad()
      predictionsFake = netD(fakeImageBatch)
      lossGenerator = loss(predictionsFake, true_label) #labels = 1
      lossGenerator.backward(retain_graph = True)
      optG.step()

      torch.autograd.set_detect_anomaly(True)
      fakeImageBatch = fakeImageBatch.detach().clone()

      # train classifier on real data
      predictions = netC(inputs)
      # print(labels.size())
      realClassifierLoss = criterion(predictions, labels)
      realClassifierLoss.backward(retain_graph=True)
      
      optC.step()
      optC.zero_grad()

      # update the classifer on fake data
      predictionsFake = netC(fakeImageBatch)
      # get a tensor of the labels that are most likely according to model
      predictedLabels = torch.argmax(predictionsFake, 1) # -> [0 , 5, 9, 3, ...]
      confidenceThresh = .2

      # psuedo labeling threshold
      probs = F.softmax(predictionsFake, dim=1)
      mostLikelyProbs = np.asarray([probs[i, predictedLabels[i]].item() for  i in range(len(probs))])
      toKeep = mostLikelyProbs > confidenceThresh
      if sum(toKeep) != 0:
          fakeClassifierLoss = criterion(predictionsFake[toKeep], predictedLabels[toKeep]) * advWeight
          fakeClassifierLoss.backward()
          
      optC.step()

      # reset the gradients
      optD.zero_grad()
      optG.zero_grad()
      optC.zero_grad()

      # save losses for graphing
      generatorLosses.append(lossGenerator.item())
      discriminatorLosses.append(lossDiscriminator.item())
      classifierLosses.append(realClassifierLoss.item())

      # get train accurcy 
      if(i % 10 == 0):
        netC.eval()
        # accuracy
        _, predicted = torch.max(predictions, 1)
        total_train += labels.size(0)
        correct_train += predicted.eq(labels.data).sum().item()
        train_accuracy = 100 * correct_train / total_train
        text = ("Train Accuracy: " + str(train_accuracy))
        print("Epoch {}, {}".format(i, text))
        file.write(text + '\n')
        netC.train()

    print("Epoch " + str(epoch) + "Complete")
    
    # save gan image
    gridOfFakeImages = torchvision.utils.make_grid(fakeImageBatch.cpu())
    torchvision.utils.save_image(gridOfFakeImages, "/content/gridOfFakeImages/" + str(epoch) + '_' + str(i) + '.png')
    validate()

def validate():
  netC.eval()
  correct = 0
  total = 0
  with torch.no_grad():
      for data in test_dataloader:
          inputs, labels = data
          inputs, labels = data[0].to(device), data[1].to(device)
          outputs = netC(inputs)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

  accuracy = (correct / total) * 100 

  print('Accuracy of the network on the 10000 test images: %d %%' % (
      100 * correct / total))

  text = ("Test Accuracy: " + str(accuracy) + "\n")
  file.write(text)
  netC.train()

train(train_dataloader)
file.close()