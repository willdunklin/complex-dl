import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ComplexLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weights = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.cfloat))

        if bias:
            self.bias = nn.Parameter(torch.randn(out_features, dtype=torch.cfloat))
        else:
            self.bias = torch.zeros(out_features, dtype=torch.cfloat)

    def forward(self, x):
        # x is a tensor of shape (batch_size, in_features)
        # weights is a tensor of shape (out_features, in_features)
        # bias is a tensor of shape (out_features)

        # x = x.unsqueeze(1)
        # weights = weights.unsqueeze(0)
        # print(x.shape, weights.shape, bias.shape)
        # exit()
        # x = torch.matmul(x, weights.t()) + bias
        x = torch.matmul(x, self.weights.t()) + self.bias
        return x


    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_features}, {self.out_features}, bias={sum(self.bias) != 0})')

class ComplexLReLU(nn.Module):
    def __init__(self):
        super(ComplexLReLU, self).__init__()
        self.proj = nn.Parameter(torch.randn(1, dtype=torch.cfloat))

    def forward(self, x):
        # in leakyrelu-like fashion, transform values with negative real and complex parts
        # multiple all (<0,<0) values by self.proj
        x[torch.logical_and(x.imag < 0, x.real < 0)] *= self.proj
        return x

class ComplexRescale(nn.Module):
    def __init__(self):
        super(ComplexRescale, self).__init__()

    def forward(self, x):
        # assuming x is a flat complex valued tensor
        max_magnitude = torch.max(torch.abs(x))
        x /= max_magnitude
        return x

class ComplexLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(ComplexLayerNorm, self).__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape
        self.gamma = nn.Parameter(torch.ones(1, normalized_shape))
        self.beta = nn.Parameter(torch.zeros(1, normalized_shape))

    def forward(self, x):
        x_real = x.real
        x_imag = x.imag

        mean_real = x_real.mean(dim=-1, keepdim=True)
        mean_imag = x_imag.mean(dim=-1, keepdim=True)
        var_real = x_real.var(dim=-1, keepdim=True)
        var_imag = x_imag.var(dim=-1, keepdim=True)

        x_real = (x_real - mean_real) / (var_real + self.eps).sqrt()
        x_imag = (x_imag - mean_imag) / (var_imag + self.eps).sqrt()

        x_real = self.gamma * x_real + self.beta
        x_imag = self.gamma * x_imag + self.beta

        return torch.complex(x_real, x_imag)

class ComplexAgent(nn.Module):
    def __init__(self):
        super(ComplexAgent, self).__init__()
        self.in_features = 1024 # 32 * 32
        self.out_features = 10

        self.clrelu = lambda x: x
        # self.clrelu = ComplexLReLU()

        self.fc1 = ComplexLinear(self.in_features, 128)
        self.fc2 = ComplexLinear(128, 80)
        self.fc3 = ComplexLinear(80, self.out_features)

        self.cln1 = ComplexLayerNorm(128)
        self.cln2 = ComplexLayerNorm(80)
        self.cln3 = ComplexLayerNorm(self.out_features)

        self.c2r = nn.Linear(self.out_features * 2, self.out_features)

    def forward(self, x):
        # print(x.shape)
        x = x.view(-1, self.in_features)
        # print(x.shape)
        # print([z.shape for z in list(self.fc1.parameters())])
        # print([z.shape for z in list(self.c2r.parameters())])
        # exit()
        x = self.clrelu(self.fc1(x))
        x = self.cln1(x)

        x = self.clrelu(self.fc2(x))
        x = self.cln2(x)

        x = self.clrelu(self.fc3(x))
        x = self.cln3(x)

        x = torch.cat((x.real, x.imag), dim=1)
        x = self.c2r(x)

        x = nn.functional.softmax(x, dim=1)

        return x


def test_complex_linear():
    in_features = 4
    out_features = 10
    c = ComplexLinear(in_features, out_features)
    b = ComplexLinear(in_features, out_features, bias=False)
    print('c', c)
    print('c.params\n', list(c.parameters()))
    print()
    print('b', b)
    print('b.params\n', list(b.parameters()))
    print()

    print()
    x = torch.randn(in_features, dtype=torch.cfloat)
    print('x:', x, '\n')
    print('c(x) =\n', c(x))
    print()
    print('b(x) =\n', b(x))
    print()


# if __name__ == '__main__':
#     test_complex_linear()

class PIL_RGB2HSV_Transform:
    def __call__(self, img):
        # assuming that img has been converted to PIL.Image.Image with transforms.ToPILImage()
        img = img.convert('HSV')
        return img


class HSV2Complex_Transform:
    def __call__(self, img):
        # assuming img is an HSV tensor

        # remap hue from 0-1 to 0-2pi
        hue = img[0] * 2 * 3.14159
        value = img[2]

        # construct complex tensor where hue is the angle and value is the magnitude
        img = torch.complex(value * torch.cos(hue), value * torch.sin(hue))

        return img

transform = transforms.Compose([
    PIL_RGB2HSV_Transform(), # convert to HSV
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    HSV2Complex_Transform(), # convert to complex tensor
])


batch_size = 8

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

for data in trainloader:
    inputs, labels = data
    # print(inputs.shape, labels.shape)
    # print(inputs)
    # print(labels)
    break

agent = ComplexAgent()
print(agent)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(agent.parameters(), lr=0.001, momentum=0.9)

best_acc = 0

for epoch in range(50):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = agent(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0
    print(f'Finished epoch {epoch + 1}')

    print('\nTesting network')
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = agent(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')

    if correct / total > best_acc:
        best_acc = correct / total
        print(f'New best accuracy: {best_acc * 100}%')

        # save the model
        path = f'./models/cifar_net_epoch{epoch}_acc{int(best_acc * 100)}.pth'
        torch.save(agent.state_dict(), path)
        print(f'Saved model to {path}')
