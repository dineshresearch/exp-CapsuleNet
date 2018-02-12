import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training hyper-parameters 
epsilon = 1e-9
batch_size = 100
routing_iter = 3
m_plus_value = 0.9
m_minus_value = 0.1
lambda_value = 0.5

# Training Data loader
print "Loading Data..."
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
print "Data Loaded"

test_loader = torch.utils.data.DataLoader(
                            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
                            batch_size=batch_size, shuffle=True)

# squash non-linear function
def squash(vector):
    '''Squashing function corresponding to Eq. 1
    Args:
        vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
    Returns:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''
    # Calcualte vector norm
    vec_squared_norm = torch.sum((vector * vector), dim=-2, keepdim=True)

    # Calculate scalar factor
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / torch.sqrt(vec_squared_norm + epsilon)
    
    # Squash vector
    vec_squashed = scalar_factor * vector

    return (vec_squashed)

# negative likelihood loss
def NLLloss(v_length, target):
    v_length = v_length.view(batch_size, 10)
    logits = F.log_softmax(v_length, dim=1)
    loss = F.nll_loss(logits, target)

    return loss

# Margin loss function
def MultiMarginloss(v_length, target):
    v_length.data = v_length.data.view(batch_size, 10)
    loss_pos = F.multi_margin_loss(v_length, target, p=2, margin=m_plus_value)
    loss_neg = F.multi_margin_loss(v_length, target, p=2, margin=m_minus_value)

    return loss_neg + loss_pos

def Marginloss(v_length, target):
    print v_length.size()
    print target.size()
    m_plus = Variable(torch.Tensor(np.array(np.tile(m_plus_value, [batch_size, 10, 1, 1]))))
    m_minus = Variable(torch.Tensor(np.array(np.tile(m_minus_value, [batch_size, 10, 1, 1]))))
    lambda_val = Variable(torch.Tensor(np.array(np.tile(lambda_value, [batch_size, 10]))))
    zero = Variable(torch.zeros(batch_size, 10, 1, 1))
    ones = Variable(torch.ones(batch_size, 10))
    # [batch_size, 10, 1, 1]
    # max_l = max(0, m_plus-||v_c||)^2
    #max_l = torch.max(zero, m_plus - v_length)
    max_l = m_plus - v_length
    max_l = torch.mul(max_l, max_l)
    # max_r = max(0, ||v_c||-m_minus)^2
    #max_r = torch.max(zero, v_length - m_minus)
    max_r = v_length - m_minus
    max_r = torch.mul(max_r, max_r)
    assert max_l.shape == (batch_size, 10, 1, 1)

    # reshape: [batch_size, 10, 1, 1] => [batch_size, 10]
    max_l.data = max_l.data.view(batch_size, 10)
    max_r.data = max_r.data.view(batch_size, 10)

    # calc T_c: [batch_size, 10]
    T_c = target
    # [batch_size, 10], element-wise multiply
    L_c = torch.mul(T_c, max_l) + torch.mul(lambda_val, torch.mul((ones - T_c), max_r))

    margin_loss = torch.mean(torch.sum(L_c, dim=1))

    return margin_loss

def ReluMarginloss(v_length, target, x, recon):
    # margin loss
    left = F.relu(0.9 - v_length, inplace=True) ** 2
    right = F.relu(v_length - 0.1, inplace=True) ** 2

    margin_loss = target * left + 0.5 * (1. - target) * right
    margin_loss = torch.mean(torch.sum(margin_loss, dim=1))

    # reconstrcution loss
    recon_loss = nn.MSELoss(size_average=False)
    
    loss = (margin_loss + 0.005 * recon_loss(recon, x)) / x.size(0)

    return loss

# Define Capsule-Net
class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()

        # Conv1: [batch_size, 20, 20, 256]
        self.conv1 = nn.Conv2d(1, 256, kernel_size=9, stride=1)

        # Cpasule layer 1: [batch_size, 1152, 8, 1]
        self.caps_1 = nn.Conv2d(256, 32*8, kernel_size=9, stride=2)

        # Capsule layer 2: [batch_size, 10, 16, 1]
        w = torch.Tensor(1, 1152, 10, 8, 16)
        nn.init.normal(w)
        self.W = nn.Parameter(w)
        b = torch.zeros(1, 1, 10, 16, 1)
        self.bias = nn.Parameter(b)

        # Reconstruction Layers
        self.recon_fc_1 = nn.Linear(16, 512)
        self.recon_fc_2 = nn.Linear(512, 1024)
        self.recon_fc_3 = nn.Linear(1024, 784)

        # Debug flat and full-connected
        self.fc_debug_0 = nn.Linear(160, 50)
        self.fc_debug_1 = nn.Linear(50, 10)

    def forward(self, x, y):
        # conv layer1
        x = F.relu(self.conv1(x))

        # capsule layer1
        x = F.relu(self.caps_1(x))
        
        # capsule layer2
        b_ij = Variable(torch.Tensor(batch_size, 1152, 10, 1, 1), requires_grad=False)
        x = self.routing(x, b_ij, self.W)
        x = torch.squeeze(x, dim=1)

        # decoder layer
        v_length = torch.sqrt(torch.sum(torch.mul(x, x),
                                        dim=2, keepdim=True) + epsilon)
        v_length = v_length.view(batch_size, 10, 1, 1)

        masked_v = torch.matmul(torch.squeeze(x).view(batch_size, 16, 10), y.view(-1, 10, 1))

        # reconstruction layer
        vector_j = masked_v.view(batch_size, 16)
        fc1 = self.recon_fc_1(vector_j)
        fc2 = self.recon_fc_2(fc1)
        reconstruction = F.sigmoid(self.recon_fc_3(fc2))

        return v_length, reconstruction

        x = x.view(-1, 160)
        x = F.relu(self.fc_debug_0(x))

        x = self.fc_debug_1(x)
        return F.log_softmax(x)

    def routing(self, x, b_IJ, W):
        # Tiling input
        x1 = x.view(batch_size, 256, 1, 6, 6)
        x_tile = x1.repeat(1, 1, 10, 1, 1)
        x_view = x_tile.view(batch_size, 1152, 10, 8, 1)
        W_tile = W.repeat(batch_size, 1, 1, 1, 1)
        W_view = W_tile.view(batch_size, 1152, 10, 16, 8)

        u_hat = torch.matmul(W_view, x_view)
        
        # clone u_hat for intermediate routing iters
        u_hat_stopped = Variable(u_hat.data.clone(), requires_grad=False)

        # routing
        #print "Start routing..."
        for r_iter in range(routing_iter):
            c_IJ = F.softmax(b_IJ, dim=2)

            # last iteration
            if r_iter == routing_iter - 1:
                s_J = torch.mul(c_IJ, u_hat)
                s_J_sum = torch.sum(s_J, dim=1, keepdim=True) + self.bias
                V_J = squash(s_J_sum)

            # routing ieration
            if r_iter < routing_iter - 1:
                #u_hat_stopped_0 = u_hat_stopped.view(batch_size, 1152, 10, 16, 1)
                #s_J_tmp = torch.mul(c_IJ, u_hat_stopped_0)
                #s_J_tmp_sum = torch.sum(s_J_tmp, dim=1, keepdim=True) + self.bias
                #V_J_tmp = squash(s_J_tmp_sum)

                # Tile V_J
                #V_J_tmp_tiled = V_J_tmp.repeat(1, 1152, 1, 1, 1)
                #u_hat_stopped_1 = u_hat_stopped_0.view(batch_size, 1152, 10, 1, 16)

                # update b_IJ
                #u_produce_v = torch.matmul(u_hat_stopped_1, V_J_tmp_tiled)
            #    assert u_produce_v.size() == (batch_size, 1152, 10, 1, 1)

                #b_IJ += u_produce_v

                # implement with numpy operations
                u_hat_stopped_tmp = u_hat_stopped.data.numpy()
                u_hat_stopped_tmp = np.reshape(u_hat_stopped_tmp, (batch_size, 1152, 10, 16, 1))
                c_IJ_tmp = c_IJ.data.numpy()
                s_J_tmp = c_IJ_tmp * u_hat_stopped_tmp
                s_J_tmp_sum = np.sum(s_J_tmp, axis=1, keepdims=True) + self.bias.data.numpy()
                V_J_tmp = squash(torch.Tensor(s_J_tmp_sum))

                V_J_tmp_tiled = np.tile(V_J_tmp.numpy(), (1, 1152, 1, 1, 1))
                u_hat_stopped_tmp = np.reshape(u_hat_stopped_tmp, (batch_size, 1152, 10, 1, 16))

                u_produce_v = np.matmul(u_hat_stopped_tmp, V_J_tmp_tiled)

                b_IJ.data += torch.Tensor(u_produce_v)

        #print "Finished routing"
        return V_J

def one_hot(labels):
    # make one hot label
    # input size: [batch_size]
    out = []
    for i, label in enumerate(labels):
        out.append(np.zeros(10))
        out[i][label] = 1
    assert np.array(out).shape == (batch_size, 10)
    
    return np.array(out)

print "Building model..."
model = CapsNet()
optimizer = optim.Adam(model.parameters())
for param in model.parameters():
    print param.size()
print "Model built"

print "Start training..."
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        target = torch.Tensor(one_hot(target))
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output, recon = model(data, target)
        #print output.size()
        #loss = F.nll_loss(output, target)
        #loss = NLLloss(output, target)
        loss = ReluMarginloss(output, target, data, recon)
        loss.backward()
        optimizer.step()
        if batch_idx % 1 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # Turn data batch into variables
        data, target = Variable(data, volatile=True), Variable(target)
        
        # Get model output
        output = model(data)

        # Computing loss
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for i in range(2):
    train(i)
test()