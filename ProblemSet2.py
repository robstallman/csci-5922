#%%
import torch
import torch.nn as nn

#%% ----- Question 1 -----
# Define tensor
x = torch.ones([2,3,3,3])
x[0,0,:,:] = torch.tensor([[1,0,0],[0,1,0],[0,0,1]])
x[0,1,:,:] = torch.tensor([[0,2,0],[0,1,0],[2,1,0]])
x[0,2,:,:] = torch.tensor([[0,1,3],[3,0,0],[1,1,2]])
x[1,0,:,:] = torch.tensor([[0,1,0],[0,1,0],[1,0,1]])
x[1,1,:,:] = torch.tensor([[0,0,2],[0,0,2],[1,1,1]])
x[1,2,:,:] = torch.tensor([[0,3,1],[1,0,0],[0,3,0]])

# Calculate mean and variance
mean_x = torch.tensor([0.389, 0.722, 1.06])
var_x = torch.tensor([0.238, 0.645, 1.39])

# Normalize x
y = torch.ones_like(x)
for n in range(2):
    print(f"Batch {n+1}")
    for c in range(3):
        print(f"Channel {c+1}")
        y[n,c,:,:] = (x[n,c,:,:] - mean_x[c])/(var_x[c]**(1/2))
        print(y[n,c,:,:])

# Calculate leaky ReLU of x and normalized y
def leaky_ReLU(x):
    return torch.clamp(x, 0.1*x)

x_act = leaky_ReLU(x)
y_act = leaky_ReLU(y)

# Calculate distance
def get_distance(x):
    d = [0 for _ in range(3)]
    for c in range(3):
        d[c] = (((x[0,c,:,:] - x[1,c,:,:])**2).sum())**(1/2)
    return torch.tensor(d)

get_distance(x_act)

# %% ----- Question 4 ----- 
xb = torch.rand((3,224,224))

downsampling = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5),
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
    nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5),
    nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5),
    nn.Conv2d(in_channels=64, out_channels=80, kernel_size=5),
    nn.Conv2d(in_channels=80, out_channels=96, kernel_size=5),
    nn.Conv2d(in_channels=96, out_channels=112, kernel_size=5),
    nn.Conv2d(in_channels=112, out_channels=128, kernel_size=5),
    nn.Conv2d(in_channels=128, out_channels=144, kernel_size=5),
    nn.Conv2d(in_channels=144, out_channels=160, kernel_size=5),
    nn.Conv2d(in_channels=160, out_channels=176, kernel_size=5),
    nn.Conv2d(in_channels=176, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5),
)
upsampling = nn.Sequential(
    nn.ConvTranspose2d(in_channels=192, out_channels=192, kernel_size=11),
    nn.ConvTranspose2d(in_channels=192, out_channels=192, kernel_size=11),
    nn.ConvTranspose2d(in_channels=192, out_channels=192, kernel_size=11),
    nn.ConvTranspose2d(in_channels=192, out_channels=192, kernel_size=11),
    nn.ConvTranspose2d(in_channels=192, out_channels=192, kernel_size=11),
    nn.ConvTranspose2d(in_channels=192, out_channels=192, kernel_size=11),
    nn.ConvTranspose2d(in_channels=192, out_channels=192, kernel_size=11),
    nn.ConvTranspose2d(in_channels=192, out_channels=192, kernel_size=11),
    nn.ConvTranspose2d(in_channels=192, out_channels=192, kernel_size=11),
    nn.ConvTranspose2d(in_channels=192, out_channels=192, kernel_size=11),
    nn.ConvTranspose2d(in_channels=192, out_channels=192, kernel_size=11),
    nn.ConvTranspose2d(in_channels=192, out_channels=192, kernel_size=11),
    nn.ConvTranspose2d(in_channels=192, out_channels=192, kernel_size=11),
    nn.ConvTranspose2d(in_channels=192, out_channels=160, kernel_size=11),
    nn.ConvTranspose2d(in_channels=160, out_channels=128, kernel_size=11),
    nn.ConvTranspose2d(in_channels=128, out_channels=96, kernel_size=11),
    nn.ConvTranspose2d(in_channels=96, out_channels=64, kernel_size=11),
    nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=11),
    nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=11),
    nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=11),
)

downsample_param_count = 0
downsample_feature_count = 0
upsample_param_count = 0
upsample_feature_count = 0
x = xb
for idx, conv in enumerate(downsampling):
    N_weights = conv.in_channels * conv.out_channels * conv.kernel_size[0] * conv.kernel_size[1]
    N_biases = conv.out_channels
    N_params = N_weights + N_biases
    N_features = x.flatten().shape[0]
    downsample_param_count += N_params
    downsample_feature_count += N_features
    print(f"Layer {idx+1}")
    print(f"{N_params} parameters")
    print(f"{N_features} input features")
    x = conv(x)

for idx, conv in enumerate(upsampling):
    N_weights = conv.in_channels * conv.out_channels * conv.kernel_size[0] * conv.kernel_size[1]
    N_biases = conv.out_channels
    N_params = N_weights + N_biases
    N_features = x.flatten().shape[0]
    upsample_param_count += N_params
    upsample_feature_count += N_features
    print(f"Layer {idx+1}")
    print(f"{N_params} parameters")
    x = conv(x)
    print(f"{N_features} output features")

print(x.shape)
# %%
