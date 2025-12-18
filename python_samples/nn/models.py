import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_extension

class TorchLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, wt, b):
        # Linear:
        # [B, out_features] = [B, in_features] x [in_features, out_features] + [out_features]
        # row-major = row-major x column-major
        y = x @ wt + b
        # Save for backward pass:
        ctx.save_for_backward(x, wt)
        return y

    @staticmethod
    def backward(ctx, do):
        # Get saved tensors:
        x, wt = ctx.saved_tensors
        # Linear backprop:
        dx = do @ wt.T      # row-major = row-major x row-major (column-major transposed)
        dwt = (do.T @ x).T  # column-major = column-major (row-major transposed) x row-major
        db = do.sum(dim=0)  # sum reduction over rows
        return dx, dwt, db

class ExtensionLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, wt, b):
        # Linear:
        # [B, out_features] = [B, in_features] x [in_features, out_features] + [out_features]
        # row-major = row-major x column-major
        y = torch_extension.matmul_bias(x, wt, b) # y = x @ wt + b
        # Save for backward pass:
        ctx.save_for_backward(x, wt)
        return y

    @staticmethod
    def backward(ctx, do):
        # Get saved tensors:
        x, wt = ctx.saved_tensors
        # Linear backprop:
        # row-major = row-major x row-major (column-major transposed)
        dx = torch_extension.matmul(do, wt.T) # dx = do @ wt.T
        # column-major = column-major (row-major transposed) x row-major
        dwt = torch_extension.matmul(do.T, x, True).T # dwt = (do.T @ x).T
        # sum reduction over rows
        db = torch_extension.matrix_reduction_row(do) # db = do.sum(dim=0)
        return dx, dwt, db

class TorchLinearReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, wt, b):
        # Linear:
        # [B, out_features] = [B, in_features] x [in_features, out_features] + [out_features]
        # row-major = row-major x column-major
        z = x @ wt + b
        # Elementwise ReLU activation:
        # y = max(0, z)
        y = z.clamp(min=0)  # y is row-major
        # Save for backward pass:
        ctx.save_for_backward(x, wt, z)
        return y

    @staticmethod
    def backward(ctx, do):
        # Get saved tensors:
        x, wt, z = ctx.saved_tensors
        # ReLU backprop:
        # dz = do.clone()  # do is row-major
        # dz[z <= 0] = 0   # dz is row-major
        dz = do * (z > 0)  # do and dz are row-major
        # Linear backprop:
        dx = dz @ wt.T      # row-major = row-major x row-major (column-major transposed)
        dwt = (dz.T @ x).T  # column-major = column-major (row-major transposed) x row-major
        db = dz.sum(dim=0)  # sum reduction over rows
        return dx, dwt, db

class ExtensionLinearReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, wt, b):
        # Linear:
        # [B, out_features] = [B, in_features] x [in_features, out_features] + [out_features]
        # row-major = row-major x column-major
        z = x @ wt + b
        # Elementwise ReLU activation:
        # y = max(0, z)
        y = z.clamp(min=0)  # y is row-major
        # Save for backward pass:
        ctx.save_for_backward(x, wt, z)
        return y

    @staticmethod
    def backward(ctx, do):
        # Get saved tensors:
        x, wt, z = ctx.saved_tensors
        # ReLU backprop:
        # dz = do.clone()  # do is row-major
        # dz[z <= 0] = 0   # dz is row-major
        dz = do * (z > 0)  # do and dz are row-major
        # Linear backprop:
        dx = dz @ wt.T      # row-major = row-major x row-major (column-major transposed)
        dwt = (dz.T @ x).T  # column-major = column-major (row-major transposed) x row-major
        db = dz.sum(dim=0)  # sum reduction over rows
        return dx, dwt, db

class CustomLinearReLU(nn.Module):
    def __init__(self, in_features, out_features, dtype, use_extension, use_relu = True):
        assert in_features > 0
        assert out_features > 0
        super().__init__()

        k = (1 / in_features) ** 0.5
        self.wt = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype).uniform_(-k, k).T)
        self.b = nn.Parameter(torch.empty(out_features, dtype=dtype).uniform_(-k, k))

        self.use_relu = use_relu

        self.LinearFunction = ExtensionLinearFunction if use_extension else TorchLinearFunction
        self.LinearReLUFunction = ExtensionLinearReLUFunction if use_extension else TorchLinearReLUFunction

    def forward(self, x):
        if self.use_relu:
            return self.LinearReLUFunction.apply(x, self.wt, self.b)
        else:
            return self.LinearFunction.apply(x, self.wt, self.b)

class CustomMLP(nn.Module):
    def __init__(self, in_features, out_features, use_extension, dtype = torch.float32):
        super(CustomMLP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc1 = CustomLinearReLU(in_features, 256, dtype=dtype, use_extension=use_extension, use_relu=False)
        self.fc2 = CustomLinearReLU(256, 128, dtype=dtype, use_extension=use_extension, use_relu=False)
        self.fc3 = CustomLinearReLU(128, out_features, dtype=dtype, use_extension=use_extension, use_relu=False)

    def forward(self, x):
        x = x.view(-1, self.in_features)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output