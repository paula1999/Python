# PyTorch

```py
import torch
```

## Crear tensor

```py
t = torch.Tensor(
    [
     [[1, 2], [3, 4]], 
     [[5, 6], [7, 8]], 
     [[9, 0], [1, 2]]
    ]
)
```

## Shape

```py
t.shape # number of elements in each dimension
t.shape[n] #size of dimension n
len(t.shape) # number of dimensions (rank)
t.numel() # number of elements
```

## Indexing

```py
t[i] # access element i
t[i, j] # access the j-th dimension of the i-th example
t[i, j].item() # scalar value
t[:, i, j] # index into the i-th element of a column
```

## Initializing

```py
torch.ones_like(t) # creates a tensow of all ones with the same shape and device as t
torch.zeros_like(t) # creates a tensor of all zeros with the same shape and device as t
torch.randn_like(t) # creates a tensor with every element sampled from a Normal (or Gaussian) distribution with the same shape and device as t
torch.randn(i, j, device = 'cpu') # initialize a tensor knowing only the shape and device
```

## Basic Functions

```py
t - a
t * n
t.mean()
t.std()
t-mean(0) # mean along a particular dimension
# Equivalently, you could also write:
# t.mean(dim=0)
# t.mean(axis=0)
# torch.mean(t, 0)
# torch.mean(t, dim=0)
# torch.mean(t, axis=0)
```


## PyTorch Neural Network Module

```py
import torch.nn as nn
```

### nn.Linear

```py
linear = nn.Linear(i, j) # i: number of input dimensions; j: number of output dimensions.
example_input = torch.randn(3, 10)
example_output = linear(example_input)
example_output
```

### nn.ReLU

```py
relu = nn.ReLU()
relu_output = relu(example_output) # sets all negative numbers to zero
relu_output
```

### nn.BatchNorm1d

```py
batchnorm = nn.BatchNorm1d(2)
batchnorm_output = batchnorm(relu_output) # normalization technique that will rescalea batch of n inputs to have a consistent mean and standard deviation between batches
batchnorm_output
```

### nn.Sequential

```py
mlp_layer = nn.Sequential( # creates a single operation that performs a sequence of operations
    nn.Linear(5, 2),
    nn.BatchNorm1d(2),
    nn.ReLU()
)

test_example = torch.randn(5,5) + 1
print("input: ")
print(test_example)
print("output: ")
print(mlp_layer(test_example))
```

## Optimization

### Optimizerss

```py
import torch.optim as optim
adam_opt = optim.Adam(
  mlp_layer.parameters(), # parameters as a list
  lr=1e-1 # learning rate
)
```


### Training Loop

A (basic) training step in PyTorch consists of four basic parts:

1.   Set all of the gradients to zero using `opt.zero_grad()`
2.   Calculate the loss, `loss`
3.   Calculate the gradients with respect to the loss using `loss.backward()`
4.   Update the parameters being optimized using `opt.step()`

```py
train_example = torch.randn(100,5) + 1
adam_opt.zero_grad()

# We'll use a simple loss function of mean distance from 1
# torch.abs takes the absolute value of a tensor
cur_loss = torch.abs(1 - mlp_layer(train_example)).mean()

cur_loss.backward()
adam_opt.step()
print(cur_loss)
```

- `requires_grad_()`: to calculate the gradient with respect to a tensor.
- `torch.no_grad()`: to prevent the gradients from being calculated in a piece of code.
- `detach()`: to calculate and use a tensor's value without calculating its gradients.

## New `nn` Classes

You can also create new classes which extend the `nn` module. For these classes, all class attributes, as in `self.layer` or `self.param` will automatically treated as parameters if they are themselves `nn` objects or if they are tensors wrapped in `nn.Parameter` which are initialized with the class. 

The `__init__` function defines what will happen when the object is created. The first line of the init function of a class, for example, `WellNamedClass`, needs to be `super(WellNamedClass, self).__init__()`. 

The `forward` function defines what runs if you create that object `model` and pass it a tensor `x`, as in `model(x)`. If you choose the function signature, `(self, x)`, then each call of the forward function, gets two pieces of information: `self`, which is a reference to the object with which you can access all of its parameters, and `x`, which is the current tensor for which you'd like to return `y`.


```py
class ExampleModule(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(ExampleModule, self).__init__()
        self.linear = nn.Linear(input_dims, output_dims)
        self.exponent = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        x = self.linear(x)

        # This is the notation for element-wise exponentiation, 
        # which matches python in general
        x = x ** self.exponent 
        
        return x
```

```py
example_model = ExampleModule(10, 2)
list(example_model.parameters())
```


```py
list(example_model.named_parameters())
```

```py
input = torch.randn(2, 10)
example_model(input)
```

## 2D Operations

*   2D convolutions: [`nn.Conv2d`](https://pytorch.org/docs/master/generated/torch.nn.Conv2d.html) requires the number of input and output channels, as well as the kernel size.
*   2D transposed convolutions (aka deconvolutions): [`nn.ConvTranspose2d`](https://pytorch.org/docs/master/generated/torch.nn.ConvTranspose2d.html) also requires the number of input and output channels, as well as the kernel size
*   2D batch normalization: [`nn.BatchNorm2d`](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) requires the number of input dimensions
*   Resizing images: [`nn.Upsample`](https://pytorch.org/docs/master/generated/torch.nn.Upsample.html) requires the final size or a scale factor. Alternatively, [`nn.functional.interpolate`](https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate) takes the same arguments.
