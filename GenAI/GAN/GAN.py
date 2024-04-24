# Build and train a GAN that can generate hand-written images of digits (0-9)

import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST # Training dataset
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0) # Set for testing purposes

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

#################################################
# Generator
#################################################

'''
Create a function to make a single layer/block for the generator's neural network. 
Each block should include a linear transformation to map to another shape, 
a batch normalization for stabilization, 
and finally a non-linear activation function (you use a ReLU here) so the output can be transformed in complex ways
'''
def get_generator_block(input_dim, output_dim):
    '''
    Function for returning a block of the generator's neural network
    given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a generator neural network layer, with a linear transformation 
          followed by a batch normalization and then a relu activation
    '''
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True)
    )



'''
The generator class will take 3 values:
*   The noise vector dimension
*   The image dimension
*   The initial hidden dimension

Using these values, the generator will build a neural network with 5 layers/blocks. 
Beginning with the noise vector, the generator will apply non-linear transformations 
via the block function until the tensor is mapped to the size of the image to be outputted 
(the same size as the real images from MNIST). 
You will need to fill in the code for final layer since it is different than the others. 
The final layer does not need a normalization or activation function, but does need to be scaled with a sigmoid function.

Finally, you are given a forward pass function that takes in a noise vector and generates an image of the output dimension using your neural network.
'''
class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
          (MNIST images are 28 x 28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        super(Generator, self).__init__()
        # Build the neural network
        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid()
        )
    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return self.gen(noise)
    
    # Needed for grading
    def get_gen(self):
        '''
        Returns:
            the sequential model
        '''
        return self.gen


#################################################
# Noise
#################################################

'''
To be able to use your generator, you will need to be able to create noise vectors. 
The noise vector z has the important role of making sure the images generated from the same class don't all look the same. 
You will generate it randomly using PyTorch by sampling random numbers from the normal distribution. 
Since multiple images will be processed per pass, you will generate all the noise vectors at once.
'''
def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim),
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    '''
    return torch.randn(n_samples, z_dim, device = device)


#################################################
# Discriminator
#################################################

'''
Create a function that builds a neural network block for the discriminator
You use leaky ReLUs to prevent the "dying ReLU" problem, 
which refers to the phenomenon where the parameters stop changing due to consistently negative values passed to a ReLU, 
which result in a zero gradient.
'''

def get_discriminator_block(input_dim, output_dim):
    '''
    Discriminator Block
    Function for returning a neural network of the discriminator given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a discriminator neural network layer, with a linear transformation 
          followed by an nn.LeakyReLU activation with negative slope of 0.2 
          (https://pytorch.org/docs/master/generated/torch.nn.LeakyReLU.html)
    '''
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(0.2)
    )


'''
The discriminator class holds 2 values:
*   The image dimension
*   The hidden dimension

The discriminator will build a neural network with 4 layers. 
It will start with the image tensor and transform it until it returns a single number (1-dimension tensor) output. 
This output classifies whether an image is fake or real. 
Note that you do not need a sigmoid after the output layer since it is included in the loss function. 
Finally, to use your discrimator's neural network you are given a forward pass function that takes in an image tensor to be classified.
'''
class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
            (MNIST images are 28x28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, im_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image):
        '''
        Function for completing a forward pass of the discriminator: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_dim)
        '''
        return self.disc(image)
    
    # Needed for grading
    def get_disc(self):
        '''
        Returns:
            the sequential model
        '''
        return self.disc



#################################################
# Training
#################################################

# Set parameters
criterion = nn.BCEWithLogitsLoss() # loss function
n_epochs = 200 # number of times you iterate through the entire dataset when training
z_dim = 64 # dimension of the noise vector
display_step = 500 # how often to display/visualize the images
batch_size = 128 # number of images per forward/backward pass
lr = 0.00001 # learning rate

# Load MNIST dataset as tensors
dataloader = DataLoader(
    MNIST('.', download=False, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True)

# Device
device = 'cuda'

# Initialize generator, discriminator and optimizers
gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to(device) 
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

# Discriminator's loss
def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
               the discriminator's predictions to the ground truth reality of the images 
               (e.g. fake = 0, real = 1)
        real: a batch of real images
        num_images: the number of images the generator should produce, 
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        disc_loss: a torch scalar loss value for the current batch
    '''
    
    #  1) Create noise vectors and generate a batch (num_images) of fake images. 
    #  Make sure to pass the device argument to the noise.
    noise = torch.randn(num_images, z_dim).to(device)
    fake = gen(noise)
  
    #  2) Get the discriminator's prediction of the fake image 
    #  and calculate the loss. Don't forget to detach the generator!
    #  (Remember the loss function you set earlier -- criterion. You need a 
    #  'ground truth' tensor in order to calculate the loss. 
    #  For example, a ground truth tensor for a fake image is all zeros.)
    #  *Important*: You should NOT write your own loss function here - use criterion(pred, true)!
    fake_pred = disc(fake.detach())  # detach the generator to avoid backpropagating through it
    fake_target = torch.zeros_like(fake_pred)  # ground truth tensor for fake images
    fake_loss = criterion(fake_pred, fake_target)
  
    #  3) Get the discriminator's prediction of the real image and calculate the loss.
    real_pred = disc(real)
    real_target = torch.ones_like(real_pred)  # ground truth tensor for real images
    real_loss = criterion(real_pred, real_target)
  
    #  4) Calculate the discriminator's loss by averaging the real and fake loss
    #  and set it to disc_loss.
    disc_loss = (fake_loss + real_loss) / 2
    
    return disc_loss

# Generator's loss
def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    '''
    Return the loss of the generator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
               the discriminator's predictions to the ground truth reality of the images 
               (e.g. fake = 0, real = 1)
        num_images: the number of images the generator should produce, 
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        gen_loss: a torch scalar loss value for the current batch
    '''

    # 1) Create noise vectors and generate a batch of fake images.
    # Remember to pass the device argument to the get_noise function.
    noise = torch.randn(num_images, z_dim).to(device)
    fake = gen(noise)
    
    # 2) Get the discriminator's prediction of the fake image.
    fake_pred = disc(fake)
    
    # 3) Calculate the generator's loss. Remember the generator wants
    # the discriminator to think that its fake images are real
    # *Important*: You should NOT write your own loss function here - use criterion(pred, true)!
    gen_target = torch.ones_like(fake_pred)
    gen_loss = criterion(fake_pred, gen_target)
    
    return gen_loss


'''
Finally, you can put everything together! For each epoch, you will process the entire dataset in batches. 
For every batch, you will need to update the discriminator and generator using their loss. 
Batches are sets of images that will be predicted on before the loss functions are calculated 
(instead of calculating the loss function after each image). 
Note that you may see a loss to be greater than 1, 
this is okay since binary cross entropy loss can be any positive number for a sufficiently confident wrong guess. 

It’s also often the case that the discriminator will outperform the generator, 
especially at the start, because its job is easier. 
It's important that neither one gets too good (that is, near-perfect accuracy), 
which would cause the entire model to stop learning. 
'''

cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
test_generator = True # Whether the generator should be tested
gen_loss = False
error = False
for epoch in range(n_epochs):
  
    # Dataloader returns the batches
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)

        # Flatten the batch of real images from the dataset
        real = real.view(cur_batch_size, -1).to(device)

        ### Update discriminator ###
        # Zero out the gradients before backpropagation
        disc_opt.zero_grad()

        # Calculate discriminator loss
        disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)

        # Update gradients
        disc_loss.backward(retain_graph=True)

        # Update optimizer
        disc_opt.step()

        # For testing purposes, to keep track of the generator weights
        if test_generator:
            old_generator_weights = gen.gen[0][0].weight.detach().clone()

        ### Update generator ###
        # 1) Zero out the gradients.
        disc_opt.zero_grad()
        # 2) Calculate the generator loss, assigning it to gen_loss.
        gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
        # 3) Backprop through the generator: update the gradients and optimizer.
        gen_loss.backward(retain_graph=True)
        gen_opt.step()

        # For testing purposes, to check that your code changes the generator weights
        if test_generator:
            try:
                assert lr > 0.0000002 or (gen.gen[0][0].weight.grad.abs().max() < 0.0005 and epoch == 0)
                assert torch.any(gen.gen[0][0].weight.detach().clone() != old_generator_weights)
            except:
                error = True
                print("Runtime tests have failed")

        # Keep track of the average discriminator loss
        mean_discriminator_loss += disc_loss.item() / display_step

        # Keep track of the average generator loss
        mean_generator_loss += gen_loss.item() / display_step

        ### Visualization code ###
        if cur_step % display_step == 0 and cur_step > 0:
            print(f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            fake = gen(fake_noise)
            show_tensor_images(fake)
            show_tensor_images(real)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        cur_step += 1
