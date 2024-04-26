!pip install diffusers==0.11.1
!pip install transformers scipy ftfy accelerate

import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)  # remove torch_dtype=torch.float16 if you want to ensure the highest possible precision, at the cost of a higher memory usage.

# Move the pipeline to GPU to have faster inference
pipe = pipe.to("cuda")



# Generate
prompt = "a photograph of an astronaut riding a horse"
image = pipe(prompt).images[0]  # image here is in [PIL format](https://pillow.readthedocs.io/en/stable/)

# Now to display an image you can either save it such as:
image.save(f"astronaut_rides_horse.png")

# or if you're in a google colab you can directly display it with 
image

# Running the above cell multiple times will give you a different image every time



# Deterministic output: pass a random seed
generator = torch.Generator("cuda").manual_seed(1024) # with the same seed, you'll have the same image result
image = pipe(prompt, generator=generator).images[0]



# Change the number of inference steps with num_inference_steps
# Results are better themore steps you use, default = 50
generator = torch.Generator("cuda").manual_seed(1024)
image = pipe(prompt, num_inference_steps=15, generator=generator).images[0]



# To increase the adherence to the conditional signal (text and sample quality): guidance_scale with values like 7 or 8.5
from PIL import Image

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

num_images = 3
prompt = ["a photograph of an astronaut riding a horse"] * num_images # generate multiple images for the same prompt
images = pipe(prompt).images
grid = image_grid(images, rows=1, cols=3)


# To generaate a grid of nxm images
num_cols = 3
num_rows = 4
prompt = ["a photograph of an astronaut riding a horse"] * num_cols
all_images = []

for i in range(num_rows):
  images = pipe(prompt).images
  all_images.extend(images)

grid = image_grid(all_images, rows=num_rows, cols=num_cols)





# Generate non-square images
'''
Stable Diffusion produces images of 512 x 512 pixels by default
Going below 512 might result in lower quality images.
Going over 512 in both directions will repeat image areas (global coherence is lost).
The best way to create non-square images is to use 512 in one dimension, and a value larger than that in the other one.
'''
prompt = "a photograph of an astronaut riding a horse"
image = pipe(prompt, height=512, width=768).images[0] # height and width must be multiples of 8




# To write your own inference pipeline with diffusers
import torch

torch_device = "cuda" if torch.cuda.is_available() else "cpu"


'''
The pre-trained model includes all the components required to setup a complete diffusion pipeline. They are stored in the following folders:

text_encoder: Stable Diffusion uses CLIP, but other diffusion models may use other encoders such as BERT.
tokenizer. It must match the one used by the text_encoder model.
scheduler: The scheduling algorithm used to progressively add noise to the image during training.
unet: The model used to generate the latent representation of the input.
vae: Autoencoder module that we'll use to decode latent representations into real images.
'''

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

# 1. Load the autoencoder model which will be used to decode the latents into image space. 
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# 2. Load the tokenizer and text encoder to tokenize and encode the text. 
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# 3. The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")


# instead of loading the pre-defined scheduler, we'll use the K-LMS scheduler
from diffusers import LMSDiscreteScheduler

scheduler = LMSDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")


# move the models to GPU
vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device) 


# define the parameters
prompt = ["a photograph of an astronaut riding a horse"]
height = 512                        # default height of Stable Diffusion
width = 512                         # default width of Stable Diffusion
num_inference_steps = 100            # Number of denoising steps
guidance_scale = 7.5                # Scale for classifier-free guidance
generator = torch.manual_seed(32)   # Seed generator to create the inital latent noise
batch_size = 1


# we get the text_embeddings for the prompt. These embeddings will be used to condition the UNet model.
text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

with torch.no_grad():
  text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]


# get the unconditional text embeddings for classifier-free guidance, 
# which are just the embeddings for the padding token (empty text). 
# They need to have the same shape as the conditional text_embeddings (batch_size and seq_length)
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer(
    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
)
with torch.no_grad():
  uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]   


# For classifier-free guidance, we need to do two forward passes. 
# One with the conditioned input (text_embeddings), and another with the unconditional embeddings (uncond_embeddings). 
# In practice, we can concatenate both into a single batch to avoid doing two forward passes.
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])


# Generate the intial random noise.
latents = torch.randn(
  (batch_size, unet.in_channels, height // 8, width // 8),
  generator=generator,
)
latents = latents.to(torch_device)
latents.shape # 64×64  is expected. The model will transform this latent representation (pure noise) into a 512 × 512 image later on.


# Initialize the scheduler
scheduler.set_timesteps(num_inference_steps)

# The K-LMS scheduler needs to multiply the latents by its sigma values
latents = latents * scheduler.init_noise_sigma

# Write denoising loop
from tqdm.auto import tqdm
from torch import autocast

for t in tqdm(scheduler.timesteps):
  # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
  latent_model_input = torch.cat([latents] * 2)

  latent_model_input = scheduler.scale_model_input(latent_model_input, t)

  # predict the noise residual
  with torch.no_grad():
    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

  # perform guidance
  noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
  noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

  # compute the previous noisy sample x_t -> x_t-1
  latents = scheduler.step(noise_pred, t, latents).prev_sample


# Decode the generated latents back into the image
# scale and decode the image latents with vae
latents = 1 / 0.18215 * latents

with torch.no_grad():
  image = vae.decode(latents).sample


# convert the image to PIL so we can display or save it
image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
images = (image * 255).round().astype("uint8")
pil_images = [Image.fromarray(image) for image in images]
pil_images[0]
