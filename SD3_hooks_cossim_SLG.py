import inspect

import torch
import numpy as np
import types
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusion3Pipeline


def get_features_from_sd3_model(model, input_latents, encoder_hidden_states, pooled_projections, timestep):
    """Extracts features from the Stable Diffusion 3 model.
    Args:
        model: The Stable Diffusion 3 model.
        input_latents: Latent representations of the input images.
        encoder_hidden_states: Encoder hidden states.
        pooled_projections: Pooled projections from the model.
        timestep: The current timestep in the diffusion process.
    Returns:
        features: A dictionary containing the extracted features from various layers of the model.
        """
    features = {}

    def hook_fn(module, input, output, name):
        """Hook function to extract features from the model.
        Args:
            module: The module being hooked.
            input: The input to the module.
            output: The output from the module.
            name: The name of the layer.
        """
        if isinstance(output, tuple) and len(output) == 2:
            features[name] = output[1].detach().cpu()  # Use the second output (hidden states)
        else:
            features[name] = output.detach().cpu()  # Use the output directly

    # Register hooks for each transformer block
    hooks = []
    for i, block in enumerate(model.transformer_blocks):
        hooks.append(
            block.register_forward_hook(
                lambda mod, inp, outp, name=f"transformer_block_{i}": hook_fn(mod, inp, outp, name)
            )
        )

    # Register hooks for the pooled projection (optional use)
    hooks.append(
        model.pos_embed.register_forward_hook(
            lambda mod, inp, outp, name="pos_embed": hook_fn(mod, inp, outp, name)
        )
    )

    # Do a forward pass through the model to extract features
    with torch.no_grad():
        model(
            hidden_states=input_latents,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep
        )

    # Remove hooks to avoid memory leaks
    for hook in hooks:
        hook.remove()

    return features


def calculate_cosine_similarity(features_text_image, features_no_text_image):
    """Calculates the cosine similarity between features of images with and without text.

    Args:
        features_text_image: Features extracted from the image with text.
        features_no_text_image: Features extracted from the image without text.
    Returns:
        similarities: A dictionary containing the cosine similarity for each transformer block.
    """
    similarities = {}

    for key in features_text_image.keys():
        # Only process transformer blocks
        if key.startswith('transformer_block_'):
            # Reshape the features to 2D tensors
            f1 = features_text_image[key].reshape(features_text_image[key].shape[0], -1)
            f2 = features_no_text_image[key].reshape(features_no_text_image[key].shape[0], -1)

            # normalize the features
            f1 = F.normalize(f1, p=2, dim=1)
            f2 = F.normalize(f2, p=2, dim=1)

            # calculate cosine similarity
            similarity = torch.mm(f1, f2.transpose(0, 1))
            similarities[key] = similarity.item()

    return similarities


def plot_similarities(similarities):
    """Plots the cosine similarities between features of images with and without text.
    Args:
        similarities: A dictionary containing the cosine similarity for each transformer block.
    """
    layers = list(similarities.keys())
    values = list(similarities.values())

    # extract layer numbers from the keys and sort
    layer_nums = [int(layer.split('_')[-1]) for layer in layers]

    sorted_indices = np.argsort(layer_nums)
    sorted_layers = [layers[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(sorted_layers)), sorted_values)
    plt.xticks(range(len(sorted_layers)), [f"Layer {l.split('_')[-1]}" for l in sorted_layers], rotation=90)
    plt.xlabel('Transformer Block')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity between Features of Images with and without Text')
    plt.tight_layout()
    plt.savefig('text_similarity_analysis.png')
    plt.close()


def generate_images(pipe, prompts, output_filenames, guidance_scale=7.5):
    """Generates images based on the provided prompts using the Stable Diffusion 3 pipeline.
    Args:
        pipe: The Stable Diffusion 3 pipeline.
        prompts: A list of text prompts for image generation.
        output_filenames: A list of filenames to save the generated images.
        guidance_scale: The guidance scale for the image generation.
        """
    for prompt, filename in zip(prompts, output_filenames):
        print(f"Generate image for prompt: '{prompt}'")
        image = pipe(
            prompt=prompt,
            guidance_scale=guidance_scale,
        ).images[0]
        image.save(filename)
        print(f"Image saved as {filename}")
    return output_filenames


def get_features_for_image(pipe, sd3_model, image_path, device="cuda"):
    """Extracts features from a given image using the Stable Diffusion 3 model.

    Args:
        pipe: The Stable Diffusion 3 pipeline.
        sd3_model: The Stable Diffusion 3 model.
        image_path: Path to the input image.
        device: The device to run the model on (default is "cuda").
    """
    # loade and preprocess the image
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # use the VAE encoder to get latents
    with torch.no_grad():
        latents = pipe.vae.encode(
            # Encode the image to latents
            image_tensor.to(dtype=pipe.vae.dtype)).latent_dist.sample() * pipe.vae.config.scaling_factor

        # Ensure latents are in the correct format
        encoder_hidden_states = torch.zeros((1, 77, 4096), device=device, dtype=torch.float16) # Placeholder for encoder hidden states
        pooled_projections = torch.zeros(1, 2048, device=device, dtype=torch.float16) # Placeholder for pooled projections
        timestep = torch.tensor([500], device=device, dtype=torch.float16) # Placeholder for timestep

        # Extract features from the SD3 model
        return get_features_from_sd3_model(
            sd3_model,
            latents,
            encoder_hidden_states,
            pooled_projections,
            timestep
        )


def implement_skip_layer_guidance(pipe, text_prompt, skip_layers, device="cuda"):
    print(f"Skip layers: {skip_layers}")

    original_forwards = {}

    for layer_idx in skip_layers:
        block = pipe.transformer.transformer_blocks[layer_idx]
        original_forwards[layer_idx] = block.forward

        def create_skip_forward(orig_idx):
            def skip_forward(self, hidden_states, encoder_hidden_states=None, *args, **kwargs):
                #print(f"Skip layer {orig_idx}")
                return encoder_hidden_states, hidden_states

            return skip_forward

        block.forward = types.MethodType(create_skip_forward(layer_idx), block)

    # Generate the image with the modified layers
    print(f"Generate image with skipped layers {skip_layers}...")
    modified_image = pipe(
        prompt=text_prompt,
        num_inference_steps=28,
    ).images[0]

    # Restore the original forward functions for the skipped layers
    for layer_idx, forward_func in original_forwards.items():
        pipe.transformer.transformer_blocks[layer_idx].forward = forward_func

    # Save the modified image
    modified_image.save("balloon example/text_image_modified.png")
    print(f"Modified image saved as {"balloon example/text_image_modified.png"}")

    return modified_image

def main():
    # loade the Stable Diffusion 3 model
    device = "cuda"
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16
    )
    pipe.to(device)
    sd3_model = pipe.transformer
    sd3_model.eval()

    # Prompts with and without text
    text_image_prompt = "A red balloon with the clear text 'Happy Birthday' written on it, floating in a blue sky"
    no_text_image_prompt = "A red balloon floating in a blue sky"

    # generate images based on the prompts
    image_files = generate_images(
        pipe,
        [text_image_prompt, no_text_image_prompt],
        ["balloon example/text_image.png", "balloon example/no_text_image.png"]
    )

    # extract features from the generated images
    print("Extract Features for images with text...")
    features_text_image = get_features_for_image(pipe, sd3_model, "balloon example/text_image.png", device)

    print("Extarct Features for images without text...")
    features_no_text_image = get_features_for_image(pipe, sd3_model, "balloon example/no_text_image.png", device)

    # calculate cosine similarity between features of images with and without text
    print("Calcuate cosine similarity")
    similarities = calculate_cosine_similarity(features_text_image, features_no_text_image)

    # visualize the similarities
    print("Visualize")
    plot_similarities(similarities)

    # Save the similarities to a text file
    with open("balloon example/text_similarity_analysis.txt", "w") as f:
        f.write("Cosine-Similarity between images with and without text:\n")
        for key, value in sorted(similarities.items(), key=lambda x: int(x[0].split('_')[-1])):
            f.write(f"{key}: {value:.4f}\n")

    # Find the layers with the largest differences
    # This will help identify which layers are most affected by the presence of text in the image.
    lowest_similarity_layers = sorted(similarities.items(), key=lambda x: x[1])[:3]
    text_processing_layers = [int(layer[0].split('_')[-1]) for layer in lowest_similarity_layers]

    print(f"Layers with the biggest difference are: {text_processing_layers}")

    # Apply Skip-Layer Guidance on the identified text-processing layers
    print("Apply Skip-Layer Guidance on the identified text-processing layers")
    implement_skip_layer_guidance(
        pipe,
        text_image_prompt,
        text_processing_layers,
        device
    )


if __name__ == "__main__":
    main()
