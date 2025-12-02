import os
import torch
import numpy as np
from PIL import Image
import io
import base64
import folder_paths
from openai import OpenAI
import httpx

# Try to import config
try:
    # Try relative import first (when used as package)
    from . import config
except ImportError:
    # Try absolute import (when used directly)
    try:
        import config
    except ImportError:
        # If config.py doesn't exist, try to load from example
        try:
            import importlib.util
            config_path = os.path.join(os.path.dirname(__file__), "config.py")
            example_path = os.path.join(os.path.dirname(__file__), "config.py.example")
            
            # Try config.py first, then fallback to example
            if os.path.exists(config_path):
                spec = importlib.util.spec_from_file_location("config", config_path)
            elif os.path.exists(example_path):
                spec = importlib.util.spec_from_file_location("config", example_path)
            else:
                raise FileNotFoundError("config.py or config.py.example not found")
            
            config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config)
        except Exception as e:
            raise ImportError("Please create config.py from config.py.example and configure your API key") from e


class NanobananaImageGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Generate a beautiful sunset over mountains"}),
                "model": ("STRING", {"default": config.DEFAULT_MODEL}),
            },
            "optional": {
                "input_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "text")
    FUNCTION = "generate"
    CATEGORY = "Nanobanana"

    def generate_empty_image(self, width=512, height=512):
        """Generate an empty placeholder image"""
        empty_image = np.ones((height, width, 3), dtype=np.float32) * 0.5
        tensor = torch.from_numpy(empty_image).unsqueeze(0)
        return tensor

    def image_tensor_to_pil(self, image_tensor):
        """Convert ComfyUI image tensor to PIL Image"""
        img_array = image_tensor[0].cpu().numpy()
        img_array = (img_array * 255).astype(np.uint8)
        return Image.fromarray(img_array)

    def pil_to_image_tensor(self, pil_image):
        """Convert PIL Image to ComfyUI image tensor"""
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        img_array = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(img_array).unsqueeze(0)

    def generate(self, prompt, model, input_image=None):
        """Generate image using OpenRouter API"""
        
        # Check if API key is configured
        if not config.OPENROUTER_API_KEY or config.OPENROUTER_API_KEY == "<YOUR_OPENROUTER_API_KEY>":
            error_msg = "Please configure OPENROUTER_API_KEY in config.py"
            print(f"[ERROR] {error_msg}")
            return (self.generate_empty_image(), error_msg)

        # Save original proxy settings (before try block so finally can access)
        original_proxies = {}
        proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 
                     'ALL_PROXY', 'all_proxy', 'SOCKS_PROXY', 'socks_proxy']
        for var in proxy_vars:
            if var in os.environ:
                original_proxies[var] = os.environ[var]

        try:
            for var in proxy_vars:
                if var in os.environ:
                    original_proxies[var] = os.environ[var]
            
            # Temporarily remove SOCKS proxy to avoid socksio dependency
            # Keep HTTP/HTTPS proxies if they exist
            socks_vars = ['SOCKS_PROXY', 'socks_proxy', 'ALL_PROXY', 'all_proxy']
            for var in socks_vars:
                if var in os.environ:
                    del os.environ[var]
            
            # Initialize OpenAI client with OpenRouter
            # Explicitly disable proxy to avoid SOCKS proxy issues
            import httpx
            http_client = httpx.Client(
                proxies=None,  # Disable proxy
                timeout=60.0,
            )
            
            client = OpenAI(
                base_url=config.OPENROUTER_BASE_URL,
                api_key=config.OPENROUTER_API_KEY,
                http_client=http_client,
            )

            # Prepare messages
            messages = []
            
            # Add input image if provided
            if input_image is not None:
                try:
                    pil_img = self.image_tensor_to_pil(input_image)
                    img_byte_arr = io.BytesIO()
                    pil_img.save(img_byte_arr, format='PNG')
                    img_bytes = img_byte_arr.getvalue()
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    })
                except Exception as e:
                    print(f"[WARNING] Failed to process input image: {e}")
                    messages.append({
                        "role": "user",
                        "content": prompt
                    })
            else:
                messages.append({
                    "role": "user",
                    "content": prompt
                })

            print(f"[INFO] Generating image with model: {model}")
            print(f"[INFO] Prompt: {prompt[:100]}...")

            # Generate image
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                extra_body={"modalities": ["image", "text"]}
            )

            # Process response
            message = response.choices[0].message
            text_output = ""
            image_tensor = None

            # Extract text content
            if hasattr(message, 'content') and message.content:
                text_output = message.content

            # Extract images - according to ref.md, message.images is a list of dicts
            if hasattr(message, 'images') and message.images:
                for image in message.images:
                    # Handle both dict and object formats
                    if isinstance(image, dict):
                        image_url = image.get('image_url', {}).get('url', '')
                    else:
                        # Object format
                        image_url = getattr(image, 'image_url', None)
                        if image_url:
                            if isinstance(image_url, dict):
                                image_url = image_url.get('url', '')
                            else:
                                image_url = getattr(image_url, 'url', '')
                    
                    if image_url and image_url.startswith('data:image'):
                        # Handle base64 data URL
                        try:
                            # Extract base64 data
                            header, encoded = image_url.split(',', 1)
                            img_data = base64.b64decode(encoded)
                            
                            # Load image
                            pil_img = Image.open(io.BytesIO(img_data))
                            image_tensor = self.pil_to_image_tensor(pil_img)
                            
                            # Save image to output directory
                            output_dir = folder_paths.get_output_directory()
                            nanobanana_dir = os.path.join(output_dir, "nanobanana_outputs")
                            os.makedirs(nanobanana_dir, exist_ok=True)
                            
                            import uuid
                            file_name = os.path.join(nanobanana_dir, f"nanobanana_{uuid.uuid4()}.png")
                            pil_img.save(file_name)
                            print(f"[INFO] Saved image to: {file_name}")
                            
                            break  # Use first image
                        except Exception as e:
                            print(f"[ERROR] Failed to process image data URL: {e}")
                            import traceback
                            traceback.print_exc()
                            continue

            if image_tensor is None:
                print("[WARNING] No image found in response, generating placeholder")
                image_tensor = self.generate_empty_image()
                if not text_output:
                    text_output = "Image generation completed but no image was returned"

            print(f"[INFO] Generation completed. Text length: {len(text_output)}")
            return (image_tensor, text_output)

        except Exception as e:
            error_msg = f"API request failed: {str(e)}"
            print(f"[ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            return (self.generate_empty_image(), error_msg)
        finally:
            # Restore original proxy settings
            for var, value in original_proxies.items():
                os.environ[var] = value
            # Remove any vars that weren't originally set
            for var in proxy_vars:
                if var not in original_proxies and var in os.environ:
                    del os.environ[var]


NODE_CLASS_MAPPINGS = {
    "NanobananaImageGenerator": NanobananaImageGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanobananaImageGenerator": "Nanobanana Image Generator",
}

