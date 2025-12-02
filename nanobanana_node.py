import os
import torch
import numpy as np
from PIL import Image
import io
import base64
import random
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
    # Maximum number of input images supported
    MAX_IMAGES = 10
    
    @classmethod
    def INPUT_TYPES(cls):
        # Fixed 4 input images
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Generate a beautiful sunset over mountains"}),
                "model": ("STRING", {"default": config.DEFAULT_MODEL}),
            },
            "optional": {
                "input_image_1": ("IMAGE",),
                "input_image_2": ("IMAGE",),
                "input_image_3": ("IMAGE",),
                "input_image_4": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
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

    def generate(self, prompt, model, seed=0, **kwargs):
        """Generate image using OpenRouter API"""
        
        # Check if API key is configured
        if not config.OPENROUTER_API_KEY or config.OPENROUTER_API_KEY == "<YOUR_OPENROUTER_API_KEY>":
            error_msg = "Please configure OPENROUTER_API_KEY in config.py"
            print(f"[ERROR] {error_msg}")
            return (self.generate_empty_image(), error_msg)
        
        # Handle seed: if seed is 0, use random; otherwise use the specified seed
        if seed == 0:
            actual_seed = random.randint(0, 2147483647)
            print(f"[INFO] Seed is 0, using random seed: {actual_seed}")
        else:
            actual_seed = seed
            print(f"[INFO] Using specified seed: {actual_seed}")
        
        # Collect input images (up to 4)
        input_images = []
        for i in range(1, 5):  # Fixed 4 inputs
            img_key = f"input_image_{i}"
            if img_key in kwargs and kwargs[img_key] is not None:
                input_images.append(kwargs[img_key])

        # Save original proxy settings (before try block so finally can access)
        original_proxies = {}
        proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 
                     'ALL_PROXY', 'all_proxy', 'SOCKS_PROXY', 'socks_proxy']
        for var in proxy_vars:
            if var in os.environ:
                original_proxies[var] = os.environ[var]

        try:
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
            
            # Process input images
            if input_images:
                try:
                    content = []
                    
                    # Add all input images
                    for img_tensor in input_images:
                        try:
                            pil_img = self.image_tensor_to_pil(img_tensor)
                            img_byte_arr = io.BytesIO()
                            pil_img.save(img_byte_arr, format='PNG')
                            img_bytes = img_byte_arr.getvalue()
                            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                            
                            content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            })
                        except Exception as e:
                            print(f"[WARNING] Failed to process input image: {e}")
                            continue
                    
                    # Add text prompt
                    if content:
                        content.append({
                            "type": "text",
                            "text": prompt
                        })
                        messages.append({
                            "role": "user",
                            "content": content
                        })
                    else:
                        # If all images failed, just use text
                        messages.append({
                            "role": "user",
                            "content": prompt
                        })
                    
                    print(f"[INFO] Processing {len(input_images)} input image(s)")
                except Exception as e:
                    print(f"[WARNING] Failed to process input images: {e}")
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

            # Debug: Print response structure (simplified)
            if hasattr(response, 'choices') and response.choices:
                print(f"[DEBUG] Response has {len(response.choices)} choice(s)")
            else:
                print(f"[DEBUG] Response has no choices")
            
            # Process response
            if not response or not hasattr(response, 'choices') or not response.choices or len(response.choices) == 0:
                error_msg = "API returned empty response or no choices"
                print(f"[ERROR] {error_msg}")
                print(f"[DEBUG] Full response object: {response}")
                if hasattr(response, '__dict__'):
                    print(f"[DEBUG] Response __dict__: {response.__dict__}")
                return (self.generate_empty_image(), error_msg)
            
            choice = response.choices[0]
            
            # Check for errors in the choice
            if hasattr(choice, 'error') and choice.error:
                error_info = choice.error
                error_msg = f"API returned error: {error_info.get('message', 'Unknown error')}"
                if 'metadata' in error_info and 'raw' in error_info['metadata']:
                    try:
                        import json
                        raw_error = json.loads(error_info['metadata']['raw'])
                        if 'error' in raw_error:
                            api_error = raw_error['error']
                            error_msg = f"API Error ({api_error.get('code', 'Unknown')}): {api_error.get('message', 'Unknown error')}"
                            print(f"[ERROR] {error_msg}")
                            print(f"[ERROR] Full error details: {raw_error}")
                    except:
                        pass
                print(f"[ERROR] {error_msg}")
                return (self.generate_empty_image(), error_msg)
            
            message = choice.message
            
            # Check for images in various locations (simplified logging)
            has_images_in_content = False
            has_images_attr = False
            
            if hasattr(message, 'content'):
                content = message.content
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'image_url':
                            has_images_in_content = True
                            break
            
            if hasattr(message, 'images') and message.images:
                has_images_attr = True
            
            print(f"[DEBUG] Message has content: {hasattr(message, 'content')}, has images attr: {has_images_attr}, images in content: {has_images_in_content}")
            
            text_output = ""
            image_tensor = None

            # Extract text content and check for images in content list
            if message and hasattr(message, 'content'):
                content = message.content
                if isinstance(content, list):
                    # Content is a list of parts (text and/or images)
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get('type') == 'text':
                                text_parts.append(item.get('text', ''))
                            elif item.get('type') == 'image_url':
                                # Found image in content list
                                print(f"[INFO] Found image in content list")
                                image_url_data = item.get('image_url', {})
                                if isinstance(image_url_data, dict):
                                    image_url = image_url_data.get('url', '')
                                else:
                                    image_url = str(image_url_data)
                                # Only log URL prefix, not full base64 data
                                url_preview = image_url[:50] + "..." if image_url and len(image_url) > 50 else image_url
                                print(f"[DEBUG] Image URL preview: {url_preview}")
                                # Process this image URL
                                if image_url and image_url.startswith('data:image'):
                                    try:
                                        header, encoded = image_url.split(',', 1)
                                        img_data = base64.b64decode(encoded)
                                        pil_img = Image.open(io.BytesIO(img_data))
                                        image_tensor = self.pil_to_image_tensor(pil_img)
                                        
                                        output_dir = folder_paths.get_output_directory()
                                        nanobanana_dir = os.path.join(output_dir, "nanobanana_outputs")
                                        os.makedirs(nanobanana_dir, exist_ok=True)
                                        
                                        import uuid
                                        file_name = os.path.join(nanobanana_dir, f"nanobanana_{uuid.uuid4()}.png")
                                        pil_img.save(file_name)
                                        print(f"[INFO] Saved image to: {file_name}")
                                        break  # Use first image
                                    except Exception as e:
                                        print(f"[ERROR] Failed to process image from content: {e}")
                                        import traceback
                                        traceback.print_exc()
                        elif hasattr(item, 'text'):
                            text_parts.append(item.text)
                    text_output = '\n'.join(text_parts)
                elif content:
                    text_output = content
                if text_output:
                    print(f"[INFO] Extracted text content length: {len(text_output)}")

            # Extract images - according to ref.md, message.images is a list of dicts
            if message and hasattr(message, 'images') and message.images:
                print(f"[INFO] Processing {len(message.images)} image(s) from message.images")
                for idx, image in enumerate(message.images):
                    # Handle both dict and object formats
                    if isinstance(image, dict):
                        image_url = image.get('image_url', {}).get('url', '')
                        if not image_url:
                            # Try alternative structure
                            image_url = image.get('url', '')
                    else:
                        # Object format
                        image_url = getattr(image, 'image_url', None)
                        if image_url:
                            if isinstance(image_url, dict):
                                image_url = image_url.get('url', '')
                            else:
                                image_url = getattr(image_url, 'url', '')
                        else:
                            # Try direct url attribute
                            image_url = getattr(image, 'url', '')
                    
                    # Only log URL prefix, not full base64 data
                    if image_url:
                        url_preview = image_url[:50] + "..." if len(image_url) > 50 else image_url
                        print(f"[DEBUG] Image {idx} URL preview: {url_preview}")
                    
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
                # Only print detailed debug info when there's an issue
                if message:
                    print(f"[DEBUG] Message has images attr: {hasattr(message, 'images')}")
                    if hasattr(message, 'images'):
                        print(f"[DEBUG] message.images value: {message.images}")
                    if hasattr(message, 'content'):
                        content = message.content
                        if isinstance(content, list):
                            print(f"[DEBUG] Content list has {len(content)} items")
                            for idx, item in enumerate(content):
                                if isinstance(item, dict):
                                    print(f"[DEBUG] Content item {idx} type: {item.get('type', 'unknown')}")
                                else:
                                    print(f"[DEBUG] Content item {idx} type: {type(item)}")
                    else:
                        print(f"[DEBUG] Message content type: {type(message.content) if hasattr(message, 'content') else 'N/A'}")
                image_tensor = self.generate_empty_image()
                if not text_output:
                    text_output = "Image generation completed but no image was returned"
                else:
                    text_output += "\n[WARNING] No image was returned from API"

            print(f"[INFO] Generation completed. Text length: {len(text_output)}")
            print(f"[INFO] Used seed: {actual_seed}")
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


# No frontend extensions needed
# WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {
    "NanobananaImageGenerator": NanobananaImageGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanobananaImageGenerator": "Nanobanana Image Generator",
}

