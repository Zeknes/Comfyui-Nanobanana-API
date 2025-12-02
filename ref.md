from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="<OPENROUTER_API_KEY>",
)

# Generate an image
response = client.chat.completions.create(
  model="google/gemini-3-pro-image-preview",
  messages=[
          {
            "role": "user",
            "content": "Generate a beautiful sunset over mountains"
          }
        ],
  extra_body={"modalities": ["image", "text"]}
)

# The generated image will be in the assistant message
response = response.choices[0].message
if response.images:
  for image in response.images:
    image_url = image['image_url']['url']  # Base64 data URL
    print(f"Generated image: {image_url[:50]}...")