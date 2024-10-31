import os
import io
import base64
import requests
import json
import gradio as gr
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
hf_api_key = os.getenv("HF_API_KEY")
TTI_ENDPOINT = os.getenv("HF_API_TTI_BASE")
ITT_ENDPOINT = os.getenv("HF_API_ITT_BASE")

# Helper functions for image encoding and decoding
def image_to_base64_str(pil_image):
    try:
        byte_arr = io.BytesIO()
        pil_image.save(byte_arr, format="PNG")
        return base64.b64encode(byte_arr.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def base64_to_pil(img_base64):
    try:
        byte_stream = io.BytesIO(base64.b64decode(img_base64))
        return Image.open(byte_stream)
    except Exception as e:
        print(f"Error decoding Base64 image: {e}")
        return None

# Function to resize image before sending to API
def resize_image(image, max_size=(512, 512)):
    return image.resize(max_size, Image.Resampling.LANCZOS)

# API request function with content-type check
def get_completion(inputs, endpoint_url):
    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }
    response = requests.post(endpoint_url, headers=headers, json={"inputs": inputs})

    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return None

    # Check if the response is JSON or binary (for images)
    content_type = response.headers.get('Content-Type')
    if content_type == "application/json":
        try:
            result = response.json()
            print("API JSON Response:", result)  # Log response for debugging
            return result
        except json.JSONDecodeError as e:
            print("Failed to decode JSON response:", e)
            print("Raw Response:", response.text)
            return None
    elif content_type == "image/jpeg":  # For JPEG images
        print("Received binary JPEG image data.")
        try:
            img = Image.open(io.BytesIO(response.content))
            return img
        except Exception as e:
            print(f"Error processing JPEG image: {e}")
            return "Failed to process JPEG image."
    else:
        print("Unexpected Content-Type:", content_type)
        return None

# Image captioning function
def captioner(image):
    resized_image = resize_image(image)  # Resize before processing
    base64_image = image_to_base64_str(resized_image)
    if not base64_image:
        return "Failed to encode image to Base64."

    result = get_completion(base64_image, ITT_ENDPOINT)
    if result and isinstance(result, list) and "generated_text" in result[0]:
        return result[0]["generated_text"]
    return "Failed to generate caption."

# Image generation function
def generate(prompt):
    output = get_completion(prompt, TTI_ENDPOINT)
    
    if not output:
        return "Image generation failed. Check API limits or input size."
    
    # Process binary image response if received
    if isinstance(output, Image.Image):  # Output is a PIL image
        return output
    elif isinstance(output, list) and "generated_image" in output[0]:
        img_base64 = output[0]["generated_image"]
        return base64_to_pil(img_base64)
    else:
        print("Unexpected output format:", output)
        return "Image generation failed."

# Combined caption and image generation function
def caption_and_generate(image):
    caption = captioner(image)
    generated_image = generate(caption)
    if not isinstance(generated_image, Image.Image):
        return caption, "Image generation failed. Try reducing image size or restarting the application."
    return caption, generated_image

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Describe-and-Generate Game 🖍️")
    
    image_upload = gr.Image(label="Upload an Image", type="pil")
    btn_caption = gr.Button("Generate Caption")
    caption = gr.Textbox(label="Generated Caption")
    btn_image = gr.Button("Generate Image")
    image_output = gr.Image(label="Generated Image")
    btn_all = gr.Button("Caption and Generate")

    # Button click events
    btn_caption.click(fn=captioner, inputs=[image_upload], outputs=[caption])
    btn_image.click(fn=generate, inputs=[caption], outputs=[image_output])
    btn_all.click(fn=caption_and_generate, inputs=[image_upload], outputs=[caption, image_output])

# Launch Gradio App
demo.launch(share=True, server_port=int(os.getenv('PORT1', 7860)), debug=True)
