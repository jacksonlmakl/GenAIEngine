import os
import json
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, BlipProcessor
from flask import Flask, request, jsonify
import requests
from io import BytesIO

# Load secrets.json and get necessary keys
def load_secrets():
    with open('secrets.json', 'r') as f:
        secrets = json.load(f)
    return secrets.get('aws_access_key'), secrets.get('aws_secret_key'), secrets.get('huggingface')

# Download the image from the given URL
def download_image(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        raise Exception(f"Failed to download image from URL {image_url}, status code: {response.status_code}")

# Load LLaMA Vision Instruct model and processor from Hugging Face
def load_llama_model():
    model_name = "meta-llama/Llama-3.2-90B-Vision-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ['HUGGINGFACE_API_KEY'])
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        token=os.environ['HUGGINGFACE_API_KEY']
    )
    processor = BlipProcessor.from_pretrained(model_name)  # Processor to handle images
    return tokenizer, model, processor

# Generate a response using LLaMA with the image and prompt
def text_prompt_with_image(image, prompt, processor, tokenizer, model):
    # Process the image and prompt
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

    # Generate output using the model
    outputs = model.generate(**inputs, max_length=512)
    
    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Flask app setup
app = Flask(__name__)

@app.route('/llama-vision-engine', methods=['POST'])
def llama_engine():
    # Get JSON data from the request
    data = request.get_json()
    image_url = data.get('image_url')  # Get image URL from request
    prompt = data.get('prompt', 'Describe this image')

    # Download the image from the provided URL
    try:
        image = download_image(image_url)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # Generate a response using the image and prompt
    response = text_prompt_with_image(image, prompt, processor, tokenizer, model)
    response_data = {
        "response": response
    }

    # Return the response as JSON
    return jsonify(response_data), 200

# Main function to run the app
if __name__ == '__main__':
    # Set environment variables for Hugging Face API key
    aws_access_key, aws_secret_key, huggingface_api_key = load_secrets()
    os.environ['HUGGINGFACE_API_KEY'] = huggingface_api_key

    # Load the model, tokenizer, and processor
    tokenizer, model, processor = load_llama_model()
    print("\nMeta-Llama model has been loaded....\n")
    
    # Run the Flask app
    app.run(debug=True, host='127.0.0.1', port=4040)
