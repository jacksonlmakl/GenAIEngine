import os
import json
import subprocess
from PIL import ImageGrab  # To capture screenshots
import time
import boto3
from botocore.exceptions import NoCredentialsError
import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from flask import Flask, request, jsonify
import requests

# Load secrets.json and get necessary keys
def load_secrets():
    with open('secrets.json', 'r') as f:
        secrets = json.load(f)
    return secrets.get('aws_access_key'), secrets.get('aws_secret_key'), secrets.get('huggingface')


# Load LLaMA model and tokenizer from Hugging Face
def load_llama_model():
    # model_name = "meta-llama/Llama-2-7b-chat-hf"
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    _tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ['HUGGINGFACE_API_KEY'])  # Update for token
    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        token=os.environ['HUGGINGFACE_API_KEY']  # Update for token
    )
    return _tokenizer, _model

# Text prompt using LLaMA
def text_prompt(prompt, tokenizer, model, chain=None):
    # Tokenize the input and return as PyTorch tensors
    # Ensure pad_token_id is set to eos_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    # Move the inputs to the correct device (GPU or CPU)
    inputs = inputs.to(model.device)
    attention_mask = inputs['attention_mask']
    # Generate the output from the model
    outputs = model.generate(inputs["input_ids"], max_length=512,attention_mask=attention_mask,pad_token_id=tokenizer.pad_token_id)  # Ensure we pass input_ids to the model
    
    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Upload an image to S3
def upload_image_to_s3(image_file_path, s3_filename):
    bucket_name = 'jackson-makl-s3'
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key,
                      aws_secret_access_key=aws_secret_key, region_name='us-east-1')

    try:
        s3.upload_file(image_file_path, bucket_name, s3_filename, ExtraArgs={'ACL': 'public-read'})
        public_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_filename}"
        return public_url

    except FileNotFoundError:
        return "The file was not found."
    except NoCredentialsError:
        return "Credentials not available."

def llama_prompt(url_base,prompt):
    url=url_base+"/llama-engine"
    if "http://" not in url and "https://" not in url:
        url="http://"+url
        
    # The prompt you want to send
    data = {
        "prompt": prompt
    }

    # Send a POST request
    response = requests.post(url, headers={'Content-Type': 'application/json'}, data=json.dumps(data))

    # Check if the request was successful
    if response.status_code == 200:
        # Print the JSON response from the API
        print("Response from API:", response.json())
        return response.json().get('response',None)
    else:
        print("Failed to get a response, Status code:", response.status_code)
        return False


app = Flask(__name__)

@app.route('/llama-engine', methods=['POST'])
def post_example():
    # Get JSON data from the request
    data = request.get_json()
    prompt = data.get('prompt','Hello World')
    response=text_prompt(prompt, tokenizer, model, chain=None)
    # Perform some operation with the data (for demonstration, just echo it back)
    response_data = {
        "response": str(response)
    }

    # Return a JSON response
    return jsonify(response_data), 200

# Main function to run the app
if __name__ == '__main__':
    # Set environment variables for Hugging Face API key
    global aws_access_key, aws_secret_key, huggingface_api_key, tokenizer, model
    aws_access_key, aws_secret_key, huggingface_api_key = load_secrets()
    os.environ['HUGGINGFACE_API_KEY'] = huggingface_api_key

    tokenizer, model = load_llama_model()
    print("\nMeta-Llama model has been loaded....\n")
    app.run(debug=True,host='127.0.0.1',port=4040 )

