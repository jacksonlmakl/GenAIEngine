import os
import openai
import json
from flask import Flask, request, jsonify
import subprocess
from PIL import ImageGrab  # To capture screenshots
import time
import boto3
from botocore.exceptions import NoCredentialsError
import datetime

# Load secrets.json and get OpenAI API key
def load_secrets():
    with open('secrets.json', 'r') as f:
        secrets = json.load(f)
    return secrets.get('openai'),secrets.get('huggingface')
    




def upload_image_to_s3(image_file_path, s3_filename):
    with open('secrets.json', 'r') as f:
        _secrets = json.load(f)
    # Set up your AWS credentials and region
    aws_access_key = _secrets.get('aws_access_key')
    aws_secret_key = _secrets.get('aws_secret_key')
    bucket_name = _secrets.get('aws_bucket_name')

    # Initialize the S3 client
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key,
                      aws_secret_access_key=aws_secret_key, region_name='us-east-1')

    try:
        # Upload the image to S3
        s3.upload_file(image_file_path, bucket_name, s3_filename, ExtraArgs={'ACL': 'public-read'})


        # Generate the public URL
        public_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_filename}"
        return public_url

    except FileNotFoundError:
        return "The file was not found."
    except NoCredentialsError:
        return "Credentials not available."


def text_prompt(prompt,chain=None):
    os.environ['OPENAI_API_KEY'] = load_secrets()[0]
    client=openai.OpenAI()
    context=f"""You are a helpful assistant, please answer this prompt:  """
    messages=[
            {"role": "system", "content": context+prompt}
        ]
    if chain:
        messages.extend(chain)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return response.choices[0].message.content



# Function to capture a screenshot and save it locally
def capture_screenshot(filename='screenshot.png'):
    screenshot = ImageGrab.grab()
    screenshot.save(filename)
    return filename




def image_text_prompt(url, user_prompt,chain=None):
    # Initialize the OpenAI API client
    client = openai.OpenAI()

    # Construct the context message
    context = """You are a helpful assistant. Please answer this prompt based on the image and text provided."""
    messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": context+user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": url,
                        }
                    },
                ],
            }
        ]
    if chain:
        messages.extend(chain)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )

    # Return the first completion's response
    return response.choices[0].message.content

#Send a screenshot of current screen & attach a prompt
def visual_prompt(user_prompt,chain=None,filename=None):
    os.environ['OPENAI_API_KEY'] = load_secrets()[0]
    if not filename:
        image_path= "screenshot__"+datetime.datetime.now().__str__().replace("-","_").replace(" ","__").replace(":","_").replace(".","_")+".png"
        capture_screenshot(filename=image_path)
    else:
        image_path=filename
    url = upload_image_to_s3(image_path, image_path)
    r=image_text_prompt(url,user_prompt,chain)
    return r


if __name__=='__main__':
    os.environ['OPENAI_API_KEY'] = load_secrets()[0]

    r=text_prompt("What is the capital of New York state?")
    print("Text Prompt Response: \n",r,'\n')

    visual_r=visual_prompt("What is this?",chain=None,filename='resources/puppy.jpg')
    print("Visual Prompt Response: \n", visual_r,'\n')



