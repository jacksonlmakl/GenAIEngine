from llama_engine import llama_prompt
from openai_engine import text_prompt, visual_prompt

#Text GPT4o Prompt
open_ai_text_response=text_prompt("What is the capital of New York state?")
print("Text Prompt Response: \n",open_ai_text_response,'\n')

#Visual GPT4o Prompt
open_ai_visual_response=visual_prompt("What is this?",chain=None,filename='resources/puppy.jpg')
print("Visual Prompt Response: \n", open_ai_visual_response,'\n')

#Text Meta-Llama Prompt
llama_response=llama_prompt("http://127.0.0.1:4040","How many people live in Los Angeles")
print("Meta-Llama Text Response", llama_response)
