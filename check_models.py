import google.generativeai as genai
import os

API_KEY = "AIzaSyCahdcqm4qGOpQZN0WxXJ4iH3sot98B9o4"
genai.configure(api_key=API_KEY)

try:
    print("Listing models...")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print(f"Error listing models: {e}")
