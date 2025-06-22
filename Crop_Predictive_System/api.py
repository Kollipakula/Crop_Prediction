# import google.generativeai as genai
# API_KEY=''
# predicted_crop=input()
# genai.configure(api_key=API_KEY)
# gemini_model=genai.GenerativeModel('gemini-1.5-flash')
# customer_prompt = f"Provide detailed information about {predicted_crop} [the uses],and the heaith benifits of this crop [information of seeling this crop in trends]"
# try:
#     response = gemini_model.generate_content(customer_prompt)
#     gemini_output=response.text
#     print(gemini_output)
# except Exception as e:
#     print("An error occurred:", e)
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

API_KEY = os.environ.get('GEMINI_API_KEY')
predicted_crop = input()

genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')
customer_prompt = f"Provide detailed information about {predicted_crop} [the uses], and the health benefits of this crop [information of selling this crop in trends]"

try:
    response = gemini_model.generate_content(customer_prompt)
    gemini_output = response.text
    print(gemini_output)
except Exception as e:
    print("An error occurred:", e)

    