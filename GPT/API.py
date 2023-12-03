import os
from openai import OpenAI
import openai
import re

client = OpenAI(
    api_key ='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',
)

def get_sentiment(input_text):
    prompt = f"Respond in the json format: {{'response':  sentiment_classification}}\nText: {input_text}\nSentiment (positive, neutral, negative):"
    response = client.chat.completions.create(
        model= 'gpt-3.5-turbo' ,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=40,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].message.content
    response_text = response.choices[0].text.strip()
    sentiment = re.search("negative|neutral|positive", response_text).group(0)


# Test example
input_text = "s this treatment ok for a 3 year old"
sentiment = get_sentiment(input_text)
print("Result\n", f"{sentiment}")




# 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

