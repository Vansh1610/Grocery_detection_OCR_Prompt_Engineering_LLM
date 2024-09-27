import os

from groq import Groq

import random
random.seed(42)

GROQ_KEY=os.environ['GROQ_KEY']
client = Groq(
    api_key=GROQ_KEY,
)


def prompt_refine(g_list):

    g_list=[i for i in g_list if len(i)>3]

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content":  f"""
                            Correct the following list of grocery items and provide the output in the format [item, quantity, unit]:

                            Standardize item names to common grocery terms.
                            If a unit is missing, use None.
                            If a quantity is missing, use 0.
                            Use only these units: Kg, g, Lb, oz, L, ml, pack, piece.
                            Here’s the list to correct: {g_list} .

            
                            Please provide only the corrected list in this format without any additional explanation."""
            }
        ],
        model="llama3-groq-70b-8192-tool-use-preview",
    )
    
    return chat_completion.choices[0].message.content