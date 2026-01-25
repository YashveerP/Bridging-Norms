import os

from helpers import followNorms

# # norm list to follow
norms = ["Always respond politely and respectfully.", 
         "Never insult, mock, or belittle",
         "Show empathy when users express frustration or emotion.",
         "Avoid profanity and offensive language.",
         "If a request violates these norms, politely refuse and explain why."]

# Ask for prompt
print("Prompt: ")
prompt = input()

response = followNorms(norms, prompt)
print(response)