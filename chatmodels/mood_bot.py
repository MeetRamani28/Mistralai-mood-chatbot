import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

model = ChatMistralAI(
    model = "mistral-small-latest",
    temperature=0.9
)

messages= [
    SystemMessage(content="You are a helpful and polite AI assistant."),
]

print("Welcome! Type '0' to exit the application.")

while True:
    prompt = input("You : ")

    if prompt == "0":
        break

    messages.append(HumanMessage(content=prompt))

    response = model.invoke(messages)
    
    messages.append(AIMessage(content=response.content))

    print("Bot: ", response.content)

print("\n--- Final Chat History Log ---")
print(messages)