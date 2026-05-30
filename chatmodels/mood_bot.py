import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

model = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.9,
    mistral_api_key=os.getenv("MISTRAL_API_KEY") 
)

print("--- Select Assistant Mood ---")
print("1. Funny")
print("2. Angry")
print("3. Sad")
print("4. Romantic")
print("5. Sarcastic")
mood_choice = input("Enter mood number (1-5) or type custom mood: ")

mood_mapping = {
    "1": "funny assistant who cracks jokes often",
    "2": "very angry assistant who gets annoyed quickly",
    "3": "sad and depressed assistant",
    "4": "romantic and poetic assistant",
    "5": "sarcastic assistant who makes fun of everything"
}

selected_mood = mood_mapping.get(mood_choice, mood_choice)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a {mood}."),
])

formatted_prompt = prompt_template.format_prompt(mood=selected_mood)

messages = formatted_prompt.to_messages()

print("\nWelcome! Type '0' to exit the application.")

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