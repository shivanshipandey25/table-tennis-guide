from ai21 import AI21Client
from ai21.models.chat.chat_message import SystemMessage, UserMessage, AssistantMessage

# Initialize client with your Studio API key
client = AI21Client(api_key="f462a241-9f04-4f74-867f-db7c41a11f19")  # Use Studio key here

# Define system instruction as a message
system = "You're a support engineer in a SaaS company"
system_message = SystemMessage(content=system, role="system")

# Initialize message history with system message
messages = [system_message]

# Intro message
print("Type 'exit' to end conversation\n")
print("Support Bot: Hi! I'm here to help you with your signup issues.\n")

# Conversation loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chat. Goodbye!")
        break

    # Add user message to history
    messages.append(UserMessage(content=user_input, role="user"))

    # Generate AI response
    response = client.chat.completions.create(
        messages=messages,
        model="jamba-mini",
        max_tokens=100,
        temperature=0.7,
        top_p=1.0,
    )

    ai_reply = response.choices[0].message.content
    print("Support Bot:", ai_reply)

    # Add assistant response to history
    messages.append(AssistantMessage(content=ai_reply, role="assistant"))
