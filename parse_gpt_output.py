def get_answer_gpt(chat):
    return chat.choices[0].message["content"]
    
def get_usage_gpt(chat):
    usage = chat.usage
    return usage