from prompts import REFLECTION_PROMPT

class Reflection():
    def __init__(self, llm):
        self.llm = llm

    def _concat_and_format_texts(self, data)->str:
        concatenatedTexts = []
        for entry in data:
            role = entry.get('role', '')
            if entry.get('parts'):
                all_texts = ' '.join(part['text'] for part in entry['parts'] )
            elif entry.get('content'):
                all_texts = entry['content'] 
            concatenatedTexts.append(f"{role}: {all_texts} \n")
        return ''.join(concatenatedTexts)


    def __call__(self, chatHistory, lastItemsConsidereds=3):
        
        if len(chatHistory) >= lastItemsConsidereds:
            chatHistory = chatHistory[len(chatHistory) - lastItemsConsidereds:]

        historyString = self._concat_and_format_texts(chatHistory)
        print("History string: ", historyString)

        higherLevelSummariesPrompt = {
            "role": "user",
            "content": REFLECTION_PROMPT.format(historyString=historyString)
        }

        completion = self.llm.generate_content([higherLevelSummariesPrompt])
        if not completion:
            print('[LOG][Warning]The reflection return None')
            return historyString
        
        return completion
    
if __name__ == '__main__':
    import os
    import sys

    # Thêm thư mục cha (parent directory) vào sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from llms import LLMs
    llm = LLMs(
        type="online",
        model_name="chatgroq",
        api_key="YOUR_GROQ_API_KEY",
        model_version="llama-3.1-8b-instant",
        base_url="https://api.groq.com"
    )
    chat_history = [
        {"role": "user", "content": "Xin chào bạn"}
,
    ]

    reflection = Reflection(llm)
    question = reflection(chat_history)
    print(question)