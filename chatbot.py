import os
import string
import requests
import wikipedia
from google import genai
from wikipedia_fetcher import fetch_wikipedia_summary
from spellchecker import SpellChecker
import re

WOLFRAM_APPID = "HPQQ9Y-734KXXQEE3"
GEMINI_API_KEY = "AIzaSyBXLFpQHahdMYY4KGWtSFEmouexhXCUtPc"
TRAINING_DATA_FILE = "training_data.txt"

spell = SpellChecker()
conversation_history = []
math_expr_pattern = re.compile(r'^\d+x\d+$', re.IGNORECASE)

def correct_query(query):
    """Correct typos in the query using pyspellchecker, but skip words that match a math expression pattern."""
    words = query.split()
    corrected_words = []
    for word in words:
        if math_expr_pattern.match(word):
            corrected_words.append(word)
        else:
            correction = spell.correction(word)
            corrected_words.append(correction if correction is not None else word)
    return " ".join(corrected_words)


def get_conversation_context():
    """Return the full conversation history as a single string."""
    return "\n".join(conversation_history)

def is_academic_query(query):
    """
    Use Gemini to determine if the query is academic in nature.
    Gemini is instructed to respond with exactly one word: 
    'Academic' if the query is academic, or 'Trivial' if it is not.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = (
        "You are an academic assistant. Evaluate the following query and determine whether it is academic "
        "(requiring detailed explanation, study help, or subject-specific analysis) or trivial "
        "(casual chit-chat or a simple greeting). Respond with exactly one word: 'Academic' if it is academic, "
        "or 'Trivial' if it is not.\n\n"
        f"Query: {query}"
    )
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    answer = response.text.strip().lower()
    return "academic" in answer

def load_training_data():
    """Load training data from file; return an empty string if file doesn't exist."""
    if os.path.exists(TRAINING_DATA_FILE):
        with open(TRAINING_DATA_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    else:
        return ""

def add_training_data(new_data):
    """Append new training data to the training file."""
    with open(TRAINING_DATA_FILE, "a", encoding="utf-8") as f:
        f.write(new_data.strip() + "\n")
    print("Training data added.\n")

def detect_wikipedia_relevance(query):
    """
    Ask Gemini to determine if the given query would benefit from additional background context 
    from Wikipedia. Respond with exactly one word: 'Yes' or 'No'.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = (
        "You are an academic assistant. Determine if the following query would benefit from additional "
        "background context from Wikipedia. If it would, respond with exactly one word: 'Yes'. Otherwise, "
        "respond with 'No'.\n\n"
        f"Query: {query}"
    )
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    answer = response.text.strip().lower()
    return "yes" in answer

def auto_add_related_wikipedia(query):
    """
    Automatically search Wikipedia for related topics if Gemini deems additional context is needed.
    For each related article, fetch a short summary and add it to training data.
    """
    if detect_wikipedia_relevance(query):
        print("Gemini indicates additional Wikipedia context is beneficial.")
        try:
            results = wikipedia.search(query, results=3)
            if results:
                print("Automatically adding related Wikipedia summaries to training data:")
                for title in results:
                    summary = fetch_wikipedia_summary(title)
                    add_training_data(f"Summary for {title}: {summary}")
        except Exception as e:
            print("Error while auto-adding Wikipedia articles:", e)
    else:
        print("Gemini determined no additional Wikipedia context is necessary.")

def query_wolfram(math_query):
    """
    Query Wolfram|Alpha’s LLM API with the math query.
    Returns a tuple (success, output).
    """
    base_url = "https://www.wolframalpha.com/api/v1/llm-api"
    params = {"appid": WOLFRAM_APPID, "input": math_query}
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return True, response.text
    else:
        return False, response.text

def detect_math_with_gemini(query):
    """
    Ask Gemini to determine if the query involves math.
    Respond with exactly one word: 'Yes' if math is involved, or 'No' if it is not.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = (
        "You are an academic expert in mathematics and science. A student has posed a question. "
        "Determine if the question involves a mathematical problem (e.g., solving equations, calculations, "
        "differentiation, integration, etc.). Respond with exactly one word: 'Yes' if it does, or 'No' if it does not.\n\n"
        f"Query: {query}"
    )
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    answer = response.text.strip().lower()
    return "yes" in answer

def summarize_with_gemini(text):
    """
    Ask Gemini to summarize the Wolfram|Alpha output concisely.
    Limit the summary to no more than 150 words. Training data and conversation history are added as context.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    training_data = load_training_data()
    conversation_context = get_conversation_context()
    prompt = (
        "You are an experienced academic tutor. Using the following training data and conversation history as context, "
        "please summarize the Wolfram|Alpha output into a clear, concise explanation limited to 150 words. "
        "Include step-by-step solutions only if necessary.\n\n"
        f"Training Data:\n{training_data}\n\n"
        f"Conversation History:\n{conversation_context}\n\n"
        f"Wolfram|Alpha Output:\n{text}"
    )
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return response.text

def direct_chat(query, conversation_context):
    """
    Query Gemini's normal chatbot functionality for nonacademic queries.
    The prompt instructs Gemini to provide a casual, conversational answer.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = (
        "You are a friendly chatbot. Here is the conversation so far:\n"
        f"{conversation_context}\n\n"
        "Now answer the following question in a casual, concise, and helpful manner:\n"
        f"Question: {query}"
    )
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return response.text

def direct_gemini(query, conversation_context):
    """
    Query Gemini as an academic tutor with full context. The prompt instructs Gemini to pull from both the 
    training data and external web sources if needed. The answer is expected to be concise (around 100–150 words)
    and educational. If the answer is not found in the training data, please perform a brief real-time Google search 
    and include the most relevant information in your answer.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    training_data = load_training_data()
    prompt = (
        "You are an academic tutor and subject matter expert with real-time access to Google search. "
        "Using the following training data and conversation history as background context, please answer the following question "
        "in a clear, comprehensive, and educational manner in about 100–150 words. "
        "If the relevant information is not present in the training data, please perform a brief real-time Google search and incorporate "
        "the most relevant, up-to-date information into your answer.\n\n"
        f"Training Data:\n{training_data}\n\n"
        f"Conversation History:\n{conversation_context}\n\n"
        f"Question: {query}"
    )
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return response.text

def main():
    global conversation_history
    print("Academic Chatbot started.")
    print("Type 'quit' or 'exit' to stop.")
    print("Commands:")
    print("  ADD TRAINING: <text>       - Add text to training data")
    print("  ADD FILE: <filepath>       - Add content from a text file to training data")
    print("  ADD IMAGE: <filepath>      - Extract and add text from an image file to training data")
    print("  ADD WIKIPEDIA: <topic>     - Fetch and add a Wikipedia summary for the given topic to training data\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        corrected_query = correct_query(user_input)
        if corrected_query.lower() != user_input.lower():
            print(f"\nInterpreting query as: {corrected_query}")
        else:
            corrected_query = user_input

        if corrected_query.lower().startswith("add training:"):
            new_data = corrected_query[len("add training:"):].strip()
            if new_data:
                add_training_data(new_data)
            else:
                print("No training data provided to add.\n")
            continue

        if corrected_query.lower().startswith("add file:"):
            file_path = corrected_query[len("add file:"):].strip()
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        file_content = f.read()
                    add_training_data(file_content)
                except Exception as e:
                    print("Error reading file:", e)
            else:
                print("File not found.\n")
            continue

        if corrected_query.lower().startswith("add image:"):
            image_path = corrected_query[len("add image:"):].strip()
            if os.path.exists(image_path):
                try:
                    from PIL import Image
                    import pytesseract
                except ImportError:
                    print("pytesseract and Pillow are required for image OCR. Please install them via 'pip install pytesseract Pillow'.\n")
                    continue
                try:
                    image = Image.open(image_path)
                    extracted_text = pytesseract.image_to_string(image)
                    if extracted_text:
                        add_training_data(extracted_text)
                    else:
                        print("No text detected in the image.\n")
                except Exception as e:
                    print("Error processing image:", e)
            else:
                print("Image file not found.\n")
            continue

        if corrected_query.lower().startswith("add wikipedia:"):
            topic = corrected_query[len("add wikipedia:"):].strip()
            if topic:
                summary = fetch_wikipedia_summary(topic)
                add_training_data(summary)
            else:
                print("No topic provided for Wikipedia lookup.\n")
            continue

        auto_add_related_wikipedia(corrected_query)
        conversation_context = get_conversation_context()

        if not is_academic_query(corrected_query):
            bot_response = direct_chat(corrected_query, conversation_context)
        else:
            if detect_math_with_gemini(corrected_query):
                print("\nMath detected by Gemini. Querying Wolfram|Alpha...")
                success, wolfram_output = query_wolfram(corrected_query)
                if success:
                    print("\nSummarizing Wolfram output with Gemini...")
                    bot_response = summarize_with_gemini(wolfram_output)
                else:
                    print("\nWolfram|Alpha returned an error. Falling back to academic Gemini answer.")
                    bot_response = direct_gemini(corrected_query, conversation_context)
            else:
                bot_response = direct_gemini(corrected_query, conversation_context)

        print("\nBot:", bot_response)
        conversation_history.append(f"You: {corrected_query}")
        conversation_history.append(f"Bot: {bot_response}")
        print()

if __name__ == '__main__':
    main()
