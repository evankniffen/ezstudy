import os
import re
import requests
import wikipedia
from google import genai
from wikipedia_fetcher import fetch_wikipedia_summary
from spellchecker import SpellChecker

WOLFRAM_APPID = "HPQQ9Y-734KXXQEE3"
GEMINI_API_KEY = "AIzaSyBXLFpQHahdMYY4KGWtSFEmouexhXCUtPc"
TRAINING_DATA_FILE = "training_data.txt"

spell = SpellChecker()
conversation_history = []
math_expr_pattern = re.compile(r'^\d+x\d+$', re.IGNORECASE)

def correct_query(query):
    """Correct typos in the query using pyspellchecker, skipping words that match math expressions."""
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
    Use Gemini to decide if the query is academic.
    Respond with one word: 'Academic' if academic, 'Trivial' if not.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = (
        "You are an academic assistant. Evaluate the following query and determine whether it is academic "
        "(requiring detailed explanation, study help, or subject-specific analysis) or trivial "
        "(casual chit-chat or a greeting). Respond with exactly one word: 'Academic' if it is academic, or 'Trivial' if it is not.\n\n"
        f"Query: {query}"
    )
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    answer = response.text.strip().lower()
    return "academic" in answer

def load_training_data():
    """Load training data from file; return an empty string if not found."""
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

def add_transcribed_text(transcribed_text):
    """
    Add previously transcribed text (provided as a string) to training data.
    Auto-add related Wikipedia context based on the transcribed text and update conversation history.
    """
    if transcribed_text.strip():
        add_training_data(transcribed_text)
        auto_add_related_wikipedia(transcribed_text)
        conversation_history.append(f"Transcribed Text: {transcribed_text}")
        print("Transcribed text added to training data.")
    else:
        print("No transcribed text provided to add.\n")

def detect_wikipedia_relevance(query):
    """
    Use Gemini to decide if the query would benefit from additional background context from Wikipedia.
    Respond with one word: 'Yes' or 'No'.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = (
        "You are an academic assistant. Determine if the following query would benefit from additional "
        "background context from Wikipedia. Respond with exactly one word: 'Yes' if it would, or 'No' if not.\n\n"
        f"Query: {query}"
    )
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    answer = response.text.strip().lower()
    return "yes" in answer

def auto_add_related_wikipedia(query):
    """
    Automatically search for related Wikipedia articles if Gemini deems additional context is needed.
    For each found article, fetch its summary and add it to training data.
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

def extract_math_problem(query):
    """
    Use Gemini to extract only the math problem part from a larger query.
    For example, given 'can you integrate 5x+4 for me', return 'integrate 5x+4'.
    If no math problem can be isolated, return an empty string.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = (
        "You are a math assistant. Given the following sentence, extract only the math problem directive, "
        "removing any conversational fluff. For example, if given 'can you integrate 5x+4 for me', return 'integrate 5x+4'. "
        "If no math problem is present, return an empty string.\n\n"
        f"Sentence: {query}\n\nMath Problem:"
    )
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    math_problem = response.text.strip()
    return math_problem

def query_wolfram(math_query):
    """
    Query Wolfram|Alpha’s LLM API with a math query.
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
    Use Gemini to decide if the query involves math.
    Respond with one word: 'Yes' if math is involved, or 'No' if not.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = (
        "You are an academic expert in mathematics and science. A student has posed a question. "
        "Determine if the question involves a mathematical problem (e.g., solving equations, calculations, differentiation, integration, etc.). "
        "Respond with exactly one word: 'Yes' if it does, or 'No' if it does not.\n\n"
        f"Query: {query}"
    )
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    answer = response.text.strip().lower()
    return "yes" in answer

def summarize_with_gemini(text):
    """
    Ask Gemini to summarize the Wolfram|Alpha output into a clear, concise explanation (max 150 words).
    Include training data and conversation history as context.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    training_data = load_training_data()
    conversation_context = get_conversation_context()
    prompt = (
        "You are an experienced academic tutor. Using the following training data and conversation history as context, "
        "please summarize the Wolfram|Alpha output into a clear, concise explanation (max 150 words). "
        "Include step-by-step solutions only if necessary.\n\nTraining Data:\n" + training_data + "\n\n"
        "Conversation History:\n" + conversation_context + "\n\n"
        "Wolfram|Alpha Output:\n" + text
    )
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return response.text

def direct_chat(query, conversation_context):
    """
    Use Gemini's general chat functionality for nonacademic queries.
    Provide a casual, conversational answer.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = (
        "You are a friendly chatbot. Here is the conversation so far:\n" + conversation_context + "\n\n"
        "Now answer the following question in a casual, concise, and helpful manner:\n"
        "Question: " + query
    )
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return response.text

def direct_gemini(query, conversation_context):
    """
    Use Gemini as an academic tutor with full context.
    Instruct Gemini to pull from training data and external sources if needed,
    and answer in a concise (100–150 word) educational manner.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    training_data = load_training_data()
    prompt = (
        "You are an academic tutor and subject matter expert with real-time access to Google search. "
        "Using the following training data and conversation history as background context, please answer the following question "
        "in a clear, comprehensive, and educational manner (around 100–150 words). "
        "If the answer is not present in the training data, please perform a brief Google search and incorporate the most relevant, up-to-date information.\n\n"
        "Training Data:\n" + training_data + "\n\n"
        "Conversation History:\n" + conversation_context + "\n\n"
        "Question: " + query
    )
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return response.text

def main():
    global conversation_history
    print("Academic Chatbot started.")
    print("Type 'quit' or 'exit' to stop.")
    print("Commands:")
    print("  ADD TRAINING: <text>         - Add text to training data")
    print("  ADD FILE: <filepath>         - Add content from a text file to training data")
    print("  ADD TRANSCRIBED: <text>      - Add previously transcribed text to training data")
    print("  ADD WIKIPEDIA: <topic>       - Fetch and add a Wikipedia summary for the given topic to training data\n")
    
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

        if corrected_query.lower().startswith("add transcribed:"):
            transcribed_text = corrected_query[len("add transcribed:"):].strip()
            add_transcribed_text(transcribed_text)
            continue

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

        if corrected_query.lower().startswith("add wikipedia:"):
            topic = corrected_query[len("add wikipedia:"):].strip()
            if topic:
                summary = fetch_wikipedia_summary(topic)
                add_training_data(summary)
            else:
                print("No topic provided for Wikipedia lookup.\n")
            continue

        # Auto-add related Wikipedia context for the query.
        auto_add_related_wikipedia(corrected_query)
        conversation_context = get_conversation_context()

        # If the query involves math, isolate the math problem using Gemini.
        if detect_math_with_gemini(corrected_query):
            math_problem = extract_math_problem(corrected_query)
            if not math_problem:
                math_problem = corrected_query
            print("\nMath detected by Gemini. Querying Wolfram|Alpha with:", math_problem)
            success, wolfram_output = query_wolfram(math_problem)
            if success:
                print("\nSummarizing Wolfram output with Gemini...")
                bot_response = summarize_with_gemini(wolfram_output)
            else:
                print("\nWolfram|Alpha returned an error. Falling back to academic Gemini answer.")
                bot_response = direct_gemini(corrected_query, conversation_context)
        else:
            if not is_academic_query(corrected_query):
                bot_response = direct_chat(corrected_query, conversation_context)
            else:
                bot_response = direct_gemini(corrected_query, conversation_context)

        print("\nBot:", bot_response)
        conversation_history.append(f"You: {corrected_query}")
        conversation_history.append(f"Bot: {bot_response}")
        print()

if __name__ == '__main__':
    main()
    
def extract_math_problem(query):
    """
    Use Gemini to extract only the math problem directive from a larger query.
    For example, if given "can you integrate 5x+4 for me", return "integrate 5x+4".
    If no math problem can be isolated, return an empty string.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = (
        "You are a math assistant. Extract only the math problem directive from the following text, removing any conversational or extraneous content. "
        "For example, if given 'can you integrate 5x+4 for me', return 'integrate 5x+4'. "
        "If no clear math problem is present, return an empty string.\n\n"
        f"Text: {query}\n\nMath Problem:"
    )
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    math_problem = response.text.strip()
    return math_problem
