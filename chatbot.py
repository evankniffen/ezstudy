import os
import requests
import wikipedia
from google import genai
from wikipedia_fetcher import fetch_wikipedia_summary

WOLFRAM_APPID = "HPQQ9Y-734KXXQEE3"
GEMINI_API_KEY = "AIzaSyBXLFpQHahdMYY4KGWtSFEmouexhXCUtPc"

TRAINING_DATA_FILE = "training_data.txt"

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

def auto_add_related_wikipedia(query):
    """
    Automatically search Wikipedia for related topics based on the user's query,
    and append a short summary for each found article to the training data.
    """
    try:
        results = wikipedia.search(query, results=3)
        if results:
            print("Automatically adding related Wikipedia summaries to training data:")
            for title in results:
                summary = fetch_wikipedia_summary(title)
                add_training_data(f"Summary for {title}: {summary}")
    except Exception as e:
        print("Error while auto-adding Wikipedia articles:", e)

def query_wolfram(math_query):
    """
    Query Wolfram|Alpha’s LLM API with the math query.
    Returns a tuple (success, output).
    """
    base_url = "https://www.wolframalpha.com/api/v1/llm-api"
    params = {
        "appid": WOLFRAM_APPID,
        "input": math_query,
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return True, response.text
    else:
        return False, response.text

def detect_math_with_gemini(query):
    """
    Ask Gemini to determine if the query involves math.
    The prompt instructs Gemini to act as an academic subject matter expert
    and respond with exactly one word: "Yes" if math is involved, or "No" if it does not.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = (
        "You are an academic expert in mathematics and science. A student has posed a question. "
        "Determine if the question involves a mathematical problem (such as solving equations, "
        "performing calculations, differentiation, integration, etc.). "
        "Respond with exactly one word: 'Yes' if the query involves math, or 'No' if it does not.\n\n"
        f"Query: {query}"
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    answer = response.text.strip().lower()
    return "yes" in answer

def summarize_with_gemini(text):
    """
    Ask Gemini to summarize the Wolfram|Alpha output.
    The prompt instructs Gemini to generate a clear, concise, and educational summary
    that is suitable for academic learning. The current training data is added as context.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    training_data = load_training_data()
    prompt = (
        "You are an experienced academic tutor. Use the following training data as additional context "
        "to help craft your explanation:\n"
        f"Training Data:\n{training_data}\n\n"
        "Please summarize the following Wolfram|Alpha output into a clear, concise, and educational explanation "
        "that would help a student understand the result. Ensure your response covers the essential points "
        "without excessive details. Do not over explain; assume the student has been taught the prerequisites. "
        "Include step-by-step solutions where necessary. Ensure all responses are concise unless prompted otherwise."
        "Do not restate phrases unless told to do so. Do not over-explain.\n\n"
        f"Wolfram|Alpha Output:\n{text}"
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    return response.text

def direct_gemini(query):
    """
    Query Gemini directly with the input query.
    The prompt instructs Gemini to respond as an academic tutor, providing a clear, comprehensive, and educational answer.
    Training data is added as context to influence the response.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    training_data = load_training_data()
    prompt = (
        "You are an academic tutor and subject matter expert. Use the following training data as background context "
        "to inform your answer:\n"
        f"Training Data:\n{training_data}\n\n"
        "Answer the following question in a clear, comprehensive, and educational manner that helps a student understand "
        "the concept. Keep the response focused and avoid unnecessary verbosity. Do not include information not provided "
        "in the query. Include step-by-step solutions where necessary. Ensure all responses are concise unless prompted otherwise."
        "Do not restate phrases unless told to do so. Do not over-explain.\n\n"
        f"Question: {query}"
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    return response.text

def main():
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

        # Special commands for adding training data manually:
        if user_input.lower().startswith("add training:"):
            new_data = user_input[len("add training:"):].strip()
            if new_data:
                add_training_data(new_data)
            else:
                print("No training data provided to add.\n")
            continue

        if user_input.lower().startswith("add file:"):
            file_path = user_input[len("add file:"):].strip()
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

        if user_input.lower().startswith("add image:"):
            image_path = user_input[len("add image:"):].strip()
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

        if user_input.lower().startswith("add wikipedia:"):
            topic = user_input[len("add wikipedia:"):].strip()
            if topic:
                summary = fetch_wikipedia_summary(topic)
                add_training_data(summary)
            else:
                print("No topic provided for Wikipedia lookup.\n")
            continue

        # Automatically add related Wikipedia summaries based on query keywords
        auto_add_related_wikipedia(user_input)
        if detect_math_with_gemini(user_input):
            print("\nMath detected by Gemini. Querying Wolfram|Alpha...")
            success, wolfram_output = query_wolfram(user_input)
            if success:
                print("\nWolfram|Alpha Output:")
                print(wolfram_output)
                print("\nSummarizing Wolfram output with Gemini...")
                summary = summarize_with_gemini(wolfram_output)
                print("\nBot:", summary)
            else:
                print("\nWolfram|Alpha returned an error. Falling back to Gemini for an academic answer.")
                gemini_result = direct_gemini(user_input)
                print("\nBot:", gemini_result)
        else:
            result = direct_gemini(user_input)
            print("\nBot:", result)
        print()

if __name__ == '__main__':
    main()
