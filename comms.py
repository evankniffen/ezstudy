import requests
from google import genai

# Replace with your actual credentials
WOLFRAM_APPID = "HPQQ9Y-734KXXQEE3"
GEMINI_API_KEY = "AIzaSyBXLFpQHahdMYY4KGWtSFEmouexhXCUtPc"

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
    and respond with exactly one word: "Yes" if math is involved, or "No" otherwise.
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
    that is suitable for academic learning, ensuring the response is neither too brief nor too verbose.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = (
        "You are an experienced academic tutor. Please summarize the following Wolfram|Alpha output "
        "into a clear, concise, and educational explanation that would help a student understand the result. "
        "Ensure your response covers the essential points without excessive details. Do not over explain,"
        " assume extremely basic knowledge of everything that would be taught in a course before the topic of the problem. "
        " Incluse step-by-steps solutions where necessary. \n\n"
        f"Wolfram|Alpha Output:\n{text}"
    )
    response = client.models.generate_content(
        model="gemini-2.0-pro",
        contents=prompt,
    )
    return response.text

def direct_gemini(query):
    """
    Query Gemini directly with the input query.
    The prompt instructs Gemini to respond as an academic tutor, providing an answer that is clear and informative,
    with sufficient context to aid student learning.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = (
        "You are an academic tutor and subject matter expert. Answer the following question in a clear, "
        "comprehensive, and educational manner that helps a student understand the concept. "
        "Keep the response focused and avoid unnecessary verbosity. Do not reference something not outputted to the"
        "user. Ensure all responses are clear and concise. Incluse step-by-step solutions where necessary. \n\n"
        f"Question: {query}"
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    return response.text

def main():
    print("Academic Chatbot started. Type 'quit' or 'exit' to stop.\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

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

if __name__ == '__main__':
    main()
