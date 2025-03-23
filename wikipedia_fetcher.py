import wikipedia

def fetch_wikipedia_summary(topic):
    """
    Fetch a summary of the given topic from Wikipedia.
    Returns a summary string, or an error message if fetching fails.
    """
    try:
        summary = wikipedia.summary(topic, sentences=5)
        return summary
    except Exception as e:
        return f"Error fetching Wikipedia article for '{topic}': {e}"

