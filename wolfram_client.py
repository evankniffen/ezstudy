import requests

def wolfram():
    return_str = ""
    appid = "HPQQ9Y-734KXXQEE3"
    base_url = "https://www.wolframalpha.com/api/v1/llm-api"
    
    math_problem = input("Enter your math problem: ").strip()
    params = {
        "appid": appid,
        "input": math_problem,
        # "maxchars": "6800"
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        return_str = response.text
    else:
        return_str = response.status_code
        return_str = response.text
    return return_str

if __name__ == '__main__':
    wolfram()
