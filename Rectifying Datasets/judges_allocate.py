import os
from dotenv import load_dotenv
import json
import google.generativeai as genai

load_dotenv()
API_KEY = os.environ["GOOGLE_API_KEY"]

# Configure Gemini API
genai.configure(api_key=API_KEY)

# Load judge data
with open("judges_data.json", "r") as file:
    judges_data = json.load(file)

def assign_judge_using_llm(case_data, judges_data):
    """Use Gemini to assign the best judge based on case details."""
    
    # Create a structured prompt for the LLM
    prompt = f"""
    You are an AI trained in legal case assignment.
    Given the following case details, assign the best judge from the list:

    **Case Details:**
    - Type: {case_data["case_type"]}
    - Subtype: {case_data["case_subtype"]}
    - Legal Sections: {", ".join(case_data["legal_sections"])}
    - Seriousness: {case_data["seriousness"]}
    - No. of Parties: {case_data["party_count"]}

    **Judges Available:**
    {json.dumps(judges_data, indent=4)}

    **Instructions:**
    - Select a judge whose specialization **matches** the case type.
    - Prioritize judges with **experience in similar past cases**.
    - Prefer judges with a **lower pending cases**.
    - Output the result in JSON format as follows:
    ```json
    {{
        "assigned_judge": "Judge Name",
        "reason": "Explain why this judge was selected"
    }}
    ```
    """
    
    # Get Gemini response
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)

    response_text = response.text.strip()

    if response_text.startswith("```json"):
        response_text = response_text[7:-3]

    try:
        assigned_judge = json.loads(response_text)
        return assigned_judge
    except json.JSONDecodeError:
        print("Error parsing Gemini response:", response_text)
        return {"assigned_judge": "Unknown", "reason": "Could not determine a judge"}


case_data = {
    "case_type": "Civil",
    "case_subtype": "Tax Refund Suit",
    "party_count": 2,
    "seriousness": "MEDIUM",
    "legal_sections": [
        "Section 3 of the Madhya Bharat Sales Tax Act (Act 30 of 1950)",
        "Article 301 of the Constitution of India"
    ]
}

assigned_judge = assign_judge_using_llm(case_data, judges_data)
print(f"Assigned Judge: {assigned_judge['assigned_judge']}")
print(f"Reason: {assigned_judge['reason']}")
