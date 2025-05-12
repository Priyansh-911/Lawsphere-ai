import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from flask import Flask, request, jsonify
import fitz
import requests


load_dotenv()
API_KEY = os.environ["GOOGLE_API_KEY"]
app = Flask(__name__)
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")


def extract_text_from_pdf(file_stream):
    # Open PDF in memory (without saving it)
    file_bytes = file_stream.read()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text.strip()


def extract_case_details(case_text: str) -> dict:
    prompt = f"""
You are a legal AI. From the case description below, extract the following:
- case_type
- case_subtype
- legal_sections
- party_count
- title

Output only a JSON like this:
{{
  "case_type": "...",
  "case_subtype": "...",
  "party_count": ...,
  "legal_sections": ["...", "..."],
  "title": "..."
}}

Case Description:
\"\"\"
{case_text}
\"\"\"
"""

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()

        if text.startswith("```json"):
            text = text[7:-3]

        return json.loads(text)
    except Exception as e:
        print(f"[Gemini error] {e}")
        # return {"case_type": "Unknown", "party_count": 2, "legal_sections": []}



def assign_best_judge(case_data, judges_data):
    prompt = f"""
You are a judge allocation AI. Assign the most suitable judge from this list for the given case.

**Case Data**:
Type: {case_data.get("case_type")}
Subtype: {case_data.get("case_subtype")}
Sections: {', '.join(case_data.get("legal_sections", []))}
Parties: {case_data.get("party_count")}

**Judges**:
{json.dumps(judges_data, indent=2)}

Output JSON:
{{
  "assigned_judge": "Justice XYZ",
  "reason": "..."
}}
"""

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:-3]
        return json.loads(text)
    except Exception as e:
        print(f"[Judge allocation error] {e}")
        return {"assigned_judge": "Unknown", "reason": "Error in allocation"}

@app.route("/")
def index():
    return jsonify({"message": "Welcome to the Judge Allocation API!"})


@app.route("/judge_allocation", methods=["GET", "POST"])
def judge_allocation():
    # data = json.loads(request.data) 
    if request.method == "GET":
        return jsonify({"message": "Will allocate a judge for you!"})
    
    if request.method == "POST":
        pdf_file = request.files.get("case_file")
        url = "https://lawsphere-back.onrender.com/lawsphere-api/v1/admin_settings/judge/?page=1&page_size=10"
        header = {
            'Authorization' : 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzUyMTU5NDE1LCJpYXQiOjE3NDY5NzU0MTUsImp0aSI6ImZjN2VhZjg1ZGRhNDQwNTFhZmRkZDZkNzNiNTJjNWVhIiwidXNlcl9pZCI6MX0.81tBCw9r4uLiCbVf7sv30rA0LLKuZva3o4w1bf7e-Cs'
        }

        res = requests.get(url, headers=header)
        res = res.json()
        judges = res.get("results", [])
        processed_judges = []
        for judge in judges:
            judge_info = {
                "id": judge["id"],
                "name": judge["name"],
                "email": judge["email"],
                "gender": judge["gender"],
                "specialization": judge["specialization"],
                "pending_cases": judge["pending_cases"],
                "court_name": judge["court_data"]["name"],
                "experience_years": 2025 - int(judge["date_of_appointment"][:4]),
                "past_cases": judge["past_cases"]
            }
        processed_judges.append(judge_info)

        # judges_data_raw = request.form.get("judges_data")

        if not pdf_file:
            return jsonify({"error": "Missing 'case_file"}), 400

        case_text = extract_text_from_pdf(pdf_file)
        case_data = extract_case_details(case_text)
        allocation_result = assign_best_judge(case_data, processed_judges)

        allocation_result["case_data"] = case_data
        return jsonify(allocation_result)



if __name__ == "__main__":
    app.run(debug=True)