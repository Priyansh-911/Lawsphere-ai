import os
import json
import random
import asyncio
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

from typing import Dict

# Load environment variables and configure Gemini
load_dotenv()
API_KEY = os.environ["GOOGLE_API_KEY"]
genai.configure(api_key=API_KEY)

# Load judge data
with open("judges_data.json", "r") as f:
    judges_data = json.load(f)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-2.0-flash')


def assign_judge_using_llm(case_data: Dict, judges_data: list) -> Dict:
    """Assign best judge based on LLM logic."""
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
    - Output the result in JSON format:
    {{
        "assigned_judge": "Judge Name",
        "reason": "Explain why this judge was selected"
    }}
    """

    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()

        if response_text.startswith("```json"):
            response_text = response_text[7:-3]

        return json.loads(response_text)
    except Exception as e:
        print("LLM Assignment Error:", e)
        return {
            "assigned_judge": "Unknown",
            "reason": "Error in assignment"
        }


def assign_random_judge(judges_data):
    return random.choice(judges_data)["name"]


async def enrich_dataset_with_judges(df):
    enriched_rows = []

    for _, row in df.iterrows():
        case_data = {
            "case_type": row["Type of Case"],
            "case_subtype": row["Subtype of Case"],
            "party_count": row["No. of Parties"],
            "seriousness": row["Seriousness"],
            "legal_sections": row["Sections"].split(', ') if pd.notna(row["Sections"]) else []
        }

        ai_result = assign_judge_using_llm(case_data, judges_data)
        random_judge = assign_random_judge(judges_data)

        enriched_rows.append({
            "Title": row["Title"],
            "Date": row["Date"],
            "Judgment": row["Judgment"],
            "Type of Case": case_data["case_type"],
            "Subtype of Case": case_data["case_subtype"],
            "No. of Parties": case_data["party_count"],
            "Seriousness": case_data["seriousness"],
            "Sections": ', '.join(case_data["legal_sections"]),
            "Random Judge": random_judge,
            "AI Assigned Judge": ai_result["assigned_judge"],
            "Assignment Reason": ai_result["reason"]
        })

        await asyncio.sleep(10)  # for rate limiting (Gemini allows 2 RPM)

    return pd.DataFrame(enriched_rows)


async def main():
    input_file = "Enriched_IndianKanoon_Cases.csv"  # From the imputer
    df = pd.read_csv(input_file)

    # Optional: Filter out already processed ones if rerunning
    # df = df[df["Type of Case"].notna()]

    final_df = await enrich_dataset_with_judges(df)

    # Save final dataset
    final_df.to_csv("Final_Enriched_Dataset.csv", index=False)
    print("✅ Final dataset saved as Final_Enriched_Dataset.csv")


if __name__ == "__main__":
    asyncio.run(main())
