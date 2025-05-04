import os
from dotenv import load_dotenv
import pandas as pd
import google.generativeai as genai
import time
import asyncio
import json
from typing import List, Dict

class LegalCaseImputer:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def _create_prompt(self, row: pd.Series) -> str:
        judgment_text = row['Judgment'] if pd.notna(row['Judgment']) else "No judgment text available."
        prompt = f"""
        You are a legal expert specializing in Indian law. Analyze this case and fill in missing details.

        Title: {row['Title']}
        Date: {row['Date']}
        Judgment Summary: {judgment_text}

        Provide ONLY a JSON object like:
        {{
            "case_type": "Criminal/Civil/Constitutional/etc.",
            "case_subtype": "Murder, Contract Dispute, etc.",
            "party_count": <numeric>,
            "seriousness": "HIGH/MEDIUM/LOW",
            "legal_sections": ["Section 302 IPC", "Section 34 CrPC"]
        }}
        """
        return prompt.strip()

    def _parse_gemini_response(self, response_text: str) -> Dict:
        try:
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]

            response_data = json.loads(response_text)

            return {
                "Type of Case": response_data.get("case_type", "Unknown"),
                "Subtype of Case": response_data.get("case_subtype", "Unknown"),
                "No. of Parties": response_data.get("party_count", 2),
                "Seriousness": response_data.get("seriousness", "Medium"),
                "Sections": ', '.join(response_data.get("legal_sections", []))
            }

        except Exception as e:
            print(f"Error parsing response: {e}\nRaw response: {response_text}")
            return {
                "Type of Case": "Unknown",
                "Subtype of Case": "Unknown",
                "No. of Parties": 2,
                "Seriousness": "Medium",
                "Sections": ""
            }

    async def impute_row(self, row: pd.Series) -> Dict:
        prompt = self._create_prompt(row)
        try:
            response = self.model.generate_content(prompt)
            if not response or not response.candidates:
                return self._default_case_data()

            response_text = response.candidates[0].content.parts[0].text.strip()
            return self._parse_gemini_response(response_text)

        except Exception as e:
            print(f"Error processing row '{row['Title']}': {e}")
            return self._default_case_data()

    def _default_case_data(self) -> Dict:
        return {
            "Type of Case": "Unknown",
            "Subtype of Case": "Unknown",
            "No. of Parties": 2,
            "Seriousness": "Medium",
            "Sections": ""
        }

    async def process_dataset(self, df: pd.DataFrame, batch_size: int = 2) -> pd.DataFrame:
        results = []
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            tasks = [self.impute_row(row) for _, row in batch.iterrows()]
            batch_results = await asyncio.gather(*tasks)

            for row, result in zip(batch.itertuples(), batch_results):
                results.append({
                    "Title": row.Title,
                    "Date": row.Date,
                    "Judgment": row.Judgment,
                    **result
                })

            await asyncio.sleep(10)
        return pd.DataFrame(results)

# ------------------- Main Runner ------------------- #

async def main():
    load_dotenv()
    API_KEY = os.environ["GOOGLE_API_KEY"]

    # Load raw input dataset
    df = pd.read_csv("IndianKanoon_Cases_100.csv")

    imputer = LegalCaseImputer(api_key=API_KEY)

    # Run Gemini to generate enriched dataset
    enriched_df = await imputer.process_dataset(df)

    # Save new full file
    enriched_df.to_csv("Enriched_IndianKanoon_Cases.csv", index=False)
    print("✅ Enriched dataset saved as 'Enriched_IndianKanoon_Cases.csv'")

if __name__ == "__main__":
    asyncio.run(main())
