# import os
# import pandas as pd
# import numpy as np
# import google.generativeai as genai
# import time
# from typing import List, Dict
# import json

# class LegalCaseImputer:
#     def __init__(self, api_key: str):
#         """Initialize the imputer with Google API key."""
#         genai.configure(api_key=api_key)
#         # Initialize Gemini-Pro model
#         self.model = genai.GenerativeModel('gemini-1.0-pro')

#         # Define known case types and subtypes for validation
#         self.case_types = [
#             "Criminal", "Civil", "Constitutional", "Administrative",
#             "Family", "Property", "Contract", "Tax", "Labor",
#             "Environmental", "Corporate", "Intellectual Property"
#         ]

#         # Map of common Indian legal sections to case types
#         self.section_patterns = {
#             "IPC": "Criminal",
#             "CrPC": "Criminal",
#             "CPC": "Civil",
#             "Companies Act": "Corporate",
#             "Income Tax Act": "Tax",
#             "Hindu Marriage Act": "Family",
#         }

#     def _create_prompt(self, row: pd.Series) -> str:
#         """Create a prompt for Gemini based on available case information."""
#         prompt = f"""You are a legal expert specialized in Indian law. Analyze this case and provide missing information:

# Title: {row['Title']}
# Date: {row['Date']}
# Judgment Summary: {row['Judgment'][:500]}...

# Provide ONLY a JSON object with these keys (no other text):
# - case_type (Criminal, Civil, Constitutional, etc.)
# - case_subtype (specific area of law)
# - party_count (numeric)
# - seriousness (HIGH/MEDIUM/LOW based on implications)
# - legal_sections (array of sections mentioned)"""

#         return prompt

#     def _parse_gemini_response(self, response: str) -> Dict:
#         """Parse the Gemini response and validate the output."""
#         try:
#             # Clean the response to handle potential formatting issues
#             response_text = response.strip()
#             if response_text.startswith("```json"):
#                 response_text = response_text[7:-3]
#             elif response_text.startswith("{") and response_text.endswith("}"):
#                 response_text = response_text
#             else:
#                 # Extract JSON if it's embedded in other text
#                 import re
#                 json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
#                 if json_match:
#                     response_text = json_match.group()

#             data = json.loads(response_text)

#             # Validate case type
#             if data['case_type'] not in self.case_types:
#                 data['case_type'] = self._find_closest_match(data['case_type'], self.case_types)

#             # Validate party count
#             try:
#                 data['party_count'] = int(data['party_count'])
#             except (ValueError, TypeError):
#                 data['party_count'] = None

#             # Validate seriousness
#             if data['seriousness'].upper() not in ['HIGH', 'MEDIUM', 'LOW']:
#                 data['seriousness'] = 'MEDIUM'

#             # Ensure legal_sections is a list
#             if not isinstance(data['legal_sections'], list):
#                 data['legal_sections'] = [str(data['legal_sections'])]

#             return data

#         except (json.JSONDecodeError, KeyError) as e:
#             print(f"Error parsing Gemini response: {e}")
#             return None

#     def _find_closest_match(self, value: str, valid_values: List[str]) -> str:
#         """Find the closest matching valid value using string similarity."""
#         from difflib import get_close_matches
#         matches = get_close_matches(value, valid_values, n=1, cutoff=0.6)
#         return matches[0] if matches else valid_values[0]

#     def _extract_sections_from_text(self, text: str) -> List[str]:
#         """Extract legal section references from text using regex."""
#         import re

#         patterns = [
#             r"Section\s+\d+(?:\([a-zA-Z]\))?\s+of\s+[A-Za-z\s]+Act",
#             r"S\.\s*\d+(?:\([a-zA-Z]\))?\s+of\s+[A-Za-z\s]+Act",
#             r"IPC\s+\d+",
#             r"CrPC\s+\d+",
#             r"Article\s+\d+(?:\([0-9]\))?"
#         ]

#         sections = []
#         for pattern in patterns:
#             matches = re.finditer(pattern, text, re.IGNORECASE)
#             sections.extend([match.group() for match in matches])

#         return list(set(sections))

#     async def impute_row(self, row: pd.Series) -> Dict:
#         """Process a single row of the dataset using Gemini."""
#         # First try to extract sections directly from text
#         extracted_sections = self._extract_sections_from_text(row['Judgment'])

#         # Generate Gemini prompt and get response
#         prompt = self._create_prompt(row)

#         try:
#             # Generate response from Gemini
#             response = await self.model.generate_content_async(prompt)

#             # Get the text from the response
#             response_text = response.text

#             parsed_data = self._parse_gemini_response(response_text)

#             if parsed_data:
#                 # Combine extracted sections with Gemini-identified sections
#                 if extracted_sections:
#                     parsed_data['legal_sections'] = list(set(
#                         extracted_sections + parsed_data.get('legal_sections', [])
#                     ))
#                 return parsed_data

#         except Exception as e:
#             print(f"Error processing row: {e}")
#             return None

#     async def process_dataset(self, df: pd.DataFrame, batch_size: int = 2) -> pd.DataFrame:
#         """Process the entire dataset with batching and rate limiting."""
#         import asyncio
#         results = []

#         for i in range(0, len(df), batch_size):
#             batch = df.iloc[i:i + batch_size]

#             # Process batch concurrently
#             tasks = [self.impute_row(row) for _, row in batch.iterrows()]
#             batch_results = await asyncio.gather(*tasks)

#             for row, result in zip(batch.itertuples(), batch_results):
#                 if result:
#                     results.append({
#                         'Tile': row[1],  # Assumes title is first column
#                         'Type of Case': result['case_type'],
#                         'Subtype of Case': result['case_subtype'],
#                         'No. of Parties': result['party_count'],
#                         'Seriousness': result['seriousness'],
#                         'Sections': ', '.join(result['legal_sections'])
#                     })

#             # Rate limiting - Gemini has different rate limits than OpenAI
#             await asyncio.sleep(10)

#         return pd.DataFrame(results)


# # Example usage
# async def main():
#     # Load API key from env
#     API_KEY = "AIzaSyDpT7ravwOs5h8nAwK9koTr_gv4M5KsWBs"

#     # Load your dataset
#     df = pd.read_csv('IndianKanoon_Cases_100.csv')

#     imputer = LegalCaseImputer(api_key=API_KEY)

#     # Process the dataset
#     completed_df = await imputer.process_dataset(df)

#     # Merge with original dataset
#     final_df = df.merge(completed_df, on='Title', how='left')

#     # Save results
#     final_df.to_csv('completed_cases.csv', index=False)


# if __name__ == "__main__":
#     import asyncio

#     asyncio.run(main())


import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import google.generativeai as genai
import time
import asyncio
import json
from typing import List, Dict

class LegalCaseImputer:
    def __init__(self, api_key: str):
        """Initialize the imputer with Google API key."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

        # Define known case types
        self.case_types = [
            "Criminal", "Civil", "Constitutional", "Administrative",
            "Family", "Property", "Contract", "Tax", "Labor",
            "Environmental", "Corporate", "Intellectual Property"
        ]

    def _create_prompt(self, row: pd.Series) -> str:
        """Create a better prompt for Gemini that ensures a useful response."""
        judgment_text = row['Judgment'] if pd.notna(row['Judgment']) else "No judgment text available."

        prompt = f"""
        You are a legal expert specializing in Indian law. Analyze this case and fill in missing details.

        If the case does not explicitly mention details, make an educated guess based on common legal knowledge.

        Title: {row['Title']}
        Date: {row['Date']}
        Judgment Summary: {judgment_text}...

        Provide ONLY a JSON object:
        {{
            "case_type": "Criminal/Civil/Constitutional/etc.",
            "case_subtype": "Murder, Contract Dispute, etc.",
            "party_count": <numeric>,
            "seriousness": "HIGH/MEDIUM/LOW",
            "legal_sections": ["Section 302 IPC", "Section 34 CrPC", etc.]
        }}
        If the judgment does not explicitly mention some details such as the case_type or case_subtype use the knowledge based on Indian Law to make it out of the Sections used in the judgement, provide the **most reasonable assumption** based on Indian law.
        """
        return prompt.strip()


    def _parse_gemini_response(self, response_text: str) -> Dict:
        """Parse and validate the JSON response from Gemini."""
        try:
            if not response_text or response_text.strip() == "":
                print("Warning: Empty response from Gemini.")
                return self._default_case_data()

            # Extract JSON if wrapped in markdown ```json ... ```
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]

            response_data = json.loads(response_text)

            return {
                "Type of Case": response_data.get("case_type", "Unknown"),
                "Subtype of Case": response_data.get("case_subtype", "Unknown"),
                "No. of Parties": response_data.get("party_count", 2),
                "Seriousness": response_data.get("seriousness", "Medium"),
                "Sections": response_data.get("legal_sections", [])
            }

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error parsing response: {e}")
            print(f"Raw response: {response_text}")
            return self._default_case_data()

    def _default_case_data(self) -> Dict:
        """Return default values when API response fails or data is missing."""
        return {
            "Type of Case": "Unknown",
            "Subtype of Case": "Unknown",
            "No. of Parties": 2,
            "Seriousness": "Medium",
            "Sections": []
        }

    async def impute_row(self, row: pd.Series) -> Dict:
        """Process a single row of the dataset using Gemini."""
        prompt = self._create_prompt(row)

        try:
            response = self.model.generate_content(prompt)
            
            if not response or not response.candidates:
                print(f"Warning: No response from Gemini for case '{row['Title']}'")
                return self._default_case_data()

            response_text = response.candidates[0].content.parts[0].text.strip()

            # Log raw response for debugging
            print(f"Raw response from Gemini:\n{response_text}\n")

            parsed_data = self._parse_gemini_response(response_text)

            # Ensure "Sections" is always a list
            if not isinstance(parsed_data["Sections"], list):
                parsed_data["Sections"] = [parsed_data["Sections"]]

            return parsed_data

        except Exception as e:
            print(f"Error processing row '{row['Title']}': {e}")
            return self._default_case_data()

    async def process_dataset(self, df: pd.DataFrame, batch_size: int = 2) -> pd.DataFrame:
        """Process dataset asynchronously with batching and error handling."""
        results = []

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            tasks = [self.impute_row(row) for _, row in batch.iterrows()]
            batch_results = await asyncio.gather(*tasks)

            for row, result in zip(batch.itertuples(), batch_results):
                if result:
                    results.append({
                        "Title": row.Title,
                        "Type of Case": result["Type of Case"],
                        "Subtype of Case": result["Subtype of Case"],
                        "No. of Parties": result["No. of Parties"],
                        "Seriousness": result["Seriousness"],
                        "Sections": ', '.join(result["Sections"])  # Convert list to string
                    })

            await asyncio.sleep(10)  # Prevent rate limit issues

        return pd.DataFrame(results)

# Example usage
async def main():
    load_dotenv()
    API_KEY = os.environ["GOOGLE_API_KEY"]

    # Load dataset
    df = pd.read_csv("IndianKanoon_Cases_100.csv")

    imputer = LegalCaseImputer(api_key=API_KEY)

    # Process missing values
    completed_df = await imputer.process_dataset(df)

    # Merge back into original dataset
    final_df = df.merge(completed_df, on="Title", how="left")

    # Save updated dataset
    final_df.to_csv("Updated_IndianKanoon_Cases.csv", index=False)
    print("Updated dataset saved as Updated_IndianKanoon_Cases.csv")

if __name__ == "__main__":
    asyncio.run(main())
