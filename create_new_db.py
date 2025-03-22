import pandas as pd
import openai
import json
import time
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

INPUT_CSV = "item_db.csv"
OUTPUT_CSV = "item_db_enriched.csv"

PROMPT_TEMPLATE = """You are given the following item:

Name: {item_name}
Description: {description}

Extract the following fields in JSON format (keys should match exactly):

- category: A short label like "Kitchen Appliance", "Furniture", "Electronics", etc.
- intended_use: A short sentence describing what the item is primarily used for.
- indoor_or_outdoor: One of "Indoor", "Outdoor", or "Both".
- portable: "Yes" or "No" depending on if it's generally portable.
- size: One of "Small", "Medium", or "Large" depending on typical size.

Keep values short and relevant. Only return a raw JSON object, no explanations or markdown."""

# original item database
df = pd.read_csv(INPUT_CSV)

# Prepare new columns
new_columns = ['category', 'intended_use', 'indoor_or_outdoor', 'portable', 'size']
for col in new_columns:
    df[col] = ""

# Iterate each row
for idx, row in tqdm(df.iterrows(), total=len(df)):
    item_name = row['item_name']
    description = row['description']

    prompt = PROMPT_TEMPLATE.format(item_name=item_name, description=description)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts structured information from product descriptions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=300
        )

        content = response.choices[0].message.content.strip()

       
        if content.startswith("```json"):
            content = content.strip("```json").strip("```").strip()
        elif content.startswith("```"):
            content = content.strip("```").strip()

        parsed = json.loads(content)

        for col in new_columns:
            df.at[idx, col] = parsed.get(col, "")

    except Exception as e:
        print(f"Error processing '{item_name}': {e}")
        for col in new_columns:
            df.at[idx, col] = ""

    #\Delay to avoid rate limits
    time.sleep(0.3)


df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Enriched item database saved to '{OUTPUT_CSV}'")