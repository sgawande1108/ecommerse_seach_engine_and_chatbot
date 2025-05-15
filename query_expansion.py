import google.generativeai as genai
import re, json

genai.configure(api_key="enter key")

llmmodel = genai.GenerativeModel("gemini-1.5-flash")

def extract_terms(context_json):
    terms = []
    def recurse(obj):
        if isinstance(obj, dict):
            for v in obj.values(): recurse(v)
        elif isinstance(obj, list):
            for item in obj: recurse(item)
        elif isinstance(obj, str):
            terms.append(obj.lower())
    recurse(context_json)
    return " ".join(terms)

def expand_query_with_llm(user_input):
    prompt = f"""
                Analyze this Amazon/Flipkart product search query: "{user_input}"

                Generate a search-optimized JSON structure containing:

                {{
                "core_terms": [
                    /* 1. Primary product type + key modifiers (<=5 terms)
                    Format: [Product, Key adjective, Key feature]
                    Example: ["bluetooth earphones", "sweatproof", "bass boost"] */
                ],
                
                "technical_specs": [
                    /* 2. Technical parameters and measurable attributes
                    - Electronics: ["driver size", "frequency response", "IPX7 rating"]
                    - Fashion: ["polyester blend", "crew neck", "preshrunk fabric"]
                    - Home: ["cordless design", "stainless steel", "2L capacity"] */
                ],
                
                "use_cases": [
                    /* 3. At least 3 usage scenarios from query context
                    Example: ["gym workouts", "running", "commuting"] */
                ],
                
                "semantic_links": [
                    /* 4. Synonyms + related product categories (mix of):
                    - Direct alternatives: ["wireless earbuds", "neckband headphones"]
                    - Hypernyms: ["audio accessories", "wearable tech"]
                    - Complementary items: ["earphone case", "sports armband"] */
                ],
                
                "commerce_signals": [
                    /* 5. E-commerce-specific search boosters:
                    - Price brackets: ["under 2000", "mid-range", "premium"]
                    - Brand variants: ["Boat", "Noise", "Sony XB series"]
                    - Platform terms: ["Amazon Prime", "Flipkart Supercoin"]
                    - Temporal terms: ["2024 model", "Diwali sale", "summer collection"] */
                ]
                }}

                Guidelines:
                - Prioritize terms appearing in Amazon/Flipkart product titles
                - Include common misspellings ("earphone" + "earphones")
                - Convert implied needs to explicit specs ("long battery" → "20hr playback")
                - Maintain 8-12 terms total per category
                - Use lowercase except brand names
                - Avoid marketing jargon ("best in class")
                """
    #response = llmmodel.generate_content(prompt.replace("{user_input}", user_input))
    response = llmmodel.generate_content(prompt)

    raw = response.candidates[0].content.parts[0].text
    cleaned = re.sub(r"```json|```", "", raw).strip()
    structured = json.loads(cleaned)
    return extract_terms(structured)
