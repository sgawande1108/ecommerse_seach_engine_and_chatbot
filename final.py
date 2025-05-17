import streamlit as st
import pandas as pd
import numpy as np
import re
import random
import requests
from sentence_transformers import SentenceTransformer
import faiss

# --- Load and Prepare Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("flipkart_laptop_dataset.csv")

    if "battery_life" not in df.columns:
        battery_life_options = ["4 hours", "5 hours", "6 hours", "7 hours", "8 hours", "10 hours", "12 hours"]
        df["battery_life"] = [random.choice(battery_life_options) for _ in range(len(df))]
        df.to_csv("flipkart_laptop_dataset.csv", index=False)

    def create_desc(row):
        return (
            f"Product: {row['product_name']}, Rating: {row['average_rating']}, Price: ₹{row['selling_price']}, "
            f"Processor: {row['processor']}, RAM: {row['ram']}, Storage: {row['SSD']}, "
            f"Display: {row['display_size']}, OS: {row['operating_system']}, Battery: {row['battery_life']}"
        )

    df["desc"] = df.apply(create_desc, axis=1)
    return df

df = load_data()

# --- Embed Products ---
@st.cache_resource
def get_model_and_index(df):
    model = SentenceTransformer('paraphrase-albert-small-v2')
    embeddings = model.encode(df["desc"].tolist(), show_progress_bar=False).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return model, index

model, index = get_model_and_index(df)


# --- Search Logic ---
def extract_price_limit(query):
    match = re.search(r"under\s*₹?\s*(\d+)", query)
    if match:
        return int(match.group(1))
    return None

def search_products(query, top_k=5):
    price_limit = extract_price_limit(query)

    filtered_df = df.copy()
    if price_limit:
        filtered_df["clean_price"] = (
            filtered_df["selling_price"]
            .astype(str)
            .str.replace(r"[^\d.]", "", regex=True)
            .replace("", np.nan)
            .astype(float)
        )
        filtered_df = filtered_df[filtered_df["clean_price"] <= price_limit]

    if filtered_df.empty:
        return []

    query_vector = model.encode([query]).astype("float32")
    filtered_embeddings = model.encode(filtered_df["desc"].tolist()).astype("float32")

    temp_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
    temp_index.add(filtered_embeddings)

    distances, indices = temp_index.search(query_vector, min(top_k, len(filtered_df)))
    filtered_data = filtered_df.to_dict(orient="records")
    return [filtered_data[i] for i in indices[0]]


# --- Generate LLM Recommendation ---
def generate_response(query, retrieved_products):
    groq_api_key = st.secrets["GROQ_API_KEY"]  # Make sure you set this in .streamlit/secrets.toml
    model_name = "llama3-8b-8192"

    if not retrieved_products:
        return "❌ Sorry, no matching laptops found for your query."

    product_lines = []
    for i, p in enumerate(retrieved_products, 1):
        summary = (
            f"A budget-friendly laptop with a {p['display_size']} display, "
            f"{p['processor']} processor, {p['ram']} RAM, {p['battery_life']} battery."
        )
        product_lines.append(
            f"{i}\nProduct Name: {p['product_name']}\n"
            f"Price: ₹{p['selling_price']}\n"
            f"Rating: {p['average_rating']}\n"
            f"Summary: {summary}\n"
        )

    context = "\n".join(product_lines)

    prompt = f"""
🔍 Top Products:

Here are the top {len(retrieved_products)} relevant products that match the user's query:

{context}

Based only on these results, recommend the most suitable laptop. Do not use any external knowledge.
"""

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful shopping assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"❌ API Error {response.status_code}: {response.text}"


# --- Streamlit UI ---
st.title("💻 AI Laptop Recommender")
st.markdown("Ask for laptops like a human (e.g. _Best laptops under ₹60000 with SSD and i5 processor_)")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("🔎 Your Query", placeholder="e.g., Best laptops under ₹70000 with SSD")

if st.button("Search") and query:
    results = search_products(query)
    st.session_state["last_results"] = results  # Save results for follow-up questions
    if not results:
        st.warning("❌ No laptops found matching your query.")
    else:
        st.success(f"🎯 Found {len(results)} matching laptops:")
        for idx, laptop in enumerate(results, 1):
            with st.container():
                st.markdown(f"**🔹 Recommendation #{idx}**")
                st.markdown(f"**📦 {laptop['product_name']}**")
                st.markdown(f"💰 **Price:** ₹{laptop['selling_price']}")
                st.markdown(f"⭐ **Rating:** {laptop['average_rating']}")
                st.markdown(f"🧠 **Processor:** {laptop['processor']}")
                st.markdown(f"💾 **Storage:** {laptop['SSD']}, {laptop['ram']} RAM")
                st.markdown(f"📺 **Display:** {laptop['display_size']}")
                st.markdown(f"🔋 **Battery:** {laptop['battery_life']}")
                st.markdown(f"🖥 **OS:** {laptop['operating_system']}")
                st.markdown("---")

        if st.button("🤖 Ask AI for the best option"):
            with st.spinner("Thinking..."):
                response = generate_response(query, results)
            st.markdown("### 🤖 LLM Recommendation")
            st.info(response)

# --- Follow-up Question Section ---
if st.session_state.get("last_results"):
    follow_up = st.text_input("Ask a follow-up question 🧠", key="follow_up_q")

    if st.button("💬 Ask Follow-up"):
        if not follow_up.strip():
            st.warning("Please enter your follow-up question.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": follow_up})
            with st.spinner("Generating follow-up answer..."):
                # Use last results for follow-up context
                response = generate_response(follow_up, st.session_state["last_results"])
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.success("Follow-up answer generated!")

# --- Chat History Display ---
if st.session_state.chat_history:
    st.subheader("🗂️ Chat History")
    for msg in st.session_state.chat_history:
        role = "🧑‍💻 You" if msg["role"] == "user" else "🤖 Bot"
        st.markdown(f"**{role}:** {msg['content']}")
