import argparse
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import requests
import json
from flask import Flask, request, jsonify
import requests
import urllib.parse
import json

def pollination_model(input_text):
    params = {
        "model": "openai",
        "seed": 42,
        "json": "true", # Optional: Get response as JSON string
        "system": "Refer to only the given context and give me the detailed answer in 2000 words. DON'T ANSWER BRIEFLY ANSWER IT, GIVE THE COMPLETE ANSWER IN DETAIL.", # Optional
        # "referrer": "MyPythonApp" # Optional for referrer-based authentication
    }
    encoded_prompt = urllib.parse.quote(input_text)
    encoded_system = urllib.parse.quote(params.get("system", "")) if "system" in params else None

    url = f"https://text.pollinations.ai/{encoded_prompt}"
    query_params = {k: v for k, v in params.items() if k != "system"} # Remove system from query params if present
    if encoded_system:
        query_params["system"] = encoded_system

    try:
        response = requests.get(url, params=query_params)
        print(response.json())
        response.raise_for_status()

        if params.get("json") == "true":
            # The response is a JSON *string*, parse it
            try:
                data = json.loads(response.text)
                print("Response (JSON parsed):", data)
            except json.JSONDecodeError:
                print("Error: API returned invalid JSON string.")
                print("Raw response:", response.text)
        else:
            print("Response (Plain Text):")
            print(response.text)

        return response

    except requests.exceptions.RequestException as e:
        print(f"Error fetching text: {e}")
        # if response is not None: print(response.text)

app = Flask(__name__)

# Load the FAISS vector store and embeddings model when the app starts
print("Loading FAISS vector store and embeddings model...")
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    print("Vector store and embeddings model loaded successfully.")

    # Create a list of all documents from the docstore in order
    all_docs = []
    if db:
        # Reconstruct the ordered list of documents from the docstore
        # This is needed to get the context chunks around a search result
        docstore_ids = [db.index_to_docstore_id[i] for i in range(len(db.index_to_docstore_id))]
        all_docs = [db.docstore._dict[doc_id] for doc_id in docstore_ids]

except Exception as e:
    print(f"Error loading the vector store: {e}")
    print("Please make sure you have run the 'chatbot.ipynb' or 'chatbot.py' script to create the 'faiss_index' directory.")
    db = None
    all_docs = []

@app.route('/query', methods=['POST'])
def query():
    if db is None:
        return jsonify({"error": "Vector store not loaded. Please check the logs."}), 500

    data = request.get_json()
    query_text = data.get('query')

    if not query_text:
        return jsonify({"error": "Query not provided."}), 400

    # Embed the query to get the vector
    query_embedding = embeddings.embed_query(query_text)

    # Perform similarity search on the FAISS index to get the indices of the results
    k = 3
    scores, indices = db.index.search(np.array([query_embedding], dtype=np.float32), k)
    
    result_indices = indices[0]

    if not result_indices.any():
        return jsonify({"answer": "No relevant documents found."})

    # Collect the indices of the context chunks (result + 2 before and 2 after)
    context_indices = set()
    for idx in result_indices:
        start = max(0, idx - 2)
        end = min(len(all_docs), idx + 3)
        for i in range(start, end):
            context_indices.add(i)
    
    # Sort the indices to maintain the order of chunks
    sorted_context_indices = sorted(list(context_indices))
    
    # Retrieve the actual documents for the context
    context_docs = [all_docs[i] for i in sorted_context_indices]

    # Prepare context for the model
    context = ""
    for doc in context_docs:
        context += f"Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}\n"
        context += f"Content: {doc.page_content}\n\n"

    # Call the Mistral model
    prompt = f"Based on the following context, please answer the question.\n\nContext:\n{context}\n\nQuestion: {query_text}"
    
    # response = model_call(prompt)
    response = pollination_model(prompt)

    if response.status_code == 200:
        response_data = response.json()
        try:
            content = response_data["choices"][0]["message"]["content"]
            return jsonify({"answer": content})
        except (KeyError, IndexError):
            return jsonify({"error": "Could not parse the model's response correctly."}), 500
    else:
        return jsonify({"error": f"Request failed with status code {response.status_code}", "details": response.text}), 500

if __name__ == '__main__':
    app.run(port=5000)
