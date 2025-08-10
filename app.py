import os
from platform import processor
from flask import Flask, jsonify, request, Response
import requests
import json
from datetime import datetime
from flask_cors import CORS
from threading import Thread
from typing import List
from langchain.schema import Document
from pdf_processor import PDFProcessor
from vector_store import VectorStoreManager

app = Flask(__name__)
CORS(app)
conversation_history = []
processor = PDFProcessor()
vector_store = VectorStoreManager(None,None)

def load_and_process_pdfs(pdf_folder_path:str)->List[Document]:
    all_documents = []
    for filename in os.listdir(pdf_folder_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(pdf_folder_path,filename)
            try:
                docs = processor.process_pdf(file_path)
                all_documents.extend(docs)
                print(f"Processed {filename}:extracted {len(docs)} chunks")
            except Exception as e :
                print(f"Failed to process {filename} : {e}")
    print(f"Total chunks extracted from all PDFs : {len(all_documents)}")
    return all_documents

def save_chat(user_message, bot_reply):
    def _save():
        chats = {"user": user_message, "bot": bot_reply, "timestamp": datetime.now().isoformat()}
        with open("chat_history.json", "a") as f:
            f.write(json.dumps(chats) + '\n')

    Thread(target=_save).start()


OLLAMA_URL = "http://localhost:11434/api/chat"
all_documents = load_and_process_pdfs("data/pdfs")
vector_store.build_index(all_documents)


@app.route('/')
def home():
    return jsonify({'message': "Backend is running !"})


@app.route('/chat', methods=['POST'])
def chat():
    try:
        session = requests.Session()
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' in request"}), 400
        text = data.get('text')
        conversation_history.append({"role": "user", "content": text})
        pdf_matches = vector_store.search(text,k=3)
        context = '\n\n'.join([doc["page_content"] for doc in pdf_matches])
        system_message ={
            "role" : "system",
            "content":(
            "You are a helpful assistant. Use only the following context to answer the question."
            "If the answer is not contained here, say you don't know.\n\n"
            f"Context:\n{context}"
            )
        }
        messages = [system_message] + conversation_history

        def generate():
            reply = session.post(
                OLLAMA_URL,
                json={
                    "model": "llama3.1",
                    "messages": messages,
                    "stream": True
                },
                stream=True,
                timeout=60
            )
            bot_reply = ""
            reply.raise_for_status()
            for line in reply.iter_lines():
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    content = chunk.get("message", {}).get("content", "")
                    bot_reply += content
                    yield content
            save_chat(text, bot_reply)
            conversation_history.append({"role": "assistant", "content": bot_reply})
        return Response(generate(), mimetype='text/event-stream')
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Ollama connection failed :{str(e)}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/history', methods=['GET'])
def history():
    chats = []
    try:
        with open("chat_history.json", 'r') as f:
            for line in f:
                chats.append(json.loads(line))
            return jsonify(chats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history", methods=['DELETE'])
def clear_history():
    with open('chat_history.json', 'w') as f:
        f.write('')
    return jsonify({"message": "Chat history cleared."})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
