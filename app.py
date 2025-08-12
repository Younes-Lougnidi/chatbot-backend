from flask import Flask, jsonify, request, Response
import requests
import json
from datetime import datetime
from flask_cors import CORS
from threading import Thread
from pdf_processor import PDFProcessor
from vector_store import VectorStoreManager


app = Flask(__name__)
CORS(app)
conversation_history = []
stop_flags = {}
processor = PDFProcessor()
vector_store = VectorStoreManager(None,None)
vector_store.load_index()

def save_chat(user_message, bot_reply):
    def _save():
        chats = {"user": user_message, "bot": bot_reply, "timestamp": datetime.now().isoformat()}
        with open("chat_history.json", "a") as f:
            f.write(json.dumps(chats) + '\n')

    Thread(target=_save).start()


OLLAMA_URL = "http://localhost:11434/api/chat"



@app.route('/')
def home():
    return jsonify({'message': "Backend is running !"})

@app.route('/stop',methods= ['POST'])
def stop_generation():
    session_id = request.json.get("session_id")
    stop_flags[session_id] = True
    return jsonify({"status": "stopping"})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        session = requests.Session()
        data = request.get_json()
        session_id = data.get("session_id","default")
        stop_flags[session_id] = False
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' in request"}), 400
        text = data.get('text')
        conversation_history.append({"role": "user", "content": text})
        pdf_matches = vector_store.search(text,k=4)
        context = '\n\n'.join([doc["page_content"] for doc in pdf_matches])
        system_message ={
            "role" : "system",
            "content":(
                f"You are an expert assistant. Use the information from the documents to answer the question.\n First, identify relevant facts.\n Then, reason step-by-step.\nFinally, provide a clear, concise answer.\nFor all math formulas use LATEX.\nFor latex expressions always write them in one single line all of them using $$ even for expressions for matrices and alignment....\nIf the documents don’t contain enough info, you can answer from you're knowledge(don't say that you don't know).\nIf the user wants to hold a normal conversation you can do so.\ncontext:{context}\nquestion:{text}"

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
                if stop_flags.get(session_id):
                    break
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
