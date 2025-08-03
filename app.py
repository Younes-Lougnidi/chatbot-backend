from flask import Flask, jsonify, request, session
import requests
import json
from datetime import datetime
from flask_cors import CORS
from threading import Thread

app = Flask(__name__)
CORS(app)
conversation_history = []


def save_chat(user_message, bot_reply):
    def _save():
       chats = {"user": user_message, "bot": bot_reply,"timestamp":datetime.now().isoformat()}
       with open("chat_history.json", "a") as f:
           f.write(json.dumps(chats) + '\n')
    Thread(target=_save).start()


OLLAMA_URL = "http://localhost:11434/api/chat"


@app.route('/')
def home():
    return jsonify({'message': "Backend is running !"})


@app.route('/chat', methods=['POST'])
def chat():
    try:
        session = requests.Session()
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error" : "Missing 'text' in request"}),400
        text = data.get('text')
        conversation_history.append({"role" : "user","content":text})
        reply = session.post(
            OLLAMA_URL,
            json={
                "model": "llama3.1",
                "messages": conversation_history,
                "stream": True
            },
            stream = True,
            timeout=60
        )
        bot_reply = ""
        reply.raise_for_status()
        for line in reply.iter_lines():
            if line:
                chunk = json.loads(line.decode('utf-8'))
                bot_reply += chunk.get("message",{}).get("content","")
        save_chat(text, bot_reply)
        conversation_history.append({"role" :"assistant","content":bot_reply})
        return jsonify({"reply": bot_reply})
    except requests.exceptions.RequestException as e :
        return jsonify({"error" :f"Ollama connection failed :{str(e)}"}),502
    except Exception as e :
        return jsonify({"error" : str(e)}),500

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

