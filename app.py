from flask import Flask, jsonify, request
import requests
import json
from datetime import datetime

app = Flask(__name__)


def save_chat(user_message, bot_reply):
    chats = {"user": user_message, "bot": bot_reply,"timestamp":datetime.now().isoformat()}
    with open("chat_history.json", "a") as f:
        f.write(json.dumps(chats) + '\n')


OLLAMA_URL = "http://localhost:11434/api/chat"


@app.route('/')
def home():
    return jsonify({'message': "Backend is running !"})


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error" : "Missing 'text' in request"}),400
        text = data.get('text')
        reply = requests.post(
            OLLAMA_URL,
            json={
                "model": "llama3.1",
                "messages": [{
                    "role": "user",
                    "content": text,
                }],
                "stream": False
            },
            timeout=30
        )
        reply.raise_for_status()
        bot_reply = reply.json()["message"]["content"]
        save_chat(text, bot_reply)
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
    app.run(debug=True)
