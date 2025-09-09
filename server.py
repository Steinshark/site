from flask import Flask, Response, request, stream_with_context, jsonify
from flask_cors import CORS
import sys 
import os 
import torch 
sys.path.append("C:/gitrepos/nlp/")
sys.path.append("C:/gitrepos/cloudGPT/")
import time 
from training import PATH,MODELS,PREV_RUNS
import json
import datetime 
import threading
from zoneinfo import ZoneInfo # For Python 3.9+
from data import load_tokenizer 
from model import LMSteinshark


RESPONSES_FILE = "C:/data/cloudGPT/finetune/crowdsource_choices.jsonl"


#Load the tokenizer
CUR_MODEL       = "C:/data/nlp/models/finetune0"
CUR_TOK         = "C:/gitrepos/cloudgpt/tokenizer"
TOP_P           = .9
TEMP            = .75
N_TOK           = 512


MODEL           = LMSteinshark.from_loadpoint(CUR_MODEL,p_override=0).bfloat16().cuda().eval()
TOKENIZER       = load_tokenizer(CUR_TOK)
app = Flask(__name__)
CORS(app)  # Enable CORS for all domains on all routes (adjust for production)

USERS_FILE = "users.txt"
TIMEZONE    = ZoneInfo("Pacific/Honolulu")


#Allows for continuous reloading of the most recent model
# def load_model():
#     MODEL           = LMSteinshark.from_loadpoint(CUR_MODEL,p_override=0).bfloat16().cuda().eval()
#     TOKENIZER       = load_tokenizer(CUR_TOK)

# def start_load_cycle():
#     thread          = threading.Thread(target=load_model)
#     thread.daemon   = True 
#     thread.start()


def check_credentials(username, password):
    try:
        with open(USERS_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                user, pwd = line.split(":", 1)
                if user == username and pwd == password:
                    return True
        return False
    except FileNotFoundError:
        # No users file yet
        return False

@app.route('/api/chat/stream', methods=['POST'])
def stream_chat():
    print(f"user connected")
    def generate():
        print(f"generating")
        prompt = request.json.get("prompt", "")
        print(f"promt: {prompt}")
        for token in MODEL.token_streamer(prompt,TOKENIZER,256,.75,None,.9,'p'):  # Replace with your generator
            yield f"data: {token}\n\n"
            time.sleep(0.05)  # Simulate streaming delay
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route("/api/chat", methods=["POST"])
def serve_chat_request():

    def generate():
        print(f"generating")
        prompt = request.json.get("prompt", "")
        for token in MODEL.token_streamer(prompt,TOKENIZER,256,.75,None,.9,'p'):  # Replace with your generator
            yield f"data: {token}\n\n"
            time.sleep(0.05)  # Simulate streaming delay

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

    print(f"Recieved chat request")
    data = request.json
    request_type = data.get("request_type","")

    if request_type == 'chat':
        print(f"serving chat")
        prompt = data.get("prompt", "")
        conversation_id = data.get("conversation_id")
        request_ip = request.remote_addr


        #Get model output 
        with torch.no_grad():
            prompt_as_tok   = tokenizer.encode(prompt).ids
            output          = model.generate(prompt_as_tok,tokenizer,256,.7,top_k=1000)

        # You would replace this with actual model logic or socket call
        return_text = tokenizer.decode(output)

        return jsonify({
            "mode": "return",
            "return_text": return_text,
            "conversation_id": conversation_id
        })
    # elif request_type == 'stats':
    #     print(f"serving stats")
    #     return model_stats()
    
    # else:
    #     print(f"serving unknown")


@app.route('/api/stats', methods=['POST'])
def model_stats():
    data = request.json
    print(f"sending stats")

    #get stats 
    stats   = MODEL.stats
    toks    = stats['tok_through']
    

    if toks > 1_000_000_000:
        toks    = f"{toks / 1_000_000_000:.3f}"
        toks    = f"{toks}B"
    else:
        toks    = f"{toks / 1_000_000:.3f}"
        toks    = f"{toks}M"

    return jsonify({
        "param_count": f"{MODEL.n_params//1_000_000}M",
        "layer_count": MODEL.n_layers,
        "embed_size": MODEL.n_embed,
        "vocab_size": 32768,
        "phase": "Fine Tuning",
        "loss": f"{sum(stats['losses'][-100:])/100:.4f}",
        "dtype":"bfloat_16",
        "tokens_trained": toks,
        "last_update": datetime.datetime.fromtimestamp(stats['time_snap'],tz=TIMEZONE)
    })

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"success": False, "error": "Missing username or password"}), 400

    if check_credentials(username, password):
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "error": "Invalid credentials"}), 401

@app.route("/submit_choice", methods=["POST"])
def submit_choice():
    """
    Accept JSON with the following format:
    {
        "prompt": "The text prompt",
        "choice": "response1" | "response2" | "both" | "neither",
        "user_id": "optional_user_identifier"
    }
    """
    print(f"received json data")
    try:
        data = request.get_json()
        if not data or "prompt" not in data or "choice" not in data:
            return jsonify({"status": "error", "message": "Invalid payload"}), 400

        # Add timestamp for logging
        data["timestamp"] = datetime.utcnow().isoformat()

        # Append JSON line to file
        with open(RESPONSES_FILE, "a") as f:
            contents    = f.read()
            if not contents:
                contents = []
            else:
                contents    = json.loads(contents)
            
            contents.append(json.dumps(data))
            f.write(contents)

        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    #start_load_cycle()
    app.run(host="0.0.0.0",debug=True, port=6969)
