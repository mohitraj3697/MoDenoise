import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from flask import Flask, render_template_string, request, Response, jsonify
from transformers import PreTrainedTokenizerFast, GPT2TokenizerFast

app = Flask(__name__)

@dataclass
class DiffusionLMConfig:
    vocab_size: int
    seq_len: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    dropout: float
    diffusion_steps: int

class DiffusionTransformerLM(nn.Module):
    def __init__(self, cfg: DiffusionLMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.time_emb = nn.Embedding(cfg.diffusion_steps + 1, cfg.d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, input_ids, timesteps, attention_mask=None):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        t_emb = self.time_emb(timesteps).unsqueeze(1)
        x = x + t_emb
        x = self.drop(x)
        src_key_padding_mask = ~attention_mask if attention_mask is not None else None
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.ln_f(x)
        return self.lm_head(x)

MODEL = None
TOKENIZER = None
CFG = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(checkpoint_dir="checkpoints/finetune_best"):
    global MODEL, TOKENIZER, CFG
    if not os.path.exists(checkpoint_dir):
        checkpoint_dir = "checkpoints/pretrain" # fallback
        if not os.path.exists(checkpoint_dir):
            raise RuntimeError(f"No checkpoint found at {checkpoint_dir}")
            
    with open(os.path.join(checkpoint_dir, "config.json"), "r") as f:
        CFG = DiffusionLMConfig(**json.load(f))
        
    MODEL = DiffusionTransformerLM(CFG).to(DEVICE)
    MODEL.load_state_dict(torch.load(os.path.join(checkpoint_dir, "model.pt"), weights_only=True, map_location=DEVICE))
    MODEL.eval()
    
    tokenizer_dir = os.path.join(checkpoint_dir, "tokenizer")
    TOKENIZER = GPT2TokenizerFast.from_pretrained(tokenizer_dir)

def chat_prompt(user_msg: str) -> str:
    return f"<|user|>\n{user_msg}\n<|assistant|>\n"

@torch.no_grad()
def diffusion_generate_stream(prompt_text: str, max_new_tokens=128, temperature=1.0, top_k=0):
    global MODEL, TOKENIZER, CFG, DEVICE
    
    prompt_ids = TOKENIZER.encode(prompt_text, add_special_tokens=True)
    prompt_ids = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    Lp = prompt_ids.size(1)
    L = min(CFG.seq_len, Lp + max_new_tokens)
    gen_len = L - Lp

    x = torch.full((1, L), TOKENIZER.mask_token_id, dtype=torch.long, device=DEVICE)
    x[:, :Lp] = prompt_ids[:, :Lp]
    fixed = torch.zeros((1, L), dtype=torch.bool, device=DEVICE)
    fixed[:, :Lp] = True
    attention_mask = torch.ones((1, L), dtype=torch.bool, device=DEVICE)

    def sample_from_logits(logits):
        if temperature != 1.0: logits = logits / temperature
        if top_k > 0:
            topk_vals, topk_idx = torch.topk(logits, k=top_k, dim=-1)
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(-1, topk_idx, topk_vals)
            logits = filtered
        probs = F.softmax(logits, dim=-1)
        flat = probs.view(-1, probs.size(-1))
        sampled = torch.multinomial(flat, num_samples=1).view(1, L)
        sampled_prob = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        return sampled, sampled_prob

    for s in range(CFG.diffusion_steps, 0, -1):
        t = torch.tensor([s], device=DEVICE, dtype=torch.long)
        logits = MODEL(x, timesteps=t, attention_mask=attention_mask)
        sampled, conf = sample_from_logits(logits)

        update_pos = ~fixed
        x[update_pos] = sampled[update_pos]

        next_ratio = float(s - 1) / float(CFG.diffusion_steps)
        target_masks = int(math.ceil(gen_len * next_ratio))
        gen_positions = torch.arange(L, device=DEVICE) >= Lp
        candidates = gen_positions & (~fixed[0])
        cand_idx = torch.where(candidates)[0]

        if target_masks > 0 and cand_idx.numel() > 0:
            cand_conf = conf[0, cand_idx]
            k = min(target_masks, cand_idx.numel())
            _, low_idx = torch.topk(cand_conf, k=k, largest=False)
            remask_positions = cand_idx[low_idx]
            x[0, remask_positions] = TOKENIZER.mask_token_id

        decoded = TOKENIZER.decode(x[0].tolist())
        # Strip system/user tags for cleaner output
        display = decoded.replace(prompt_text, "").replace("[MASK]", "█")
        yield f"data: {json.dumps({'step': CFG.diffusion_steps - s + 1, 'total': CFG.diffusion_steps, 'text': display})}\n\n"

    final = TOKENIZER.decode(x[0].tolist()).replace(prompt_text, "").replace("<|end|>", "").strip()
    yield f"data: {json.dumps({'step': CFG.diffusion_steps, 'total': CFG.diffusion_steps, 'text': final, 'done': True})}\n\n"


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MoDenoise</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #000000;
            --bg-light: #111111;
            --glass-bg: rgba(255, 255, 255, 0.02);
            --glass-border: rgba(255, 255, 255, 0.15);
            --primary: #ffffff;
            --primary-glow: rgba(255, 255, 255, 0.4);
            --text-main: #ffffff;
            --text-dim: #888888;
        }

        * { box-sizing: border-box; }

        body {
            margin: 0;
            padding: 0;
            min-height: 100vh;
            background: var(--bg-dark);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        /* Subtle static noise overlay */
        body::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background-image: url('data:image/svg+xml,%3Csvg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg"%3E%3Cfilter id="noiseFilter"%3E%3CfeTurbulence type="fractalNoise" baseFrequency="0.65" numOctaves="3" stitchTiles="stitch"/%3E%3C/filter%3E%3Crect width="100%25" height="100%25" filter="url(%23noiseFilter)" opacity="0.05"/%3E%3C/svg%3E');
            pointer-events: none;
            z-index: 0;
        }

        .app-container {
            position: relative;
            z-index: 1;
            width: 95vw;
            max-width: 1400px;
            height: 90vh;
            display: flex;
            gap: 24px;
            padding: 24px;
            background: var(--bg-light);
            border: 1px solid var(--glass-border);
            border-radius: 0; /* Sharp corners for monochrome theme */
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8);
        }

        .main-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: var(--bg-dark);
            border: 1px solid var(--glass-border);
            overflow: hidden;
        }

        .header {
            padding: 20px 24px;
            border-bottom: 1px solid var(--glass-border);
            display: flex;
            align-items: center;
            gap: 12px;
            background: var(--bg-light);
        }

        .header h1 {
            margin: 0;
            font-size: 1.25rem;
            font-weight: 600;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: var(--primary);
        }

        .pulse-dot {
            width: 8px;
            height: 8px;
            background: var(--primary);
            box-shadow: 0 0 10px var(--primary);
            animation: blink 2s infinite;
        }

        @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.2; } }

        .chat-container {
            flex: 1;
            padding: 24px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 20px;
            scroll-behavior: smooth;
            background: var(--bg-dark);
        }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.3); }
        ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.6); }

        .message {
            max-width: 85%;
            padding: 16px 20px;
            font-family: 'Fira Code', monospace;
            font-size: 0.95rem;
            line-height: 1.6;
            word-wrap: break-word;
            animation: slideUp 0.3s ease-out forwards;
            opacity: 0;
            transform: translateY(10px);
            border: 1px solid var(--glass-border);
        }

        @keyframes slideUp {
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            align-self: flex-end;
            background: var(--primary);
            color: var(--bg-dark);
            border-color: var(--primary);
        }

        .message.bot {
            align-self: flex-start;
            background: var(--bg-light);
            color: var(--text-main);
        }

        .input-panel {
            padding: 20px;
            border-top: 1px solid var(--glass-border);
            background: var(--bg-light);
        }

        .input-wrapper {
            display: flex;
            gap: 12px;
            background: var(--bg-dark);
            border: 1px solid var(--glass-border);
            padding: 8px;
            transition: all 0.2s ease;
        }

        .input-wrapper:focus-within {
            border-color: var(--primary);
        }

        input[type="text"] {
            flex: 1;
            background: transparent;
            border: none;
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
            font-size: 1rem;
            padding: 12px;
            outline: none;
        }

        input[type="text"]::placeholder { color: var(--text-dim); }

        button.send-btn {
            background: var(--primary);
            color: var(--bg-dark);
            border: none;
            padding: 0 24px;
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        button.send-btn:hover:not(:disabled) {
            background: #e0e0e0;
        }

        button.send-btn:disabled {
            background: #333333;
            color: #666666;
            cursor: not-allowed;
        }

        .side-panel {
            width: 320px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .settings-card {
            background: var(--bg-dark);
            border: 1px solid var(--glass-border);
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 24px;
        }

        .settings-card h3 {
            margin: 0;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: var(--text-dim);
            display: flex;
            align-items: center;
            gap: 8px;
            border-bottom: 1px solid var(--glass-border);
            padding-bottom: 12px;
        }

        .control-group {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .control-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
            color: var(--text-dim);
        }

        .control-val {
            color: var(--primary);
            font-family: 'Fira Code', monospace;
        }

        input[type="range"] {
            -webkit-appearance: none;
            width: 100%;
            height: 2px;
            background: var(--glass-border);
            outline: none;
        }

        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 12px;
            height: 12px;
            border-radius: 0;
            background: var(--primary);
            cursor: pointer;
            transition: transform 0.1s;
        }

        input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.4);
        }

        .progress-card {
            background: var(--bg-dark);
            border: 1px solid var(--glass-border);
            padding: 24px;
            display: none;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

        .progress-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            color: var(--text-dim);
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .progress-track {
            height: 4px;
            background: var(--glass-border);
            position: relative;
        }

        .progress-fill {
            height: 100%;
            width: 0%;
            background: var(--primary);
            transition: width 0.1s linear;
        }
        
        .bot-cursor {
            display: inline-block;
            width: 8px;
            height: 15px;
            background-color: var(--primary);
            margin-left: 2px;
            vertical-align: middle;
            animation: blink 1s step-end infinite;
        }
    </style>
</head>
<body>
    <div class="app-container">
        <div class="main-panel">
            <div class="header">
                <div class="pulse-dot"></div>
                <h1>MoDenoise</h1>
            </div>
            
            <div class="chat-container" id="chat">
                <div class="message bot">System active. Ready for input.</div>
            </div>
            
            <div class="input-panel">
                <div class="input-wrapper">
                    <input type="text" id="prompt" placeholder="Type a message..." autocomplete="off" onkeypress="if(event.key === 'Enter') generate()">
                    <button id="sendBtn" class="send-btn" onclick="generate()">SEND</button>
                </div>
            </div>
        </div>
        
        <div class="side-panel">
            <div class="settings-card">
                <h3>Parameters</h3>
                
                <div class="control-group">
                    <div class="control-header">
                        <span>Temperature</span>
                        <span class="control-val" id="tempVal">1.0</span>
                    </div>
                    <input type="range" id="temp" min="0.1" max="2.0" step="0.1" value="1.0" oninput="document.getElementById('tempVal').innerText=this.value">
                </div>
                
                <div class="control-group">
                    <div class="control-header">
                        <span>Top-K Sampling</span>
                        <span class="control-val" id="topkVal">50</span>
                    </div>
                    <input type="range" id="topk" min="0" max="200" step="10" value="50" oninput="document.getElementById('topkVal').innerText=this.value">
                </div>
                
                <div class="control-group">
                    <div class="control-header">
                        <span>Max Tokens</span>
                        <span class="control-val" id="maxTokensVal">128</span>
                    </div>
                    <input type="range" id="maxTokens" min="32" max="256" step="16" value="128" oninput="document.getElementById('maxTokensVal').innerText=this.value">
                </div>
            </div>

            <div class="progress-card" id="progressWrap">
                <div class="progress-header">
                    <span>Denoising Step</span>
                    <span id="progressText">0 / 128</span>
                </div>
                <div class="progress-track">
                    <div class="progress-fill" id="progressBar"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function generate() {
            const promptInput = document.getElementById('prompt');
            const sendBtn = document.getElementById('sendBtn');
            const chat = document.getElementById('chat');
            const progressWrap = document.getElementById('progressWrap');
            const progressBar = document.getElementById('progressBar');
            const progressText = document.getElementById('progressText');
            
            const prompt = promptInput.value.trim();
            if (!prompt) return;
            
            // Add user message
            const userMsg = document.createElement('div');
            userMsg.className = 'message user';
            userMsg.innerText = prompt;
            chat.appendChild(userMsg);
            
            // Add bot message placeholder
            const botMsg = document.createElement('div');
            botMsg.className = 'message bot';
            chat.appendChild(botMsg);
            
            chat.scrollTop = chat.scrollHeight;
            
            // Lock UI
            promptInput.value = '';
            promptInput.disabled = true;
            sendBtn.disabled = true;
            progressWrap.style.display = 'block';
            progressBar.style.width = '0%';
            
            const temp = document.getElementById('temp').value;
            const topk = document.getElementById('topk').value;
            const maxTokens = document.getElementById('maxTokens').value;
            
            const params = new URLSearchParams({ prompt, temp, topk, maxTokens });
            const eventSource = new EventSource('/generate_stream?' + params.toString());
            
            eventSource.onmessage = function(e) {
                const data = JSON.parse(e.data);
                
                // Keep the raw text plus a blinking cursor effect during generation
                botMsg.innerText = data.text;
                if (!data.done) {
                    const cursor = document.createElement('span');
                    cursor.className = 'bot-cursor';
                    botMsg.appendChild(cursor);
                }
                
                const pct = (data.step / data.total) * 100;
                progressBar.style.width = pct + '%';
                progressText.innerText = `${data.step} / ${data.total}`;
                
                chat.scrollTop = chat.scrollHeight;
                
                if (data.done) {
                    eventSource.close();
                    promptInput.disabled = false;
                    sendBtn.disabled = false;
                    setTimeout(() => { progressWrap.style.display = 'none'; }, 1000);
                    promptInput.focus();
                }
            };
            
            eventSource.onerror = function() {
                eventSource.close();
                botMsg.innerText = "[Connection Error]";
                botMsg.style.color = "#ffffff";
                promptInput.disabled = false;
                sendBtn.disabled = false;
            };
        }
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/generate_stream")
def generate_stream():
    prompt = request.args.get("prompt", "Write a short story.")
    temp = float(request.args.get("temp", 1.0))
    topk = int(request.args.get("topk", 50))
    max_tokens = int(request.args.get("maxTokens", 128))
    
    formatted_prompt = chat_prompt(prompt)
    return Response(diffusion_generate_stream(formatted_prompt, max_tokens, temp, topk), mimetype="text/event-stream")

if __name__ == "__main__":
    print("Loading model...")
    try:
        load_model()
        print(f"Model loaded successfully. Running on {DEVICE}.")
        app.run(host="0.0.0.0", port=5000, debug=False)
    except Exception as e:
        print(f"Failed to start server: {e}")
