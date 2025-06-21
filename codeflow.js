document.addEventListener("DOMContentLoaded", () => {
    const codePanel = document.querySelector(".code-stream");
    if (!codePanel) return;
  
    const codeLines = [
        "# Fine-tune transformer with custom LR scheduler",
        "class Transformer(nn.Module):",
        "    def __init__(self, layers, d_model):",
        "        self.attn = MultiHeadAttention(d_model)",
        "        self.norm = LayerNorm(d_model)",
        "        self.ff = FeedForward(d_model)",
        "    def forward(self, x):",
        "        return self.norm(self.ff(self.attn(x)))",
        "optimizer = torch.optim.AdamW(params, lr=2e-4)",
        "scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=10000)",
        "outputs = model(input_ids)",
        "loss = loss_fn(outputs, targets)",
        "loss.backward(); optimizer.step()",
        "embedding = nn.Embedding(vocab_size, embedding_dim)",
        "pos_embedding = nn.Parameter(torch.zeros(1, max_len, embedding_dim))",
        "x = embedding(input_ids) + pos_embedding[:, :input_ids.size(1), :]",
        "attention_mask = (input_ids != pad_token_id).unsqueeze(1).unsqueeze(2)",
        "for epoch in range(num_epochs):",
        "    model.train()",
        "    optimizer.zero_grad()",
        "    logits = model(inputs)",
        "    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))",
        "    loss.backward()",
        "    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)",
        "    optimizer.step()",
    ];

    let lastTop = null; // persist between calls
  
    function tokenizeLine(line) {
        const tokens = [];
        const patterns = [
            { regex: /#.*$/, type: "comment" },
            { regex: /\b(def|class|return|if|for|in|with|super|self|import|from|as|else|torch|nn|not|and|or|is|lambda|yield|try|except|finally|raise|print|for|range|while|break|continue)\b/, type: "keyword" },
            { regex: /\b\d+(\.\d+)?\b/, type: "number" },
            { regex: /(['"])(?:(?=(\\?))\2.)*?\1/, type: "string" },
            { regex: /\b(\w+)(?=\()/, type: "function" }
        ];
    
        let pos = 0;
        while (pos < line.length) {
            let matched = false;
            for (const { regex, type } of patterns) {
                regex.lastIndex = pos;
                const result = regex.exec(line);
                if (result && result.index === pos) {
                    for (const char of result[0]) {
                        tokens.push([char, type]);
                    }
                    pos += result[0].length;
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                tokens.push([line[pos], null]);
                pos++;
            }
        }
        return tokens;
    }
  
    function typeLine(rawText) {
        const div = document.createElement("div");
        div.className = "code-line";
    
        const panelWidth = codePanel.clientWidth;
        const panelHeight = codePanel.clientHeight;
    
        let top;
        let attempts = 0;
        const minSpacing = 20;

        do {
            top = Math.random() * (codePanel.clientHeight - 20);
            attempts++;
        } while (lastTop !== null && Math.abs(top - lastTop) < minSpacing && attempts < 10);

        lastTop = top;

        const left = panelWidth / 3 - 50 + Math.random() * 100;
    
        div.style.position = "absolute";
        div.style.top = `${top}px`;
        div.style.left = `${left}px`;
        div.style.opacity = "1";
        div.style.transition = "opacity 1.5s ease, transform 1.5s ease";
    
        codePanel.appendChild(div);
    
        const tokens = tokenizeLine(rawText);
        let i = 0;
    
        function typeChar() { /* Defined inside of another function because it's a helper function */
            if (i < tokens.length) {
                const [char, type] = tokens[i++];
                const span = document.createElement("span");
                span.textContent = char;
                if (type) span.classList.add("token", type);
                div.appendChild(span);
                setTimeout(typeChar, 200);
            } else {
                // Typing complete — start fade out and move up
                setTimeout(() => {
                    div.style.opacity = "0";
                    div.style.transform = "translateY(-10px)";
                }, 100); // Wait 1s fully visible after typing
        
                // Remove after fade out transition duration
                setTimeout(() => {
                    div.remove();
                    // After removal, try to spawn next line from queue
                    spawnNextLine();
                }, 2500);
            }
        }
    
        typeChar();
    }
  
    // Queue to hold pending lines
    const lineQueue = [];
  
    // Spawn a line if less than 2 are visible
    function spawnNextLine() {
        const currentLines = codePanel.querySelectorAll(".code-line").length;
        if (currentLines < 2 && lineQueue.length > 0) {
            const nextLine = lineQueue.shift();
            typeLine(nextLine);
        }
    }
  
    // Push a new line every 4.5 seconds into the queue and try spawning
    // Immediately queue and show a first line without waiting
    const firstLine = codeLines[Math.floor(Math.random() * codeLines.length)];
    lineQueue.push(firstLine);
    spawnNextLine();

    // Then continue as normal every 3.5 seconds
    setInterval(() => {
        const line = codeLines[Math.floor(Math.random() * codeLines.length)];
        lineQueue.push(line);
        spawnNextLine();
    }, 4500);

  
    // Also check frequently if we can spawn new lines from queue (in case lines got removed)
    setInterval(() => {
      spawnNextLine();
    }, 1000);
  
  });
  