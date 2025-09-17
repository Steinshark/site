import torch
import sys
sys.path.append("C:/gitrepos")
try:
    from ..cloudGPT.model import LMSteinshark
    from ..cloudGPT.data import load_tokenizer, RESERVE_1
except ImportError:
    from cloudGPT.model import LMSteinshark
    from cloudGPT.data import load_tokenizer, RESERVE_1
import os 

if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    print(f"init api")
    # === CONFIG ===
    MODEL_PATH              = "//Steinpc/S/nlp/models/FineTune"
    TOKENIZER_PATH          = "//Steinpc/s/nlp/tokenizer"
    BUCKETS                 = [128, 256, 2048]

    # === LOAD MODEL & TOKENIZER ===
    tokenizer   = load_tokenizer(TOKENIZER_PATH)
    device      = 'cuda'
    device      = 'cpu'
    dtype       = torch.bfloat16
    dtype       = torch.float
    model       = LMSteinshark.from_loadpoint(MODEL_PATH,p_override=0,device=torch.device(device))
    model.device= torch.device(device)
    model.send_to_cpu()
    # Move to GPU, BF16, eval, no grad
    model = model.eval().to(device, dtype=dtype)
    for p in model.parameters():
        p.requires_grad_(False)

    torch.cuda.empty_cache()
    # Compile model
    #model = torch.compile(model,backend='eager')

    # Warmup compile graphs for each bucket
#     with torch.inference_mode():
#         for L in BUCKETS:
#             dummy = torch.randint(0, tokenizer.get_vocab_size(), (1, L), device="cuda")
#             dummy_mask = torch.ones_like(dummy).bool()

#             _ = model(dummy,dummy,dummy_mask)



# === INFERENCE WRAPPER ===
def generate_tokens(
    prompt: str,
    max_tok,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    verbose: bool = False
):
    inputs = tokenizer.encode(prompt).ids

    # if inputs.ndim == 1:
    #     inputs = inputs.unsqueeze(0)

    #Add 
    

    with torch.inference_mode():
        for token in model.token_streamer(
            inputs,
            tokenizer=tokenizer,
            n_tokens=max_tok,
            temperature=temperature,
            topk=top_k,
            topp=top_p,
            mode='p',
            verbose=verbose,
            tokenized=True,
            device=device
            ):
            yield token
