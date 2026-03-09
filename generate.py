import torch
import tiktoken
from model import GPT, GPTConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'
enc = tiktoken.get_encoding('gpt2')

print('Waking up the neural network...')
checkpoint = torch.load('ckpt_final.pt', map_location=device)
config = GPTConfig()
model = GPT(config)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()
print('Neural network is ready!')

prompt = input("Enter a prompt (or press Enter for default): ").strip()
if not prompt:
    prompt = "Once upon a time"

tokens = enc.encode(prompt)
x = torch.tensor([tokens], dtype=torch.long, device=device)

max_new_tokens = int(input("How many tokens to generate? (default 500): ").strip() or "500")

print(f"\nGenerating {max_new_tokens} tokens from prompt: '{prompt}'\n")
print("=" * 50)

with torch.no_grad():
    for _ in range(max_new_tokens):
        logits, _ = model(x)
        logits = logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        # Append the new word to the running sequence
        x = torch.cat((x, next_token), dim=1)
        # Crop to block_size so we don't exceed the positional embedding table
        if x.size(1) > config.block_size:
            x = x[:, -config.block_size:]

generated_tokens = x[0].tolist()
generated_text = enc.decode(generated_tokens)
print(generated_text)
print("=" * 50)
