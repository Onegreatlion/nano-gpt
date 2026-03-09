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

prompt = 'Once upon a time'
tokens = enc.encode_ordinary(prompt)
x = torch.tensor([tokens], dtype=torch.long, device=device)

max_new_tokens = 100
temperature = 0.8

print(f'\nPrompt: {prompt}')
print('Generating text...\n')

with torch.no_grad():
    for _ in range(max_new_tokens):
        logits, _ = model(x)
       
        logits = logits[:, -1, :] / temperature
        
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        next_token = torch.multinomial(probs, num_samples=1)
        # Append the new word to the running sequence
        x = torch.cat((x, next_token), dim=1)


generated_text = enc.decode(x[0].tolist())
print(generated_text)
