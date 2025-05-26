import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

x = torch.randn(3, 3).to(device)
y = torch.randn(3, 3).to(device)
z = x + y

print(z)
print("Device of z:", z.device)
