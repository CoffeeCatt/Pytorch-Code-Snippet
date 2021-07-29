device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(1)
print("Using device:", device)
