import torch

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('GPU is available')
    else:
        device = torch.device('cpu')
        print('GPU not available, using CPU')
        print(f"Using {device} device")
    
    return device