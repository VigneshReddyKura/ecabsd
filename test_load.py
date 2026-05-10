import torch
from models.ecabsd_model import ECABSDModel

ck = torch.load('checkpoints/best_model.pt', map_location='cpu', weights_only=False)
cfg = ck['config']['model']
print("Checkpoint config:", cfg)

model = ECABSDModel(
    input_dim=cfg['input_dim'],
    hidden_dim=cfg['hidden_dim'],
    num_heads=cfg['num_heads'],
    dropout=0.0,
)

result = model.load_state_dict(ck['model_state_dict'], strict=False)
print("Missing keys:", len(result.missing_keys))
print("Unexpected keys:", len(result.unexpected_keys))
print("Model loaded with strict=False")