"""Probe: does MobileNetV3 small head replacement work correctly?"""
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

m = mobilenet_v3_small(weights=None)
print("classifier:", m.classifier)
m.classifier[3] = nn.Linear(1024, 2)
m.eval()

for label in ["animal", "empty"]:
    d = sorted(Path(f"/home/ubuntu/project/datasets/cct_subset/val/{label}").iterdir())[:3]
    xs = torch.stack([tf(Image.open(f).convert("RGB")) for f in d])
    with torch.no_grad():
        out = m(xs)
        p = torch.softmax(out, 1)
    print(label, "logits mean:", out.mean(0).tolist(), "prob mean:", p.mean(0).tolist())
