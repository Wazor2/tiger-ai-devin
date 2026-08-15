"""Diagnose why val_acc stays at 50%: compute train accuracy and per-class
confusion on val, plus check a few raw predictions."""
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

MODELS = Path("/home/ubuntu/project/models")
CCT = Path("/home/ubuntu/project/datasets/cct_subset")

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

ckpt = torch.load(MODELS / "blank_filter.pth", map_location="cpu", weights_only=False)
print("ckpt keys:", list(ckpt.keys()))
print("val_accuracy in ckpt:", ckpt.get("val_accuracy"))

from torchvision.models import mobilenet_v3_small as _m
from torchvision.models import MobileNet_V3_Small_Weights
import torch.nn as nn
model = _m(weights=MobileNet_V3_Small_Weights.DEFAULT)
model.classifier[3] = nn.Linear(1024, 2)
model.load_state_dict(ckpt["model_state"])
model.eval()

def eval_split(split):
    counts = {"animal": {"tp": 0, "fp": 0}, "empty": {"tn": 0, "fn": 0}}
    for label in ["animal", "empty"]:
        d = sorted((CCT / split / label).iterdir())
        for f in d[:100]:
            x = tf(Image.open(f).convert("RGB")).unsqueeze(0)
            p = torch.softmax(model(x), 1)[0]
            pred = "animal" if p[1] > 0.5 else "empty"
            if label == "animal":
                if pred == "animal":
                    counts["animal"]["tp"] += 1
                else:
                    counts["animal"]["fp"] += 1
            else:
                if pred == "empty":
                    counts["empty"]["tn"] += 1
                else:
                    counts["empty"]["fn"] += 1
    a = counts["animal"]
    e = counts["empty"]
    print(split, f"animal: tp={a['tp']} fp={a['fp']} | empty: tn={e['tn']} fn={e['fn']}")
    acc = (a["tp"] + e["tn"]) / 200
    print(split, "acc =", round(acc, 3))
    # confidence stats
    conf_animal, conf_empty = [], []
    for label in ["animal", "empty"]:
        d = sorted((CCT / split / label).iterdir())[:100]
        for f in d:
            x = tf(Image.open(f).convert("RGB")).unsqueeze(0)
            p = torch.softmax(model(x), 1)[0, 1].item()
            (conf_animal if label == "animal" else conf_empty).append(p)
    import statistics
    print(split, "P(animal) mean: animal-label=", round(statistics.mean(conf_animal), 3),
          "empty-label=", round(statistics.mean(conf_empty), 3))

eval_split("train")
eval_split("val")
