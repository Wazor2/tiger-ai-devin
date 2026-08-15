# Status notes 2

Reid: epoch2 val rank1=0.5169 (stable vs 0.5228 at epoch1). Training continuing, 18 epochs, val every 3.

Blank v2: epoch 6 f1=0.30 fE=0.17. Oscillating badly. Saved checkpoint on disk = best epoch so far (epoch 1: f1 0.91 fE 0.0067). NOTE: epoch 2/3 reported f1=1.0 with acc near 0 — that indicates ALL val predictions fell in review band (lo=0.15, hi=0.85). The f1=1.0 at epoch2/3 was from tp=0, precision 0/0 → code gave 1.0? compute_metrics f1 = 2*prec*rec/(prec+rec); prec=0 rec=0 → 0.0. But log shows f1=1.0. Because precision=0/0 → 0.0; rec=0.0; f1=0. Hmm actually f1=1.0000 logged at epoch 2 & 3! That can only happen if tp>0, fp=0, fn=0: i.e., some animal images predicted 'animal' confidently while NO animal image predicted 'empty' (fE=0.0) and NO empty image predicted 'animal'... but acc=0.0083. If pred==labels mean=0.008, most predictions are 'review' (2); acc counts only exact matches so review predictions lower acc. TP+TN small, f1 could be 1 with tiny tp. So epoch 2/3 = mostly-review regime (fE=0 safe but useless). Epoch 1 = best usable checkpoint (f1 0.91, fE 0.007, recall_animal 0.88).

Decision: accept epoch-1 checkpoint. After run ends, if final calibrated metrics beat v1 (fE<=48.3%), fine; else document recommendation.
