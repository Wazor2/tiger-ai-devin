# Dataset Research Notes — Pench Tiger Camera Trap System

## Module 1: Blank Image Filtering — Caltech Camera Traps (CCT)
- Source: https://beerys.github.io/CaltechCameraTraps/ ; hosted at lila.science/datasets/caltech-camera-traps
- 243,187 images, 140 camera locations, binary animal/no-animal labels, COCO-format annotations
- Perfect for training a blank-image classifier (Stage A two-stage cascade). ECCV18 paper "Recognition in Terra Incognita"
- iWildCam 2018 challenge binary classification metric
- Data hosted on LILA BC (also in the SpeciesNet ecosystem)

## Module 2: Tiger Re-ID — ATRW (Amur Tiger Re-identification in the Wild)
- Source: https://lila.science/datasets/atrw/ (arXiv 1906.05586, ACM MM 2020)
- 8,000+ video clips, 92 individual Amur tigers, ~9,500 bounding boxes + COCO pose keypoints, ~3,600 with individual ID
- GCP links (smaller, reliable):
  - Re-ID train images (132MB): https://storage.googleapis.com/public-datasets-lila/cvwc2019/train/atrw_reid_train.tar.gz
  - Re-ID train annotations: https://storage.googleapis.com/public-datasets-lila/cvwc2019/train/atrw_anno_reid_train.tar.gz
  - Re-ID test images (90MB): https://storage.googleapis.com/public-datasets-lila/cvwc2019/test/atrw_reid_test.tar.gz
  - Re-ID test annotations: https://storage.googleapis.com/public-datasets-lila/cvwc2019/test/atrw_anno_reid_test.tar.gz
  - Pose train images (255MB): https://storage.googleapis.com/public-datasets-lila/cvwc2019/train/atrw_pose_train.tar.gz
  - Detection train images (2GB) — optional
- License: CC BY-NC-SA 4.0 (images owned by MakerCollider/WWF). Citation: Li et al., arXiv:1906.05586
- Re-ID task at CVWC 2019 challenge: "plain re-ID" (with pose/bbox) and "wild re-ID" (without)
- Pre-cropped flank-style images exist in Re-ID split (cropped tiger patches with pose keypoints for flank side)

## Alternative / complementary
- MegaDetector (Microsoft AI for Earth) pretrained: https://github.com/agentmorris/MegaDetector — can be used as Stage B detector instead of training from scratch
- Snapshot Serengeti: huge, but download is heavier
- CCT download page: https://lila.science/datasets/caltech-camera-traps (COCO annotations, ~243k images)

## Plan
- Module 1: train lightweight MobileNetV3/EfficientNet classifier on subset of CCT (animal/no-animal binary)
- Module 2: train Siamese/triplet metric-learning Re-ID model on ATRW Re-ID + pose subsets (stripe embedding, flank-aware)
- Use ultralytics YOLOv8 for tiger detection (fine-tune on ATRW detection split) or use pretrained MegaDetector
- Modules 3-4: MCP + AKDE home range, rule-engine deviation alerts (geo data simulated in Pench geometry)

## ATRW Re-ID actual structure (verified)
- reid_list_train.csv: [tiger_id_int, filename] pairs — 1,887 rows, 107 unique tiger IDs (note: ATRW docs say 92, but this split has 107 labeled ids... actually official number of labeled identities; test set has unlabeled images)
- reid_keypoints_train.json: {filename: [x1,y1,v1,...,x15,y15,v15]} — 15 COCO keypoints per image, many fully zero (visible flag v=0) — keypoints sparse
- reid_list_test.csv: 1,764 unlabeled filenames (wild re-ID test)
- reid_keypoints_test.json: keypoints dict
- Images are in atrw/train and atrw/test directories (5,156 jpgs)

=> Strategy: train metric-learning Re-ID (Siamese/triplet) on 1,887 labeled train images across 107 IDs; test set serves as probe gallery. Images are already tiger crops/patches.
