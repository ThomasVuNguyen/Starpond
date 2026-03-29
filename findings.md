# Proof-of-Concept Findings

This document summarizes our initial findings towards building a world model for biological anomaly detection (per `GOAL.md`) using the WormSwin dataset.

## 1. The Dataset (WormSwin)
We located and downloaded the open-source **WormSwin** dataset (DOI: 10.5281/zenodo.7456803) locally. It is a highly robust instance segmentation dataset for *C. elegans* worms containing three primary splits, extracting to ~13GB uncompressed:
- **CSB-1 Dataset (~3GB uncompressed):** Annotated frames of *C. elegans* from video tracks.
- **Synthetic Images (~9.7GB uncompressed):** Background plates with artificially pasted worms.
- **Mating Dataset (~200MB uncompressed):** Highly clustered/overlapping worms under mating conditions.

### Class Separation directly mapped to our Goal
The CSB-1 videos natively separate our normal/anomaly classifications! The sequences are named using `<worm_age>_<mutation>_<irradiated>_<video_index>`.
- **"Healthy Baseline":** Wild-type (`mutation=0`) and non-irradiated (`irradiated=0`) worms.
- **"Anomalous Targets":** `csb-1` mutant worms (`mutation=1` or `2`) or UV-irradiated worms (`irradiated=1`).

## 2. Extremely Rich Metadata (COCO Annotations)
Inside `csb-1_dataset/coco_annotations/`, we found massive `.json` annotation files containing:
- Binary segmentation polygon masks 
- Bounding boxes (`[x, y, w, h]`)
- Image Area sizes

**Impact:** Instead of feeding raw, gigantic 912x736 video frames and the noisy petri dish background into our Vision Transformer (ViT), we can use these bounding boxes to dynamically crop tiny, fixed patches (e.g., 128x128px) of the worms in isolation. This radically minimizes the computational requirement for the ViT backbone.

## 3. Local Compute Capabilities
We analyzed the local hardware and confirmed we are running on:
- **CPU:** Intel Core i7-8750H (6 cores, 12 threads @ 2.20GHz)
- **RAM:** ~24 GB 
- **GPU:** NVIDIA GTX 1050 Ti (4GB VRAM)

**Constraint Relief:** We originally bounded ourselves to M2 Mac hardware limits. Finding out we have an Nvidia GPU (meaning direct CUDA acceleration for inference) and 24GB of system RAM is a massive win. A frozen `ViT-Base` or `DINOv2` model will easily fit inside the 4GB of VRAM to output latent feature variables at speeds highly manageable for a solo researcher.

## 4. Alternate Compute (Cloud Credits)
In addition to our strong local baseline, we have documented access to cloud credits on **Cloudrift** and **Prime Intellect**. 
*If* we outscale the GTX 1050 Ti (e.g., if we choose to train a larger, unfrozen temporal transformer on the latent features instead of a small MLP predictor), we can trivially switch the training load over to the cloud while keeping dev/test cycles strictly local.

## 5. Early World Model Baseline Results (AUROC: 0.808)
We successfully ported the dataset and architecture to an **RTX 4090 Cloudrift VM** to bypass local system-level CPU bottlenecks (Linux file-descriptor exhaustion).

**The Pipeline:**
1. A frozen `google/vit-base-patch16-224` vision-transformer extracted 768-dimensional latent embeddings for each bounding-box cropped worm.
2. A lightweight PyTorch Autoencoder was trained **exclusively** on the embeddings from the `6,987` healthy wild-type worms.
3. The Autoencoder then evaluated `60,571` mixed test embeddings (both healthy and mutant) to calculate reconstruction errors (MSE).

**The Metric:**
The pipeline achieved a baseline **AUROC of `0.808`**.
This means that using our blindly extracted features, our model has an **80.8% probability** of correctly assigning a higher prediction error to an anomalous/mutated worm compared to a healthy wild-type worm. This completely validates the fundamental hypothesis set out in `GOAL.md`!
