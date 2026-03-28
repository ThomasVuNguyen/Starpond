# Goal

Build a proof-of-concept world model that detects biological anomalies in 
microscopic organism imagery — as a stepping stone toward early cancer 
detection.

## The Core Idea

Train a world model exclusively on images of healthy organisms. Measure 
prediction error in latent space. Anomalous organisms produce elevated 
prediction error — flagged as deviation from the healthy baseline.

This is architecturally identical to how we would eventually detect early 
cancer signals in a human biomarker stream. We prove the framework works 
here first, cheaply, before scaling.

## What Success Looks Like

A world model trained only on "healthy" organism images that demonstrably 
separates healthy from anomalous organisms via prediction error alone — 
with no labels used at inference time. A single strong quantitative result 
(e.g. AUROC) that can anchor a grant application.

## Phases

**Phase 1 — Proof of concept (now, $0)**
Use existing public microscopic organism imagery datasets. Pick one organism 
class as "healthy normal", treat everything else as anomaly. Prove the 
world model framework detects the separation.

**Phase 2 — Live organism stream (grant-funded)**
Continuous video of real organisms over their full lifetime. Healthy strain 
vs tumor/stressed strain. World model trained on healthy lifetime 
trajectories detects deviation before it's visually obvious.

**Phase 3 — Human translation (longer term)**
Replace organism video with human biomarker streams. Same architecture, 
different sensor modality.

## Constraints

- Solo researcher, limited compute (Intel i7-8750H, 24GB RAM, GTX 1050 Ti 4GB VRAM)
- Local compute preferred, but cloud credits available on Cloudrift and Prime Intellect if needed
- Frozen pretrained backbone — do not train from scratch
- Keep it simple — this is a proof of concept not a paper implementation

## Starting Points to Investigate

- Public microscopic organism datasets (plankton, zooplankton, c. elegans)
- Frozen ViT or video foundation model as perception backbone
- Latent prediction error as anomaly signal
- AUROC as primary evaluation metric

## The Grant Pitch

We demonstrated that a world model trained on healthy organism imagery 
detects biological anomalies — using only public data and consumer hardware. 
The same framework applied to live organisms with induced tumors is a novel, 
credible, and fundable approach to early cancer detection.