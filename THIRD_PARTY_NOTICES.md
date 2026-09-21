# Provenance and third-party terms

- DINOv2: [Facebook Research official source](https://github.com/facebookresearch/dinov2),
  Apache-2.0 source license. The downloader retains LICENSE. All registered research
  Python files/hubconf.py match revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8`;
  hashes are in `configs/dinov2_source.json`. The historical Torch Hub cache did
  not retain a git commit: this records verified source parity, not a recovered log.
- CLIP: [OpenAI official code](https://github.com/openai/CLIP) and official
  `openai/clip-vit-base-patch32`, revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.
  Downloads use recorded SHA256 values. Upstream model/code terms apply; no large
  weights are vendored or relicensed.
- NumPy, SciPy, OpenCV, Pillow, PyTorch, torchvision, scikit-learn, timm and
  transformers remain separately licensed installed dependencies.
- Controlled clustering/fitting derive from the preceding GCLIP research export;
  features/semantic routines from its semantic_assignment.py. This is a portability/test refactor, not a new invention claim.


No blanket project license is inferred from storage visibility. Code, GMM,
prompts, synthetic/real images and third-party figures have separate rights.
Public redistribution requires corresponding authorization.

- PAMR: `third_party/pamr.py` is the unchanged research-used upstream TU Darmstadt
  (2020) implementation from https://github.com/visinf/1-stage-wseg/blob/master/models/mods/pamr.py.
  Original copyright is retained; Apache-2.0 license text is in
  `third_party/PAMR_LICENSE.txt`. The integration is in `sjr.py`.
- pydensecrf is installed separately from https://github.com/lucasb-eyer/pydensecrf
  at pinned revision 2723c7fa4f2ead16ae1ce3d8afe977724bb8f87f. The wrapper and
  bundled DenseCRF/Eigen dependencies retain their upstream terms. No native
  binaries are redistributed in this repository.
