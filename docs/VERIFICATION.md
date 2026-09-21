# Verification scope

The SJR implementation was checked in Python 3.11, PyTorch 2.5.0+cu124, NumPy
1.26.4 and OpenCV 4.11.0 on an RTX 3090, using the existing native pydensecrf build.
On 28 real images (RiceP 3, CVRP 22, Paddy 3), SJR masks matched the archived
full-cohort SJR (560) masks pixel for pixel. The sample covers every CVRP native
image size in the evaluated cohort. Raw-GCLIP masks also matched exactly.
Component-to-foreground mapping and area fractions were checked against masks.

The preceding core passed 22 CPU protocol tests. This repository deliberately
omits research experiment runners, scoring scripts, test images and raw logs.
Those previous checks do not establish a fresh GPU installation, full-source
refit, CPU/GPU bitwise portability or new external validation. The documented
pydensecrf build route needs the upstream C++ build prerequisites.

The original SJR (560) research run evaluated all 2,735 images. The portable
release check above uses 28 images; it is not a new full-cohort experiment.
Target-mask training is absent, but labeled development screening informed
configuration selection. Small panicles and domain shifts remain limitations.
