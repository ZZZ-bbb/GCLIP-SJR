# Fixed GCLIP-SJR (560) protocol

1. EXIF-transpose RGB and letterbox to 1120 x 1120 with preserved aspect ratio.
   ImageNet normalization; frozen DINOv2-S/14 final 384-D patch tokens; patch L2,
   float16 cache round trip, float32 bilinear dense interpolation, second L2.
2. Evaluate the bound diagonal K=3 GMM in 64-row chunks and keep all three
   posterior channels. Remove letterbox padding.
3. Halve valid dimensions with Python round, minimum one. RGB uses OpenCV area
   resize, probabilities use PyTorch area resize. Longest working side: 560.
4. Fixed PAMR anchor: ten updates, dilations [1,1,2,4,6,12], repeated offsets
   intentionally retained. RGB weights are softmax over neighbors of the
   channel mean of -abs(difference)/(1e-8 + 0.1*local_std), with RGB / 255.
5. Resize original valid DINO features directly to half the working dimensions,
   bilinear with align_corners=False, and L2-normalize. Cosine similarities to
   the three normalized frozen GMM means define three semantic channels.
   Normalize means on the inference device to match the research computation.
6. Real DenseCRF messages on a longest-side-280 grid. For working size (H,W) and
   CRF size (h,w), Gaussian sxy=(3*w/W*0.5,3*h/H*0.5), compatibility=3.
   Bilateral features are [x/(80*w/W*0.5),y/(80*h/H*0.5),R/13,G/13,B/13,
   c0/0.1,c1/0.1,c2/0.1], compatibility=10. Both kernels use DIAG_KERNEL and
   NORMALIZE_SYMMETRIC.
7. Initialize Q=A (the fixed PAMR anchor). Five joint updates with damping 0.5:

   Q_next = 0.5*Q + 0.5*softmax(log(max(A,1e-5)) + 0.25*global(Q) + local(Q)).

   global(Q): area-resize Q, perform one zero-unary DenseCRF step, clamp at 1e-20,
   take log, bilinear-resize back. local(Q): propagation using fixed RGB weights.
   Temperature and local weight are 1. No scene gates or morphology are used.
8. Bilinear-resize the final three-class posterior to the valid 1120 grid before
   argmax. Nearest-resize labels to native dimensions, then use the bound
   foreground component set (component 2 only for this exact released model).

This is a heuristic fixed-iteration message coupling, not an exact energy solver
or a proven convergent optimization algorithm. CLIP operates offline on source
images and is absent at target inference. No target masks are read by prediction.

RiceP/Paddy images: native 512 x 512, encoder 1120 x 1120, SJR 560 x 560,
CRF 280 x 280. For a portrait image of width x height 1080 x 1920: valid encoder
630 x 1120, SJR 315 x 560, CRF 158 x 280. Landscape dimensions reverse.

Area fraction is measured on the final native binary mask. Recorded runtime
includes decoding and image writes, excludes model initialization, and should
not be equated with a warm-up-controlled latency benchmark.
