"""Frozen GCLIP-SJR (560): PAMR anchor and DINO-guided DenseCRF messages.

No fitting, target masks, scene gates, or substitute CRF backends. The update
is a heuristic message coupling; exact energy minimization is not claimed.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from third_party.pamr import PAMR


PROTOCOL = dict(
    name="GCLIP-SJR (560)", version=1, input_side=1120, refinement_side=560,
    crf_side=280, pamr_iterations=10, dilations=[1, 1, 2, 4, 6, 12],
    rgb_temperature=0.1, joint_iterations=5, damping=0.5,
    global_weight=0.25, local_weight=1.0, anchor_temperature=1.0,
    gaussian_sxy=3.0, gaussian_compat=3, bilateral_sxy=80.0,
    bilateral_srgb=13.0, bilateral_compat=10, semantic_scale=0.1,
    spatial_scale_reference=0.5, unary_floor=1e-5, message_floor=1e-20,
    output="bilinear posterior to valid 1120 grid, argmax, nearest native labels",
)


class SemanticJointRefiner(PAMR):
    """The validated three-component configuration, with no tunable CLI defaults."""

    def __init__(self, means, device):
        try:
            import pydensecrf.densecrf as dcrf
        except ImportError as exc:
            raise ImportError(
                "SJR requires the real pydensecrf backend. Install requirements-sjr.txt "
                "as described in README.md, or explicitly select --refinement none."
            ) from exc
        super().__init__(10, [1, 1, 2, 4, 6, 12])
        means = np.asarray(means)
        if means.shape != (3, 384) or not np.isfinite(means).all():
            raise ValueError("SJR (560) requires three finite 384-D GMM means")
        self.dcrf = dcrf
        self.register_buffer("centers", F.normalize(torch.as_tensor(
            means, dtype=torch.float32, device=device), dim=1))
        self.eval().requires_grad_(False).to(device)

    @torch.inference_mode()
    def forward(self, dense, posterior, boxed, pad):
        """Return native-resolution component labels, preserving all three classes.

        posterior: float32 NumPy [1120,1120,3]; dense: [1,384,1120,1120].
        pad is the tuple returned by features.letterbox, including native size.
        """
        top, left, nh, nw, h, w = pad
        if posterior.shape != (1120, 1120, 3):
            raise ValueError("Expected the three-class 1120-square GMM posterior")
        if tuple(dense.shape) != (1, 384, 1120, 1120):
            raise ValueError("Expected the original dense DINO feature grid")
        device = dense.device
        lh, lw = max(1, round(nh * 0.5)), max(1, round(nw * 0.5))
        rgb_np = boxed[top:top + nh, left:left + nw].copy()
        rgb_small = np.ascontiguousarray(cv2.resize(
            rgb_np, (lw, lh), interpolation=cv2.INTER_AREA))
        rgb = torch.from_numpy(rgb_small).permute(2, 0, 1)[None].float().to(device) / 255
        pp = torch.from_numpy(posterior[top:top + nh, left:left + nw].copy())
        pp = pp.permute(2, 0, 1)[None].to(device)
        down = F.interpolate(pp, size=(lh, lw), mode="area")
        weights = (-self.aff_x(rgb) / (1e-8 + 0.1 * self.aff_std(rgb)))
        weights = weights.mean(1, keepdim=True).softmax(2)
        anchor = down
        for _ in range(10):
            anchor = (self.aff_m(anchor) * weights).sum(2)

        # Directly resize the original valid DINO grid, not an RGB-grid descriptor.
        shape = (max(1, round(lh / 2)), max(1, round(lw / 2)))
        f = F.normalize(F.interpolate(dense[:, :, top:top + nh, left:left + nw],
                        size=shape, mode="bilinear", align_corners=False), dim=1)
        proto = torch.einsum("kc,bchw->bkhw", self.centers, f)
        scale = min(1.0, 280 / max(lh, lw))
        ch, cw = max(1, round(lh * scale)), max(1, round(lw * scale))
        small = np.ascontiguousarray(cv2.resize(
            rgb_small, (cw, ch), interpolation=cv2.INTER_AREA))
        crf = self.dcrf.DenseCRF2D(cw, ch, 3)
        crf.addPairwiseGaussian(
            sxy=(3 * cw / lw * 0.5, 3 * ch / lh * 0.5), compat=3,
            kernel=self.dcrf.DIAG_KERNEL, normalization=self.dcrf.NORMALIZE_SYMMETRIC)
        sem = F.interpolate(proto, size=(ch, cw), mode="bilinear",
                            align_corners=False)[0].cpu().numpy()
        yy, xx = np.mgrid[:ch, :cw].astype(np.float32)
        pairwise = np.concatenate([
            xx[None] / (80 * cw / lw * 0.5), yy[None] / (80 * ch / lh * 0.5),
            small.transpose(2, 0, 1).astype(np.float32) / 13, sem / 0.1], axis=0)
        crf.addPairwiseEnergy(np.ascontiguousarray(pairwise.reshape(8, -1)),
                             compat=10, kernel=self.dcrf.DIAG_KERNEL,
                             normalization=self.dcrf.NORMALIZE_SYMMETRIC)
        crf.setUnaryEnergy(np.zeros((3, ch * cw), dtype=np.float32))
        q_crf, tmp1, tmp2 = crf.startInference()
        q = anchor.clone()
        unary_logits = anchor.clamp_min(1e-5).log() / 1.0
        for _ in range(5):
            np.asarray(q_crf)[...] = F.interpolate(
                q, size=(ch, cw), mode="area")[0].cpu().numpy().reshape(3, -1)
            crf.stepInference(q_crf, tmp1, tmp2)
            message = torch.from_numpy(np.asarray(q_crf).copy())
            message = message.reshape(1, 3, ch, cw).to(device).clamp_min(1e-20).log()
            message = F.interpolate(message, size=(lh, lw), mode="bilinear", align_corners=False)
            local = (self.aff_m(q) * weights).sum(2)
            proposal = (unary_logits + 0.25 * message + 1.0 * local).softmax(1)
            q = (1 - 0.5) * q + 0.5 * proposal
        q = F.interpolate(q, size=(nh, nw), mode="bilinear", align_corners=False)
        label = q[0].argmax(0).byte().cpu().numpy()
        return cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
