"""
Revision patch (applied after r34_patch.py): adds three published test-time adaptation baselines to tta_px1.run_one
and a second label definition for Kuro Siwo.

  eata   EATA (Niu et al., 2022): entropy minimisation on batch-norm weights, only on reliable pixels
         (entropy < 0.4 ln 2), re-weighted by exp(E0 - entropy), with a Fisher penalty that anchors the weights.
         The Fisher information is estimated with the unadapted model on the first 20 target batches.
  sar    SAR (Niu et al., 2023): sharpness-aware entropy minimisation on batch-norm weights, reliable pixels only.
  cotta  CoTTA (Wang et al., 2022): mean teacher, augmentation-averaged teacher predictions, all weights updated,
         stochastic restore of 1% of the weights to their source values at every step.

All three use the same crops, batch size, number of steps and optimiser as the TENT baseline.

Kuro Siwo: histogram slot 1 (unused for Kuro Siwo) now holds the scores with permanent water ignored
(only flood water counts as water), so both label definitions come from one run.
"""
import os
import sys

NEW = '''    if method in ("eata", "sar"):
        import math
        E0 = 0.4 * math.log(2)
        model.train()
        params = [p for mod in model.modules() if isinstance(mod, torch.nn.BatchNorm2d) for p in (mod.weight, mod.bias)]
        for p in model.parameters():
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)

        def ent(r, t):
            p = model(inp(r, t)).softmax(1)
            return -(p * torch.log(p.clamp_min(1e-6))).sum(1)

        opt = torch.optim.Adam(params, lr=LR_TENT)
        if method == "eata":
            anchor = [p.detach().clone() for p in params]
            fisher = [torch.zeros_like(p) for p in params]
            for _ in range(20):
                r, t = crops()
                logits = model(inp(r, t))
                g = torch.autograd.grad(F.cross_entropy(logits, logits.argmax(1)), params)
                for f, gi in zip(fisher, g):
                    f += gi.detach() ** 2 / 20
        sel_share = []
        for _ in range(args.steps):
            r, t = crops()
            H = ent(r, t)
            sel = (H < E0).detach()
            sel_share.append(float(sel.float().mean()))
            if method == "eata":
                w = torch.exp(E0 - H.detach())
                loss = (H * w * sel).sum() / sel.sum().clamp_min(1)
                loss = loss + 2000.0 * sum((f * (p - a) ** 2).sum() for f, p, a in zip(fisher, params, anchor))
                opt.zero_grad(); loss.backward(); opt.step()
            else:
                loss = (H * sel).sum() / sel.sum().clamp_min(1)
                opt.zero_grad(); loss.backward()
                gn = torch.sqrt(sum((p.grad ** 2).sum() for p in params)) + 1e-12
                eps = [0.05 * p.grad / gn for p in params]
                with torch.no_grad():
                    for p, e_ in zip(params, eps):
                        p.add_(e_)
                H2 = ent(r, t)
                sel2 = sel & (H2 < E0).detach()
                loss2 = (H2 * sel2).sum() / sel2.sum().clamp_min(1)
                opt.zero_grad(); loss2.backward()
                with torch.no_grad():
                    for p, e_ in zip(params, eps):
                        p.sub_(e_)
                opt.step()
        print(f"    reliable share of pixels {np.mean(sel_share):.3f}", flush=True)
        for p in model.parameters():
            p.requires_grad_(True)
    cotta_teacher = None
    if method == "cotta":
        source = {k: p.detach().clone() for k, p in model.named_parameters()}
        cotta_teacher = copy.deepcopy(model).train()
        for p in cotta_teacher.parameters():
            p.requires_grad_(False)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=LR_ST)
        for _ in range(args.steps):
            r, t = crops()
            with torch.no_grad():
                acc = 0
                for k in range(4):
                    off = (torch.rand(r.shape[0], 1, 1, 1, device=device) * 3 - 1.5) if k else 0.0
                    x = inp(r + off, t)
                    o = cotta_teacher(x.flip(3)).softmax(1).flip(3) if k % 2 else cotta_teacher(x).softmax(1)
                    acc = acc + o / 4
            loss = -(acc * F.log_softmax(model(inp(r, t)), 1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for pt_, ps_ in zip(cotta_teacher.parameters(), model.parameters()):
                    pt_.mul_(EMA).add_(ps_.detach(), alpha=1 - EMA)
                for k, p in model.named_parameters():
                    mask = torch.rand_like(p) < 0.01
                    p.copy_(torch.where(mask, source[k], p))
    final = model
    if cotta_teacher is not None:
        final = cotta_teacher
'''

HIST_OLD = '''            for sub in ([0, 1] if ev["clean"][i] else [0]):
                for c in (0, 1):
                    h[sub, c] += torch.bincount(b[lab == c], minlength=BINS).cpu().numpy()
'''
HIST_NEW = HIST_OLD + '''            if "lab_alt" in ev:
                lab = torch.from_numpy(ev["lab_alt"][i].astype(np.int64)).to(device)
                for c in (0, 1):
                    h[1, c] += torch.bincount(b[lab == c], minlength=BINS).cpu().numpy()
'''

PATCHES = {
    "tta_px1.py": [
        ("ALL_METHODS = METHODS + ABLATIONS + COMBOS + ADAPTIVE\n",
         'ALL_METHODS = METHODS + ABLATIONS + COMBOS + ADAPTIVE + ["eata", "sar", "cotta", "ft"]\n'),
        ("    final = model\n    if base in (", NEW + "    if base in ("),
        (HIST_OLD, HIST_NEW),
    ],
    "ks_eval.py": [
        ('        ev["lab"].append(lab)\n',
         '        ev["lab"].append(lab)\n        alt = lab.copy()\n        alt[(m == 1) & (lab != 255)] = 255\n'
         '        ev.setdefault("lab_alt", []).append(alt)\n'),
    ],
}


def main():
    code, check = sys.argv[1], "--check" in sys.argv
    for f, ps in PATCHES.items():
        p = os.path.join(code, f)
        s = open(p).read()
        for old, new in ps:
            n = s.count(old)
            if n != 1:
                raise SystemExit(f"patch mismatch in {f}: {old.strip()[:60]!r} found {n} times")
            s = s.replace(old, new)
        if not check:
            open(p, "w").write(s)
        print(f"{'checked' if check else 'patched'} {f} ({len(ps)} changes)", flush=True)
    print("REV_PATCH_OK", flush=True)


if __name__ == "__main__":
    main()
