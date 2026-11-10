"""
Second backbone (ResNet-34 U-Net, ImageNet-pretrained encoder) for PX1, as small exact patches to the existing code.

The architecture is chosen by the environment variable PX1_ARCH (unset or "unet" = the original U-Net, "r34" =
segmentation_models_pytorch Unet with a ResNet-34 encoder). Every patch is checked to match exactly once, so a
mismatch with the bundle's code fails immediately instead of silently doing something else.

Also adds --variants, --part and --nparts to ks_eval.py so Kuro Siwo can be split across GPUs and run on V0 only.

Usage: python r34_patch.py <code dir> [--check]
"""
import os
import sys

BUILD = '''

def build_model(cin=2, pretrained=False):
    """PX1_ARCH=unet (default): the original U-Net. PX1_ARCH=r34: U-Net with a ResNet-34 encoder (ImageNet weights when training)."""
    arch = os.environ.get("PX1_ARCH", "unet")
    if arch == "unet":
        return UNet(cin=cin)
    import segmentation_models_pytorch as smp
    enc = {"r34": "resnet34", "r50": "resnet50"}[arch]
    return smp.Unet(enc, encoder_weights="imagenet" if pretrained else None, in_channels=cin, classes=2)

'''

PATCHES = {
    "sen1floods11_unet_baseline.py": [
        ("\n\n# ---------------------------------------------------------------- evaluation\n",
         BUILD + "\n# ---------------------------------------------------------------- evaluation\n"),
    ],
    "train_px1.py": [
        ("from sen1floods11_unet_baseline import IGNORE, UNet, channel_stats,",
         "from sen1floods11_unet_baseline import IGNORE, UNet, build_model, channel_stats,"),
        ("    model = UNet(cin=4 if use_dem else 2).to(device)\n",
         "    model = build_model(cin=4 if use_dem else 2, pretrained=True).to(device)\n"),
    ],
    "tta_px1.py": [
        ("    from sen1floods11_unet_baseline import IGNORE, UNet\n",
         "    from sen1floods11_unet_baseline import IGNORE, UNet, build_model\n"),
        ("    model = UNet(cin=4 if use_dem else 2).to(device)\n",
         "    model = build_model(cin=4 if use_dem else 2).to(device)\n"),
    ],
    "ks_eval.py": [
        ("    events = events_of(args.ks)\n",
         "    events = dict(list(events_of(args.ks).items())[args.part::args.nparts])\n"),
        ("        todo = [(m, v) for v in VARIANTS for m in args.methods",
         "        todo = [(m, v) for v in args.variants for m in args.methods"),
        ("    a.methods = a.methods.split(\",\")\n",
         "    a.methods = a.methods.split(\",\")\n    a.variants = a.variants.split(\",\")\n"),
        ("    ap.add_argument(\"--analyse-only\", action=\"store_true\")\n",
         "    ap.add_argument(\"--analyse-only\", action=\"store_true\")\n    ap.add_argument(\"--variants\", default=\"V0,V1\")\n"
         "    ap.add_argument(\"--part\", type=int, default=0)\n    ap.add_argument(\"--nparts\", type=int, default=1)\n"),
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
    print("PATCH_OK", flush=True)


if __name__ == "__main__":
    main()
