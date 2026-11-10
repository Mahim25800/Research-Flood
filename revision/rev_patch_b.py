"""
Backbone patch (applied after r34_patch.py): PX1_ARCH=r50s1 builds a U-Net with a ResNet-50 encoder whose weights were
pretrained on Sentinel-1 (VV, VH) without labels (SSL4EO-S12, MoCo; Wang et al., 2023), loaded through torchgeo.
"""
import os
import sys

OLD = '    enc = {"r34": "resnet34", "r50": "resnet50"}[arch]\n'
NEW = '''    if arch == "r50s1":
        m = smp.Unet("resnet50", encoder_weights=None, in_channels=cin, classes=2)
        if pretrained:
            from torchgeo.models import ResNet50_Weights
            sd = ResNet50_Weights.SENTINEL1_ALL_MOCO.get_state_dict(progress=False)
            assert sd["conv1.weight"].shape[1] == cin, sd["conv1.weight"].shape
            missing, unexpected = m.encoder.load_state_dict(sd, strict=False)
            assert not [k for k in missing if not k.startswith("fc.")], missing[:5]
            print(f"SSL4EO-S12 Sentinel-1 MoCo weights: {len(sd)} tensors loaded, {len(unexpected)} unused", flush=True)
        return m
''' + OLD


def main():
    p = os.path.join(sys.argv[1], "sen1floods11_unet_baseline.py")
    s = open(p).read()
    assert s.count(OLD) == 1, "apply r34_patch.py first"
    open(p, "w").write(s.replace(OLD, NEW))
    print("BACKBONE_PATCH_OK", flush=True)


if __name__ == "__main__":
    main()
