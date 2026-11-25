"""Freeze a private Kaggle payload from the original ImpactMesh local archives."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile


REV = Path(__file__).resolve().parents[1]
HERE = REV / "fresh_sar_test"
RUN = HERE / "frozen_comparison_v2"
OUT = REV / "generated/impactmesh_kaggle_followup"
OWNER = "mostofaseum"
SLUG = "px1-impactmesh-ablation-crossover-inputs"
ARCHIVE = "px1_impactmesh_followup.zip"

MASK = '''def darkness_only_water_mask(raw, terrain, tv, th):
    if terrain.shape[1] != 5:
        raise ValueError("Darkness-only mode requires the explicit radar-validity mask")
    return (raw[:, 0] < tv) & (raw[:, 1] < th) & (terrain[:, 3] > 0.5)


'''


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def patch_mask(code):
    anchor = "def boost_adaptive_water("
    call = "dark_low = adaptive_water_mask(r, t, tv, th)"
    assert code.count(anchor) == code.count(call) == 1
    return code.replace(anchor, MASK + anchor).replace(
        call,
        "mask_fn = darkness_only_water_mask if getattr(args, 'darkness_only', False) else adaptive_water_mask\n"
        "                    dark_low = mask_fn(r, t, tv, th)")


def stage():
    OUT.mkdir(exist_ok=True)
    dataset = OUT / "dataset"
    dataset.mkdir(exist_ok=False)
    artifact = dataset / ARCHIVE
    original = RUN / "protocol_snapshot.zip"
    frozen_digest = (RUN / "snapshot.sha256").read_text().split()[0]
    assert sha(original) == frozen_digest
    frozen = {k.replace("\\", "/"): v for k, v in read(RUN / "freeze.json")["files"].items()}
    roster = read(HERE / "selected.json")
    inputs = read(HERE / "input_verification.json")
    chips = [c for members in roster.values() for c in members]
    assert len(roster) == 13 and len(chips) == len(set(chips)) == 516 and set(chips) == set(inputs)

    # This archived calibration file contains only Sen1Floods11 source information.
    source = read(REV / "geoid_source_reference.json")
    assert set(source["threshold_bins"].values()) == {208}
    checkpoints = read(REV / "geoid_checkpoint_sha256.json")
    refs = {"resnet50": read(HERE / "source_calibration.json"), "unet": dict(
        mean=source["mean"], std=source["std"], threshold=.208)}
    assert refs["resnet50"]["threshold"] == .430
    protocol = dict(
        task="ImpactMesh darkness-only ablation and U-Net crossover",
        design="13 original activations, seeds 0–2, 200 steps per arm, same Kaggle GPU family",
        contrasts=["darkness_only - ordinary", "joint - darkness_only", "joint - ordinary"],
        paired_unit="activation after averaging the three seeds", primary_metric="floodwater AP",
        threshold_metrics=["IoU", "miss", "dry-pixel FPR", "FDP"],
        annotation="2=floodwater, 0=dry, 1=permanent water excluded, 255=nodata excluded",
        steps=200, original_snapshot_sha256=frozen_digest, code_sha256={}, models={},
    )
    with ZipFile(artifact, "x", compression=ZIP_STORED) as out, ZipFile(original) as archive:
        for name in ("selected.json", "input_verification.json", "source_calibration.json"):
            path = HERE / name
            assert sha(path) == frozen[name], name
            out.write(path, name)
        allowlist = ("src/tta_px1.py", "src/sen1floods11_unet_baseline.py", "src/conformal_px1.py",
                     "src/train_px1.py", "src/hand_fetch.py", "revision/fresh_sar_test/experiment.py")
        for name in allowlist:
            code = archive.read(name)
            key = "../../" + name if name.startswith("src/") else Path(name).name
            assert digest(code) == frozen[key], name
            if name == "src/tta_px1.py":
                code = patch_mask(code.decode()).encode()
            compile(code, name, "exec")
            out.writestr(name, code)
            protocol["code_sha256"][name] = digest(code)
        for chip in chips:
            radar = HERE / "prepared" / f"{chip}.npz"
            label = RUN / "labels" / f"{chip}.npy"
            assert sha(radar) == inputs[chip]["input_sha256"], chip
            assert sha(label) == inputs[chip]["label_sha256"], chip
            out.write(radar, f"prepared/{radar.name}")
            out.write(label, f"labels/{label.name}")
        for model, ref in refs.items():
            pins = {}
            for seed in range(3):
                if model == "unet":
                    key = f"V0_seed{seed}/model.pt"
                    path = REV.parent.parent / "runs" / key
                    expected = checkpoints[key]
                else:
                    key = f"source_models/SSL4EO_seed{seed}/model.pt"
                    path = HERE / key
                    expected = frozen[key]
                assert sha(path) == expected, path
                name = f"checkpoints/{model}/seed{seed}.pt"
                out.write(path, name)
                pins[str(seed)] = {"path": name, "sha256": expected}
            protocol["models"][model] = {key: ref[key] for key in ("mean", "std", "threshold")}
            protocol["models"][model]["checkpoints"] = pins
        out.writestr("protocol.json", json.dumps(protocol, indent=2))

    checksum = sha(artifact)
    dump(OUT / "protocol.json", protocol)
    (OUT / "dataset.sha256").write_text(checksum + "\n", encoding="ascii")
    dump(dataset / "dataset-metadata.json", dict(
        title="PX1 ImpactMesh ablation and crossover inputs", id=f"{OWNER}/{SLUG}",
        licenses=[{"name": "other"}]))
    stage_kernels()
    print(f"Frozen {len(chips)} inputs, six source checkpoints, four Kaggle jobs. Archive SHA-256 {checksum}")


def stage_kernels():
    checksum = (OUT / "dataset.sha256").read_text().strip()
    with ZipFile(OUT / "dataset" / ARCHIVE) as archive:
        protocol_sha = digest(archive.read("protocol.json"))
        metadata = {name: digest(archive.read(name)) for name in
                    ("selected.json", "input_verification.json", "source_calibration.json")}
    runner = (REV / "analysis/impactmesh_kaggle_run.py").read_text(encoding="utf-8")
    for tag, seed in (("smoke", 0), ("seed0", 0), ("seed1", 1), ("seed2", 2)):
        folder = OUT / tag
        folder.mkdir(exist_ok=True)
        code = runner + f"\nrun({seed}, {tag == 'smoke'}, {checksum!r}, {protocol_sha!r}, {metadata!r})\n"
        compile(code, "run.py", "exec")
        (folder / "run.py").write_text(code, encoding="utf-8")
        dump(folder / "kernel-metadata.json", dict(
            id=f"{OWNER}/px1-impactmesh-ablation-crossover-{tag}",
            title=f"PX1 ImpactMesh ablation crossover {tag}", code_file="run.py",
            language="python", kernel_type="script", is_private=True, enable_gpu=True,
            enable_internet=False, dataset_sources=[f"{OWNER}/{SLUG}"], competition_sources=[],
            kernel_sources=[]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels-only", action="store_true")
    if parser.parse_args().kernels_only:
        stage_kernels()
    else:
        stage()
