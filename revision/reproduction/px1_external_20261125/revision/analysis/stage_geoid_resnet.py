"""Freeze the Kaggle ResNet-50 follow-up without modifying the original GEOID run."""

import ast
import argparse
import json
from zipfile import ZipFile

from stage_impactmesh_kaggle import ARCHIVE, HERE, OUT as IMPACT, OWNER, REV, SLUG, digest, dump, read, sha


ORIGINAL = REV / "generated/geoid_v4_results_20261118/geoid_evaluation_v4"
FROZEN = REV / "generated/geoid_frozen_execution_20261117"
OUT = REV / "generated/geoid_resnet50_followup_20261125"
KERNEL = f"{OWNER}/px1-geoid-flood-resnet50-followup"
RUNNER = REV / "analysis/run_geoid_resnet.py"


def shared_setup():
    source = (REV / "run_geoid_kaggle.py").read_text(encoding="utf-8")
    imports = {"hashlib", "json", "pathlib", "re", "shutil", "sys", "tarfile", "time",
               "zipfile", "numpy", "rasterio", "requests", "torch"}
    constants = {"RELEASE", "COMMIT", "PROTOCOL_HASH", "HF", "PROTOCOL_URL"}
    functions = {"sha", "save", "fetch", "extract"}
    blocks = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            retain = all(n.name in imports for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            retain = node.module in imports
        elif isinstance(node, ast.Assign):
            retain = any(isinstance(n, ast.Name) and n.id in constants for n in node.targets)
        else:
            retain = isinstance(node, ast.FunctionDef) and node.name in functions
        if retain:
            blocks.append(ast.get_source_segment(source, node))
    return "\n\n".join(blocks)


def stage(folder=OUT):
    prior = read(ORIGINAL / "run_manifest.json")
    original_result = ORIGINAL / "results/results.json"
    assert prior["status"] == "completed" and sha(original_result) == prior["results_sha256"]
    assert read(ORIGINAL / "verification.json")["verified"]
    protocol = FROZEN / "px1-geoid-protocol-20261117-v4.zip"
    assert sha(protocol) == prior["protocol_sha256"]
    selected = {e: row["selected"] for e, row in read(ORIGINAL / "chip_selection.json").items()}
    events = read(original_result)["events"]
    assert set(selected) == set(events) and len(selected) == 9
    assert sum(map(len, selected.values())) == 424
    assert all(selected[e] == events[e]["selected"] for e in events)
    wanted = {name for chips in selected.values() for name, _, _ in chips}
    inventory = read(ORIGINAL / "inventory.json")
    tiles = {row["id"]: row for row in inventory["tiles"] if row["id"] in wanted}
    assert set(tiles) == wanted and inventory["release"] == prior["release"]
    hashes = read(ORIGINAL / "raster_sha256.json")
    names = {tile[name] for tile in tiles.values() for name in ("s1_post", "dem", "label", "hand") if tile[name]}
    support = {}
    for event, row in events.items():
        dry, water = row["methods"]["0:none_valid"]["flood"]["histogram"]
        support[event] = [sum(water), sum(dry)]
    spec_ref = read(IMPACT / "protocol.json")
    report = HERE / "source_calibration.json"
    calibration = {key: read(report)[key] for key in ("mean", "std", "threshold")}
    assert calibration["threshold"] == .430
    models = spec_ref["models"]["resnet50"]
    assert all(calibration[k] == models[k] for k in calibration)
    files = {pin["path"]: pin["sha256"] for pin in models["checkpoints"].values()}
    files["revision/fresh_sar_test/experiment.py"] = spec_ref["code_sha256"]["revision/fresh_sar_test/experiment.py"]
    files["src/tta_px1.py"] = spec_ref["code_sha256"]["src/tta_px1.py"]
    archive_path = IMPACT / "dataset" / ARCHIVE
    checksum = (IMPACT / "dataset.sha256").read_text().strip()
    with ZipFile(archive_path) as archive:
        assert read(IMPACT / "protocol.json") == json.loads(archive.read("protocol.json"))
        impact_protocol = digest(archive.read("protocol.json"))
        assert all(digest(archive.read(name)) == fingerprint for name, fingerprint in files.items())
        assert sha(report) == digest(archive.read("source_calibration.json"))
    shards = read(ORIGINAL / "shard_manifest.json")
    assert len(shards) == 7 and all(len(entry["lfs"]["oid"]) == 64 for entry in shards)
    spec = dict(release=prior["release"], geoid_protocol_sha256=prior["protocol_sha256"],
                geoid_results_sha256=prior["results_sha256"], selected=selected, support=support, tiles=tiles,
                raster_hashes={name: hashes[name] for name in sorted(names)}, shards=shards,
                hand_sources=read(ORIGINAL / "hand_source_manifest.json"), calibration=calibration,
                impactmesh_archive_sha256=checksum, impactmesh_protocol_sha256=impact_protocol,
                impactmesh_files=files, steps=200, adaptation_seed=0)
    spec_sha = digest(json.dumps(spec, sort_keys=True).encode())
    runner = shared_setup() + "\n\n" + RUNNER.read_text(encoding="utf-8")
    runner += f"\nrun_resnet({spec!r}, {spec_sha!r})\n"
    compile(runner, "run.py", "exec")
    folder.mkdir(exist_ok=False)
    job = folder / "kernel"
    job.mkdir()
    (job / "run.py").write_bytes(runner.encode())
    dump(job / "kernel-metadata.json", dict(
        id=KERNEL, title="PX1 GEOID-Flood ResNet50 followup", code_file="run.py",
        language="python", kernel_type="script", is_private=True, enable_gpu=True,
        enable_internet=True, dataset_sources=[f"{OWNER}/{SLUG}"],
        competition_sources=[], kernel_sources=[]))
    dump(folder / "protocol.json", spec)
    dump(folder / "freeze.json", dict(protocol_sha256=spec_sha, runner_sha256=sha(job / "run.py"),
                                      kernel=KERNEL, cohort_count=len(selected), seed_count=3))
    print(f"Staged {len(selected)} activations, {sum(map(len, selected.values()))} windows and {len(names)} raster hashes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="store_true")
    args = parser.parse_args()
    stage(OUT.with_name(OUT.name + "_candidate") if args.candidate else OUT)
