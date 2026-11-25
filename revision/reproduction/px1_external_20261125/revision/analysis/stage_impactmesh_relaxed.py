"""Stage isolated private Kaggle jobs for the fixed relaxed ordinary control."""

from zipfile import ZipFile

from stage_impactmesh_kaggle import ARCHIVE, OWNER, OUT as PRIOR, REV, SLUG, digest, dump, sha


OUT = REV / "generated/impactmesh_relaxed_control_20261124"
PREFIX = f"{OWNER}/px1-impactmesh-relaxed-ordinary-control-"
FPR = (0.02, 0.05)
JOBS = {"smoke": 0, "seed0": 0, "seed1": 1, "seed2": 2}


def stage():
    checksum = (PRIOR / "dataset.sha256").read_text().split()[0]
    with ZipFile(PRIOR / "dataset" / ARCHIVE) as archive:
        metadata = {name: digest(archive.read(name)) for name in
                    ("protocol.json", "selected.json", "input_verification.json", "source_calibration.json")}
    protocol_sha = metadata.pop("protocol.json")
    runner = (REV / "analysis/impactmesh_kaggle_run.py").read_text(encoding="utf-8")
    OUT.mkdir(exist_ok=False)
    script_digests = {}
    for tag, seed in JOBS.items():
        folder = OUT / tag
        folder.mkdir()
        code = runner + f"\nrun({seed}, {tag == 'smoke'}, {checksum!r}, {protocol_sha!r}, {metadata!r}, relaxed=True)\n"
        compile(code, "run.py", "exec")
        raw = code.encode("utf-8")
        (folder / "run.py").write_bytes(raw)
        script_digests[tag] = digest(raw)
        dump(folder / "kernel-metadata.json", dict(
            id=PREFIX + tag, title=f"PX1 ImpactMesh relaxed ordinary control {tag}",
            code_file="run.py", language="python", kernel_type="script", is_private=True,
            enable_gpu=True, enable_internet=False, dataset_sources=[f"{OWNER}/{SLUG}"],
            competition_sources=[], kernel_sources=[]))
    dump(OUT / "protocol.json", dict(
        task="Post-hoc relaxed ordinary ImpactMesh control",
        changes="Set strict positive teacher cutoff to 0.7 for ordinary self-training only.",
        positive_cutoff=.7, negative_cutoff=.1, steps=200, adabn_batches=50,
        backbone="SSL4EO-S12 ResNet-50", activations=13, chips=516, seeds=[0, 1, 2],
        archive_sha256=checksum, protocol_sha256=protocol_sha, input_metadata_sha256=metadata,
        scripts_sha256=script_digests, source_threshold=.430, matched_fpr=FPR,
        reference_report_sha256=sha(REV / "analysis/impactmesh_saved_scores.json"),
        references=["archived ordinary (0.9)", "archived physics"],
        metric_support="floodwater versus dry, excluding permanent water and nodata",
        aggregation="Average three seeds per activation, then weight all activations equally."))


if __name__ == "__main__":
    stage()
