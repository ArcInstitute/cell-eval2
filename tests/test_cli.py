import os
import shlex
import stat
from functools import partial

import polars as pl
import pytest

from cell_eval2.cli import main
from cell_eval2.config import EvalConfig


def test_cli_run_writes_results(synthetic_pair, tmp_path):
    pred, real = synthetic_pair
    pp = tmp_path / "pred.h5ad"
    rp = tmp_path / "real.h5ad"
    pred.write_h5ad(pp)
    real.write_h5ad(rp)
    de_real = tmp_path / "real_de.parquet"
    de_pred = tmp_path / "pred_de.parquet"
    pl.DataFrame({
        "target": ["GENE1", "GENE1", "GENE2", "GENE3"],
        "feature": ["g0", "g1", "g2", "g3"],
        "log2_fold_change": [2.0, 1.0, 0.5, 1.0],
        "p_adj": [0.01, 0.2, 0.03, 0.01],
    }).write_parquet(de_real)
    pl.DataFrame({
        "target": ["GENE1", "GENE1", "GENE2", "GENE3"],
        "feature": ["g0", "g1", "g2", "g3"],
        "log2_fold_change": [1.5, 0.8, 0.7, 0.9],
        "p_adj": [0.02, 0.3, 0.04, 0.01],
    }).write_parquet(de_pred)
    out = tmp_path / "out"
    main(["run", "-ap", str(pp), "-ar", str(rp), "--profile", "vcc",
          "--pert-col", "target", "--control", "non-targeting",
          "--input-type", "lognorm",
          "--de-pred", str(de_pred), "--de-real", str(de_real), "-o", str(out)])
    assert (out / "results.csv").exists()
    assert (out / "run_params.yaml").exists()


def test_cli_write_degenes_emits_de_tables(synthetic_pair, tmp_path):
    pred, real = synthetic_pair
    pp = tmp_path / "pred.h5ad"
    rp = tmp_path / "real.h5ad"
    pred.write_h5ad(pp)
    real.write_h5ad(rp)
    de_real = tmp_path / "real_de.parquet"
    de_pred = tmp_path / "pred_de.parquet"
    pl.DataFrame({
        "target": ["GENE1", "GENE1", "GENE2", "GENE3"],
        "feature": ["g0", "g1", "g2", "g3"],
        "log2_fold_change": [2.0, 1.0, 0.5, 1.0],
        "p_adj": [0.01, 0.2, 0.03, 0.01],
    }).write_parquet(de_real)
    pl.DataFrame({
        "target": ["GENE1", "GENE1", "GENE2", "GENE3"],
        "feature": ["g0", "g1", "g2", "g3"],
        "log2_fold_change": [1.5, 0.8, 0.7, 0.9],
        "p_adj": [0.02, 0.3, 0.04, 0.01],
    }).write_parquet(de_pred)
    out = tmp_path / "out"
    main(["run", "-ap", str(pp), "-ar", str(rp), "--profile", "vcc",
          "--pert-col", "target", "--control", "non-targeting",
          "--input-type", "lognorm", "--write-degenes",
          "--de-pred", str(de_pred), "--de-real", str(de_real), "-o", str(out)])
    assert (out / "de_real.parquet").exists()
    assert (out / "de_pred.parquet").exists()
    dr = pl.read_parquet(out / "de_real.parquet")
    dp = pl.read_parquet(out / "de_pred.parquet")
    assert {"target", "feature"}.issubset(dr.columns)
    assert {"target", "feature"}.issubset(dp.columns)
    # Each emitted table must be the side it was SUPPLIED as. The column assertions above
    # hold just as well if the two sides are swapped, so compare the contents: the fixtures
    # differ in log2_fold_change precisely so a swap is visible.
    assert dr.equals(pl.read_parquet(de_real))
    assert dp.equals(pl.read_parquet(de_pred))


def test_cli_write_degenes_does_not_clobber_a_supplied_input(synthetic_pair, tmp_path):
    """A supplied --de-* path that aliases an emitted file must not corrupt the run.

    de_real.parquet is written first, so reading de_pred only afterwards let that write
    destroy the file --de-pred names. Loading both frames before writing either fixed the
    emitted FILES -- but discarding those frames left _prepare_de_cached re-reading the
    original paths, which the writes had just swapped, so the metrics were still scored
    with the sides reversed while the files on disk looked right. Assert both.
    """
    pred, real = synthetic_pair
    pp, rp = tmp_path / "pred.h5ad", tmp_path / "real.h5ad"
    pred.write_h5ad(pp)
    real.write_h5ad(rp)

    def _table(n_sig):
        # n_sig significant features per target, so nsig_counts_real/pred differ per side
        # and a swap is visible in results.csv, not just in the parquet files.
        #
        # The first three features are the TARGET LABELS themselves: `--profile de` now
        # includes the eleven chance-corrected direction metrics (#195), which resolve each
        # target against this table's `feature` column and fail loud when none resolves.
        # Their target-gene exclusion applies identically to the reference and the crossed
        # run below, so the `got == expected` comparison this test rests on is unaffected.
        tgts = ["GENE1", "GENE2", "GENE3"]
        feats = [*tgts, *(f"g{i}" for i in range(3, 10))]
        return pl.DataFrame({
            "target": [t for t in tgts for _ in feats],
            "feature": feats * len(tgts),
            "log2_fold_change": [2.0] * (len(tgts) * len(feats)),
            "p_adj": ([0.001] * n_sig + [0.9] * (len(feats) - n_sig)) * len(tgts),
        })

    real_side, pred_side = _table(8), _table(2)

    def _run(outdir, de_real, de_pred):
        outdir.mkdir(parents=True, exist_ok=True)
        main(["run", "-ap", str(pp), "-ar", str(rp), "--profile", "de",
              "--pert-col", "target", "--control", "non-targeting",
              "--input-type", "lognorm", "--write-degenes",
              "--de-real", str(de_real), "--de-pred", str(de_pred), "-o", str(outdir)])
        return pl.read_csv(outdir / "results.csv")

    # Reference: the same two tables supplied from paths that alias nothing.
    src = tmp_path / "src"
    src.mkdir()
    real_side.write_parquet(src / "r.parquet")
    pred_side.write_parquet(src / "p.parquet")
    expected = _run(tmp_path / "ref", src / "r.parquet", src / "p.parquet")

    # Crossed: each side is supplied from the file the OTHER side is written to.
    out = tmp_path / "out"
    out.mkdir()
    real_side.write_parquet(out / "de_pred.parquet")
    pred_side.write_parquet(out / "de_real.parquet")
    got = _run(out, out / "de_pred.parquet", out / "de_real.parquet")

    assert pl.read_parquet(out / "de_real.parquet").equals(real_side)
    assert pl.read_parquet(out / "de_pred.parquet").equals(pred_side)
    assert got.equals(expected)  # ...and the METRICS are not scored on the swapped tables


@pytest.mark.parametrize("break_at", ["write", "replace"])
def test_write_de_tables_publishes_atomically_or_not_at_all(tmp_path, monkeypatch, break_at):
    """Both DE tables are published as a PAIR, or neither is.

    Writing straight to the two destinations publishes them one at a time, so a failure on the
    second leaves a new de_real.parquet beside a stale de_pred.parquet -- and in the aliasing
    case the previous test covers, that half-write lands ON the file the other side was
    supplied from, destroying the caller's own input with no way back.

    Two failure points, because staging alone only covers the first (codex checkpoint-2 P1,
    rounds 1-2): a failed staged WRITE never touches a destination, and a failed second
    os.replace must roll the first one back from its saved copy.
    """
    from cell_eval2 import run as run_mod

    outdir = tmp_path / "out"
    outdir.mkdir()
    keep = pl.DataFrame({"target": ["OLD"], "feature": ["g0"],
                         "log2_fold_change": [9.0], "p_adj": [0.5]})
    keep.write_parquet(outdir / "de_real.parquet")  # the pre-existing / supplied file
    keep.write_parquet(outdir / "de_pred.parquet")

    src = tmp_path / "src"
    src.mkdir()
    new = pl.DataFrame({"target": ["NEW"], "feature": ["g1"],
                        "log2_fold_change": [1.0], "p_adj": [0.01]})
    new.write_parquet(src / "r.parquet")
    new.write_parquet(src / "p.parquet")

    cfg = EvalConfig(outdir=str(outdir))
    call = partial(run_mod._write_de_tables, str(src / "r.parquet"), str(src / "p.parquet"),
                   cfg=cfg)

    if break_at == "write":
        real_write, calls = pl.DataFrame.write_parquet, {"n": 0}

        def flaky(self, path, *a, **kw):
            calls["n"] += 1
            if calls["n"] == 2:                  # the SECOND staged write blows up
                raise OSError("disk full")
            return real_write(self, path, *a, **kw)

        monkeypatch.setattr(pl.DataFrame, "write_parquet", flaky)
    else:
        real_replace, calls = os.replace, {"n": 0}

        def flaky(src_path, dst_path, *a, **kw):
            calls["n"] += 1
            # 1: dest -> bak, 2: tmp -> dest (de_real published), 3: dest -> bak, 4: BOOM
            if calls["n"] == 4:
                raise OSError("disk full")
            return real_replace(src_path, dst_path, *a, **kw)

        monkeypatch.setattr(run_mod.os, "replace", flaky)

    with pytest.raises(OSError, match="disk full"):
        call()
    monkeypatch.undo()

    # Neither destination moved, and no staging or backup debris was left behind.
    assert pl.read_parquet(outdir / "de_real.parquet").equals(keep)
    assert pl.read_parquet(outdir / "de_pred.parquet").equals(keep)
    assert sorted(p.name for p in outdir.iterdir()) == ["de_pred.parquet", "de_real.parquet"]

    # And the happy path still publishes both, under the caller's umask rather than the 0600 a
    # tempfile.mkstemp staging would carry onto the published artifact.
    call()
    assert pl.read_parquet(outdir / "de_real.parquet").equals(new)
    assert pl.read_parquet(outdir / "de_pred.parquet").equals(new)
    assert sorted(p.name for p in outdir.iterdir()) == ["de_pred.parquet", "de_real.parquet"]
    reference = src / "r.parquet"                # written by polars with no staging at all
    want = stat.S_IMODE(reference.stat().st_mode)
    for name in ("de_real.parquet", "de_pred.parquet"):
        assert stat.S_IMODE((outdir / name).stat().st_mode) == want, name


def _de_sources(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    tbl = pl.DataFrame({"target": ["A"], "feature": ["g0"],
                        "log2_fold_change": [1.0], "p_adj": [0.01]})
    tbl.write_parquet(src / "r.parquet")
    tbl.write_parquet(src / "p.parquet")
    return str(src / "r.parquet"), str(src / "p.parquet")


def test_write_de_tables_reports_an_unrestorable_backup(tmp_path, monkeypatch):
    """A failed publish that ALSO fails to restore must name where the old file survives.

    Otherwise the destination is missing, its only copy is a backup under a generated name,
    and the cleanup path deletes it -- an intact file nobody can find is lost in practice
    (codex checkpoint-2 round 3 P1; same argument as real_bundle.py's restore failure).
    """
    from cell_eval2 import run as run_mod

    outdir = tmp_path / "out"
    outdir.mkdir()
    keep = pl.DataFrame({"target": ["OLD"], "feature": ["g0"],
                         "log2_fold_change": [9.0], "p_adj": [0.5]})
    keep.write_parquet(outdir / "de_real.parquet")

    real_replace, calls = os.replace, {"n": 0}

    def flaky(src_path, dst_path, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:      # 1: de_real -> bak, 2: publish BOOM
            raise OSError("disk full")
        if calls["n"] == 3:      # ...and the restore fails too
            raise OSError("still full")
        return real_replace(src_path, dst_path, *a, **kw)

    monkeypatch.setattr(run_mod.os, "replace", flaky)
    with pytest.raises(ValueError, match="previous contents survive only as") as exc:
        run_mod._write_de_tables(*_de_sources(tmp_path), cfg=EvalConfig(outdir=str(outdir)))
    monkeypatch.undo()

    bak = [p for p in outdir.iterdir() if ".bak." in p.name]
    assert len(bak) == 1                          # PRESERVED, not swept up by the cleanup
    assert str(bak[0]) in str(exc.value)          # ...and findable from the message
    assert pl.read_parquet(bak[0]).equals(keep)


def test_write_de_tables_reports_a_published_file_it_could_not_withdraw(tmp_path, monkeypatch):
    """With no pre-existing destinations there is nothing to restore -- but the first table is
    already published when the second replace fails, and if withdrawing it ALSO fails the pair
    is half-published. Best-effort cleanup there would rethrow the original error and say
    nothing about it (codex checkpoint-2 round 4 P1)."""
    from cell_eval2 import run as run_mod

    outdir = tmp_path / "out"
    outdir.mkdir()
    real_replace, calls = os.replace, {"n": 0}

    def flaky(src_path, dst_path, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:                      # 1: de_real published, 2: de_pred BOOM
            raise OSError("disk full")
        return real_replace(src_path, dst_path, *a, **kw)

    real_unlink = os.unlink

    def stubborn(path, *a, **kw):
        if os.path.basename(str(path)) == "de_real.parquet":
            raise OSError("read-only")           # ...and it cannot be withdrawn
        return real_unlink(path, *a, **kw)

    monkeypatch.setattr(run_mod.os, "replace", flaky)
    monkeypatch.setattr(run_mod.os, "unlink", stubborn)
    with pytest.raises(ValueError, match="could not fully roll back") as exc:
        run_mod._write_de_tables(*_de_sources(tmp_path), cfg=EvalConfig(outdir=str(outdir)))
    monkeypatch.undo()

    assert "de_real.parquet" in str(exc.value)   # NAMED, not silently left behind
    assert "published and could not be removed" in str(exc.value)
    assert isinstance(exc.value.__cause__, OSError)   # the original failure is still attached


def test_write_de_tables_does_not_report_a_rollback_that_succeeded(tmp_path, monkeypatch):
    """Withdrawing a published table that HAS a backup is redundant, and reporting a failed
    withdrawal there is a false alarm.

    `os.replace(bak, dest)` overwrites whatever sits at the destination, so unlinking it first
    buys nothing and only opens a window where the file is missing. When the unlink fails but
    the restore then succeeds, the rollback worked -- raising "could not fully roll back" would
    describe a failure that did not happen (Gemini, PR #292).
    """
    from cell_eval2 import run as run_mod

    outdir = tmp_path / "out"
    outdir.mkdir()
    keep = pl.DataFrame({"target": ["OLD"], "feature": ["g0"],
                         "log2_fold_change": [9.0], "p_adj": [0.5]})
    keep.write_parquet(outdir / "de_real.parquet")   # exists -> gets a backup
    keep.write_parquet(outdir / "de_pred.parquet")

    real_replace, calls = os.replace, {"n": 0}

    def flaky(src_path, dst_path, *a, **kw):
        calls["n"] += 1
        # 1: de_real -> bak, 2: de_real published, 3: de_pred -> bak, 4: BOOM
        if calls["n"] == 4:
            raise OSError("disk full")
        return real_replace(src_path, dst_path, *a, **kw)

    def never(path, *a, **kw):
        raise OSError("read-only")                   # any unlink of a destination would fail

    monkeypatch.setattr(run_mod.os, "replace", flaky)
    monkeypatch.setattr(run_mod.os, "unlink", never)
    # The ORIGINAL error, not a rollback-failure report: both destinations had backups, so the
    # restores put them back and nothing needed withdrawing.
    with pytest.raises(OSError, match="disk full"):
        run_mod._write_de_tables(*_de_sources(tmp_path), cfg=EvalConfig(outdir=str(outdir)))
    monkeypatch.undo()

    assert pl.read_parquet(outdir / "de_real.parquet").equals(keep)
    assert pl.read_parquet(outdir / "de_pred.parquet").equals(keep)


def test_write_de_tables_refuses_a_destination_that_is_not_a_file(tmp_path):
    """os.replace would move a DIRECTORY aside and leave it as undeletable backup debris."""
    from cell_eval2 import run as run_mod

    outdir = tmp_path / "out"
    (outdir / "de_pred.parquet").mkdir(parents=True)
    with pytest.raises(ValueError, match="not a regular file"):
        run_mod._write_de_tables(*_de_sources(tmp_path), cfg=EvalConfig(outdir=str(outdir)))
    assert (outdir / "de_pred.parquet").is_dir()          # untouched
    assert not (outdir / "de_real.parquet").exists()      # ...and nothing was published


def test_write_de_tables_refuses_an_outdir_that_is_not_a_directory(tmp_path):
    """`os.makedirs` reports this as a bare `FileExistsError: [Errno 17] File exists: <path>`
    (measured) with no hint of what the caller did wrong -- the same door-check real_bundle.py
    grew in #290 (Copilot, PR #292)."""
    from cell_eval2 import run as run_mod

    notadir = tmp_path / "notadir"
    notadir.write_text("x")
    with pytest.raises(ValueError, match="exists and is not a directory"):
        run_mod._write_de_tables(*_de_sources(tmp_path), cfg=EvalConfig(outdir=str(notadir)))
    assert notadir.read_text() == "x"          # untouched


def test_write_de_tables_validates_the_schema_before_publishing(tmp_path):
    """`_prepare_de_cached` rejects a schema-invalid table anyway -- but it runs AFTER the
    write, so the run used to die leaving two published files describing an input it had
    already rejected (codex checkpoint-2 round 3 P2)."""
    from cell_eval2 import run as run_mod

    outdir = tmp_path / "out"
    outdir.mkdir()
    good, _ = _de_sources(tmp_path)
    bad = tmp_path / "bad.parquet"
    pl.DataFrame({"target": ["A"], "feature": ["g0"]}).write_parquet(bad)   # no lfc / p_adj
    with pytest.raises(ValueError, match="missing required columns"):
        run_mod._write_de_tables(good, str(bad), cfg=EvalConfig(outdir=str(outdir)))
    assert sorted(p.name for p in outdir.iterdir()) == []


def test_cli_config_without_outdir_writes_run_params(synthetic_pair, tmp_path):
    pred, real = synthetic_pair
    pp = tmp_path / "pred.h5ad"
    rp = tmp_path / "real.h5ad"
    pred.write_h5ad(pp)
    real.write_h5ad(rp)
    cfg = tmp_path / "cfg.yaml"
    # config deliberately omits outdir -> CLI must fall back to -o
    EvalConfig(metrics="pds", pert_col="target", control="non-targeting",
               input_type="lognorm").to_yaml(str(cfg))
    out = tmp_path / "out"
    main(["run", "-ap", str(pp), "-ar", str(rp), "--config", str(cfg), "-o", str(out)])
    assert (out / "results.csv").exists()
    assert (out / "run_params.yaml").exists()


def test_cli_version_flag_emits_v1_names(synthetic_pair, tmp_path):
    pred, real = synthetic_pair
    pp, rp = tmp_path / "pred.h5ad", tmp_path / "real.h5ad"
    pred.write_h5ad(pp)
    real.write_h5ad(rp)
    out = tmp_path / "out"
    main(["run", "-ap", str(pp), "-ar", str(rp), "--profile", "pds",
          "--version", "v1", "--pert-col", "target", "--control", "non-targeting",
          "--input-type", "lognorm", "-o", str(out)])
    df = pl.read_csv(out / "results.csv")
    assert "discrimination_score_l1" in df["metric"].unique().to_list()  # v1 label


def test_cli_version_flag_overrides_config(synthetic_pair, tmp_path):
    # config says v2; explicit --version v1 must win (output carries v1 labels)
    pred, real = synthetic_pair
    pp, rp = tmp_path / "pred.h5ad", tmp_path / "real.h5ad"
    pred.write_h5ad(pp)
    real.write_h5ad(rp)
    cfg = tmp_path / "cfg.yaml"
    EvalConfig(metrics="pds", version="v2", pert_col="target", control="non-targeting",
               input_type="lognorm").to_yaml(str(cfg))
    out = tmp_path / "out"
    main(["run", "-ap", str(pp), "-ar", str(rp), "--config", str(cfg), "--version", "v1", "-o", str(out)])
    df = pl.read_csv(out / "results.csv")
    assert "discrimination_score_l1" in df["metric"].to_list()  # v1 label => flag won


def test_cli_outdir_flag_overrides_config(synthetic_pair, tmp_path):
    # config has its own outdir; explicit -o must win (finding 3)
    pred, real = synthetic_pair
    pp, rp = tmp_path / "pred.h5ad", tmp_path / "real.h5ad"
    pred.write_h5ad(pp)
    real.write_h5ad(rp)
    cfgdir = tmp_path / "cfgout"
    cfg = tmp_path / "cfg.yaml"
    EvalConfig(metrics="pds", outdir=str(cfgdir), pert_col="target", control="non-targeting",
               input_type="lognorm").to_yaml(str(cfg))
    flagdir = tmp_path / "flagout"
    main(["run", "-ap", str(pp), "-ar", str(rp), "--config", str(cfg), "-o", str(flagdir)])
    assert (flagdir / "results.csv").exists()           # wrote to the flag dir
    assert not (cfgdir / "results.csv").exists()        # not the config dir


def test_cli_run_with_cache_dirs(tmp_path, synthetic_pair):
    from cell_eval2.cli import main
    pred, real = synthetic_pair
    pp, rp = tmp_path / "pred.h5ad", tmp_path / "real.h5ad"
    pred.write_h5ad(pp)
    real.write_h5ad(rp)
    cr, cp = tmp_path / "cr", tmp_path / "cp"
    # `minimal`, not `anndata`: as of #257 the anndata profile carries
    # `expr_mse_unbiased_capped_norm`, and this fixture draws every perturbation from ONE
    # distribution, so the panel has no real effect and `sum(expr_distance_unbiased)` is
    # non-positive -- which `run._derived_value` refuses by design, on the reference rather
    # than on the submission. The profile is incidental to what this test asserts;
    # `tests/test_run_derived_shape.py` covers a profile carrying the derived metric on a
    # panel that has an effect.
    main(["run", "-ap", str(pp), "-ar", str(rp), "--profile", "minimal",
          "--input-type", "lognorm", "--cache-real", str(cr), "--cache-pred", str(cp),
          "-o", str(tmp_path / "out")])
    assert (cr / "manifest.json").exists()
    assert list(cp.glob("results*.parquet"))  # content-addressed filename


def test_cli_prep_cache_real(tmp_path, synthetic_pair):
    from cell_eval2.cli import main
    _, real = synthetic_pair
    rp = tmp_path / "real.h5ad"
    real.write_h5ad(rp)
    cr = tmp_path / "cr"
    main(["prep-cache", "--side", "real", "--adata", str(rp),
          "--profile", "anndata", "--input-type", "lognorm",
          "--comparator", "lognorm", "--cache-real", str(cr)])
    # The `anndata` profile carries moment-consuming expression metrics (#198), so this writes the
    # MOMENTS artifact -- its own key prefix and its own `.moments.npz` extension, which is
    # exactly what keeps a moments run and a plain run from invalidating each other.
    assert list(cr.glob("pseudobulk_moments_lognorm*.moments.npz"))  # content-addressed filename
    assert not list(cr.glob("pseudobulk_lognorm*.npz"))


def test_cli_no_cache_strict_overrides_config(tmp_path, synthetic_pair, monkeypatch):
    import polars as pl
    import cell_eval2.cli as cli
    from cell_eval2 import EvalConfig
    pred, real = synthetic_pair
    pp, rp = tmp_path / "p.h5ad", tmp_path / "r.h5ad"
    pred.write_h5ad(pp)
    real.write_h5ad(rp)
    cfg_yaml = tmp_path / "c.yaml"
    # The stub below deliberately emits no metric rows. Keep its selection to a profile with
    # no derived metric: an empty result cannot honestly build a selected derived aggregate.
    EvalConfig(metrics="minimal", cache_strict=True, input_type="lognorm").to_yaml(str(cfg_yaml))
    seen = {}

    def fake_compute(p, r, *, config, de_pred=None, de_real=None, **kw):
        seen["strict"] = config.cache_strict
        return pl.DataFrame(schema={"perturbation": pl.Utf8, "metric": pl.Utf8, "value": pl.Float64})

    monkeypatch.setattr(cli, "compute_metrics", fake_compute)
    cli.main(["run", "-ap", str(pp), "-ar", str(rp), "--config", str(cfg_yaml),
              "--no-cache-strict", "-o", str(tmp_path / "o")])
    assert seen["strict"] is False  # explicit --no-cache-strict overrode config's true


def _capture_cfg(monkeypatch, tmp_path, argv_extra):
    """Run `cell-eval2 run` with compute_metrics stubbed; return the EvalConfig it built."""
    import cell_eval2.cli as cli
    seen = {}

    def fake_compute(p, r, *, config, de_pred=None, de_real=None, **kw):
        seen["cfg"] = config
        return pl.DataFrame(schema={"perturbation": pl.Utf8, "metric": pl.Utf8, "value": pl.Float64})

    monkeypatch.setattr(cli, "build_run_meta", lambda *args, **kwargs: {})
    monkeypatch.setattr(cli, "compute_metrics", fake_compute)
    # This helper tests config parsing while its compute stub returns no rows. Selecting a
    # derived metric would correctly make aggregation raise, obscuring the config assertion.
    cli.main(["run", "-ap", "x.h5ad", "-ar", "y.h5ad", "--profile", "minimal",
              "-o", str(tmp_path / "o"), *argv_extra])
    return seen["cfg"]


def test_cli_set_single_override(monkeypatch, tmp_path):
    cfg = _capture_cfg(monkeypatch, tmp_path, ["--set", "de.min_abs_log2fc=0.25"])
    assert cfg.de.min_abs_log2fc == 0.25 and isinstance(cfg.de.min_abs_log2fc, float)


def test_cli_set_multiple_and_yaml_value_types(monkeypatch, tmp_path):
    cfg = _capture_cfg(monkeypatch, tmp_path,
                       ["--set", "de.backend=pdex", "--set", "device=cpu", "--set", "target_sum=null"])
    assert cfg.de.backend == "pdex"     # str
    assert cfg.device == "cpu"          # str (top-level, no dedicated flag)
    assert cfg.target_sum is None       # YAML null -> None


def test_cli_set_overrides_config(monkeypatch, tmp_path):
    cfgfile = tmp_path / "c.yaml"
    EvalConfig(metrics="de", input_type="lognorm").to_yaml(str(cfgfile))  # de.p_adj_threshold defaults 0.05
    cfg = _capture_cfg(monkeypatch, tmp_path,
                       ["--config", str(cfgfile), "--set", "de.p_adj_threshold=0.1"])
    assert cfg.de.p_adj_threshold == 0.1  # --set wins over the config's 0.05


def test_cli_set_invalid_value_raises(tmp_path):
    with pytest.raises(SystemExit):  # fails DEParams validation via from_dict
        main(["run", "-ap", "x", "-ar", "y", "--set", "de.min_abs_log2fc=-1", "-o", str(tmp_path / "o")])


def test_cli_set_unknown_path_raises(tmp_path):
    with pytest.raises(SystemExit):
        main(["run", "-ap", "x", "-ar", "y", "--set", "de.nope=1", "-o", str(tmp_path / "o")])


def test_cli_set_malformed_raises(tmp_path):
    with pytest.raises(SystemExit):
        main(["run", "-ap", "x", "-ar", "y", "--set", "noequals", "-o", str(tmp_path / "o")])


def test_cli_set_malformed_yaml_value_raises(tmp_path):
    with pytest.raises(SystemExit):  # yaml.YAMLError -> clean SystemExit, not a traceback
        main(["run", "-ap", "x", "-ar", "y", "--set", "de.min_abs_log2fc={", "-o", str(tmp_path / "o")])


def test_cli_set_whole_section_target_raises(tmp_path):
    # scalar over a section (late crash) AND dict over a section (drops siblings) both rejected
    for bad in ("de=1", "de={min_abs_log2fc: 0.5}"):
        with pytest.raises(SystemExit):
            main(["run", "-ap", "x", "-ar", "y", "--set", bad, "-o", str(tmp_path / "o")])


def test_cli_ceiling_only_writes_only_ceiling_outputs(synthetic_counts_pair, tmp_path):
    """--ceiling without -ap is ceiling-only: the ceiling is a property of the real
    data, so no prediction is needed and no results.csv is produced."""
    _pred, real = synthetic_counts_pair
    rp = tmp_path / "real.h5ad"
    real.write_h5ad(rp)
    out = tmp_path / "out"
    main(["run", "-ar", str(rp), "--ceiling", "--profile", "anndata",
          "--pert-col", "target", "--control", "non-targeting", "-o", str(out)])
    assert (out / "ceiling_results.csv").exists()
    assert (out / "ceiling_agg.csv").exists()
    assert not (out / "results.csv").exists()  # main scoring skipped


def test_real_data_only_mode_still_requires_a_reason(capsys):
    """The message survived the `_build_parser` extraction (#276 part C): `main` no longer
    holds the `run` subparser, so the usage error is raised through `args._run_parser`.
    `test_cli_requires_pred_unless_ceiling` pins the exit code; this pins the text, which a
    silently-swallowed error would lose while still exiting 2."""
    with pytest.raises(SystemExit) as e:
        main(shlex.split("run -ar r.h5ad"))
    assert e.value.code == 2
    assert "real-data-only mode" in capsys.readouterr().err


def test_cli_requires_pred_unless_ceiling(tmp_path):
    """Omitting -ap without --ceiling leaves nothing to compute. argparse usage
    error: message + exit 2, not a traceback."""
    with pytest.raises(SystemExit) as exc:
        main(["run", "-ar", "x", "-o", str(tmp_path / "o")])
    assert exc.value.code == 2


def test_cli_ceiling_only_matches_ceiling_with_a_prediction(synthetic_counts_pair, tmp_path):
    """Supplying a prediction must not move the ceiling: same seed, same values."""
    pred, real = synthetic_counts_pair
    pp, rp = tmp_path / "pred.h5ad", tmp_path / "real.h5ad"
    pred.write_h5ad(pp)
    real.write_h5ad(rp)
    # `minimal`, not `anndata`: see test_cli_run_with_cache_dirs (#257, null panel).
    common = ["--ceiling", "--profile", "minimal", "--pert-col", "target",
              "--control", "non-targeting"]
    both, only = tmp_path / "both", tmp_path / "only"
    main(["run", "-ap", str(pp), "-ar", str(rp), *common, "-o", str(both)])
    main(["run", "-ar", str(rp), *common, "-o", str(only)])
    assert (both / "results.csv").exists()  # with a prediction the main run still happens
    a = pl.read_csv(both / "ceiling_agg.csv")
    b = pl.read_csv(only / "ceiling_agg.csv")
    assert a.equals(b)


def test_cli_ceiling_only_warns_on_flags_it_ignores(synthetic_counts_pair, tmp_path, caplog):
    """--de-* feed only the skipped main scoring, and the ceiling disables caching so
    --cache-* cannot apply either. Both warn instead of failing or going quiet."""
    _pred, real = synthetic_counts_pair
    rp = tmp_path / "real.h5ad"
    real.write_h5ad(rp)
    de = tmp_path / "de.parquet"
    pl.DataFrame({"target": ["GENE1"], "feature": ["g0"],
                  "log2_fold_change": [1.0], "p_adj": [0.01]}).write_parquet(de)
    out = tmp_path / "out"
    with caplog.at_level("WARNING"):
        main(["run", "-ar", str(rp), "--ceiling", "--profile", "anndata",
              "--pert-col", "target", "--control", "non-targeting",
              "--de-real", str(de), "--de-pred", str(de),
              "--cache-real", str(tmp_path / "cr"), "--cache-pred", str(tmp_path / "cp"),
              "--write-degenes", "-o", str(out)])
    msgs = " ".join(r.getMessage() for r in caplog.records if r.levelname == "WARNING")
    for flag in ("--de-pred", "--de-real", "--cache-real", "--cache-pred", "--write-degenes"):
        # #208: renamed from "ceiling-only" -- --lfc-nmae-ref reaches this branch too,
        # so a message naming the ceiling would be wrong for half the callers.
        assert f"{flag} is ignored in real-data-only mode" in msgs, flag
    assert (out / "ceiling_agg.csv").exists()  # still completes
    assert sorted(out.glob("de_*.parquet")) == []  # ...and wrote no DE tables


def test_cli_write_degenes_ceiling_warning_fires_exactly_once(synthetic_counts_pair, tmp_path,
                                                              caplog):
    """The --write-degenes warning belongs to the real-data-only BRANCH, not to the per-flag
    cache loop it sits next to.

    `test_cli_ceiling_only_warns_on_flags_it_ignores` supplies both --cache-* flags and only
    asserts the substring appears, so it would stay green with the warning nested one level
    deeper inside that loop -- emitting it once per cache flag, and not at all when neither is
    given. This pins the scope directly (codex checkpoint-2 P2): no cache flags at all, and
    exactly one matching record.
    """
    _pred, real = synthetic_counts_pair
    rp = tmp_path / "real.h5ad"
    real.write_h5ad(rp)
    out = tmp_path / "out"
    with caplog.at_level("WARNING"):
        main(["run", "-ar", str(rp), "--ceiling", "--profile", "anndata",
              "--pert-col", "target", "--control", "non-targeting",
              "--write-degenes", "-o", str(out)])
    hits = [r for r in caplog.records if r.levelname == "WARNING"
            and "--write-degenes is ignored in real-data-only mode" in r.getMessage()]
    assert len(hits) == 1, [r.getMessage() for r in caplog.records]
    assert sorted(out.glob("de_*.parquet")) == []


def test_cli_ceiling_does_not_overwrite_run_params(synthetic_counts_pair, tmp_path):
    """`run --ceiling` runs compute_metrics twice into one outdir. run_params.yaml must
    describe the MAIN run: the inner half-split run narrows metrics to the SB subset and
    disables caching, so an inherited outdir would leave provenance that reproduces the
    wrong evaluation."""
    pred, real = synthetic_counts_pair
    pp, rp = tmp_path / "pred.h5ad", tmp_path / "real.h5ad"
    pred.write_h5ad(pp)
    real.write_h5ad(rp)
    out, cache_r = tmp_path / "out", tmp_path / "cache_r"
    # `minimal`, not `anndata`: see test_cli_run_with_cache_dirs (#257, null panel).
    main(["run", "-ap", str(pp), "-ar", str(rp), "--ceiling", "--profile", "minimal",
          "--pert-col", "target", "--control", "non-targeting",
          "--cache-real", str(cache_r), "-o", str(out)])

    written = EvalConfig.from_yaml(str(out / "run_params.yaml"))
    assert written.metrics == "minimal"  # the profile the user asked for, not sb_run
    assert written.cache_real == str(cache_r)  # caching as the user configured it
    assert (out / "ceiling_agg.csv").exists()  # ceiling still produced


def test_run_lfc_nmae_ref_writes_both_files(tmp_path, monkeypatch):
    """The flag writes the two CSVs and prints the agg path. compute_lfc_nmae_reference is
    stubbed -- this test is about the CLI wiring, not the arithmetic (Task 4 covers that)."""
    import polars as pl
    from cell_eval2 import cli

    called = {}

    def _fake(real, *, config=None, seed=0, **kw):
        called["seed"] = seed
        res = pl.DataFrame({"perturbation": ["A"], "nmae_ref_raw": [0.9],
                            "nmae_ref_sqrt2": [0.6364], "n_gate": [12]})
        agg = pl.DataFrame({"statistic": ["mean"], "nmae_ref_raw": [0.9],
                            "nmae_ref_sqrt2": [0.6364], "n_perturbations": [1]})
        return res, agg

    monkeypatch.setattr("cell_eval2.lfc_nmae_ref.compute_lfc_nmae_reference", _fake)
    outdir = tmp_path / "out"
    cli.main(["run", "-ar", "y.h5ad", "-o", str(outdir),
              "--lfc-nmae-ref", "--lfc-nmae-ref-seed", "7"])
    assert (outdir / "lfc_nmae_ref.csv").exists()
    assert (outdir / "lfc_nmae_ref_agg.csv").exists()
    assert called["seed"] == 7


def test_run_lfc_nmae_ref_passes_de_real_through(tmp_path, monkeypatch, caplog):
    """--de-real must reach the reference (it supplies the full-real gate), and must NOT be
    reported as ignored -- that warning was written for --ceiling, which does recompute."""
    import polars as pl
    from cell_eval2 import cli

    seen = {}

    def _fake(real, *, config=None, seed=0, de_real=None, **kw):
        seen["de_real"] = de_real
        empty = pl.DataFrame({"statistic": ["mean"], "nmae_ref_raw": [0.9],
                              "nmae_ref_sqrt2": [0.64], "n_perturbations": [1]})
        return pl.DataFrame({"perturbation": ["A"], "nmae_ref_raw": [0.9],
                             "nmae_ref_sqrt2": [0.64], "n_gate": [12]}), empty

    monkeypatch.setattr("cell_eval2.lfc_nmae_ref.compute_lfc_nmae_reference", _fake)
    with caplog.at_level("WARNING"):
        cli.main(["run", "-ar", "y.h5ad", "-o", str(tmp_path / "o"),
                  "--de-real", "de.csv", "--lfc-nmae-ref"])
    assert seen["de_real"] == "de.csv"
    # NEGATIVE assertion: the "ignored in real-data-only mode" warning was written for
    # --ceiling, which recomputes DE on its own halves. It is false here and must be gone.
    assert "--de-real is ignored" not in " ".join(r.getMessage() for r in caplog.records)


def test_de_real_still_warned_as_ignored_for_ceiling_only(tmp_path, monkeypatch, caplog):
    """...and the warning must SURVIVE for the case it was written for, or narrowing it
    would have silently removed a correct message."""
    from cell_eval2 import cli

    monkeypatch.setattr(cli, "compute_ceiling",
                        lambda *a, **k: (__import__("polars").DataFrame(),
                                         __import__("polars").DataFrame()))
    with caplog.at_level("WARNING"):
        cli.main(["run", "-ar", "y.h5ad", "-o", str(tmp_path / "o2"),
                  "--de-real", "de.csv", "--ceiling"])
    assert "--de-real is ignored" in " ".join(r.getMessage() for r in caplog.records)


def _write_nondegenerate_baseline(user_agg_csv, out_csv):
    """A hand-written baseline for the anchor CLI tests.

    `baseline --emit tile` on `synthetic_pair_with_effect` produces a DEGENERATE
    `expr_mse_unbiased_capped_norm` (measured: -0.0150), and `build_generic_baseline`
    refuses to write it -- correctly. `--emit dispersed` is counts-only and raises on
    lognorm input (baseline.py:1042). The plan's stated fallback applies: hand-write a
    baseline with the same columns and plainly non-degenerate values rather than weaken the
    assertion, because what these tests must prove is that the CLI plumbs `--anchor` /
    `--anchor-cache` through to a frame carrying the three new columns.

    Each value is chosen from the metric's OWN policy so `is_degenerate` cannot fire:
    anchor-0 lower-is-better needs a finite positive base, anchor-1 higher-is-better needs
    a base below 1, and an anchorless metric needs |base| finite and positive.
    """
    import polars as pl

    from cell_eval2.catalog import CATALOG, _NAME_TO_CANONICAL

    user = pl.read_csv(user_agg_csv)
    row = {"statistic": ["mean"]}
    for name in user.columns:
        if name == "statistic":
            continue
        spec = CATALOG.get(_NAME_TO_CANONICAL.get(name, name))
        if spec is None:
            row[name] = [1.0]
            continue
        pol = spec.scoring
        if pol.anchor == 1.0 and pol.direction == "higher":
            row[name] = [0.1]          # D = 1 - 0.1 = 0.9
        else:
            row[name] = [1.0]          # D = 1.0 for anchor-0 and for anchorless
    pl.DataFrame(row).write_csv(out_csv)
    return out_csv


def test_run_anchor_writes_the_artifact(tmp_path, synthetic_pair_with_effect):
    """--profile, not --metrics (there is no --metrics flag). The effect-carrying fixture is
    required for any profile containing expr_mse_unbiased_capped_norm."""
    import os

    from cell_eval2.cli import main

    pred, real = synthetic_pair_with_effect
    pp, rp = str(tmp_path / "p.h5ad"), str(tmp_path / "r.h5ad")
    pred.write_h5ad(pp)
    real.write_h5ad(rp)
    out = str(tmp_path / "out")
    main(["run", "-ap", pp, "-ar", rp, "-o", out, "--anchor", "--anchor-splits", "2",
          "--profile", "anndata", "--pert-col", "target", "--control", "non-targeting",
          "--input-type", "lognorm"])
    for name in ("anchor_agg.parquet", "anchor_splits.parquet", "anchor_meta.json"):
        assert os.path.exists(os.path.join(out, name)), f"{name} not written"


def test_anchor_only_mode_needs_no_prediction_and_does_not_IndexError(
        tmp_path, synthetic_pair_with_effect):
    """`--anchor` alone, with no -ap: the real-data-only guard must accept it, and the
    "ignored flags" message must not index an empty list. Before this task cli.py:403 raised
    IndexError on `_on[0]` because _on knew only about --ceiling and --lfc-nmae-ref."""
    import os

    from cell_eval2.cli import main

    _pred, real = synthetic_pair_with_effect
    rp = str(tmp_path / "r.h5ad")
    real.write_h5ad(rp)
    out = str(tmp_path / "out")
    main(["run", "-ar", rp, "-o", out, "--anchor", "--anchor-splits", "2",
          "--profile", "anndata", "--pert-col", "target", "--control", "non-targeting",
          "--input-type", "lognorm", "--de-pred", "unused.csv"])
    assert os.path.exists(os.path.join(out, "anchor_meta.json"))


def test_the_anchor_is_CACHED_cold_then_warm_through_the_CLI(tmp_path,
                                                             synthetic_pair_with_effect,
                                                             monkeypatch):
    """`run --anchor` must go through `cached_anchor`, not straight to
    `compute_replicate_anchor`. Without this the cache is written by nothing and the cached
    door in `resolve_anchor` is unreachable code."""
    import cell_eval2.anchor as anchor_mod
    from cell_eval2.cli import main

    _pred, real = synthetic_pair_with_effect
    rp = str(tmp_path / "r.h5ad")
    real.write_h5ad(rp)
    cache = str(tmp_path / "cr")
    calls = {"n": 0}
    inner = anchor_mod.compute_replicate_anchor

    def counted(*a, **k):
        calls["n"] += 1
        return inner(*a, **k)

    monkeypatch.setattr(anchor_mod, "compute_replicate_anchor", counted)
    argv = ["run", "-ar", rp, "--anchor", "--anchor-splits", "2", "--profile", "anndata",
            "--pert-col", "target", "--control", "non-targeting",
            "--input-type", "lognorm", "--cache-real", cache]
    main([*argv, "-o", str(tmp_path / "out1")])
    assert calls["n"] == 1, "cold CLI run did not compute the anchor"
    main([*argv, "-o", str(tmp_path / "out2")])
    assert calls["n"] == 1, (
        "the second CLI run recomputed: `run --anchor` is not going through the cache, so "
        "nothing ever fills it and score's cached door can never open"
    )


def test_the_CACHED_door_opens_end_to_end(tmp_path, synthetic_pair_with_effect):
    """The whole loop, as one thing: `run --anchor --cache-real` fills the cache and records
    the descriptor in run_meta.json, then `score --anchor-cache` opens it and stamps
    anchor_source == "cached". Without this test the cached branch can be "implemented" and
    still be unreachable."""
    import json

    import polars as pl

    from cell_eval2.cli import main

    _pred, real = synthetic_pair_with_effect
    rp = str(tmp_path / "r.h5ad")
    real.write_h5ad(rp)
    cache = str(tmp_path / "cr")
    common = ["--profile", "anndata", "--pert-col", "target", "--control",
              "non-targeting", "--input-type", "lognorm"]
    user_out = str(tmp_path / "user")

    # ONE run that both scores the submission and builds the anchor -- the ordinary flow,
    # and the only one that can write the cache descriptor (params need base_seed/n_splits).
    main(["run", "-ap", rp, "-ar", rp, "-o", user_out, "--anchor", "--anchor-splits", "2",
          "--cache-real", cache, "--cache-strict", *common])
    meta = json.loads(open(f"{user_out}/run_meta.json").read())
    assert meta["anchor_cache"]["root"] == cache
    assert meta["anchor_cache"]["params"] and meta["anchor_cache"]["fingerprint"]

    base_agg = _write_nondegenerate_baseline(f"{user_out}/agg_results.csv",
                                             str(tmp_path / "baseline_agg.csv"))
    scored = str(tmp_path / "scored.csv")
    main(["score", "--user-agg", f"{user_out}/agg_results.csv",
          "--baseline-agg", base_agg,
          "--anchor-cache", cache, "-o", scored])          # NO --anchor: cached door only
    df = pl.read_csv(scored)
    assert set(df["anchor_source"].drop_nulls().to_list()) == {"cached"}
    assert df["from_replicate"].drop_nulls().len() > 0


def test_score_anchor_flag_is_wired_through(tmp_path, synthetic_pair_with_effect):
    """`score --anchor <dir>` must reach score_metrics as the supplied door, with the
    expectations derived from run_meta.json rather than from the anchor's own sidecar."""
    import polars as pl

    from cell_eval2.anchor import read_anchor
    from cell_eval2.cli import main

    _pred, real = synthetic_pair_with_effect
    rp = str(tmp_path / "r.h5ad")
    real.write_h5ad(rp)
    anchor_out = str(tmp_path / "anchor")
    common = ["--profile", "anndata", "--pert-col", "target", "--control",
              "non-targeting", "--input-type", "lognorm"]
    main(["run", "-ar", rp, "-o", anchor_out, "--anchor", "--anchor-splits", "2", *common])

    # A user run (writes agg_results.csv + run_meta.json) and a GENUINE baseline.
    #
    # NOT the user aggregate copied as its own baseline: `run -ap rp -ar rp` predicts the
    # real data perfectly, so expr_mae is exactly 0.0, and `_check_baseline_config`'s
    # sibling `_check_baseline_statistic` raises SystemExit on a decisive degenerate
    # baseline (cli.py:235) before score_metrics is ever called.
    user_out = str(tmp_path / "user")
    main(["run", "-ap", rp, "-ar", rp, "-o", user_out,
          "--cache-strict", *common])          # strict: the anchor gate requires it
    base_agg = _write_nondegenerate_baseline(f"{user_out}/agg_results.csv",
                                             str(tmp_path / "baseline_agg.csv"))
    _f, _s, meta = read_anchor(anchor_out)

    scored = str(tmp_path / "scored.csv")
    main(["score", "--user-agg", f"{user_out}/agg_results.csv",
          "--baseline-agg", base_agg,
          "--anchor", anchor_out, "-o", scored])
    cols = pl.read_csv(scored).columns
    assert "from_replicate" in cols
    assert "anchor_source" in cols and "anchor_digest" in cols
    assert meta["metric_names"]


def test_a_user_run_without_cache_strict_is_REFUSED_for_anchor_scoring(
        tmp_path, synthetic_pair_with_effect):
    """The gate is the strict content hash, and a default run does not have one
    (baseline.py:793). Refuse and name the flag rather than degrade to the metadata hash,
    under which two datasets with identical structure and different X are the same anchor."""
    from cell_eval2.cli import main

    _pred, real = synthetic_pair_with_effect
    rp = str(tmp_path / "r.h5ad")
    real.write_h5ad(rp)
    common = ["--profile", "anndata", "--pert-col", "target", "--control",
              "non-targeting", "--input-type", "lognorm"]
    anchor_out, user_out = str(tmp_path / "anchor"), str(tmp_path / "user")
    main(["run", "-ar", rp, "-o", anchor_out, "--anchor", "--anchor-splits", "2", *common])
    main(["run", "-ap", rp, "-ar", rp, "-o", user_out, *common])   # no --cache-strict
    base_agg = _write_nondegenerate_baseline(f"{user_out}/agg_results.csv",
                                             str(tmp_path / "baseline_agg.csv"))

    with pytest.raises(SystemExit, match="cache-strict|cache_strict"):
        main(["score", "--user-agg", f"{user_out}/agg_results.csv",
              "--baseline-agg", base_agg,
              "--anchor", anchor_out, "-o", str(tmp_path / "s.csv")])


def test_anchor_cache_without_a_descriptor_is_REFUSED_not_silently_ignored(
        tmp_path, synthetic_pair_with_effect):
    """A run that did not build the anchor records no `anchor_cache` block, so the cached
    door cannot be opened. Passing the flag anyway must RAISE, not fall through: without
    the refusal `score_metrics`' `anchor is not None or anchor_cache is not None` guard is
    False and the user gets NO from_replicate column and NO error -- a silent no-op on the
    flag they passed."""
    from cell_eval2.cli import main

    _pred, real = synthetic_pair_with_effect
    rp = str(tmp_path / "r.h5ad")
    real.write_h5ad(rp)
    common = ["--profile", "anndata", "--pert-col", "target", "--control",
              "non-targeting", "--input-type", "lognorm"]
    user_out = str(tmp_path / "user")
    # NO --anchor on this run, so run_meta.json carries no anchor_cache descriptor.
    main(["run", "-ap", rp, "-ar", rp, "-o", user_out, "--cache-strict", *common])
    base_agg = _write_nondegenerate_baseline(f"{user_out}/agg_results.csv",
                                             str(tmp_path / "baseline_agg.csv"))

    with pytest.raises(SystemExit, match="anchor_cache"):
        main(["score", "--user-agg", f"{user_out}/agg_results.csv",
              "--baseline-agg", base_agg,
              "--anchor-cache", str(tmp_path / "cr"), "-o", str(tmp_path / "s.csv")])


def test_a_FAILED_anchor_cannot_leave_a_STALE_run_meta_beside_a_NEW_aggregate(
        tmp_path, synthetic_pair_with_effect, monkeypatch):
    """`run_meta.json` is written twice on purpose, and this is the second window.

    The anchor block must run BEFORE the meta is written, because it mutates `run_meta` with
    the cache descriptor. But deferring the ONLY write past it means: the new
    `agg_results.csv` lands, the anchor raises, and in a REUSED output directory the previous
    run's `run_meta.json` is still sitting beside it -- so `score` would certify an aggregate
    that metadata does not describe (codex checkpoint-2 P1). The descriptor-free write after
    the aggregate closes it.
    """
    import json
    import os

    import cell_eval2.anchor as anchor_mod
    from cell_eval2.cli import main

    pred, real = synthetic_pair_with_effect
    pp, rp = str(tmp_path / "p.h5ad"), str(tmp_path / "r.h5ad")
    pred.write_h5ad(pp)
    real.write_h5ad(rp)
    out = str(tmp_path / "out")
    common = ["--profile", "anndata", "--pert-col", "target", "--control",
              "non-targeting", "--input-type", "lognorm"]

    # A FIRST, successful run leaves a real run_meta.json in the outdir...
    main(["run", "-ap", pp, "-ar", rp, "-o", out, *common])
    first = json.loads(open(os.path.join(out, "run_meta.json")).read())
    # ...which we mark so the assertion below cannot be satisfied by the stale file.
    stale = dict(first, source="STALE-FROM-THE-FIRST-RUN")
    with open(os.path.join(out, "run_meta.json"), "w") as fh:
        json.dump(stale, fh)

    def boom(*a, **k):
        raise RuntimeError("anchor production failed")

    monkeypatch.setattr(anchor_mod, "compute_replicate_anchor", boom)
    with pytest.raises(RuntimeError, match="anchor production failed"):
        main(["run", "-ap", pp, "-ar", rp, "-o", out, "--anchor", "--anchor-splits", "2",
              *common])

    after = json.loads(open(os.path.join(out, "run_meta.json")).read())
    assert after["source"] != "STALE-FROM-THE-FIRST-RUN", (
        "the failed --anchor run left the PREVIOUS run's run_meta.json beside the new "
        "agg_results.csv; `score` would certify an aggregate this metadata does not describe"
    )
    # The descriptor is absent, correctly: the anchor never completed.
    assert "anchor_cache" not in after
