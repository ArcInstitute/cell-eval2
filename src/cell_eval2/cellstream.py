"""Out-of-core scoring of cell-layout ``.shad`` archives on the ``partition_inmem`` engine.

``CellBatchSource`` adapts a ``cellstream.cell`` archive to the ``partition_inmem.PertBatchSource``
protocol (perturbation-complete AnnData batches of stored values), and ``score_cellstream``
orchestrates one context through ``_build_reference_streaming_core`` / ``score_piece``. All
normalization (counts CPM / lognorm expm1) happens downstream per ``cfg.input_type`` -- this
module adds none. See internal:docs/internal/superpowers/specs/2026-07-21-cellstream-scoring-unification-design.md.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from . import norm as _norm
from ._cell_archive import open_cell_store
from .cell_source import cell_fingerprint
from ._threads import resolve_gather_threads
from .h5ad_manifest import plan_pert_batches

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)


def _sanitize_cell_adata(adata):
    """Cast pandas string-extension (StringArray/ArrowStringArray) indices and columns in
    ``obs``/``var`` to numpy ``object`` dtype in place, and return ``adata``.

    cellstream reads ``obs``/``var`` from parquet, which yields pandas ``string`` (nullable
    StringArray) dtypes -- notably the ``var`` index (gene names). anndata's h5ad writer (used by
    the partition reference cache -- ``real_control.h5ad`` / ``pred_control.h5ad``,
    ``partition_inmem.py``) refuses to write these (``allow_write_nullable_strings=False``).
    h5ad-sourced AnnData carry plain ``object`` strings, so this makes cell-sourced batches match
    and keeps the shared downstream partition/gpudge path unchanged.
    """
    for names_attr in ("obs_names", "var_names"):
        idx = getattr(adata, names_attr)
        if pd.api.types.is_extension_array_dtype(idx.dtype):
            setattr(adata, names_attr, idx.astype(object))
    for df in (adata.obs, adata.var):
        for col in df.columns:
            dtype = df[col].dtype
            # Only the nullable/Arrow string extension arrays are the h5ad-write problem; leave
            # categoricals alone (anndata writes them fine, and is_string_dtype can be True for a
            # string-category dtype on some pandas versions) -- Gemini review.
            if (pd.api.types.is_extension_array_dtype(dtype)
                    and pd.api.types.is_string_dtype(dtype)
                    and not isinstance(dtype, pd.CategoricalDtype)):
                df[col] = df[col].astype(object)
    return adata


class CellBatchSource:
    """A ``PertBatchSource`` over one context of a ``layout='cell'`` cellstream archive.

    ``context=None`` (default; the only mode used in this work) treats the whole archive as one
    context. Reads stored values verbatim -- normalization is applied downstream.
    """

    def __init__(self, store_or_path, *, pert_col, control, context=None, context_col="context",
                 gather_threads=-1):
        self.store = open_cell_store(store_or_path)
        self.pert_col = str(pert_col)
        self.control = str(control)
        self.context = context
        # EvalConfig.gather_threads, stored VERBATIM (-1 = auto), NOT int()-coerced: coercion
        # would turn True/1.5 into 1, a new silent-serial path. It is resolved per gather --
        # against that gather's row count -- and resolve_gather_threads raises on a bad value.
        self.gather_threads = gather_threads
        self.stream_tag = f"{cell_fingerprint(self.store)}|ctx:{context}"
        obs = self.store.obs
        self._labels = obs[self.pert_col].to_numpy().astype(str)
        if context is None:
            self._rows_all = np.arange(int(self.store.n_obs), dtype=np.int64)
        else:
            ctx = obs[context_col].to_numpy().astype(str)
            self._rows_all = np.flatnonzero(ctx == str(context)).astype(np.int64)

    def read_control_block(self) -> "ad.AnnData":
        rows = self._rows_all[self._labels[self._rows_all] == self.control]
        if rows.size == 0:
            raise ValueError(
                f"no control rows for control={self.control!r} in pert_col={self.pert_col!r}; "
                f"available groups: {sorted(set(self._labels[self._rows_all].tolist()))}"
            )
        rows = np.sort(rows)
        return _sanitize_cell_adata(self.store.gather_rows_adata(
            rows, n_threads=resolve_gather_threads(rows.size, self.gather_threads)))

    def pert_universe(self) -> list[str]:
        """Sorted unique non-control perturbation labels of this context, without streaming a
        single batch -- the labels are already resident in ``self._labels`` from ``__init__``.

        The optional ``PertBatchSource.pert_universe()`` member (see that Protocol's docstring):
        used by ``_build_reference_streaming_core`` on a ground-truth cache HIT with no
        pseudobulk metric selected, where the ordinary stream pass never runs and nothing else
        would supply the perturbation universe the bundle manifest records.
        """
        labels = self._labels[self._rows_all]
        return sorted(set(labels.tolist()) - {self.control})

    def iter_pert_batches(self, mem_budget) -> "Iterator[tuple[list[str], ad.AnnData]]":
        ctx_rows = self._rows_all
        if ctx_rows.size == 0:      # empty context slice -> yield nothing (multi-context future)
            return
        ctx_labels = self._labels[ctx_rows]
        order = np.argsort(ctx_labels, kind="stable")
        sorted_labels = ctx_labels[order]
        # sorted_labels is already sorted -> find group boundaries in O(N) instead of the redundant
        # internal sort np.unique would do on this (up to millions-of-cells) array (Gemini review).
        change = np.concatenate(([True], sorted_labels[1:] != sorted_labels[:-1]))
        uniq = sorted_labels[change]
        first = np.flatnonzero(change)
        counts = np.diff(np.concatenate([first, [sorted_labels.size]]))
        rows_by = {u: ctx_rows[order[first[k]:first[k] + counts[k]]] for k, u in enumerate(uniq)}
        counts_by = dict(zip(uniq.tolist(), counts.tolist()))
        ctrl_n = int(counts_by.get(self.control, 0))
        sizes = [(str(u), int(c)) for u, c in zip(uniq.tolist(), counts.tolist())
                 if u != self.control]
        # Derive itemsize from the archive's materialized X dtype instead of hardcoding f32, so the
        # batch memory budget stays accurate for float64 archives (Copilot review; mirrors
        # h5ad_manifest.iter_h5ad_pert_batches deriving it from backed.X.dtype).
        x_dtype = getattr(self.store, "_x_out_dtype", None) or \
            self.store.manifest.get("value_dtype_on_disk", "float32")
        itemsize = np.dtype(x_dtype).itemsize
        for batch_perts in plan_pert_batches(sizes, n_genes=int(self.store.n_vars),
                                             itemsize=itemsize, control_cells=ctrl_n,
                                             mem_budget=mem_budget):
            rows = np.sort(np.concatenate([rows_by[p] for p in batch_perts]))
            yield batch_perts, _sanitize_cell_adata(self.store.gather_rows_adata(
                rows, n_threads=resolve_gather_threads(rows.size, self.gather_threads)))

    def close(self):
        self.store.close()


def _control_label(store, cfg) -> str:
    """The control/reference label. Prefer the archive's uniform ``control_value`` obs column
    (these archives carry it), else fall back to ``cfg.control``.

    #177: whichever path supplies it, the label is then CHECKED for membership in
    ``obs[cfg.pert_col]``. The fallback used to be silent, so a preset's label
    (``cell-eval-0.7.6`` ships ``control: non-targeting``) could be adopted for a dataset that
    labels its controls anything else, and the mistake surfaced downstream as empty or wrong
    metrics, or as a confusing error out of the DE path -- never at the point the wrong decision
    was taken. One membership check covers all three ways to get there: the column is absent,
    the column is present but non-uniform, or ``cfg.control`` is simply wrong.
    """
    obs = store.obs
    source = f"config.control={cfg.control!r}"
    label = str(cfg.control)
    if "control_value" in obs.columns:
        # uniform check in O(N) without sorting / string-materializing the whole column (Gemini)
        vals = obs["control_value"]
        if not vals.empty and vals.nunique(dropna=False) == 1:   # robust O(N) uniform check (Gemini)
            label = str(vals.iloc[0])
            source = "the archive's uniform control_value column"
        else:
            # Present but non-uniform: cellstream scores ONE context, so a control_value that
            # varies across rows means either a multi-context archive being scored as one or a
            # malformed write. Falling back is still the right resolution -- the membership check
            # below is the real guard -- but doing it silently hides a genuinely anomalous
            # archive, and this state cannot occur for anything a conforming writer produced.
            logger.warning(
                "cellstream: the archive's control_value column is present but NOT uniform "
                "(%d distinct values); falling back to %s. cellstream scores a single context, "
                "so a varying control_value suggests a multi-context or malformed archive.",
                int(vals.nunique(dropna=False)), source,
            )
    else:
        logger.info("cellstream: no control_value column in the archive; using %s", source)

    if cfg.pert_col not in obs.columns:
        raise ValueError(
            f"perturbation column {cfg.pert_col!r} missing from the archive's obs; "
            f"got {sorted(map(str, obs.columns))}"
        )
    present = obs[cfg.pert_col].astype(str)
    if not (present == label).any():
        available = sorted(set(present.unique()))
        raise ValueError(
            f"the resolved control label {label!r} (from {source}) does not appear in "
            f"obs[{cfg.pert_col!r}], so scoring would run against a control that does not "
            f"exist -- the reference block would be empty and the failure would surface "
            f"downstream instead of here (#177). {len(available)} label(s) present, e.g. "
            f"{available[:8]}."
        )
    return label


def cell_archive_input_type(archive, *, config=None, strict: bool = False,
                            peek_rows: int = 2000, allow_discrete: bool | None = None) -> str:
    """The expression space a cell-layout ``cellstream.cell`` archive's stored values are in.

    Public entry point for #179. A downstream caller imported ``_resolve_input_type_cell`` to do
    exactly
    this and then compared the answer against its own declared ``input_type`` itself; freezing an
    underscore-private symbol is a promise not expressed in the code, and the caller's real
    question was the AGREEMENT check rather than the peek. So this exposes the outcome:

    * ``strict=False`` (default) -- return the resolved space, ``"counts"`` or ``"lognorm"``.
    * ``strict=True`` -- additionally RAISE when it disagrees with ``config.input_type``.

    ``archive`` may be a path or an already-open cell store; a path is opened and closed here, so
    a caller needs neither ``open_cell_store`` nor any other private import. ``config`` defaults
    to ``EvalConfig()``; only ``input_type``, ``version``, ``allow_discrete`` and
    ``gather_threads`` are read.

    ⚠️ The peek always autodetects (``autodetect=True``), independent of
    ``config.autodetect_input_type``. That is deliberate for every cell/shard driver -- the
    accumulators take no ``input_type`` argument, so a declaration cannot be trusted here -- and
    it is why ``strict=True`` is a genuine check rather than a tautology: the resolved value comes
    from the DATA, and ``config.input_type`` is the claim being tested against it.

    ``allow_discrete=False`` makes the answer independent of ``config.allow_discrete``, which
    otherwise short-circuits the resolver to "counts" without inspecting a value -- pass it when the
    answer is feeding a safety decision rather than an ordinary read. ``None`` (default) uses the
    config's own value.

    ``strict`` is the CONSUMER's policy, not the drivers'. ``score_cellstream`` deliberately
    adopts the peeked value over the declaration (``piece_cfg`` below), because it scores whatever
    space the archive is in. A caller who wants a mismatch to be an error asks for it here.
    """
    from .config import EvalConfig
    cfg = EvalConfig() if config is None else config
    # strict=True is a SAFETY question, so it must not inherit the allow_discrete bypass:
    # `resolve_input_type` answers "counts" the moment that flag is set, without inspecting a value,
    # which would make the advertised agreement check pass for a genuine lognorm archive
    # (codex-review round 2). An explicit allow_discrete= still wins.
    if strict and allow_discrete is None:
        allow_discrete = False
    store, opened = (archive, False)
    if not hasattr(archive, "gather_rows_adata"):
        store, opened = open_cell_store(archive), True
    try:
        resolved = _resolve_input_type_cell(store, cfg, peek_rows=peek_rows,
                                            allow_discrete=allow_discrete)
    finally:
        if opened:
            store.close()
    if strict and resolved != cfg.input_type:
        raise ValueError(
            f"the archive's stored values resolve to input_type={resolved!r}, but the config "
            f"declares {cfg.input_type!r}. The cell-layout accumulators take no input_type "
            "argument and treat stored values according to the resolved space, so scoring on "
            "would silently normalize in the wrong one. Re-declare input_type, or write the "
            "archive in the declared space."
        )
    return resolved


def _resolve_input_type_cell(store, cfg, *, peek_rows: int = 2000,
                             allow_discrete: bool | None = None) -> str:
    """Peek-autodetect counts vs lognorm from a row block of the archive (mirrors
    ``h5ad_manifest._resolve_input_type_h5ad``). The downstream ``_build_reference_streaming_core`` /
    ``score_piece`` validation (``_validate_input_once``) is the guard against a mis-detected type.

    Kept as the private name #179 records an external consumer for; ``cell_archive_input_type``
    above is the supported entry point, and takes a path as well as an open store.

    ``allow_discrete=None`` uses ``cfg.allow_discrete`` -- the behaviour every existing caller
    wants, because for them this is an ORDINARY type resolution and ``allow_discrete`` is a
    legitimate policy for reading an ambiguous integer matrix. A SAFETY GATE must pass ``False``:
    ``resolve_input_type`` returns "counts" the moment the flag is set, without looking at a single
    value (its ``if allow_discrete: return "counts"`` short-circuit), so honouring it would let a
    genuinely lognorm archive declare its way through a raw-counts guard (codex-review). Splitting
    the two here rather than changing the resolver keeps ``score_cellstream`` -- which legitimately
    handles lognorm and threads the resolved type downstream -- unaffected.
    """
    from . import norm as _norm
    stop = min(int(store.n_obs), int(peek_rows))
    peek = store.gather_rows_adata(
        np.arange(stop, dtype=np.int64),
        n_threads=resolve_gather_threads(stop, cfg.gather_threads))
    return _norm.resolve_input_type(
        peek, declared=cfg.input_type, version=cfg.version,
        allow_discrete=cfg.allow_discrete if allow_discrete is None else allow_discrete,
        autodetect=True,
    )


def _uniform_obs(store, col, default) -> str:
    """The single value of ``obs[col]`` if the archive is uniform in it (these archives are
    single-dataset/context); else ``default``. Used only to tag the output frame."""
    obs = store.obs
    if col in obs.columns:
        vals = obs[col]
        if not vals.empty and vals.nunique(dropna=False) == 1:   # robust O(N) uniform check (Gemini)
            return str(vals.iloc[0])
    return default


#: Everything one ``score_cellstream(write_de=True)`` call publishes into ``outdir``, in order --
#: plus ``_GT_SIDECAR``, staged right after the two DE tables, when the ground-truth cache is on.
_WRITE_DE_OUTPUTS = ("de_real.parquet", "de_pred.parquet",
                     "per_pert.parquet", "per_context.parquet", "overall.parquet")

#: The ground-truth DE cache's provenance sidecar (docs/ground_truth_cache.md §10): published by a
#: ``write_de=True`` run only while that cache is on. With the cache off, ``outdir`` receives
#: exactly ``_WRITE_DE_OUTPUTS``.
_GT_SIDECAR = "ground_truth_de_cache.json"


def _producer_dict() -> dict:
    """``{"cell_eval2": ..., "gpudge": ... | None}`` -- the producer block of the
    ``ground_truth_de_cache.json`` sidecar."""
    import importlib.metadata

    from . import __version__

    try:
        gpudge_version = importlib.metadata.version("gpudge")
    except importlib.metadata.PackageNotFoundError:
        gpudge_version = None
    return {"cell_eval2": __version__, "gpudge": gpudge_version}


class _StagedOutputs:
    """Files staged under temporary names in ``outdir``, then published TOGETHER with
    rollback-on-error -- the discipline ``run._write_de_tables`` uses for the in-memory
    ``--write-degenes`` pair, extended to every file one ``score_cellstream(write_de=True)`` call
    emits (``_WRITE_DE_OUTPUTS``, plus ``_GT_SIDECAR`` while the ground-truth cache is on). A
    failure anywhere before or during publication therefore never
    leaves this run's DE tables beside another run's score files, or the reverse.

    ``stage(name, write)`` registers ``<outdir>/.<name>.tmp.<tag>`` BEFORE calling ``write(tmp)``,
    so a partial temp file left by a failing writer is still cleaned up. ``publish()`` moves each
    pre-existing destination aside as ``<dest>.bak.<tag>``, ``os.replace``s every temp into place
    in staging order, and removes the backups only once all have published. On ANY exception
    before the commit point, ``KeyboardInterrupt`` included, it unlinks every destination it
    published that had no backup, removes the remaining temps, and restores every backup. Each
    step is recorded BEFORE the rename it describes, and the rollback reads off the file system
    whether the rename happened -- a destination now carrying the staged file's (st_dev, st_ino)
    was published -- so an interruption between a rename and its bookkeeping is still undone,
    and a temp that vanished before its rename is not mistaken for a published one. A restore that itself fails
    raises a ``RuntimeError`` naming the stuck ``.bak`` path(s) -- an intact file nobody can find
    is lost in practice, so nothing on that path may be best-effort. ``discard()`` removes the
    temps of a staging that will never publish.

    The publish COMMITS when its last rename completes. Nothing after that is rolled back -- not
    even an interrupt that lands before ``publish()`` returns -- and removing the backups is best
    effort: one that cannot be removed, or that an interrupt leaves behind, stays beside the new
    outputs as a ``<dest>.bak.<tag>`` copy of the previous one (logged; safe to delete).

    **The guarantee is rollback-on-error, not atomic visibility as a group**: a reader in another
    process can see some new files beside old ones while ``publish()`` runs, and a crash between
    two ``os.replace`` calls leaves exactly that mix on disk. Closing either needs a
    directory-level commit no filesystem here offers. Recovery also relies on the filesystem
    behaving while it runs: recognising the commit needs readable metadata, and a rollback that
    itself fails or is interrupted can leave files to inspect by hand (named in its
    ``RuntimeError`` when it knows). That residual risk is accepted, not coded around.
    """

    def __init__(self, outdir):
        import os
        from uuid import uuid4

        self.check_destinations(outdir, ())
        os.makedirs(outdir, exist_ok=True)
        self.outdir = outdir
        self.tag = f"{os.getpid()}.{uuid4().hex}"
        self._staged: list[tuple[str, str]] = []    # (tmp, dest), in publish order

    @staticmethod
    def check_destinations(outdir, names):
        """Refuse an ``outdir`` that exists and is not a directory, and any destination in it
        that exists and is not a regular file -- the door-checks ``run._write_de_tables`` makes.
        Publishing over a directory would move it aside and leave it behind as backup debris
        that cleanup cannot remove, and a failed publish could not put it back."""
        import os

        # lexists, not exists: a DANGLING symlink is an existing non-directory entry too.
        if os.path.lexists(outdir) and not os.path.isdir(outdir):
            raise ValueError(
                f"outdir {outdir!r} exists and is not a directory, so write_de cannot publish "
                "there. Remove it or choose another output directory."
            )
        for name in names:
            dest = os.path.join(outdir, name)
            # islink: a symlink is refused even when it points at a regular file -- publishing
            # would replace the LINK with a new file and leave its target untouched.
            if os.path.islink(dest) or (os.path.lexists(dest) and not os.path.isfile(dest)):
                raise ValueError(
                    f"{dest!r} exists and is not a regular file, so write_de cannot publish "
                    "there. Remove it or choose another output directory."
                )

    @staticmethod
    def _rm(path):
        import os

        try:
            if os.path.lexists(path):
                os.remove(path)
        except OSError:
            pass  # never let cleanup mask the original error

    def stage(self, name, write):
        """Write ``name`` to its temp path through ``write(tmp_path)``."""
        import os

        self.check_destinations(self.outdir, (name,))
        dest = os.path.join(self.outdir, name)
        tmp = os.path.join(self.outdir, f".{name}.tmp.{self.tag}")
        self._staged.append((tmp, dest))   # registered BEFORE any I/O: a partial tmp is cleaned
        write(tmp)

    def discard(self):
        for tmp, _ in self._staged:
            self._rm(tmp)

    def publish(self) -> list[str]:
        import os

        backups: list[tuple[str, str]] = []           # (dest, bak), recorded before the move-aside
        published: list[tuple[str, str, tuple]] = []  # (tmp, dest, identity), before the replace
        try:
            for tmp, dest in self._staged:
                st = os.stat(tmp)   # the staged file's identity: a rename keeps (st_dev, st_ino)
                if os.path.lexists(dest):
                    bak = f"{dest}.bak.{self.tag}"
                    backups.append((dest, bak))
                    os.replace(dest, bak)
                published.append((tmp, dest, (st.st_dev, st.st_ino)))
                os.replace(tmp, dest)
        except BaseException as exc:
            if (len(published) == len(self._staged)
                    and self._landed(*published[-1][1:]) is True):
                # The last rename completed before the exception (an interrupt landing just
                # after it): the publish has COMMITTED and is not rolled back.
                self._remove_backups(backups)
                raise
            self._roll_back(backups, published, exc)
            raise
        # COMMIT POINT: the last rename completed, so every destination holds its new file.
        self._remove_backups(backups)
        return [dest for _, dest, _ident in published]

    def _remove_backups(self, backups):
        """After the commit, remove the moved-aside previous outputs. Best effort: a backup that
        cannot be removed is a stale copy, not a failure of the publish, so it is logged rather
        than raised (an interrupt here likewise leaves the remaining ones behind)."""
        import os

        left = []
        for _, bak in backups:
            try:
                if os.path.lexists(bak):
                    os.remove(bak)
            except OSError:
                left.append(bak)
        if left:
            logger.warning(
                "score_cellstream published into %r but could not remove the backup(s) of the "
                "previous outputs: %s -- stale copies, safe to delete", self.outdir,
                ", ".join(left))

    @staticmethod
    def _landed(dest, identity):
        """Whether ``dest`` IS the staged file now -- by identity, not by the temp's absence: a
        temp that vanished before its rename must not read as published. ``None`` when ``dest``
        exists but cannot be inspected: the state is UNKNOWN, which is never read as committed
        and is reported, not guessed, by the rollback."""
        import os

        try:
            st = os.stat(dest, follow_symlinks=False)
        except FileNotFoundError:
            return False
        except OSError:
            return None
        return (st.st_dev, st.st_ino) == identity

    def _roll_back(self, backups, published, exc):
        import os

        # ⚠️ Nothing below may be best-effort except where a backup already covers the loss (see
        # run._write_de_tables, whose rollback this mirrors). Whether a recorded rename HAPPENED
        # is read off the file system: a destination holding the staged file's identity was
        # published, and a backup that does not exist was never moved aside.
        states = [(dest, self._landed(dest, ident)) for _tmp, dest, ident in published]
        done = [dest for dest, landed in states if landed]
        backed_up = {dest for dest, _ in backups}
        # A destination whose state cannot be read is restored by its backup when it has one;
        # without one, nobody can say whether it holds this run's file, so it is REPORTED.
        unknown = [dest for dest, landed in states if landed is None and dest not in backed_up]
        stuck = []
        for dest in done:
            if dest in backed_up:
                continue    # restored by its own backup just below
            try:
                os.unlink(dest)
            except OSError:
                stuck.append(dest)
        self.discard()
        unrestored = []
        for dest, bak in backups:
            try:
                os.lstat(bak)
            except FileNotFoundError:
                continue    # never moved aside: dest still holds the original
            except OSError:
                pass        # unreadable, so it may hold the previous output: try the restore
            try:
                os.replace(bak, dest)
            except OSError:
                unrestored.append((dest, bak))
        problems = []
        if unknown:
            problems.append("the state of these could not be read, so they may hold the new "
                            "file: " + ", ".join(repr(p) for p in unknown))
        if stuck:
            problems.append("these were published and could not be removed: "
                            + ", ".join(repr(p) for p in stuck))
        if unrestored:
            pub = set(done)
            problems.append("the previous contents survive only as " + "; ".join(
                f"{bak!r}, which belongs at {dest!r} ("
                + ("that path currently holds the newly written file"
                   if dest in pub else "that path is now missing") + ")"
                for dest, bak in unrestored))
        if problems:
            raise RuntimeError(
                f"score_cellstream failed to publish into {self.outdir!r} and could not fully "
                f"roll back -- {'; and '.join(problems)}. Resolve by hand before trusting "
                "anything in that directory."
            ) from exc


def _stage_de_outputs(staging, *, real_de_path, piece_paths, real_de_sha256=None) -> None:
    """Stage ``de_real.parquet`` and ``de_pred.parquet`` into ``staging`` (a ``_StagedOutputs``).

    Both go through ``de_export``'s pinned writer contract, one source row group resident at a
    time (never a whole-table ``pl.concat`` -- a production ``de_pred`` is ~16.5M rows):

    * ``de_real.parquet`` -- the reference builder's ``real_de.parquet`` rewritten row group by
      row group (``merge_de_pieces([real_de_path])``): the same rows, values, column types and
      order, re-encoded under the contract. The reference bundle's own file is left as it was.
      With ``real_de_sha256`` -- the ground-truth DE cache supplied the table, so
      ``real_de_path`` links to a committed entry that the same pinned writer produced -- it is
      instead a BYTE COPY of that entry, checked against ``real_de_sha256`` as it is copied
      (``_copy_verified``). Re-encoding would replace the entry's writer footer whenever the entry
      came from another pyarrow version, and the published file would then not be the bytes the
      ``ground_truth_de_cache.json`` sidecar names.
    * ``de_pred.parquet`` -- the per-piece pred DE tables merged **in the order given**, so the
      caller must pass batch generation order, never a glob-then-sort over piece file names
      (lexicographic sort would put ``ctx_10`` before ``ctx_2``).
    """
    from . import de_export

    if real_de_sha256 is None:
        # report=False: nothing here reads the reports, and each one rehashes its whole output.
        staging.stage("de_real.parquet",
                      lambda tmp: de_export.merge_de_pieces([real_de_path], tmp, report=False))
    else:
        staging.stage("de_real.parquet",
                      lambda tmp: _copy_verified(real_de_path, tmp, real_de_sha256))
    staging.stage("de_pred.parquet",
                  lambda tmp: de_export.merge_de_pieces(piece_paths, tmp, report=False))


def _copy_verified(src, dst, sha256_hex) -> None:
    """Copy ``src`` to ``dst`` byte for byte, hashing as it streams, and raise
    ``GroundTruthCacheCorruptError`` unless the bytes are the ones ``sha256_hex`` names -- the
    ground-truth cache entry changed after this run validated it. One read of ``src`` serves both
    the copy and the check."""
    import hashlib
    import os

    from .ground_truth_cache import GroundTruthCacheCorruptError

    h = hashlib.sha256()
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        for chunk in iter(lambda: fin.read(1 << 20), b""):
            h.update(chunk)
            fout.write(chunk)
    if h.hexdigest() != sha256_hex:
        raise GroundTruthCacheCorruptError(
            f"ground-truth cache entry table {os.path.realpath(src)!r} changed after this run "
            f"validated it: SHA-256 {h.hexdigest()} != {sha256_hex}. Nothing was published; "
            "inspect the entry with `cell-eval2 cache inspect`.")


def _check_root_outside_outputs(root, outdir) -> None:
    """Refuse a ground-truth cache root that IS, or lies inside, one of the six files a cache-on
    ``write_de`` run publishes into ``outdir`` (``_WRITE_DE_OUTPUTS`` and ``_GT_SIDECAR``),
    whatever this run's ``write_de``. The readwrite store would create a directory there, and the
    clash would surface only at publication, after the whole scoring run. Both sides are compared
    after ``realpath``, so a symlinked ``outdir`` cannot hide the alias. ``outdir`` itself, or any
    other directory in it, is a valid root: ``ground-truth-deg/`` coexists with the outputs. A
    relative root is left to the store's constructor, which refuses it."""
    import os

    expanded = os.path.expanduser(os.fspath(root))
    if not os.path.isabs(expanded):
        return
    root_real = os.path.realpath(expanded)
    for name in (*_WRITE_DE_OUTPUTS, _GT_SIDECAR):
        dest = os.path.realpath(os.path.join(os.fspath(outdir), name))
        try:
            inside = os.path.commonpath([root_real, dest]) == dest
        except ValueError:   # different drives (Windows): the two paths cannot overlap
            continue
        if inside:
            raise ValueError(
                f"ground_truth_cache {os.fspath(root)!r} is, or lies inside, {dest!r}, a file "
                "score_cellstream publishes into outdir; choose a root elsewhere (outdir itself, "
                "or another directory in it, is fine)")


def score_cellstream(pred, real, *, config=None, mem_budget, outdir=None, write_de=False,
                     ground_truth_cache=None, ground_truth_cache_policy="readwrite"):
    """Score a pair of cell-layout ``.shad`` archives (single context) on the ``partition_inmem``
    engine. Handles counts and lognorm (via the shared in-memory ``compute_de`` + ``_side_bulks``).
    Returns a ``ScoreResult`` (per_pert / per_context / overall). GPU-only (gpudge backend).

    ``write_de`` (#387; hardened in the #389 split -- ``de_export``): also emit the raw gene-level
    DE tables to ``outdir`` as ``de_real.parquet`` + ``de_pred.parquet``. ``de_real`` is the
    reference builder's full real DE table; ``de_pred`` is the per-piece pred DE tables merged in
    BATCH GENERATION ORDER (the order each ``ctx_{k}`` piece was scored in this loop, never a
    glob-then-sort over piece file names -- at 12+ batches lexicographic sort would put ``ctx_10``
    before ``ctx_2``). Both are written through ``de_export``'s pinned contract (zstd level 3,
    statistics, 250k-row row groups) one source row group at a time, never a ``pl.concat`` of
    everything -- a production ``de_pred`` is ~16.5M rows. All five output files (the two DE tables
    and the three ``ScoreResult`` frames) are staged first and then published together,
    rollback-on-error (``_StagedOutputs`` -- NOT atomic visibility as a group), so a failure before
    or during publication leaves ``outdir`` as it was. Requires ``outdir`` and at least one DE
    metric in the selection. Off by default, so existing callers are unaffected.

    ``ground_truth_cache`` / ``ground_truth_cache_policy`` (the ground-truth DE cache --
    ``docs/ground_truth_cache.md``): ``ground_truth_cache=None`` (the default) leaves the cache
    OFF, and the run is the cache-free one -- nothing fingerprinted, no cache I/O, no sidecar.
    With a root (absolute after ``os.path.expanduser``) the real DE table comes from a
    content-addressed, immutable entry under ``<root>/ground-truth-deg/v1/``: ``"readwrite"``
    (default) reads a validated entry or computes and publishes one under the entry's lock;
    ``"readonly"`` only reads, and a miss raises ``GroundTruthCacheMissError``. The REAL archive
    is fingerprinted before any DE runs, so one without a strong ``payload_sha256`` fails fast
    (``GroundTruthFingerprintError``); a committed entry that fails validation raises
    ``GroundTruthCacheCorruptError`` and is never overwritten. With ``write_de=True`` the publish
    carries a sixth file, ``ground_truth_de_cache.json``, recording the lookup's outcome, and
    ``de_real.parquet`` is a verified byte copy of the cache entry. Neither argument is an
    ``EvalConfig`` field, so neither can move a scoring digest.
    """
    import json
    import os
    import tempfile
    from dataclasses import replace
    from time import perf_counter

    import polars as pl

    from .catalog import CATALOG, resolve_metrics
    # Imported HERE, per call, not at module load: a test patches de_compute._resolve_backend and
    # must reach this binding (partition_inmem binds its own copy at import time).
    from .de_compute import _resolve_backend
    from .ground_truth_cache import (
        CACHE_POLICIES,
        GroundTruthDEGCache,
        GroundTruthDEGCacheRequest,
        GroundTruthFingerprintError,
        build_real_de_semantics,
        fingerprint_ground_truth_cell_archive,
        resolve_de_precision,
    )
    from .partition import aggregate_partials
    from .partition_inmem import (
        RealDEArtifact,
        _RefBundle,
        _build_pred_control_reference_core,
        _build_reference_streaming_core,
        _require_partition_config,
        score_piece,
    )
    from .run import _cache_device, _resolve_config
    from .h5ad_manifest import _assemble_score_result, _nsig_names

    # Fail fast, before any archive I/O, on the one combination that cannot be honoured: the DE
    # tables have nowhere to land without an outdir (#387).
    if write_de and not outdir:   # None, and "" -- which os.makedirs would refuse only at the end
        raise ValueError(
            "write_de=True requires outdir; de_real.parquet/de_pred.parquet are written there"
        )
    # The ground-truth DE cache's two arguments, checked with the other pure argument checks and
    # before any archive I/O (spec §4). An unknown policy is refused whether or not the cache is
    # on, and "readonly" with no root is refused rather than ignored: a cache knob that silently
    # does nothing is the hazard that kept these out of EvalConfig.
    if ground_truth_cache_policy not in CACHE_POLICIES:
        raise ValueError(
            f"ground_truth_cache_policy must be one of {CACHE_POLICIES}, got "
            f"{ground_truth_cache_policy!r}"
        )
    if ground_truth_cache is None and ground_truth_cache_policy == "readonly":
        raise ValueError(
            "ground_truth_cache_policy='readonly' needs ground_truth_cache (the cache root); "
            "without a root the cache is off and a readonly policy has nothing to read"
        )
    gt_cache_on = ground_truth_cache is not None

    # Fail fast + normalize the AUC floor (spec §5.3): the core builders each guard internally,
    # but validating once up front raises before any archive I/O when the config is ineligible
    # (e.g. fdr_scope!='per_pert', or no gpudge backend -- score_cellstream is GPU-only).
    cfg = _require_partition_config(_resolve_config(config, {}))
    if cfg.pert_col == "target":
        cfg = replace(cfg, pert_col="perturbation")
    # Before any archive is read (#395): this driver carries no cells.
    from .run import _refuse_cell_metrics_up_front
    cfg = _refuse_cell_metrics_up_front(cfg, driver="score_cellstream (partitioned in-memory)")

    # Fail fast on the other unhonourable write_de combination: a metric selection that resolves
    # to no DE-kind metric would score_piece every batch, write zero {piece_id}__de_pred.parquet
    # files, and only THEN fail deep inside the with-block below. Moved here (ground-truth DEG
    # cache series), before any archive I/O; the defensive check further down stays, for a selection
    # that resolves a DE metric yet still yields zero pieces.
    if write_de:
        _names, _missing = resolve_metrics(cfg.metrics, version=cfg.version)
        if not any(CATALOG[n].kind == "de" for n in _names):
            raise ValueError(
                "write_de=True requires at least one DE metric in the metric selection "
                f"(resolved metrics={_names!r} contain no DE-kind metric); de_real.parquet/"
                "de_pred.parquet cannot be produced without one"
            )
        # And refuse an outdir it could not publish into (a non-directory outdir, or a
        # destination that is not a regular file) now, not after the whole scoring run.
        _StagedOutputs.check_destinations(
            outdir, _WRITE_DE_OUTPUTS + ((_GT_SIDECAR,) if gt_cache_on else ()))
    # A root that is, or lies inside, one of outdir's output files is refused before the
    # readwrite store would create a directory there (spec §4).
    if gt_cache_on and outdir is not None:   # the publication path's own condition
        _check_root_outside_outputs(ground_truth_cache, outdir)

    # The store is built after every pure check, and before the real archive is opened, so a bad
    # root fails with no archive I/O (spec §4, D2): a relative root raises
    # GroundTruthContractError, a readonly root that does not exist raises
    # GroundTruthCacheFilesystemError, and readwrite creates <root>/ground-truth-deg/v1/.
    gt_store = (GroundTruthDEGCache(os.fspath(ground_truth_cache), policy=ground_truth_cache_policy)
                if gt_cache_on else None)

    real_store = open_cell_store(real)
    try:
        control = _control_label(real_store, cfg)
        # Fingerprint the REAL archive before any DE runs (cache on only): an archive without a
        # strong payload_sha256 must fail HERE -- fingerprint_ground_truth_cell_archive never
        # falls back to a structural fingerprint -- not after a full real-side DE pass.
        gt_fp = fingerprint_seconds = None
        if gt_cache_on:
            _t0 = perf_counter()
            gt_fp = fingerprint_ground_truth_cell_archive(real_store, pert_col=cfg.pert_col)
            fingerprint_seconds = perf_counter() - _t0
        real_itype = _resolve_input_type_cell(real_store, cfg)
        dataset_tag = _uniform_obs(real_store, "dataset", "cellstream")
        context_tag = _uniform_obs(real_store, "context", "all")
    finally:
        real_store.close()
    pred_store = open_cell_store(pred)
    try:
        pred_itype = _resolve_input_type_cell(pred_store, cfg)
    finally:
        pred_store.close()
    comparator = _norm.resolve_comparator(
        version=cfg.version,
        pred_input_type=pred_itype,
        real_input_type=real_itype,
    )

    # The real-DE provider, handed to the reference builder ONLY while the cache is on (spec §6,
    # ruling 6). It keys the lookup with the semantic document built from what this run resolved,
    # then either reads a validated entry (a HIT: the builder's real-DE pass never runs) or
    # computes the table once under the entry's lock and publishes it (a MISS). gt_result carries
    # the outcome back for the log line and the sidecar.
    gt_result = None

    def _real_de_provider(*, cfg_resolved, control_ad, compute):
        nonlocal gt_result
        # A contract violation here (a missing producer distribution, de.replicate_col set) is
        # fatal: the cache cannot key an entry without the semantic document.
        sem = build_real_de_semantics(
            cfg_resolved, control=cfg_resolved.control, input_type=cfg_resolved.input_type,
            target_sum=cfg_resolved.target_sum, device=_cache_device(cfg_resolved),
            de_backend=_resolve_backend(cfg_resolved.de.backend), native_gpu_normalize=True,
            precision=resolve_de_precision(
                control_ad.X.dtype, input_type=cfg_resolved.input_type,
                native_gpu_normalize=True, target_sum=cfg_resolved.target_sum,
            ),
        )
        req = GroundTruthDEGCacheRequest(fingerprint=gt_fp, semantics=sem)
        # readonly + a miss raises GroundTruthCacheMissError and never calls compute().
        res = gt_store.get_or_compute(req, compute)
        gt_result = replace(res, fingerprint_seconds=fingerprint_seconds)
        return RealDEArtifact(
            path=res.table_path,
            provenance={"kind": res.status, "cache_id": res.cache_id,
                        "entry_path": res.entry_dir, "de_real_sha256": res.de_real_sha256,
                        "de_real_rows": res.de_real_rows},
        )

    nsig_spearman, nsig_real, nsig_pred = _nsig_names(cfg)
    staging = None   # a _StagedOutputs once write_de has staged its DE tables
    # ONE cleanup scope from here to the publish: the temporary directories' teardown and the
    # score-frame assembly both run after the DE tables are staged, so a failure (or an
    # interrupt) in either must still remove the staged temps from outdir.
    try:
        with tempfile.TemporaryDirectory() as ref_dir, tempfile.TemporaryDirectory() as parts_dir, \
                tempfile.TemporaryDirectory() as de_parts_dir:
            real_src = CellBatchSource(real, pert_col=cfg.pert_col, control=control,
                                       gather_threads=cfg.gather_threads)
            try:
                if gt_cache_on:
                    # The builder streams from real_src's OWN open of the archive, not the handle
                    # fingerprinted above. Re-fingerprint it: an archive replaced in between must
                    # neither key a new entry nor hit an old one under the wrong identity.
                    fp_again = fingerprint_ground_truth_cell_archive(real_src.store,
                                                                     pert_col=cfg.pert_col)
                    if fp_again.sha256 != gt_fp.sha256:
                        raise GroundTruthFingerprintError(
                            f"the real archive {os.fspath(real)!r} changed while this run read "
                            f"it: ground-truth fingerprint {gt_fp.sha256} at the first open, "
                            f"{fp_again.sha256} at the second. Nothing was computed or cached.")
                _build_reference_streaming_core(
                    real_src, config=cfg, cache_dir=ref_dir, mem_budget=mem_budget,
                    input_type=real_itype, native_gpu_normalize=True, comparator=comparator,
                    real_de_provider=_real_de_provider if gt_cache_on else None)
            finally:
                real_src.close()
            if gt_cache_on:
                if gt_result is None:   # a builder that skipped the provider would bypass the cache
                    raise RuntimeError(
                        "the ground-truth cache is on but the reference builder never called the "
                        "real-DE provider")
                logger.info(
                    "cellstream: ground-truth DE %s (policy=%s cache_id=%s... lock_wait=%.2fs "
                    "compute=%s)", gt_result.status, gt_result.policy, gt_result.cache_id[:12],
                    gt_result.lock_wait_seconds,
                    "n/a" if gt_result.compute_seconds is None
                    else f"{gt_result.compute_seconds:.2f}s")

            if cfg.control_source == "pred":
                pc_src = CellBatchSource(pred, pert_col=cfg.pert_col, control=control,
                                         gather_threads=cfg.gather_threads)
                try:
                    _build_pred_control_reference_core(pc_src, config=cfg, cache_dir=ref_dir,
                                                       input_type=pred_itype,
                                                       comparator=comparator)
                finally:
                    pc_src.close()

            piece_cfg = replace(cfg, control=control, input_type=pred_itype)
            # One bundle per context: the reference artifacts are written above and never change
            # during this loop, so re-reading them per batch was pure waste (#153). Built BEFORE the
            # source is opened -- constructing it after would leak pred_src if it raised, since the
            # try/finally that closes the source has not been entered yet.
            bundle = _RefBundle(ref_dir, piece_cfg)
            # Generation-order piece paths (P1): appended AS each piece scores, so de_pred.parquet's
            # eventual row order is batch order, not a lexicographic re-sort of piece file names
            # (which would place ctx_10 before ctx_2 at 10+ batches).
            pred_parts: list[str] = []
            pred_src = CellBatchSource(pred, pert_col=cfg.pert_col, control=control,
                                       gather_threads=cfg.gather_threads)
            try:
                for k, (batch_perts, batch_ad) in enumerate(pred_src.iter_pert_batches(mem_budget)):
                    piece_id = f"ctx_{k}"
                    score_piece(batch_ad, ref_dir, config=piece_cfg,
                                piece_id=piece_id, partial_out=parts_dir, bundle=bundle,
                                comparator=comparator,
                                de_out=de_parts_dir if write_de else None,
                                # counts -> gpudge normalizes on-GPU (skips the CPU _to_linear copy,
                                # issue #142); lognorm falls through to _to_linear inside compute_de.
                                native_gpu_normalize=True)
                    if write_de:
                        pred_parts.append(
                            os.path.join(de_parts_dir, f"{piece_id}__de_pred.parquet"))
            finally:
                pred_src.close()
                # Release the cached control pool / DE table before aggregation. A Python local is
                # not scoped to the block, so without this the bundle stays live through
                # aggregate_partials -- at a large control pool that is GBs held for no reason (#153).
                del bundle

            # Before aggregation, which would otherwise fail first with an opaque "no partial
            # sidecars" error: an empty prediction yields no pieces, so no pred DE table exists.
            if write_de and not pred_parts:
                raise ValueError(
                    "write_de=True but the prediction yielded no perturbation batches, so there "
                    "is no pred DE table to write -- is the pred archive empty, or did its "
                    "context select no perturbed cells?"
                )

            full, _agg = aggregate_partials(
                parts_dir, reduce_nsig_spearman=True, nsig_spearman_metric=nsig_spearman,
                nsig_real_metric=nsig_real, nsig_pred_metric=nsig_pred)

            # DE-table staging, INSIDE the with-block: ref_dir (real DE) and de_parts_dir (per-piece
            # pred DE) are both torn down when it exits, so the source files must be read out here
            # (#387). Nothing is published yet -- the two DE tables are published together with the
            # three ScoreResult frames below, so no failure in between can leave a mixed outdir.
            if write_de:
                # Every listed piece must actually be on disk -- a missing piece (a de_out wiring bug,
                # or a score_piece that silently skipped writing one) must fail loudly here, naming
                # the piece, rather than merge_de_pieces raising an opaque FileNotFoundError deeper in
                # de_export.
                missing = [p for p in pred_parts if not os.path.exists(p)]
                if missing:
                    raise ValueError(
                        "write_de=True but the following pred DE piece(s) were never written: "
                        f"{missing!r}"
                    )
                staging = _StagedOutputs(outdir)
                _stage_de_outputs(staging, real_de_path=os.path.join(ref_dir, "real_de.parquet"),
                                  piece_paths=pred_parts,
                                  real_de_sha256=(gt_result.de_real_sha256
                                                  if gt_result is not None else None))
                if gt_result is not None:
                    # The sixth file: staged with the DE tables, so it publishes -- and rolls
                    # back -- with them and the three score frames (spec §6.6). Its
                    # de_real_sha256 is the cache entry's, and the de_real.parquet staged above
                    # IS that entry: a byte copy, verified against it as it was copied.
                    sidecar = gt_result.to_sidecar_dict(producer=_producer_dict())

                    def _write_sidecar(tmp, sidecar=sidecar):
                        with open(tmp, "w", encoding="utf-8") as fh:
                            fh.write(json.dumps(sidecar, sort_keys=True, indent=2))

                    staging.stage(_GT_SIDECAR, _write_sidecar)

        full = full.with_columns(
            pl.lit(dataset_tag).alias("dataset"),
            # int sentinel: manifest orchestrators set an int panel_id, so keep the ScoreResult
            # schema type-consistent (avoids a Utf8-vs-Int64 Polars error if outputs are
            # concatenated) -- Gemini
            pl.lit(-1, dtype=pl.Int64).alias("panel_id"),
            pl.lit(context_tag).alias("context"),
        )
        result = _assemble_score_result([full])
        if staging is not None:
            # write_de: the score frames join the staged DE tables and all five publish together.
            for name, frame in (("per_pert.parquet", result.per_pert),
                                ("per_context.parquet", result.per_context),
                                ("overall.parquet", result.overall)):
                staging.stage(name, frame.write_parquet)
            staging.publish()
        elif outdir is not None:
            os.makedirs(outdir, exist_ok=True)
            result.per_pert.write_parquet(os.path.join(outdir, "per_pert.parquet"))
            result.per_context.write_parquet(os.path.join(outdir, "per_context.parquet"))
            result.overall.write_parquet(os.path.join(outdir, "overall.parquet"))
    except BaseException:
        if staging is not None:
            staging.discard()
        raise
    return result
