# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "basic-pitch==0.4.0",
#   "setuptools<81",
#   "spleeter==2.4.2",
# ]
# ///
"""Run the preregistered representation, separation and robustness experiments.

    uv run --script experiences/_shared/pitch_ablation.py

Intermediate audio and inference caches stay outside data/dataset in sandbox/.
Each stage is restartable; numerical CSVs and Markdown reports are retained.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from experiences._shared.midi_similarity import (
    NoteEvent, WindowVectors, build_window_vectors, compute_similarity_matrix,
    load_midi_events,
)
from experiences._shared.pitch_features import average_frames, fold_octaves, sample_midi_activity
from experiences._shared.chroma_experiment import (
    ChromaTrack, TEMPO_RATIOS, audio_paths, best_threshold, build_windows,
    classification_metric, compare_pair as compare_chroma, evaluate_scalar,
    extract_chroma, extract_chroma_samples, markdown_table, split_fold,
    threshold_candidates, track_key,
)
from itertools import combinations

CACHE = ROOT / "sandbox" / "pitch-experiments"
OUT = ROOT / "experiences" / "10"
ROB_OUT = ROOT / "experiences" / "12"
DOCS = ROOT / "experiences" / "10"
DATASET = ROOT / "data" / "dataset"
FEATURE_VERSION = 1
STEMS = ("bass", "other")
STRATEGIES = (
    "stems-cqt", "stems-probability", "stems-midi", "stems-midi-bass035",
    "stems-probability-onsets", "stems-midi-onsets",
    "mix-probability", "mix-midi", "mix-probability-onsets", "mix-midi-onsets",
)
SCORES = OUT / "pitch-ablation-scores.csv"
DECISIONS = OUT / "pitch-ablation-decisions.csv"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def duration(path):
    import soundfile as sf
    return sf.info(str(path)).duration


def signature(path):
    stat = path.stat()
    return json.dumps([FEATURE_VERSION, str(path.relative_to(ROOT)), stat.st_size,
                       stat.st_mtime_ns])


def cache_path(name, component):
    return CACHE / f"{name}--{component}.npz"


class Extractor:
    """Use identical Basic Pitch windows for the CQT, predictions and MIDI."""

    def __init__(self, with_cqt=True):
        import tensorflow as tf
        from basic_pitch import ICASSP_2022_MODEL_PATH, models
        from basic_pitch.constants import AUDIO_N_SAMPLES
        from basic_pitch.inference import Model
        self.model = Model(ICASSP_2022_MODEL_PATH)
        self.cqt = None
        if with_cqt:
            inputs = tf.keras.Input(shape=(AUDIO_N_SAMPLES, 1))
            self.cqt = tf.keras.Model(inputs, models.get_cqt(inputs, 8, False))
        self.model_sha = hashlib.sha256(
            (Path(ICASSP_2022_MODEL_PATH) / "saved_model.pb").read_bytes()
        ).hexdigest()
        model_dir = Path(ICASSP_2022_MODEL_PATH)
        self.model_files = {
            str(p.relative_to(model_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(model_dir.rglob("*")) if p.is_file()
        }

    def extract(self, path, destination):
        if destination.exists():
            with np.load(destination) as cached:
                if str(cached["signature"]) == signature(path):
                    return
        from basic_pitch.constants import AUDIO_N_SAMPLES, FFT_HOP
        from basic_pitch.inference import get_audio_input, unwrap_output
        from basic_pitch.note_creation import model_frames_to_time, model_output_to_notes

        started = time.monotonic()
        print(f"[inference] {path.relative_to(ROOT)}", flush=True)
        hop = AUDIO_N_SAMPLES - 30 * FFT_HOP
        pieces = {key: [] for key in ("note", "onset", "contour")}
        cqts = []
        for window, _, original_length in get_audio_input(path, 30 * FFT_HOP, hop):
            predicted = self.model.predict(window)
            for key, value in predicted.items():
                pieces[key].append(value)
            if self.cqt is not None:
                # get_cqt returns batch x time x frequency x channel.
                cqt = self.cqt(window, training=False).numpy()[..., 0]
                assert cqt.shape[1] == predicted["note"].shape[1]
                assert cqt.shape[2] >= 88 * 3
                cqts.append(cqt[:, :, :264].reshape(1, cqt.shape[1], 88, 3).mean(-1))
        output = {
            key: unwrap_output(np.concatenate(value), original_length, 30)
            for key, value in pieces.items()
        }
        times = model_frames_to_time(len(output["note"]))
        arrays = {
            "signature": signature(path), "model_sha256": self.model_sha,
            "duration": duration(path), "times": times,
            "note": output["note"].copy(), "onset": output["onset"].copy(),
        }
        if cqts:
            arrays["cqt"] = unwrap_output(np.concatenate(cqts), original_length, 30)
        for threshold, key in ((0.30, "events"), (0.35, "events035")):
            midi, _ = model_output_to_notes(
                {k: v.copy() for k, v in output.items()}, onset_thresh=0.5,
                frame_thresh=threshold, min_note_len=11,
            )
            # Round-trip through MIDI, including velocity and tick quantization.
            temporary = destination.with_suffix(".mid")
            midi.write(str(temporary))
            events = load_midi_events(temporary)
            temporary.unlink()
            arrays[key] = np.asarray(
                [(e.start, e.end, e.pitch, e.velocity) for e in events],
                dtype=np.float64,
            ).reshape(-1, 4)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **arrays)
        print(f"[cached] {path.name}: {len(times)} frames, "
              f"{len(arrays['events'])} notes, {time.monotonic()-started:.1f}s", flush=True)


def extract_dataset(limit=None):
    CACHE.mkdir(parents=True, exist_ok=True)
    extractor = Extractor()
    queries, references = audio_paths()
    inputs = [(p.stem, "mix", p) for p in [*queries, *references]]
    inputs += [(p.stem, stem, p.with_suffix("") / f"{stem}.wav")
               for p in [*queries, *references] for stem in STEMS]
    for index, (name, component, path) in enumerate(inputs[:limit], 1):
        print(f"[extract {index}/{len(inputs[:limit])}]", flush=True)
        extractor.extract(path, cache_path(name, component))
    environment = {
        "python": sys.version, "feature_version": FEATURE_VERSION,
        "model_saved_model_sha256": extractor.model_sha,
        "model_files_sha256": extractor.model_files,
        "protocol_sha256": hashlib.sha256(
            (DOCS / "protocole.md").read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for _, _, p in inputs
        },
        "packages": dict(sorted((d.metadata["Name"], d.version)
                                for d in importlib.metadata.distributions())),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pitch-experiments-environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8")


def common_features(path, representation, onsets=False, bass035=False):
    with np.load(path) as raw:
        times = raw["times"]
        seconds = float(raw["duration"])
        if representation == "midi":
            events = raw["events035" if bass035 else "events"]
            activity = sample_midi_activity(events, times)
        else:
            activity = raw["cqt" if representation == "cqt" else "note"]
        folded = average_frames(fold_octaves(activity), times, seconds)
        result = folded[:, None, :]
        if onsets:
            if representation == "probability":
                attack = average_frames(fold_octaves(raw["onset"]), times, seconds)
            else:
                attack = np.zeros_like(folded)
                for start, _, pitch, velocity in events:
                    frame = int(start / 0.5)
                    if 0 <= frame < len(attack):
                        attack[frame, int(pitch) % 12] = max(
                            attack[frame, int(pitch) % 12], velocity)
            result = np.stack([folded, attack], axis=1)
    return result, seconds


def make_windows(features, seconds, ratio):
    # Use the existing sampling/interpolation convention for every representation.
    channels = [build_windows(ChromaTrack(features[:, c, :], 0.5, seconds), ratio)
                for c in range(features.shape[1])]
    values = np.stack([c.values for c in channels], axis=2)
    if values.shape[2] == 1:
        values = np.concatenate([values, np.zeros_like(values)], axis=2)
    return WindowVectors(channels[0].starts, values)


def decode_strategy(strategy):
    return (
        ("mix",) if strategy.startswith("mix-") else STEMS,
        "probability" if "probability" in strategy else
        "cqt" if "cqt" in strategy else "midi",
        strategy.endswith("onsets"), strategy.endswith("bass035"),
    )


def compare_vectors(query_components, reference_components):
    best = dict(score=-1.0, query_start=0.0, reference_start=0.0,
                tempo_ratio=1.0, bass_transposition=0, other_transposition=0)
    for ratio in TEMPO_RATIOS:
        matrices, shifts = [], []
        for query, references in zip(query_components, reference_components):
            matrix, shift = compute_similarity_matrix(query, references[ratio])
            matrices.append(matrix)
            shifts.append(shift)
        fused = np.mean(matrices, axis=0)
        i, j = np.unravel_index(int(np.argmax(fused)), fused.shape)
        score = float(fused[i, j])
        if score > best["score"]:
            best.update(score=score, query_start=float(query_components[0].starts_seconds[i]),
                        reference_start=float(reference_components[0][ratio].starts_seconds[j]),
                        tempo_ratio=ratio, bass_transposition=int(shifts[0][i, j]),
                        other_transposition=int(shifts[-1][i, j]))
    return best


def evaluate_representations():
    queries, references = audio_paths()
    rows = []
    for strategy in STRATEGIES:
        print(f"[scores] {strategy}", flush=True)
        components, representation, onsets, bass035 = decode_strategy(strategy)
        vectors = {}
        for path in [*queries, *references]:
            for component in components:
                features, seconds = common_features(
                    cache_path(path.stem, component), representation, onsets,
                    bass035 and component == "bass")
                ratios = (1.0,) if path in queries else TEMPO_RATIOS
                vectors[path.stem, component] = {
                    ratio: make_windows(features, seconds, ratio) for ratio in ratios}
        for query in queries:
            for reference in references:
                result = compare_vectors(
                    [vectors[query.stem, c][1.0] for c in components],
                    [vectors[reference.stem, c] for c in components])
                rows.append(dict(strategy=strategy, query=query.stem,
                                 reference=reference.stem,
                                 expected_pair=str(track_key(query)==track_key(reference)).lower(),
                                 **result))
    write_csv(SCORES, rows)
    report_ablation(rows)


def score_pairs(rows):
    pairs = {}
    for row in rows:
        pair = pairs.setdefault((row["query"], row["reference"]),
                                {"expected": float(str(row["expected_pair"]) == "true")})
        pair[row["strategy"]] = float(row["score"])
    families = sorted({track_key(q) for q, _ in pairs})
    return families, pairs


def low_fpr_threshold(values, labels, target=0.1):
    negatives = sum(not label for label in labels)
    feasible = [t for t in threshold_candidates(values)
                if sum(not y and x >= t for x, y in zip(values, labels)) / negatives <= target]
    return max(feasible, key=lambda t: (
        sum(y and x >= t for x, y in zip(values, labels)), -t))


def ablation_decisions(rows):
    families, pairs = score_pairs(rows)
    decisions = []
    for held_out in combinations(families, 2):
        training, testing = split_fold(pairs, held_out)
        labels = [bool(features["expected"]) for _, _, features in training]
        for strategy in STRATEGIES:
            values = [features[strategy] for _, _, features in training]
            for policy, threshold in (
                ("balanced", best_threshold(values, labels)),
                ("train-fpr10", low_fpr_threshold(values, labels)),
            ):
                for query, reference, features in testing:
                    decisions.append(dict(
                        held_out="|".join(held_out), policy=policy, strategy=strategy,
                        query=query, reference=reference,
                        expected_pair=str(bool(features["expected"])).lower(),
                        score=features[strategy], threshold=threshold,
                        predicted=str(features[strategy] >= threshold).lower()))
    return decisions


def count_metrics(rows):
    tp = sum(str(r["expected_pair"]) == "true" and str(r["predicted"]) == "true" for r in rows)
    fp = sum(str(r["expected_pair"]) == "false" and str(r["predicted"]) == "true" for r in rows)
    p = sum(str(r["expected_pair"]) == "true" for r in rows)
    n = len(rows) - p
    return tp, fp, p, n, 2*tp/(p+tp+fp) if p+tp+fp else 0


def report_ablation(rows=None):
    rows = read_csv(SCORES) if rows is None else rows
    families, pairs = score_pairs(rows)
    decisions = ablation_decisions(rows)
    write_csv(DECISIONS, decisions)
    tables = {}
    for policy in ("balanced", "train-fpr10"):
        table = []
        for strategy in STRATEGIES:
            selected = [r for r in decisions if r["strategy"] == strategy and r["policy"] == policy]
            tp, fp, p, n, f1 = count_metrics(selected)
            table.append((strategy, f"{tp}/{p}", f"{p-tp}/{p}", f"{fp}/{n}",
                          f"{tp/p:.3f}", f"{fp/n:.3f}", f"{f1:.3f}"))
        tables[policy] = markdown_table(("Variante", "VP", "FN", "FP", "Rappel", "FPR", "F1"), table)
    family_table = []
    for family in families:
        entries = [family]
        for strategy in STRATEGIES:
            selected = [r for r in decisions if r["strategy"] == strategy and
                        r["policy"] == "balanced" and r["expected_pair"] == "true"
                        and track_key(r["query"]) == family]
            entries.append(f"{sum(r['predicted']=='true' for r in selected)}/{len(selected)}")
        family_table.append(tuple(entries))
    metrics = {s: evaluate_scalar(families, pairs, s) for s in STRATEGIES}
    cqt, probability, midi = [metrics[f"stems-{s}"] for s in ("cqt", "probability", "midi")]
    text = f"""# CQT, probabilités Basic Pitch, MIDI et séparation

## Question et protocole

Ces deux expériences suivent le [protocole fixé avant calcul](protocole.md).
On compare chaque chanson IA GEMA à chaque original : 64 paires uniques,
dont 8 correspondances attendues et 56 négatifs. Aucun classement de catalogue
n'entre dans le verdict. Les mêmes 28 folds par famille servent à calibrer
les seuils sur les six familles d'entraînement, puis à tester les deux autres.
Les 56 décisions positives sont des répétitions de huit paires, pas 56 cas
indépendants. Ces données ont déjà servi à explorer le pipeline : aucun test
final indépendant n'est disponible.

La CQT est celle du frontal Basic Pitch 0.4.0 (TensorFlow), avant batch
normalization et convolutions, sur les mêmes fenêtres audio que l'inférence.
On moyenne ses trois bandes par demi-ton et conserve les 88 notes A0–C8.
Le réseau bénéficie aussi du contexte des harmoniques au-delà de cette
plage de sortie ; ce contrôle compare l'utilité de deux représentations,
sans isoler chaque mécanisme interne du réseau.
Les trois représentations regroupent les octaves par maximum puis moyennent
sur 500 ms. Le MIDI est décodé à partir de l'inférence comparée, avec vélocité
et temps quantifiés par un aller-retour MIDI. Les probabilités sont des
activations du modèle, pas des probabilités calibrées de plagiat.

La comparaison utilise des fenêtres de 10 s, un pas de 2 s, 12 transpositions
et les six rapports temporels historiques. La fusion `bass` + `other` moyenne
les matrices avant de choisir leur maximum commun. La transposition peut
différer entre stems, comme dans la baseline historique. Le score local et
la fusion ne sont pas réoptimisés pour cette expérience.

## Résultats avec seuil d'exactitude équilibrée

{tables['balanced']}

![Rappel et faux positifs par représentation](../../experiences/10/pitch-representation-ablation.svg)

## Résultats avec cible de FPR d'entraînement ≤ 10 %

{tables['train-fpr10']}

Le seuil est choisi sur l'entraînement uniquement. Le FPR mesuré sur les
familles tenues à l'écart n'est donc pas garanti inférieur à 10 %.

## Résultats par famille

{markdown_table(('Famille', *STRATEGIES), family_table)}

## Interprétation

Sur les stems identiques, la CQT atteint F1 **{cqt.f1:.3f}**, les sorties
continues `note` **{probability.f1:.3f}**, et les événements MIDI **{midi.f1:.3f}**.
La différence probabilités moins CQT est **{probability.f1-cqt.f1:+.3f}** ;
MIDI moins probabilités est **{midi.f1-probability.f1:+.3f}**. Cela mesure
l'utilité de ces étapes pour la tâche, pas la fidélité des notes transcrites :
aucune partition de référence n'a été annotée.

À représentation identique, séparer avant Basic Pitch change le F1 de
**{metrics['stems-probability'].f1-metrics['mix-probability'].f1:+.3f}** pour
les probabilités, et de **{metrics['stems-midi'].f1-metrics['mix-midi'].f1:+.3f}**
pour le MIDI. Le contrôle principal utilise partout le seuil de trame 0.30.
La variante `bass035` reproduit le réglage 0.35 de la basse historique.
La séparation n'a donc pas un effet bénéfique universel : son intérêt dépend
de la représentation et du décodage utilisés en aval. Le passage au MIDI
sur stems conserve ici le même nombre de vrais positifs que les probabilités,
avec moins de faux positifs. À l'inverse, décoder en MIDI le master complet
dégrade les résultats dans ce contrôle.
L'ajout des attaques est secondaire ; sa pondération relative n'a pas été
optimisée et les distributions des impulsions MIDI et des activations réseau
diffèrent. Un petit écart ne suffit pas à conclure sur huit familles.

Ces résultats ne remplacent pas automatiquement le pipeline fusionné
historique : cette expérience isole des représentations, avec une agrégation
commune différente de la rasterisation MIDI historique (présence sur une
trame contre activité moyenne). Les scores des deux protocoles ne sont pas
interchangeables et leurs seuils sont calibrés séparément.

## Fichiers et reproduction

- Scores : `experiences/10/pitch-ablation-scores.csv`.
- Décisions et seuils par fold : `experiences/10/pitch-ablation-decisions.csv`.
- Versions et empreinte du modèle : `experiences/10/pitch-experiments-environment.json`.
- Caches régénérables et sorties continues : `sandbox/pitch-experiments/`.

```sh
uv run --script experiences/_shared/pitch_ablation.py --stage extract
uv run --script experiences/_shared/pitch_ablation.py --stage evaluate
```

Sources audio et logiciels : [CREDITS.md](../../CREDITS.md),
[GEMA](https://www.gema.de/en/news/ai-and-music/ai-lawsuit/audio-samples-suno),
[Basic Pitch](https://github.com/spotify/basic-pitch),
[article Basic Pitch](https://arxiv.org/abs/2203.09893),
[Spleeter](https://github.com/deezer/spleeter).
"""
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "rapport.md").write_text(text, encoding="utf-8")
    print(tables["balanced"], flush=True)


def conditions():
    result = [("unchanged", "control", 0.0), ("reconstructed", "control", 0.0)]
    for kind in ("global-pitch", "vocal-pitch"):
        result.extend((f"{kind}-{semitones:+d}", kind, float(semitones))
                      for semitones in (-4, -2, -1, 1, 2, 4))
    result.extend((f"tempo-{rate:.1f}", "tempo", rate) for rate in (0.8, 0.9, 1.1, 1.2))
    return result


def transform_audio(source, target, *, semitones=0, tempo=1.0):
    # FFmpeg's filter produces floating samples without the CLI's automatic
    # gain reduction on clipping. Explicit float WAV avoids PCM clipping.
    filter_ = (f"rubberband=pitch={2**(semitones/12):.12f}:tempo={tempo}:"
               "formant=preserved:channels=together:pitchq=quality")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source),
                    "-af", filter_, "-c:a", "pcm_f32le", "-map_metadata", "-1",
                    str(target)], check=True)


def robust_build(limit=None):
    import soundfile as sf
    CACHE.mkdir(parents=True, exist_ok=True)
    _, references = audio_paths()
    manifest = []
    for reference in references:
        start, length = 30*44100, 20*44100
        audio, sr = sf.read(reference, start=start, frames=length, dtype="float32", always_2d=True)
        assert sr == 44100 and len(audio) == length
        parts = {}
        for stem in ("vocals", "drums", "bass", "piano", "other"):
            part, part_sr = sf.read(reference.with_suffix("") / f"{stem}.wav",
                                    start=start, frames=length, dtype="float32", always_2d=True)
            assert part_sr == sr and part.shape == audio.shape
            parts[stem] = part
        reconstructed = sum(parts.values())
        accompaniment = sum(v for k, v in parts.items() if k != "vocals")
        prefix = f"{reference.stem}--robust"
        unchanged = CACHE / f"{prefix}-unchanged.wav"
        vocal_source = CACHE / f"{prefix}-vocal-source.wav"
        if not unchanged.exists():
            sf.write(unchanged, audio, sr, subtype="FLOAT")
        if not vocal_source.exists():
            sf.write(vocal_source, parts["vocals"], sr, subtype="FLOAT")
        for condition, kind, value in conditions():
            target = CACHE / f"{prefix}-{condition}.wav"
            if not target.exists():
                print(f"[transform] {reference.stem} {condition}", flush=True)
                if condition == "reconstructed":
                    sf.write(target, reconstructed, sr, subtype="FLOAT")
                elif kind == "global-pitch":
                    transform_audio(unchanged, target, semitones=value)
                elif kind == "tempo":
                    transform_audio(unchanged, target, tempo=value)
                elif kind == "vocal-pitch":
                    temporary = CACHE / f"{prefix}-temporary-vocal.wav"
                    transform_audio(vocal_source, temporary, semitones=value)
                    shifted, _ = sf.read(temporary, dtype="float32", always_2d=True)
                    # Align the start exactly; only boundary padding/cropping if
                    # the time-preserving stretcher differs by a few samples.
                    aligned = np.zeros_like(accompaniment)
                    count = min(len(aligned), len(shifted))
                    aligned[:count] = shifted[:count]
                    sf.write(target, accompaniment + aligned, sr, subtype="FLOAT")
                    temporary.unlink()
            info = sf.info(target)
            assert info.subtype == "FLOAT"
            manifest.append(dict(
                query=target.stem, family=track_key(reference), condition=condition,
                group=kind, value=value, path=str(target.relative_to(ROOT)),
                excerpt_start=30.0, source_excerpt_duration=20.0, duration=info.duration,
                reconstruction_rmse=float(np.sqrt(np.mean((audio-reconstructed)**2))),
            ))
    write_csv(ROB_OUT / "pitch-robustness-manifest.csv", manifest)
    os.environ.setdefault("MODEL_PATH", str(ROOT / "sandbox" / "spleeter-models"))
    import tensorflow as tf
    from spleeter.separator import Separator, create_estimator
    separator = Separator("spleeter:5stems", multiprocess=False)
    selected = manifest[:limit]
    def exists(entry):
        return all((CACHE / f"{entry['query']}--{s}.wav").exists() for s in STEMS)
    pending = [e for e in selected if not exists(e)]
    check_entry = next((e for e in selected if exists(e)), None)
    # One estimator graph/session for variable-length excerpts, rather than
    # rebuilding and restoring the same model for each call to separate().
    stream = ([check_entry] if check_entry else []) + pending
    if not stream:
        return
    def examples():
        for entry in stream:
            audio, sr = sf.read(ROOT / entry["path"], dtype="float32", always_2d=True)
            assert sr == 44100
            yield {"waveform": audio, "audio_id": entry["query"]}
    def input_fn():
        return tf.data.Dataset.from_generator(examples, output_signature={
            "waveform": tf.TensorSpec((None, 2), tf.float32),
            "audio_id": tf.TensorSpec((), tf.string)})
    estimator = create_estimator(separator._params.copy(), False)
    predictions = estimator.predict(input_fn, yield_single_examples=False)
    for index, (entry, separated) in enumerate(zip(stream, predictions), 1):
        name = separated["audio_id"].decode("utf-8")
        assert name == entry["query"]
        print(f"[separate {index}/{len(stream)}] {name}", flush=True)
        if entry is check_entry:
            differences = {}
            for stem in STEMS:
                previous, _ = sf.read(CACHE / f"{name}--{stem}.wav", dtype="float32", always_2d=True)
                np.testing.assert_allclose(separated[stem], previous, rtol=1e-4, atol=1e-5)
                differences[stem] = float(np.max(np.abs(separated[stem]-previous)))
            (ROB_OUT / "pitch-spleeter-stream-check.json").write_text(
                json.dumps(dict(query=name, max_absolute_error=differences), indent=2)+"\n")
        else:
            for stem in STEMS:
                sf.write(CACHE / f"{name}--{stem}.wav", separated[stem], 44100, subtype="FLOAT")


def robust_extract(limit=None):
    entries = read_csv(ROB_OUT / "pitch-robustness-manifest.csv")
    extractor = Extractor(with_cqt=False)
    for index, entry in enumerate(entries[:limit], 1):
        print(f"[robust inference {index}/{len(entries[:limit])}]", flush=True)
        for stem in STEMS:
            path = CACHE / f"{entry['query']}--{stem}.wav"
            extractor.extract(path, cache_path(entry["query"], stem))


def legacy_windows(events, seconds, ratio):
    return build_window_vectors(
        events, duration=seconds, window_seconds=10*ratio,
        hop_seconds=2*ratio, frame_seconds=0.5*ratio)


def legacy_reference_cache():
    _, references = audio_paths()
    result, chromas = {}, {}
    for reference in references:
        seconds = duration(reference)
        for stem in STEMS:
            events = load_midi_events(reference.with_suffix("") / f"{stem}_basic_pitch.mid")
            result[reference.stem, stem] = {
                ratio: legacy_windows(events, seconds, ratio) for ratio in TEMPO_RATIOS}
        track = extract_chroma(reference)
        chromas[reference.stem] = {r: build_windows(track, r) for r in TEMPO_RATIOS}
    return references, result, chromas


def robust_evaluate(limit=None):
    import soundfile as sf
    entries = read_csv(ROB_OUT / "pitch-robustness-manifest.csv")
    references, reference_vectors, reference_chromas = legacy_reference_cache()
    path = ROB_OUT / "pitch-robustness-scores.csv"
    rows = []
    for index, entry in enumerate(entries[:limit], 1):
        print(f"[robust scores {index}/{len(entries[:limit])}] {entry['query']}", flush=True)
        query_vectors = []
        for stem in STEMS:
            with np.load(cache_path(entry["query"], stem)) as raw:
                events = [NoteEvent(float(start), float(end), int(pitch), float(velocity))
                          for start, end, pitch, velocity in
                          raw["events035" if stem=="bass" else "events"]]
                query_vectors.append(legacy_windows(events, float(raw["duration"]), 1.0))
        audio, sr = sf.read(ROOT / entry["path"], dtype="float32", always_2d=True)
        query_track = extract_chroma_samples(audio.mean(axis=1), sr)
        query_chroma = build_windows(query_track, 1.0)
        for reference in references:
            midi = compare_vectors(query_vectors,
                                   [reference_vectors[reference.stem, s] for s in STEMS])
            chroma = compare_chroma(query_chroma, reference_chromas[reference.stem])["chroma-diagonal-3"]
            rows.append(dict(
                query=entry["query"], family=entry["family"], reference=reference.stem,
                condition=entry["condition"], group=entry["group"], value=entry["value"],
                expected_pair=str(entry["family"]==track_key(reference)).lower(),
                midi_score=midi["score"], chroma_score=chroma.score,
                midi_query_start=midi["query_start"], midi_reference_start=midi["reference_start"],
                midi_tempo_ratio=midi["tempo_ratio"],
                chroma_query_start=chroma.query_start, chroma_reference_start=chroma.reference_start,
                chroma_tempo_ratio=chroma.tempo_ratio,
            ))
    write_csv(path, rows)
    if limit is None:
        report_robustness(rows)


def historical_pairs():
    pairs = {}
    for row in read_csv(ROOT / "experiences" / "08" / "full-mix-chroma-experiment.csv"):
        if row["strategy"] == "chroma-diagonal-3":
            pairs[row["query"], row["reference"]] = {
                "expected": float(row["expected_pair"]=="true"), "chroma": float(row["score"])}
    for row in read_csv(ROOT / "experiences" / "06" / "multistem-fusion-experiment.csv"):
        if row["strategy"] == "sync-mean-bass-other":
            pairs[row["query"], row["reference"]]["midi"] = float(row["score"])
    return sorted({track_key(q) for q, _ in pairs}), pairs


def historical_thresholds():
    families, pairs = historical_pairs()
    thresholds = {}
    audit = []
    for held_out in combinations(families, 2):
        training, testing = split_fold(pairs, held_out)
        labels = [bool(f["expected"]) for _, _, f in training]
        values = {name: [f[name] for _, _, f in training] for name in ("midi", "chroma")}
        individual = {name: best_threshold(v, labels) for name, v in values.items()}
        candidates = []
        for first in threshold_candidates(values["midi"]):
            for second in threshold_candidates(values["chroma"]):
                predictions = [m >= first or c >= second
                               for m, c in zip(values["midi"], values["chroma"])]
                candidates.append((*classification_metric(labels, predictions), first, second))
        *_, first, second = max(candidates)
        thresholds[held_out] = (individual["midi"], individual["chroma"], first, second)
        for q, r, f in testing:
            for branch, prediction in (
                ("midi", f["midi"] >= individual["midi"]),
                ("chroma", f["chroma"] >= individual["chroma"]),
                ("OR", f["midi"] >= first or f["chroma"] >= second),
            ):
                audit.append(dict(held_out="|".join(held_out), branch=branch, query=q,
                                  reference=r, expected_pair=str(bool(f["expected"])).lower(),
                                  predicted=str(prediction).lower()))
    # Check that calibration reconstructs the already-published reference.
    tp, fp, p, n, _ = count_metrics([r for r in audit if r["branch"] == "OR"])
    if (tp, fp, p, n) != (47, 8, 56, 56):
        raise RuntimeError(f"historical OR calibration drift: {(tp, fp, p, n)}")
    return thresholds, audit


def report_robustness(rows):
    thresholds, audit = historical_thresholds()
    write_csv(ROB_OUT / "pitch-robustness-calibration-audit.csv", audit)
    decisions = []
    for held_out, (midi_threshold, chroma_threshold, or_midi, or_chroma) in thresholds.items():
        for row in rows:
            if row["family"] not in held_out or track_key(row["reference"]) not in held_out:
                continue
            m, c = float(row["midi_score"]), float(row["chroma_score"])
            for branch, prediction in (
                ("midi", m >= midi_threshold), ("chroma", c >= chroma_threshold),
                ("OR", m >= or_midi or c >= or_chroma),
            ):
                decisions.append(dict(
                    held_out="|".join(held_out), branch=branch, query=row["query"],
                    family=row["family"], reference=row["reference"], condition=row["condition"],
                    group=row["group"], value=row["value"], expected_pair=row["expected_pair"],
                    midi_score=m, chroma_score=c,
                    midi_threshold=or_midi if branch=="OR" else midi_threshold,
                    chroma_threshold=or_chroma if branch=="OR" else chroma_threshold,
                    predicted=str(prediction).lower()))
    write_csv(ROB_OUT / "pitch-robustness-decisions.csv", decisions)
    summary, table = [], []
    scores_by_key = {(r["family"], r["reference"], r["condition"]): r for r in rows}
    for condition, kind, value in conditions():
        result = [condition]
        for branch in ("midi", "chroma", "OR"):
            selected = [r for r in decisions if r["condition"]==condition and r["branch"]==branch]
            tp, fp, p, n, f1 = count_metrics(selected)
            summary.append(dict(condition=condition, group=kind, value=value, branch=branch,
                                true_positives=tp, false_positives=fp, positives=p, negatives=n,
                                recall=tp/p, fpr=fp/n, f1=f1))
            result.extend([f"{tp}/{p}", f"{fp}/{n}"])
        baseline = "reconstructed" if kind=="vocal-pitch" else "unchanged"
        positives = [r for r in rows if r["condition"]==condition and r["expected_pair"]=="true"]
        for branch in ("midi", "chroma"):
            differences = [float(r[f"{branch}_score"]) -
                           float(scores_by_key[r["family"], r["reference"], baseline][f"{branch}_score"])
                           for r in positives]
            result.append(f"{np.mean(differences):+.4f}")
        table.append(tuple(result))
    write_csv(ROB_OUT / "pitch-robustness-summary.csv", summary)
    fusion = {r["condition"]: r for r in summary if r["branch"] == "OR"}
    tempo_results = [r for r in summary if r["branch"] == "OR" and r["group"] == "tempo"]
    family_table = []
    families = sorted({r["family"] for r in rows})
    for family in families:
        family_row = [family]
        for condition, _, _ in conditions():
            selected = [r for r in decisions if r["family"]==family and r["condition"]==condition
                        and r["branch"]=="OR" and r["expected_pair"]=="true"]
            family_row.append(f"{sum(r['predicted']=='true' for r in selected)}/{len(selected)}")
        family_table.append(tuple(family_row))
    rasputin = []
    for condition, _, _ in conditions():
        item = [condition]
        for branch in ("midi", "chroma", "OR"):
            selected = [r for r in decisions if r["family"]=="rasputin-boney-m"
                        and r["condition"]==condition and r["branch"]==branch
                        and r["expected_pair"]=="true"]
            item.append(f"{sum(r['predicted']=='true' for r in selected)}/{len(selected)}")
        rasputin.append(tuple(item))
    headers = ("Condition", "VP MIDI", "FP MIDI", "VP chroma", "FP chroma", "VP OR", "FP OR",
               "Δ score MIDI", "Δ score chroma")
    text = f"""# Robustesse du pipeline aux transformations contrôlées

## Protocole et calibration

Le [protocole fixé avant calcul](../10/protocole.md) prend un seul
extrait de 20 s, à partir de 30 s, dans chacun des huit originaux. Les
positions ne sont pas choisies selon la qualité de la détection. Chaque
extrait est comparé aux huit masters originaux complets.

Les 18 conditions comprennent l'extrait inchangé, la somme des cinq stems,
six transpositions globales, six transpositions de voix seule et quatre
changements de tempo. Les transformations utilisent le filtre Rubber Band
de FFmpeg, avec conservation des formants et traitement stéréo conjoint.
L'audio est stocké en WAV flottant, sans normalisation ni réduction automatique
de gain. Pour la voix seule, le signal transformé remplace la voix dans la
somme des stems ; il est comparé au témoin reconstruit. Les artefacts Spleeter
et Rubber Band font partie des limites de ce contrôle.

Chaque version repasse réellement dans Spleeter puis Basic Pitch. La branche
MIDI utilise la rasterisation historique, seuil de trame 0.35 pour la basse
et 0.30 pour `other`. La branche chroma utilise l'extracteur historique sur
les échantillons flottants. Le score MIDI est le maximum synchronisé ; le
score chroma impose trois fenêtres consécutives. La règle OR est inchangée.

Pour chaque fold, les seuils sont calculés sur les six familles GEMA
non transformées d'entraînement. Ils sont ensuite figés pour toutes les
transformations des deux autres familles. Le contrôle de calibration
reproduit exactement **47/56 VP et 8/56 FP** de la règle OR historique.
Les résultats ci-dessous ne réajustent aucun seuil aux copies synthétiques.

## Résultats

{markdown_table(headers, table)}

![Courbes de robustesse](../../experiences/12/pitch-transformation-robustness.svg)

Les écarts de score sont calculés sur les huit correspondances positives
uniques, contre l'extrait inchangé pour les transformations globales, et
contre la reconstruction inchangée pour les voix transposées. Une hausse
du score seule n'est pas une réussite si les faux positifs augmentent.
Le tempo indique le facteur de vitesse (0.8 = plus lent ; 1.2 = plus rapide).
Les rapports temporels testés par le détecteur restent ceux du POC ; une
dégradation peut donc venir de leur discrétisation autant que de l'extraction.

## Conclusions de ce contrôle

Le témoin inchangé donne **{fusion['unchanged']['true_positives']}/56**
détections positives et **{fusion['unchanged']['false_positives']}/56** faux
positifs pour la fusion. La reconstruction des stems donne les mêmes comptes
dans ce corpus ; elle constitue donc ici un témoin utile pour la voix seule.

Pour les changements de tempo, le rappel de la fusion va de
**{min(r['recall'] for r in tempo_results):.1%} à {max(r['recall'] for r in tempo_results):.1%}**.
La chroma maintient les correspondances que la branche MIDI peut perdre.
La fusion garde néanmoins entre
**{min(r['false_positives'] for r in tempo_results)}/56 et {max(r['false_positives'] for r in tempo_results)}/56**
faux positifs : conserver les positifs ne signifie pas classer toutes les
paires correctement.

La transposition globale est plus problématique : à -4 demi-tons, la fusion
ne conserve que **{fusion['global-pitch--4']['true_positives']}/56** positifs
avec **{fusion['global-pitch--4']['false_positives']}/56** faux positifs ; à
+4 demi-tons, elle conserve **{fusion['global-pitch-+4']['true_positives']}/56**
positifs avec **{fusion['global-pitch-+4']['false_positives']}/56** faux positifs.
Les échecs positifs concernent ici *Brother Louie* et *You're My Heart,
You're My Soul*. La voix seule transposée de -2 demi-tons fait également
tomber le nombre de détections à **{fusion['vocal-pitch--2']['true_positives']}/56**.

Tester les douze décalages de classes de hauteur dans le comparateur ne
garantit donc pas l'invariance de la chaîne à une transformation audio.
La transformation, la séparation et la transcription peuvent modifier les
activations au-delà d'un simple décalage circulaire. Ce contrôle constate
la dégradation sans attribuer précisément sa cause à l'une de ces étapes.

## Rasputin

{markdown_table(('Condition', 'VP MIDI', 'VP chroma', 'VP OR'), rasputin)}

Il s'agit d'un extrait de l'original transformé de manière contrôlée, pas
de l'extrait IA GEMA. Une réussite ne prouve pas une résistance à toutes
les transformations d'une génération IA.

## Stabilité par famille de la règle OR

{markdown_table(('Famille', *(c[0] for c in conditions())), family_table)}

## Limites et fichiers

On dispose toujours de huit familles indépendantes et d'un seul passage
par original. Les 56 décisions positives par condition répètent chaque
paire sept fois avec différents seuils, et les 56 négatifs sont les paires
entre morceaux différents. Les variantes ne sont pas des observations
indépendantes. Le contrôle synthétique mesure la robustesse à des
transformations connues ; il ne mesure pas le taux d'erreur sur de nouveaux
plagiats IA. Il ne permet pas non plus de tester la robustesse à des
modifications importantes de mélodie, d'arrangement ou de structure.

Les scores, décisions par fold, résumés, manifeste des extraits et audit de
calibration sont dans `experiences/12/pitch-robustness-*.csv`. Les audios
et caches régénérables sont dans `sandbox/pitch-experiments/`. Le dataset
original est conservé intact.

```sh
uv run --script experiences/_shared/pitch_ablation.py --stage robust-build
uv run --script experiences/_shared/pitch_ablation.py --stage robust-extract
uv run --script experiences/_shared/pitch_ablation.py --stage robust-evaluate
```

Sources : [CREDITS.md](../../CREDITS.md),
[GEMA](https://www.gema.de/en/news/ai-and-music/ai-lawsuit/audio-samples-suno),
[Basic Pitch](https://github.com/spotify/basic-pitch),
[Spleeter](https://github.com/deezer/spleeter),
[Rubber Band](https://breakfastquay.com/rubberband/).
"""
    (ROB_OUT / "rapport.md").write_text(text, encoding="utf-8")
    print(markdown_table(headers, table), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("extract", "evaluate", "report", "robust-build",
                                            "robust-extract", "robust-evaluate", "robust-report",
                                            "all"), default="all")
    parser.add_argument("--limit", type=int, help="smoke-test extraction on the first N files")
    args = parser.parse_args()
    if args.stage == "extract":
        extract_dataset(args.limit)
    elif args.stage == "evaluate":
        evaluate_representations()
    elif args.stage == "report":
        report_ablation()
    elif args.stage == "robust-report":
        report_robustness(read_csv(ROB_OUT / "pitch-robustness-scores.csv"))
    elif args.stage == "all":
        for stage in ("extract", "evaluate", "robust-build", "robust-extract", "robust-evaluate"):
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--stage", stage], check=True)
    else:
        globals()[args.stage.replace("-", "_")](args.limit)
    environment_path = OUT / "pitch-experiments-environment.json"
    if environment_path.exists():
        environment = json.loads(environment_path.read_text())
        environment["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        for tool, command in (("ffmpeg", ["ffmpeg", "-version"]),
                              ("rubberband", ["rubberband", "--version"])):
            version = subprocess.run(command, capture_output=True, text=True, check=True)
            environment[f"{tool}_version"] = (version.stdout or version.stderr).splitlines()[0]
        environment_path.write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
