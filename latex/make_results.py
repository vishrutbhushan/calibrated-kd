"""
Builds every results-dependent part of the paper from results/*.json.

Inputs (results/, falling back to ../outputs/main_experiments/):
  <dataset>_results.json       written by scripts/main_experiments/run_main_experiments.py
                               (<dataset>_all_methods.json from run_single_benchmark.py also works)
  gold_counts.json             ground-truth test class counts (fixed by the seed-42 split)

Outputs (generated/):
  results_table.tex            main results table (best/worst highlighted automatically)
  results_macros.tex           every number quoted in the text (\\AvgGainStd, \\WinsStd, ...)
  collapse_table.tex           classes that some distillation method never predicts
  fig_main_f1_accuracy.pdf     Macro-F1 / accuracy bar chart
  fig_class_distributions.pdf  predicted class counts vs. ground truth

If any of the six result files is missing, placeholders are written instead
(red XX.XX values in the table and text, placeholder boxes for the figures).

Usage:  python3 make_results.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = HERE / "generated"
OUT.mkdir(exist_ok=True)

DATASETS = [  # (json key, display name, number of classes)
    ("student_questions", "Student Questions", 4),
    ("trec6", "TREC-6", 6),
    ("app_reviews", "App Reviews", 5),
    ("twitter_emotion", "Twitter Emotion", 6),
    ("tweeteval_emotion", "TweetEval Emotion", 4),
    ("bbc_news", "BBC News", 5),
]
METHODS = [  # (json key, group label, method label)
    ("teacher_raw", "Teacher", "Zero-shot, raw"),
    ("teacher_calibrated", "(Gemma-3-4B-IT)", "Zero-shot, null-calibrated"),
    ("standard_kd", "Student", "Standard KD"),
    ("batch_calibrated_kd", "(RoBERTa-base)", "Batch-Calibrated KD"),
    ("calibrated_kd", "", "Null Calibrated KD (ours)"),
]
STUDENT = ["standard_kd", "batch_calibrated_kd", "calibrated_kd"]
DIST_PLOTS = [("bbc_news", "(a) BBC News"), ("app_reviews", "(b) App Reviews"),
              ("tweeteval_emotion", "(c) TweetEval Emotion")]

# Palette from scripts/plotting/*.py
COL = {"teacher_raw": ("#94A3B8", "#475569"), "standard_kd": ("#E15759", "#B91C1C"),
       "batch_calibrated_kd": ("#8DA0CB", "#4F46E5"), "calibrated_kd": ("#66C2A5", "#0D9488")}
LEG = {"teacher_raw": "Teacher, zero-shot (Gemma-3-4B-IT)", "standard_kd": "Standard KD",
       "batch_calibrated_kd": "Batch-Calibrated KD", "calibrated_kd": "Null Calibrated KD (ours)"}
PLOTTED = ["teacher_raw", "standard_kd", "batch_calibrated_kd", "calibrated_kd"]

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10.5, "axes.titlesize": 12.0,
                     "axes.titleweight": "bold", "axes.labelsize": 11, "axes.labelweight": "bold",
                     "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 10.0})

# Every macro the text uses, with its placeholder.
MACROS = {
    "AvgGainStd": r"\ph{XX.XX}", "AvgAccGainStd": r"\ph{XX.XX}", "WinsStd": r"\ph{X}",
    "MaxGainStd": r"\ph{XX.XX}", "MaxGainStdData": r"\ph{[dataset]}",
    "MinGainStd": r"\ph{XX.XX}", "MinGainStdData": r"\ph{[dataset]}",
    "AvgGainBatch": r"\ph{XX.XX}", "WinsBatch": r"\ph{X}",
    "AvgTeacherCalGain": r"\ph{XX.XX}", "WinsTeacherCal": r"\ph{X}", "WinsTeacher": r"\ph{X}",
    "NumCollapsedStd": r"\ph{X}", "CollapsedList": r"\ph{[classes]}", "NumRecovered": r"\ph{X}",
    "NumStillCollapsedBatch": r"\ph{X}",
    "GapStd": r"\ph{XXX}", "GapBatch": r"\ph{XXX}", "GapCKD": r"\ph{XXX}",
    "NotRecoveredText": r"\ph{[Classes that Null-Calibrated KD does not recover.]}",
    "OverCorrectionText": r"\ph{[Classes that only Null-Calibrated KD never predicts, with the null-vector gap.]}",
}


def load():
    """Returns (data, gold); data is None when any result file is missing (placeholder mode)."""
    gold = json.loads((RES / "gold_counts.json").read_text())
    def find(k):
        # results/ first, then the experiment output folder of this repository
        for folder in (RES, HERE.parent / "outputs" / "main_experiments"):
            for name in (f"{k}_results.json", f"{k}_all_methods.json"):
                if (folder / name).exists():
                    return folder / name
        return None
    missing = [k for k, _, _ in DATASETS if find(k) is None]
    if missing:
        print("Missing results for: " + ", ".join(missing) + " -> writing placeholders.")
        return None, gold
    data = {k: json.loads(find(k).read_text()) for k, _, _ in DATASETS}
    for k, _, _ in DATASETS:
        for m, _, _ in METHODS:
            if m not in data[k]:
                sys.exit(f"{k}: result file has no '{m}' entry")
    return data, gold


def write_table(data):
    rows = []
    for i, (m, group, label) in enumerate(METHODS):
        cells = []
        for key, _, _ in DATASETS:
            if data is None:
                cells.append(r"\ph{XX.XX} (\ph{XX.XX})")
                continue
            f1s = {mm: data[key][mm]["macro_f1"] for mm, _, _ in METHODS}
            r = data[key][m]
            txt = f"{r['macro_f1']:.2f} ({r['acc']:.2f})"
            if m == max(f1s, key=f1s.get):
                txt = f"\\best{{{txt}}}"
            elif m == min(f1s, key=f1s.get):
                txt = f"\\worst{{{txt}}}"
            cells.append(txt)
        rows.append(f"{group} & {label} & " + " & ".join(cells) + r" \\")
        if i == 1:
            rows.append(r"\midrule")
    head = " & ".join(rf"\textbf{{{n}}}" for _, n, _ in DATASETS)
    kline = " & ".join(f"($K={k}$)" for _, _, k in DATASETS)
    (OUT / "results_table.tex").write_text("\n".join([
        r"\begin{table*}[t]", r"\centering", r"\small", r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{llcccccc}", r"\toprule",
        rf" & \textbf{{Method}} & {head} \\", rf" & & {kline} \\", r"\midrule", *rows,
        r"\bottomrule", r"\end{tabular}", "}",
        r"\caption{Macro-F1 with accuracy in parentheses, both in \%, on the 400 test examples of each dataset. "
        r"Green marks the highest and red the lowest Macro-F1 in each column.}",
        r"\label{tab:main_results}", r"\end{table*}", ""]))


def compute_macros(data, gold):
    names = {k: n for k, n, _ in DATASETS}
    keys = [k for k, _, _ in DATASETS]
    f1 = lambda k, m: data[k][m]["macro_f1"]
    acc = lambda k, m: data[k][m]["acc"]
    g_std = {k: f1(k, "calibrated_kd") - f1(k, "standard_kd") for k in keys}
    g_bat = {k: f1(k, "calibrated_kd") - f1(k, "batch_calibrated_kd") for k in keys}
    kmax, kmin = max(g_std, key=g_std.get), min(g_std, key=g_std.get)
    mac = {
        "AvgGainStd": f"{np.mean(list(g_std.values())):.2f}",
        "AvgAccGainStd": f"{np.mean([acc(k, 'calibrated_kd') - acc(k, 'standard_kd') for k in keys]):.2f}",
        "WinsStd": str(sum(g > 0 for g in g_std.values())),
        "MaxGainStd": f"{g_std[kmax]:.2f}", "MaxGainStdData": names[kmax],
        "MinGainStd": f"{g_std[kmin]:.2f}", "MinGainStdData": names[kmin],
        "AvgGainBatch": f"{np.mean(list(g_bat.values())):.2f}",
        "WinsBatch": str(sum(g > 0 for g in g_bat.values())),
        "AvgTeacherCalGain": f"{np.mean([f1(k, 'teacher_calibrated') - f1(k, 'teacher_raw') for k in keys]):.2f}",
        "WinsTeacherCal": str(sum(f1(k, "teacher_calibrated") > f1(k, "teacher_raw") for k in keys)),
        "WinsTeacher": str(sum(f1(k, "calibrated_kd") > f1(k, "teacher_raw") for k in keys)),
    }
    collapsed = []  # (dataset, class, ckd count, batch count)
    for k in keys:
        cnt = data[k]["standard_kd"].get("counts")
        if cnt is None:
            print(f"  warning: {k} standard_kd has no counts; collapse statistics skip it")
            continue
        for i, (c, n) in enumerate(zip(data[k]["class_names"], cnt)):
            if n == 0:
                collapsed.append((names[k], c, data[k]["calibrated_kd"]["counts"][i],
                                  data[k]["batch_calibrated_kd"]["counts"][i]))
    items = [f"\\textit{{{c}}} on {d}" for d, c, _, _ in collapsed]
    lst = (", ".join(items[:-1]) + (", and " if len(items) > 2 else " and ") + items[-1]) if len(items) > 1 \
        else (items[0] if items else "no class")
    mac.update({
        "NumCollapsedStd": str(len(collapsed)), "CollapsedList": lst,
        "NumRecovered": str(sum(c > 0 for _, _, c, _ in collapsed)),
        "NumStillCollapsedBatch": str(sum(b == 0 for _, _, _, b in collapsed)),
    })
    notrec = [(d, c) for d, c, ck, _ in collapsed if ck == 0]
    if notrec:
        what = " and ".join(f"\\textit{{{c}}} on {d}" for d, c in notrec)
        mac["NotRecoveredText"] = (f"Null-Calibrated KD does not recover {what}. "
                                   "Its predictions for this class remain at zero.")
    else:
        mac["NotRecoveredText"] = "Null-Calibrated KD recovers every class that standard KD never predicts."
    sents = []
    for k in keys:
        d = data[k]
        for i, c in enumerate(d["class_names"]):
            if d["calibrated_kd"]["counts"][i] == 0 and d["standard_kd"]["counts"][i] > 0:
                z = d["null_logits"]
                gap = z[i] - max(v for j, v in enumerate(z) if j != i)
                letter = "ABCDEF"[i]
                if gap > 0:
                    sents.append(f"On {names[k]}, the null vector scores option {letter} (\\textit{{{c}}}) "
                                 f"{gap:.2f} logits above every other option.")
                else:
                    sents.append(f"On {names[k]}, the null vector scores option {letter} (\\textit{{{c}}}) "
                                 f"{-gap:.2f} logits below the highest option.")
                sents.append(f"After subtraction, the student never predicts \\textit{{{c}}}, "
                             f"although {gold[k][i]} test examples belong to this class.")
                sents.append(f"Its Macro-F1 on {names[k]} is {f1(k, 'calibrated_kd'):.2f}, "
                             f"compared with {f1(k, 'standard_kd'):.2f} for standard KD.")
    mac["OverCorrectionText"] = (" ".join(sents) if sents else
                                 "We find no class that Null-Calibrated KD never predicts while standard KD predicts it.")
    write_collapse_table(data, gold)
    for macro, m in [("GapStd", "standard_kd"), ("GapBatch", "batch_calibrated_kd"), ("GapCKD", "calibrated_kd")]:
        mac[macro] = str(int(sum(np.abs(np.array(data[k][m]["counts"]) - np.array(gold[k])).sum()
                                 for k, _ in DIST_PLOTS)))
    return mac, collapsed


def write_collapse_table(data, gold):
    names = {k: n for k, n, _ in DATASETS}
    rows = []
    if data is None:
        rows.append(r"\ph{[dataset]} & \ph{[class]} & \ph{X} & \ph{X} & \ph{X} & \ph{X} & \ph{X} \\")
    else:
        for k, _, _ in DATASETS:
            d = data[k]
            for i, c in enumerate(d["class_names"]):
                if any(d[m]["counts"][i] == 0 for m in STUDENT):
                    cells = [str(d[m]["counts"][i]) for m in ["teacher_raw"] + STUDENT]
                    cells = [f"\\textbf{{{v}}}" if v == "0" else v for v in cells]
                    rows.append(f"{names[k]} & {c} & {gold[k][i]} & " + " & ".join(cells) + r" \\")
    (OUT / "collapse_table.tex").write_text("\n".join([
        r"\begin{table}[t]", r"\centering", r"\small", r"\resizebox{\columnwidth}{!}{",
        r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"\textbf{Dataset} & \textbf{Class} & \textbf{True} & \textbf{Teacher} & \textbf{Std.} & \textbf{Batch} & \textbf{Null} \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", "}",
        r"\caption{Number of test predictions for every class that at least one distillation method never predicts. "
        r"True is the number of test examples in the class. Std., Batch and Null denote standard, "
        r"Batch-Calibrated and Null Calibrated KD. Bold marks zero predictions.}",
        r"\label{tab:collapse}", r"\end{table}", ""]))


def write_macros(mac):
    lines = ["% generated by make_results.py"] + [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items()]
    (OUT / "results_macros.tex").write_text("\n".join(lines) + "\n")


def bar_chart(data):
    labels = [f"{n} (K={k})" for _, n, k in DATASETS]
    x = np.arange(len(DATASETS)); w = 0.185
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.2))
    for ax, metric, title, ylab in [(axes[0], "macro_f1", "(a) Macro-F1 (%)", "Macro-F1 (%)"),
                                    (axes[1], "acc", "(b) Accuracy (%)", "Accuracy (%)")]:
        ax.set_facecolor("#FAFAFA"); ax.grid(axis="y", linestyle="--", alpha=0.5, color="#CBD5E1"); ax.set_axisbelow(True)
        for j, m in enumerate(PLOTTED):
            ax.bar(x + (j - 1.5) * w, [data[k][m][metric] for k, _, _ in DATASETS], w, color=COL[m][0],
                   edgecolor=COL[m][1], linewidth=1.2 if m == "calibrated_kd" else 0.9, zorder=3, label=LEG[m])
        ax.set_title(title, pad=12, color="#0F172A"); ax.set_ylabel(ylab, color="#0F172A")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", color="#1E293B")
        ax.set_ylim(0, 100); ax.set_yticks(np.arange(0, 101, 10))
        for s in ax.spines.values():
            s.set_color("#CBD5E1")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=True,
               facecolor="#F8FAFC", edgecolor="#E2E8F0")
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    plt.savefig(OUT / "fig_main_f1_accuracy.pdf", bbox_inches="tight"); plt.close()


def dist_chart(data, gold):
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.5))
    handles = None
    for ax, (key, title) in zip(axes, DIST_PLOTS):
        d = data[key]; cls = d["class_names"]; x = np.arange(len(cls)); w = 0.185
        ax.set_facecolor("#FAFAFA"); ax.grid(axis="y", linestyle="--", alpha=0.5, color="#CBD5E1"); ax.set_axisbelow(True)
        hs = [ax.bar(x + (j - 1.5) * w, d[m]["counts"], w, color=COL[m][0], edgecolor=COL[m][1],
                     linewidth=1.2 if m == "calibrated_kd" else 0.8, zorder=3) for j, m in enumerate(PLOTTED)]
        g, = ax.plot(x, gold[key], color="#111827", linestyle=":", linewidth=2.4, marker="o", markersize=6.5,
                     markerfacecolor="#F3F4F6", markeredgecolor="#111827", markeredgewidth=1.8, zorder=5)
        handles = handles or [g] + hs
        ax.set_title(f"{title} (K={len(cls)})", pad=12, color="#0F172A")
        ax.set_ylabel(f"Predicted count (N={sum(gold[key])})", color="#0F172A")
        ax.set_xticks(x); ax.set_xticklabels([c if len(c) < 12 else c[:9] for c in cls], rotation=20, ha="right")
        ax.set_ylim(0, max(max(max(d[m]["counts"]) for m in PLOTTED), max(gold[key])) + 40)
        for s in ax.spines.values():
            s.set_color("#CBD5E1")
    fig.legend(handles, ["True count"] + [LEG[m] for m in PLOTTED], loc="upper center",
               bbox_to_anchor=(0.5, 0.99), ncol=5, frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    plt.tight_layout(rect=[0, 0.04, 1, 0.90])
    plt.savefig(OUT / "fig_class_distributions.pdf", bbox_inches="tight"); plt.close()


def text_checks(mac, collapsed):
    """Flags results that would make a sentence in the paper wrong."""
    print("Numbers quoted in the text:")
    for k, v in mac.items():
        print(f"  \\{k} = {v}")
    warn = []
    if float(mac["AvgGainStd"]) <= 0:
        warn.append("Average gain over Standard KD is not positive: the abstract, introduction and conclusion say 'improves'.")
    if float(mac["AvgGainBatch"]) <= 0:
        warn.append("Average gain over Batch-Calibrated KD is not positive: check Section 5.1.")
    if not collapsed:
        warn.append("No class collapses under Standard KD: rewrite the collapse paragraph in Section 5.2.")
    for w in warn:
        print("WARNING: " + w)


if __name__ == "__main__":
    data, gold = load()
    write_table(data)
    if data is None:
        write_macros(MACROS)
        write_collapse_table(None, gold)
        for f in ("fig_main_f1_accuracy.pdf", "fig_class_distributions.pdf"):
            (OUT / f).unlink(missing_ok=True)
        print(f"Wrote placeholders to {OUT}; figures will show as placeholder boxes.")
        sys.exit(0)
    mac, collapsed = compute_macros(data, gold)
    assert set(mac) == set(MACROS), set(MACROS) ^ set(mac)
    write_macros(mac)
    bar_chart(data)
    dist_chart(data, gold)
    text_checks(mac, collapsed)
    print(f"Wrote {OUT}/results_table.tex, results_macros.tex and both figures.")
