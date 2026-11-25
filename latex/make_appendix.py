"""
Builds the appendix tables and figures from the experiment outputs.

Inputs:
  ../outputs/main_experiments/<dataset>_results.json   main experiments (per-class F1, counts, null logits)
  ../outputs/ablations/*/                              ablation results
  results/gold_counts.json                             ground-truth test class counts

Outputs (generated/):
  app_null_logits.tex          null-probe logit vector per dataset
  app_per_class_f1.tex         per-class F1 of every method on every dataset
  app_counts.tex               predicted counts per class vs. ground truth, every dataset
  app_ablation1.tex            cross-architecture distillation
  app_ablation2.tex            multi-task distillation (average and per task)
  app_ablation3.tex            hard pseudo-labels vs. soft targets
  app_ablation4.tex            permutation-averaged teacher vs. CKD
  app_ablation5.tex            three-seed mean and standard deviation
  fig_app_null_logits.pdf      centred null logits per dataset
  fig_app_class_dist_all.pdf   predicted class counts on all six datasets
  fig_app_multitask.pdf        multi-task Macro-F1 per task
  fig_app_multiseed.pdf        three-seed Macro-F1 with error bars

Usage:  python3 make_appendix.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
MAIN = HERE.parent / "outputs" / "main_experiments"
ABL = HERE.parent / "outputs" / "ablations"
OUT = HERE / "generated"
OUT.mkdir(exist_ok=True)

DATASETS = [
    ("student_questions", "Student Questions"),
    ("trec6", "TREC-6"),
    ("app_reviews", "App Reviews"),
    ("twitter_emotion", "Twitter Emotion"),
    ("tweeteval_emotion", "TweetEval Emotion"),
    ("bbc_news", "BBC News"),
]
NAME = dict(DATASETS)
METHODS = [
    ("teacher_raw", "Teacher, raw"),
    ("teacher_calibrated", "Teacher, null-cal."),
    ("standard_kd", "Standard KD"),
    ("batch_calibrated_kd", "Batch-Cal. KD"),
    ("calibrated_kd", "Null-Cal. KD (ours)"),
]
STUDENT = ["standard_kd", "batch_calibrated_kd", "calibrated_kd"]

COL = {"teacher_raw": ("#94A3B8", "#475569"), "standard_kd": ("#E15759", "#B91C1C"),
       "batch_calibrated_kd": ("#8DA0CB", "#4F46E5"), "calibrated_kd": ("#66C2A5", "#0D9488")}
LEG = {"teacher_raw": "Teacher, zero-shot (Gemma-3-4B-IT)", "standard_kd": "Standard KD",
       "batch_calibrated_kd": "Batch-Calibrated KD", "calibrated_kd": "Null Calibrated KD (ours)"}
PLOTTED = ["teacher_raw", "standard_kd", "batch_calibrated_kd", "calibrated_kd"]

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10.5, "axes.titlesize": 12.0,
                     "axes.titleweight": "bold", "axes.labelsize": 11, "axes.labelweight": "bold",
                     "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 10.0})


def jload(p):
    return json.loads(Path(p).read_text())


def f(x):
    return f"{x:.2f}"


def bold_if(x, cond):
    return rf"\textbf{{{f(x)}}}" if cond else f(x)


def delta(x):
    s = f"{x:+.2f}"
    return rf"\textcolor{{textgreen}}{{{s}}}" if x > 0 else rf"\textcolor{{textred}}{{{s}}}"


def style(ax):
    ax.set_facecolor("#FAFAFA")
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="#CBD5E1")
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("#CBD5E1")


def write(name, body):
    (OUT / name).write_text(body.strip() + "\n")
    print(f"  wrote generated/{name}")


# ---------------------------------------------------------------- main-experiment extras
def null_logits_table(data):
    rows = []
    for k, n in DATASETS:
        d = data[k]
        z = np.array(d["null_logits"])
        top = d["class_names"][int(z.argmax())]
        cells = ", ".join(f"{c}~{v:.2f}" for c, v in zip(d["class_names"], z))
        rows.append(rf"{n} & {cells} & {top} & {z.max() - z.min():.2f} \\")
    write("app_null_logits.tex", r"""
\begin{table*}[t]
\centering
\small
\resizebox{\textwidth}{!}{
\begin{tabular}{l>{\raggedright\arraybackslash}p{0.62\textwidth}lc}
\toprule
\textbf{Dataset} & \textbf{Null logits $z_{\text{null}}$ (class~logit)} & \textbf{Favoured class} & \textbf{Gap (max$-$min)} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
}
\caption{Gemma-3-4B-IT logits for each option letter when the prompt contains ``N/A'' instead of a real input. The favoured class is the one the teacher prefers before it reads any text. A larger gap means a stronger prior bias.}
\label{tab:app_null_logits}
\end{table*}
""")


def null_logits_fig(data):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2))
    for ax, (k, n) in zip(axes.flat, DATASETS):
        d = data[k]
        z = np.array(d["null_logits"]); zc = z - z.mean()
        cols = ["#E15759" if v > 0 else "#8DA0CB" for v in zc]
        ax.bar(np.arange(len(z)), zc, color=cols, edgecolor="#334155", linewidth=0.7, zorder=3)
        ax.axhline(0, color="#111827", linewidth=1)
        style(ax)
        ax.set_title(n, pad=8, color="#0F172A")
        ax.set_xticks(np.arange(len(z)))
        ax.set_xticklabels([c if len(c) < 12 else c[:9] for c in d["class_names"]], rotation=20, ha="right")
        ax.set_ylabel(r"$z_{null} - \mathrm{mean}$")
    plt.tight_layout()
    plt.savefig(OUT / "fig_app_null_logits.pdf", bbox_inches="tight"); plt.close()
    print("  wrote generated/fig_app_null_logits.pdf")


def per_class_f1_table(data):
    blocks = []
    for k, n in DATASETS:
        d = data[k]; cls = d["class_names"]
        best = [max(d[m]["per_class_f1"][i] for m, _ in METHODS) for i in range(len(cls))]
        lines = [rf"\multicolumn{{{len(cls) + 2}}}{{l}}{{\textit{{{n}}}}} \\",
                 " & " + " & ".join(rf"\textbf{{{c}}}" for c in cls) + r" & \textbf{Worst} \\"]
        for m, lab in METHODS:
            pc = d[m]["per_class_f1"]
            cells = [rf"\worst{{0.00}}" if v == 0 else bold_if(v, abs(v - best[i]) < 1e-9) for i, v in enumerate(pc)]
            lines.append(f"{lab} & " + " & ".join(cells) + f" & {f(min(pc))} \\\\")
        blocks.append((len(cls), "\n".join(lines)))
    # six-class blocks set the column count; shorter blocks leave trailing cells empty
    ncol = max(b[0] for b in blocks) + 2
    body = "\n\\midrule\n".join(b for _, b in blocks)
    write("app_per_class_f1.tex", r"""
\begin{table*}[t]
\centering
\small
\resizebox{\textwidth}{!}{
\begin{tabular}{l""" + "c" * (ncol - 1) + r"""}
\toprule
""" + body + r"""
\bottomrule
\end{tabular}
}
\caption{Per-class F1 (\%) on the 400 test examples of each dataset. Bold marks the best method for each class, red marks a class with F1 of 0 (never predicted correctly). Worst is the lowest per-class F1 of the row.}
\label{tab:app_per_class_f1}
\end{table*}
""")


def counts_table(data, gold):
    rows = []
    for k, n in DATASETS:
        d = data[k]
        for i, c in enumerate(d["class_names"]):
            cells = []
            for m, _ in METHODS:
                v = d[m]["counts"][i]
                cells.append(r"\textbf{0}" if v == 0 else str(v))
            ds = n if i == 0 else ""
            rows.append(f"{ds} & {c} & {gold[k][i]} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    write("app_counts.tex", r"""
\begin{table*}[p]
\centering
\small
\begin{tabular}{llrrrrrr}
\toprule
\textbf{Dataset} & \textbf{Class} & \textbf{True} & \textbf{Teacher} & \textbf{Teacher+Null} & \textbf{Std.} & \textbf{Batch} & \textbf{Null (ours)} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Number of test predictions per class for every dataset and method. True is the number of test examples in the class. Bold marks a class that is never predicted.}
\label{tab:app_counts}
\end{table*}
""")


def class_dist_all(data, gold):
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 10))
    handles = None
    for ax, (k, n) in zip(axes.flat, DATASETS):
        d = data[k]; cls = d["class_names"]; x = np.arange(len(cls)); w = 0.185
        style(ax)
        hs = [ax.bar(x + (j - 1.5) * w, d[m]["counts"], w, color=COL[m][0], edgecolor=COL[m][1],
                     linewidth=1.2 if m == "calibrated_kd" else 0.8, zorder=3) for j, m in enumerate(PLOTTED)]
        g, = ax.plot(x, gold[k], color="#111827", linestyle=":", linewidth=2.2, marker="o", markersize=6,
                     markerfacecolor="#F3F4F6", markeredgecolor="#111827", markeredgewidth=1.6, zorder=5)
        handles = handles or [g] + hs
        ax.set_title(f"{n} (K={len(cls)})", pad=10, color="#0F172A")
        ax.set_ylabel(f"Predicted count (N={sum(gold[k])})")
        ax.set_xticks(x); ax.set_xticklabels([c if len(c) < 12 else c[:9] for c in cls], rotation=20, ha="right")
        ax.set_ylim(0, max(max(max(d[m]["counts"]) for m in PLOTTED), max(gold[k])) + 30)
    fig.legend(handles, ["True count"] + [LEG[m] for m in PLOTTED], loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUT / "fig_app_class_dist_all.pdf", bbox_inches="tight"); plt.close()
    print("  wrote generated/fig_app_class_dist_all.pdf")


# ---------------------------------------------------------------- ablations
def ablation1():
    d = jload(ABL / "1_cross_architecture" / "cross_architecture_summary.json")
    labels = {"config1_same_size": "Same-size", "config2_cross_family": "Cross-family",
              "config3_small_slm": "Small SLM"}
    rows = []
    for key, lab in labels.items():
        r = d[key]; s, c = r["standard_kd"], r["calibrated_kd"]
        rows.append(f"{lab} & {r['teacher']} & {r['student']} & {NAME[r['dataset']]} & "
                    f"{bold_if(s['macro_f1'], s['macro_f1'] > c['macro_f1'])} ({f(s['acc'])}) & "
                    f"{bold_if(c['macro_f1'], c['macro_f1'] > s['macro_f1'])} ({f(c['acc'])}) & "
                    f"{delta(r['delta_f1'])} \\\\")
    write("app_ablation1.tex", r"""
\begin{table*}[t]
\centering
\small
\resizebox{\textwidth}{!}{
\begin{tabular}{llllccc}
\toprule
\textbf{Setting} & \textbf{Teacher} & \textbf{Student} & \textbf{Dataset} & \textbf{Standard KD} & \textbf{Null-Cal. KD (ours)} & \textbf{$\Delta$ F1 (pp)} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
}
\caption{Ablation 1: other teacher and student architectures. Macro-F1 with accuracy in parentheses, both in \%. The LoRA students are causal LMs trained to predict the option letter.}
\label{tab:app_ablation1}
\end{table*}
""")


def ablation2():
    combos = []
    for p in sorted((ABL / "2_multitask").glob("combo*.json")):
        combos.append(jload(p))
    combos.sort(key=lambda c: c["combo_id"])
    rows = []
    for c in combos:
        tasks = " + ".join(NAME[t] for t in c["tasks"])
        rows.append(f"{c['combo_id']} & {tasks} & {f(c['std_kd_macro_f1'])} & "
                    rf"\textbf{{{f(c['ckd_macro_f1'])}}} & {delta(c['delta_macro_f1'])} \\")
    write("app_ablation2.tex", r"""
\begin{table*}[t]
\centering
\small
\resizebox{\textwidth}{!}{
\begin{tabular}{clccc}
\toprule
\textbf{Combo} & \textbf{Tasks sharing one student} & \textbf{Standard KD} & \textbf{Null-Cal. KD (ours)} & \textbf{$\Delta$ F1 (pp)} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
}
\caption{Ablation 2: multi-task distillation into one RoBERTa-base encoder with a separate classification head per task. Macro-F1 (\%) averaged over the tasks of each combination.}
\label{tab:app_ablation2}
\end{table*}
""")
    # per-task breakdown
    rows = []
    for c in combos:
        for i, t in enumerate(c["tasks"]):
            s = c["details"]["std_kd"]["per_task"][t]; k = c["details"]["ckd"]["per_task"][t]
            cid = str(c["combo_id"]) if i == 0 else ""
            rows.append(f"{cid} & {NAME[t]} & {f(s['macro_f1'])} ({f(s['accuracy'])}) & "
                        f"{f(k['macro_f1'])} ({f(k['accuracy'])}) & {delta(k['macro_f1'] - s['macro_f1'])} \\\\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    write("app_ablation2_per_task.tex", r"""
\begin{table}[t]
\centering
\small
\resizebox{\columnwidth}{!}{
\begin{tabular}{clccc}
\toprule
\textbf{Combo} & \textbf{Task} & \textbf{Standard KD} & \textbf{Null-Cal. KD} & \textbf{$\Delta$ F1} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
}
\caption{Ablation 2, per task: Macro-F1 with accuracy in parentheses, both in \%.}
\label{tab:app_ablation2_per_task}
\end{table}
""")
    # figure
    fig, axes = plt.subplots(1, len(combos), figsize=(3.6 * len(combos), 4.4), sharey=True,
                             gridspec_kw={"width_ratios": [len(c["tasks"]) for c in combos]})
    for ax, c in zip(axes, combos):
        style(ax)
        x = np.arange(len(c["tasks"])); w = 0.38
        s = [c["details"]["std_kd"]["per_task"][t]["macro_f1"] for t in c["tasks"]]
        k = [c["details"]["ckd"]["per_task"][t]["macro_f1"] for t in c["tasks"]]
        ax.bar(x - w / 2, s, w, color=COL["standard_kd"][0], edgecolor=COL["standard_kd"][1], zorder=3, label=LEG["standard_kd"])
        ax.bar(x + w / 2, k, w, color=COL["calibrated_kd"][0], edgecolor=COL["calibrated_kd"][1], zorder=3, label=LEG["calibrated_kd"])
        ax.set_xticks(x); ax.set_xticklabels([NAME[t].replace(" ", "\n", 1) for t in c["tasks"]], fontsize=8.5)
        ax.set_title(f"Combo {c['combo_id']}", pad=8)
    axes[0].set_ylabel("Macro-F1 (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04), frameon=True,
               facecolor="#F8FAFC", edgecolor="#CBD5E1")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(OUT / "fig_app_multitask.pdf", bbox_inches="tight"); plt.close()
    print("  wrote generated/fig_app_multitask.pdf")


def ablation3():
    d = jload(ABL / "3_hard_vs_soft" / "hard_vs_soft_results.json")
    regimes = [
        ("raw_hard", r"Raw hard labels", r"$\arg\max z_{\text{raw}}$", "Cross-entropy"),
        ("calibrated_hard", r"Calibrated hard labels", r"$\arg\max (z_{\text{raw}} - z_{\text{null}})$", "Cross-entropy"),
        ("standard_soft_kd", r"Standard soft KD", r"$\text{softmax}(z_{\text{raw}})$", "KL"),
        ("calibrated_soft_kd", r"Null-Cal. soft KD (ours)", r"$\text{softmax}(z_{\text{raw}} - z_{\text{null}})$", "KL"),
    ]
    rows = []
    for k in d:
        best = max(d[k][r]["macro_f1"] for r, *_ in regimes)
        for i, (r, lab, tgt, loss) in enumerate(regimes):
            v = d[k][r]
            ds = NAME[k] if i == 0 else ""
            rows.append(f"{ds} & {lab} & {tgt} & {loss} & {bold_if(v['macro_f1'], v['macro_f1'] == best)} & {f(v['acc'])} \\\\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    write("app_ablation3.tex", r"""
\begin{table*}[t]
\centering
\small
\begin{tabular}{lllccc}
\toprule
\textbf{Dataset} & \textbf{Regime} & \textbf{Target} & \textbf{Loss} & \textbf{Macro-F1} & \textbf{Accuracy} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Ablation 3: hard pseudo-labels (the teacher's top class) against soft targets (the teacher's full distribution), with and without null calibration. RoBERTa-base student, results in \%.}
\label{tab:app_ablation3}
\end{table*}
""")


def ablation4():
    d = jload(ABL / "4_permutation_kd" / "permutation_kd_results.json")
    rows = []
    for k in d:
        r = d[k]; passes = r["permutation_averaged_kd"]["passes"]
        ms = ["standard_kd", "permutation_averaged_kd", "calibrated_kd"]
        bf = max(r[m]["macro_f1"] for m in ms); ba = max(r[m]["acc"] for m in ms)
        rows.append(f"{NAME[k]} & Macro-F1 & " + " & ".join(bold_if(r[m]["macro_f1"], r[m]["macro_f1"] == bf) for m in ms) + r" \\")
        rows.append(" & Accuracy & " + " & ".join(bold_if(r[m]["acc"], r[m]["acc"] == ba) for m in ms) + r" \\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    write("app_ablation4.tex", r"""
\begin{table}[t]
\centering
\small
\resizebox{\columnwidth}{!}{
\begin{tabular}{llccc}
\toprule
\textbf{Dataset} & \textbf{Metric} & \textbf{Std. KD} & \textbf{Perm.-Avg. KD} & \textbf{Null-Cal. KD} \\
 & & ($1\times$) & ($""" + str(passes) + r"""\times$) & ($1\times$) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
}
\caption{Ablation 4: permutation-averaged teacher targets against null calibration. The number in parentheses is the number of teacher passes per training example. Results in \%.}
\label{tab:app_ablation4}
\end{table}
""")


def ablation5():
    d = jload(ABL / "5_multi_seed" / "multi_seed_summary.json")
    rows = []
    for k, n in DATASETS:
        r = d[k]
        best = max(r[m]["f1_mean"] for m in STUDENT)
        cells = []
        for m in STUDENT:
            v = r[m]
            s = rf"{v['f1_mean']:.2f} $\pm$ {v['f1_std']:.2f} ({v['acc_mean']:.2f} $\pm$ {v['acc_std']:.2f})"
            cells.append(rf"\textbf{{{s}}}" if v["f1_mean"] == best else s)
        wins = sum(a["macro_f1"] > b["macro_f1"] for a, b in zip(r["calibrated_kd"]["seeds"], r["standard_kd"]["seeds"]))
        rows.append(f"{n} & " + " & ".join(cells) + f" & {delta(r['delta_f1_mean'])} & {wins}/{len(r['calibrated_kd']['seeds'])} \\\\")
    write("app_ablation5.tex", r"""
\begin{table*}[t]
\centering
\small
\resizebox{\textwidth}{!}{
\begin{tabular}{lccccc}
\toprule
\textbf{Dataset} & \textbf{Standard KD} & \textbf{Batch-Cal. KD} & \textbf{Null-Cal. KD (ours)} & \textbf{$\Delta$ F1 vs. Std.} & \textbf{Seed wins} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
}
\caption{Ablation 5: mean $\pm$ standard deviation over student seeds 42, 43 and 44. Macro-F1 with accuracy in parentheses, both in \%. The teacher targets are the same for every seed. Seed wins counts the seeds on which null-calibrated KD has a higher Macro-F1 than standard KD.}
\label{tab:app_ablation5}
\end{table*}
""")
    fig, ax = plt.subplots(figsize=(13, 4.4))
    style(ax)
    x = np.arange(len(DATASETS)); w = 0.26
    for j, m in enumerate(STUDENT):
        ax.bar(x + (j - 1) * w, [d[k][m]["f1_mean"] for k, _ in DATASETS], w,
               yerr=[d[k][m]["f1_std"] for k, _ in DATASETS], capsize=4,
               color=COL[m][0], edgecolor=COL[m][1], zorder=3, label=LEG[m], error_kw={"elinewidth": 1.2})
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in DATASETS])
    ax.set_ylabel("Macro-F1 (%), mean of 3 seeds")
    ax.legend(loc="upper left", frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    plt.tight_layout()
    plt.savefig(OUT / "fig_app_multiseed.pdf", bbox_inches="tight"); plt.close()
    print("  wrote generated/fig_app_multiseed.pdf")


def main():
    data = {k: jload(MAIN / f"{k}_results.json") for k, _ in DATASETS}
    gold = jload(HERE / "results" / "gold_counts.json")
    null_logits_table(data)
    null_logits_fig(data)
    per_class_f1_table(data)
    counts_table(data, gold)
    class_dist_all(data, gold)
    ablation1()
    ablation2()
    ablation3()
    ablation4()
    ablation5()


if __name__ == "__main__":
    main()
