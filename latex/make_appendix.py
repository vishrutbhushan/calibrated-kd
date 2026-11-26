"""
Builds the appendix tables and figures from the experiment outputs.

Every table and figure is sized for one column, so the appendix floats can be
placed next to the text instead of being pushed to the top of a later page.

Inputs:
  ../outputs/main_experiments/<dataset>_results.json   main experiments (per-class F1, counts, null logits)
  ../outputs/ablations/*/                              ablation results
  results/gold_counts.json                             ground-truth test class counts

Outputs (generated/):
  app_null_logits.tex          null-probe logit vector per dataset
  app_per_class_f1.tex         per-class F1 of every method on every dataset
  app_counts.tex               predicted counts per class vs. ground truth, every dataset
  app_ablation1.tex            cross-architecture distillation
  app_ablation2.tex            multi-task distillation (average over tasks)
  app_ablation2_per_task.tex   multi-task distillation (each task)
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
SHORT_CLASS = {"Abbreviation": "Abbr.", "Description": "Desc.", "Entertainment": "Entert."}

COL = {"teacher_raw": ("#94A3B8", "#475569"), "standard_kd": ("#E15759", "#B91C1C"),
       "batch_calibrated_kd": ("#8DA0CB", "#4F46E5"), "calibrated_kd": ("#66C2A5", "#0D9488")}
LEG = {"teacher_raw": "Teacher, zero-shot", "standard_kd": "Standard KD",
       "batch_calibrated_kd": "Batch-Calibrated KD", "calibrated_kd": "Null Calibrated KD (ours)"}
PLOTTED = ["teacher_raw", "standard_kd", "batch_calibrated_kd", "calibrated_kd"]

# Figures are drawn 7 in wide and shown one column (about 3 in) wide, so fonts are set large.
COLUMN_FIG_W = 7.0
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 14, "axes.titlesize": 15,
                     "axes.titleweight": "bold", "axes.labelsize": 14, "axes.labelweight": "bold",
                     "xtick.labelsize": 12.5, "ytick.labelsize": 12.5, "legend.fontsize": 13})

# Shrinks a table to the column width when it is wider, never enlarges it.
FIT = r"\resizebox{\ifdim\width>\columnwidth\columnwidth\else\width\fi}{!}{%"


def jload(p):
    return json.loads(Path(p).read_text())


def f(x):
    return f"{x:.2f}"


def bold_if(x, cond):
    return rf"\textbf{{{f(x)}}}" if cond else f(x)


def delta(x):
    s = f"{x:+.2f}"
    return rf"\textcolor{{textgreen}}{{{s}}}" if x > 0 else rf"\textcolor{{textred}}{{{s}}}"


def stack(*lines, align="c"):
    """Several lines in one table cell."""
    return rf"\begin{{tabular}}[c]{{@{{}}{align}@{{}}}}" + r"\\".join(lines) + r"\end{tabular}"


def short(c):
    return SHORT_CLASS.get(c, c)


def style(ax):
    ax.set_facecolor("#FAFAFA")
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="#CBD5E1")
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("#CBD5E1")


def table(name, colspec, header, rows, caption, label, size=r"\footnotesize"):
    write(name, rf"""
\begin{{table}}[!ht]
\centering
{size}
\setlength{{\tabcolsep}}{{4pt}}
{FIT}
\begin{{tabular}}{{{colspec}}}
\toprule
{header}
\midrule
""" + "\n".join(rows) + rf"""
\bottomrule
\end{{tabular}}}}
\caption{{{caption}}}
\label{{{label}}}
\end{{table}}
""")


def write(name, body):
    (OUT / name).write_text(body.strip() + "\n")
    print(f"  wrote generated/{name}")


def save(fig_name):
    plt.savefig(OUT / fig_name, bbox_inches="tight"); plt.close()
    print(f"  wrote generated/{fig_name}")


def group(n, ncol):
    return rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\textit{{{n}}}}} \\"


# ---------------------------------------------------------------- main-experiment extras
def null_logits_table(data):
    rows = []
    for k, n in DATASETS:
        d = data[k]
        z = np.array(d["null_logits"])
        top = short(d["class_names"][int(z.argmax())])
        cells = ", ".join(f"{short(c)}~{v:.2f}" for c, v in zip(d["class_names"], z))
        rows.append(rf"{n} & {cells} & {top} & {z.max() - z.min():.2f} \\")
    table("app_null_logits.tex",
          r"@{}>{\raggedright\arraybackslash}p{0.2\columnwidth}>{\raggedright\arraybackslash}p{0.48\columnwidth}lc@{}",
          r"\textbf{Dataset} & \textbf{Null logits $z_{\text{null}}$} & \textbf{Top} & \textbf{Gap} \\",
          rows,
          r"Gemma-3-4B-IT logits for each option when the prompt contains ``N/A'' instead of a real input. Top is the class the teacher prefers before it reads any text. Gap is the largest minus the smallest logit; a larger gap means a stronger prior bias.",
          "tab:app_null_logits", size=r"\scriptsize")


def null_logits_fig(data):
    fig, axes = plt.subplots(3, 2, figsize=(COLUMN_FIG_W, 7.2))
    for ax, (k, n) in zip(axes.flat, DATASETS):
        d = data[k]
        z = np.array(d["null_logits"]); zc = z - z.mean()
        cols = ["#E15759" if v > 0 else "#8DA0CB" for v in zc]
        ax.bar(np.arange(len(z)), zc, color=cols, edgecolor="#334155", linewidth=0.7, zorder=3)
        ax.axhline(0, color="#111827", linewidth=1)
        style(ax)
        ax.set_title(n, pad=6, color="#0F172A")
        ax.set_xticks(np.arange(len(z)))
        ax.set_xticklabels([short(c) for c in d["class_names"]], rotation=30, ha="right")
    for ax in axes[:, 0]:
        ax.set_ylabel("Centred logit")
    plt.tight_layout(h_pad=0.6, w_pad=0.8)
    save("fig_app_null_logits.pdf")


def per_class_f1_table(data):
    ncol = max(len(data[k]["class_names"]) for k, _ in DATASETS) + 2
    rows = []
    for k, n in DATASETS:
        d = data[k]; cls = d["class_names"]; pad = " &" * (ncol - len(cls) - 2)
        best = [max(d[m]["per_class_f1"][i] for m, _ in METHODS) for i in range(len(cls))]
        rows.append(rf"\textit{{{n}}} & " + " & ".join(rf"\textit{{{short(c)}}}" for c in cls) + pad + r" & \textit{Worst} \\")
        for m, lab in METHODS:
            pc = d[m]["per_class_f1"]
            cells = [r"\worst{0.00}" if v == 0 else bold_if(v, abs(v - best[i]) < 1e-9) for i, v in enumerate(pc)]
            rows.append(f"{lab} & " + " & ".join(cells) + pad + f" & {f(min(pc))} \\\\")
        rows.append(r"\midrule")
    table("app_per_class_f1.tex", "@{}l" + "r" * (ncol - 1) + "@{}",
          r"\textbf{Method} & \multicolumn{" + str(ncol - 2) + r"}{c}{\textbf{F1 of each class}} & \textbf{Worst} \\",
          rows[:-1],
          r"Per-class F1 (\%) on the 400 test examples of each dataset. Bold marks the best method for each class, red marks a class with F1 of 0. Worst is the lowest per-class F1 of the row.",
          "tab:app_per_class_f1", size=r"\scriptsize")


def counts_table(data, gold):
    rows = []
    for k, n in DATASETS:
        d = data[k]
        rows.append(group(n, 7))
        for i, c in enumerate(d["class_names"]):
            cells = [r"\textbf{0}" if d[m]["counts"][i] == 0 else str(d[m]["counts"][i]) for m, _ in METHODS]
            rows.append(rf"\quad {c} & {gold[k][i]} & " + " & ".join(cells) + r" \\")
    table("app_counts.tex", "@{}lrrrrrr@{}",
          r"\textbf{Class} & \textbf{True} & \textbf{Teach.} & \textbf{T.+Null} & \textbf{Std.} & \textbf{Batch} & \textbf{Null} \\",
          rows,
          r"Number of test predictions per class for every dataset and method. True is the number of test examples in the class. Teach.\ and T.+Null are the raw and null-calibrated teacher; Std., Batch and Null are the three distillation methods. Bold marks a class that is never predicted.",
          "tab:app_counts", size=r"\scriptsize")


def class_dist_all(data, gold):
    fig, axes = plt.subplots(3, 2, figsize=(COLUMN_FIG_W, 9.0))
    handles = None
    for ax, (k, n) in zip(axes.flat, DATASETS):
        d = data[k]; cls = d["class_names"]; x = np.arange(len(cls)); w = 0.2
        style(ax)
        hs = [ax.bar(x + (j - 1.5) * w, d[m]["counts"], w, color=COL[m][0], edgecolor=COL[m][1],
                     linewidth=0.9 if m == "calibrated_kd" else 0.6, zorder=3) for j, m in enumerate(PLOTTED)]
        g, = ax.plot(x, gold[k], color="#111827", linestyle=":", linewidth=1.8, marker="o", markersize=4.5,
                     markerfacecolor="#F3F4F6", markeredgecolor="#111827", markeredgewidth=1.3, zorder=5)
        handles = handles or [g] + hs
        ax.set_title(n, pad=6, color="#0F172A")
        ax.set_xticks(x); ax.set_xticklabels([short(c) for c in cls], rotation=30, ha="right")
        ax.set_ylim(0, max(max(max(d[m]["counts"]) for m in PLOTTED), max(gold[k])) * 1.08)
    for ax in axes[:, 0]:
        ax.set_ylabel("Predictions")
    fig.legend(handles, ["True count"] + [LEG[m] for m in PLOTTED], loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    plt.tight_layout(rect=[0, 0, 1, 0.9], h_pad=0.6, w_pad=0.8)
    save("fig_app_class_dist_all.pdf")


# ---------------------------------------------------------------- ablations
def f1_acc(r, bold=False):
    top = rf"\textbf{{{f(r['macro_f1'])}}}" if bold else f(r["macro_f1"])
    return stack(top, f"({f(r['acc'])})")


def ablation1():
    d = jload(ABL / "1_cross_architecture" / "cross_architecture_summary.json")
    labels = {"config1_same_size": "Same-size", "config2_cross_family": "Cross-family",
              "config3_small_slm": "Small SLM"}
    rows = []
    for key, lab in labels.items():
        r = d[key]; s, c = r["standard_kd"], r["calibrated_kd"]
        setting = stack(rf"\textbf{{{lab}}}: {r['teacher']} $\rightarrow$ {r['student']}",
                        rf"\textit{{{NAME[r['dataset']]}}}", align="l")
        rows.append(f"{setting} & {f1_acc(s, s['macro_f1'] > c['macro_f1'])} & "
                    f"{f1_acc(c, c['macro_f1'] > s['macro_f1'])} & {delta(r['delta_f1'])} \\\\[2pt]")
    table("app_ablation1.tex", "@{}lccc@{}",
          r"\textbf{Teacher $\rightarrow$ student, dataset} & \textbf{Std. KD} & \textbf{Null-Cal.} & \textbf{$\Delta$ F1} \\",
          rows,
          r"Ablation 1: other teacher and student architectures. Macro-F1 with accuracy in parentheses, both in \%. The LoRA students are causal LMs trained to predict the option letter.",
          "tab:app_ablation1")


def ablation2():
    combos = sorted((jload(p) for p in (ABL / "2_multitask").glob("combo*.json")), key=lambda c: c["combo_id"])
    rows = []
    for c in combos:
        tasks = " + ".join(NAME[t] for t in c["tasks"])
        rows.append(f"{c['combo_id']} & {tasks} & {f(c['std_kd_macro_f1'])} & "
                    rf"\textbf{{{f(c['ckd_macro_f1'])}}} & {delta(c['delta_macro_f1'])} \\")
    table("app_ablation2.tex", r"@{}c>{\raggedright\arraybackslash}p{0.42\columnwidth}ccc@{}",
          r"\textbf{\#} & \textbf{Tasks sharing one student} & \textbf{Std. KD} & \textbf{Null-Cal.} & \textbf{$\Delta$ F1} \\",
          rows,
          r"Ablation 2: multi-task distillation into one RoBERTa-base encoder with a separate classification head per task. Macro-F1 (\%) averaged over the tasks of each combination.",
          "tab:app_ablation2")
    rows = []
    for c in combos:
        for i, t in enumerate(c["tasks"]):
            s = c["details"]["std_kd"]["per_task"][t]; k = c["details"]["ckd"]["per_task"][t]
            cid = str(c["combo_id"]) if i == 0 else ""
            rows.append(f"{cid} & {NAME[t]} & {f(s['macro_f1'])} ({f(s['accuracy'])}) & "
                        f"{f(k['macro_f1'])} ({f(k['accuracy'])}) & {delta(k['macro_f1'] - s['macro_f1'])} \\\\")
        rows.append(r"\midrule")
    table("app_ablation2_per_task.tex", "@{}clccc@{}",
          r"\textbf{\#} & \textbf{Task} & \textbf{Std. KD} & \textbf{Null-Cal.} & \textbf{$\Delta$ F1} \\",
          rows[:-1],
          r"Ablation 2, per task: Macro-F1 with accuracy in parentheses, both in \%.",
          "tab:app_ablation2_per_task", size=r"\scriptsize")
    # one axis, tasks grouped by combination
    fig, ax = plt.subplots(figsize=(COLUMN_FIG_W, 3.8))
    style(ax)
    xs, labels, pos = [], [], 0.0
    for c in combos:
        start = pos
        for t in c["tasks"]:
            xs.append(pos); labels.append(NAME[t].replace("Emotion", "Emo.").replace("Reviews", "Rev."))
            pos += 1
        ax.text((start + pos - 1) / 2, 101, f"Combo {c['combo_id']}", ha="center", va="bottom", fontsize=12, fontweight="bold")
        if c is not combos[-1]:
            ax.axvline(pos - 0.25, color="#94A3B8", linewidth=0.9)
        pos += 0.5
    s = [c["details"]["std_kd"]["per_task"][t]["macro_f1"] for c in combos for t in c["tasks"]]
    k = [c["details"]["ckd"]["per_task"][t]["macro_f1"] for c in combos for t in c["tasks"]]
    xs = np.array(xs); w = 0.38
    ax.bar(xs - w / 2, s, w, color=COL["standard_kd"][0], edgecolor=COL["standard_kd"][1], zorder=3, label=LEG["standard_kd"])
    ax.bar(xs + w / 2, k, w, color=COL["calibrated_kd"][0], edgecolor=COL["calibrated_kd"][1], zorder=3, label=LEG["calibrated_kd"])
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=11)
    ax.set_ylim(0, 100); ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    ax.set_ylabel("Macro-F1 (%)")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.28), frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    save("fig_app_multitask.pdf")


def ablation3():
    d = jload(ABL / "3_hard_vs_soft" / "hard_vs_soft_results.json")
    regimes = [
        ("raw_hard", "Raw hard", r"$\arg\max z_{\text{raw}}$", "CE"),
        ("calibrated_hard", "Calibrated hard", r"$\arg\max (z_{\text{raw}} - z_{\text{null}})$", "CE"),
        ("standard_soft_kd", "Standard soft", r"$\text{softmax}(z_{\text{raw}})$", "KL"),
        ("calibrated_soft_kd", "Null-cal. soft (ours)", r"$\text{softmax}(z_{\text{raw}} - z_{\text{null}})$", "KL"),
    ]
    rows = []
    for k in d:
        best = max(d[k][r]["macro_f1"] for r, *_ in regimes)
        best_acc = max(d[k][r]["acc"] for r, *_ in regimes)
        rows.append(group(NAME[k], 5))
        for r, lab, tgt, loss in regimes:
            v = d[k][r]
            rows.append(rf"\quad {lab} & {tgt} & {loss} & {bold_if(v['macro_f1'], v['macro_f1'] == best)} & {bold_if(v['acc'], v['acc'] == best_acc)} \\")
    table("app_ablation3.tex", "@{}lllcc@{}",
          r"\textbf{Regime} & \textbf{Target} & \textbf{Loss} & \textbf{F1} & \textbf{Acc.} \\",
          rows,
          r"Ablation 3: hard pseudo-labels (the teacher's top class, cross-entropy loss) against soft targets (the teacher's full distribution, KL loss), with and without null calibration. RoBERTa-base student, Macro-F1 and accuracy in \%.",
          "tab:app_ablation3")


def ablation4():
    d = jload(ABL / "4_permutation_kd" / "permutation_kd_results.json")
    ms = ["standard_kd", "permutation_averaged_kd", "calibrated_kd"]
    rows = []
    passes = 0
    for k in d:
        r = d[k]; passes = r["permutation_averaged_kd"]["passes"]
        bf = max(r[m]["macro_f1"] for m in ms); ba = max(r[m]["acc"] for m in ms)
        rows.append(f"{NAME[k]} & Macro-F1 & " + " & ".join(bold_if(r[m]["macro_f1"], r[m]["macro_f1"] == bf) for m in ms) + r" \\")
        rows.append(" & Accuracy & " + " & ".join(bold_if(r[m]["acc"], r[m]["acc"] == ba) for m in ms) + r" \\")
        rows.append(r"\midrule")
    table("app_ablation4.tex", "@{}llccc@{}",
          r"\textbf{Dataset} & \textbf{Metric} & \textbf{Std. KD} & \textbf{Perm.-Avg.} & \textbf{Null-Cal.} \\" + "\n"
          + rf" & \textit{{Teacher passes}} & $1\times$ & ${passes}\times$ & $1\times$ \\",
          rows[:-1],
          r"Ablation 4: permutation-averaged teacher targets against null calibration. Results in \%.",
          "tab:app_ablation4")


def ablation5():
    d = jload(ABL / "5_multi_seed" / "multi_seed_summary.json")
    rows = []
    for k, n in DATASETS:
        r = d[k]
        best = max(r[m]["f1_mean"] for m in STUDENT)
        cells = []
        for m in STUDENT:
            v = r[m]
            top = rf"{v['f1_mean']:.2f} $\pm$ {v['f1_std']:.2f}"
            cells.append(stack(rf"\textbf{{{top}}}" if v["f1_mean"] == best else top,
                               rf"({v['acc_mean']:.2f} $\pm$ {v['acc_std']:.2f})"))
        seeds = r["calibrated_kd"]["seeds"]
        wins = sum(a["macro_f1"] > b["macro_f1"] for a, b in zip(seeds, r["standard_kd"]["seeds"]))
        rows.append(f"{n} & " + " & ".join(cells) + f" & {delta(r['delta_f1_mean'])} & {wins}/{len(seeds)} \\\\[2pt]")
    table("app_ablation5.tex", "@{}lccccc@{}",
          r"\textbf{Dataset} & \textbf{Std. KD} & \textbf{Batch-Cal.} & \textbf{Null-Cal.} & \textbf{$\Delta$ F1} & \textbf{Wins} \\",
          rows,
          r"Ablation 5: mean $\pm$ standard deviation over student seeds 42, 43 and 44. Macro-F1 with accuracy in parentheses, both in \%. The teacher targets are the same for every seed. $\Delta$ F1 is null-calibrated minus standard KD. Wins counts the seeds on which null-calibrated KD has a higher Macro-F1 than standard KD.",
          "tab:app_ablation5", size=r"\scriptsize")
    fig, ax = plt.subplots(figsize=(COLUMN_FIG_W, 3.6))
    style(ax)
    x = np.arange(len(DATASETS)); w = 0.26
    for j, m in enumerate(STUDENT):
        ax.bar(x + (j - 1) * w, [d[k][m]["f1_mean"] for k, _ in DATASETS], w,
               yerr=[d[k][m]["f1_std"] for k, _ in DATASETS], capsize=3,
               color=COL[m][0], edgecolor=COL[m][1], zorder=3, label=LEG[m], error_kw={"elinewidth": 1.1})
    ax.set_xticks(x); ax.set_xticklabels([n.replace(" ", "\n", 1) for _, n in DATASETS], fontsize=11.5)
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_ylim(0, 85)
    ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0), frameon=True, facecolor="#F8FAFC",
              edgecolor="#CBD5E1", fontsize=10.5, handlelength=1.2, columnspacing=0.8)
    save("fig_app_multiseed.pdf")


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
