# Calibrated Knowledge Distillation (CKD)

---

## TL;DR

Distilling knowledge from a large language model ‘teacher’ into a smaller model ‘student’ leads to the transfer of the teachers’ bias as well. We study the effect of ‘Null’ prompting on the teacher and utilising it to minimise the bias transference, which in turn makes the student perform better!

---

## Background

Large language models are getting better every day at classification tasks. Be it topical, inferential or pragmatic. But there is a significant drawback in deploying these larger models in scenarios where we require only a specific classification job. This comes in the form of higher compute requirements and longer inference time.

As such, a better alternative is to train a smaller, lightweight language model on a single or a specific set of tasks. They help reduce the time and compute required while still being able to perform at par with larger LLM models on the specific tasks they are trained on.

But this comes with a separate challenge for us! How do you train these models? Training requires a large amount of labelled/annotated data, i.e., that which contains the ground truth. This is the biggest bottleneck, as it often requires human judges manually annotating/labelling the data.

To overcome this challenge, we utilise a technique known as Knowledge Distillation (Hinton et al., 2015). We utilise already available LLMs (termed as “Teacher”) and ask them to label/categorise on an unlabelled training set and capture the output it produces. The output is then used to fine-tune existing small base models like Bert (Devlin et al., 2019) / RoBERTa (Liu et al., 2019) termed as “Student”.

Pretty simple, right ?

Yes, but it comes with its own set of challenges. Now assume you have a teacher who is already biased towards a certain class/category from within the set of possible classes for that task. When a student learns from the teacher, it will undoubtedly acquire some of that bias, and in many cases, it will lead to a complete class collapse.

To mitigate this, we need to come up with a way to somehow capture that bias and remove it before the student can see it. Fortunately, we already have seen work done on how we can capture the same. (Zhao et al. ,2021) suggests that we can probe the teacher using a null prompt and capture the probabilities it gives to each class/category. Then, at inference, you simply subtract the same from the model’s output before applying softmax.

Null Calibrated KD utilises this same concept in the context of knowledge distillation to prevent bias transference. We utilise KL Divergence (Kullback & Leibler, 1951) as the loss function which we try to minimise when fine tuning the student. The teachers’ logits are modified by subtraction of the null logits and then plugged in to the KLD formula.

---

## Method

First, we need to extract the teacher models’ bias without probing it with an actual input. For that we pass it a null string, like "N/A" using the exact same classification prompt, which will be used for normal inference & knowledge distillation (Zhao et al., 2021).

Because there is no real input to classify, we can start with the assumption that whatever the teacher outputs here is its baseline bias. We extract these unnormalised logit values for all the candidate classes and call this the ‘Null’ vector.

Once we have captured this, we run the teacher model over our unlabelled training set and then extract the raw logit vectors $z_{\text{raw}}$ for each class. Now, instead of directly plugging them into the loss function, we first perform a simple logit subtraction and get the debiased target.

This is a low-to-zero-cost $O(1)$ operation which doesn’t impact the compute required or time taken. After this, we apply the softmax function, which turns these logits into actual probabilities.

Now we can train our compact student model (RoBERTa-base) using forward Kullback–Leibler (KL) divergence as our loss function. It helps in training the student model by bringing the student’s predicted probability distribution as close as possible to the teacher’s debiased target distribution.

We will be comparing metrics captured for null-calibrated KD against a few existing methods as well to see if we actually see any improvements.

* **Teacher Zero-Shot:** The teacher LLM evaluated both with and without null calibration.
* **Standard Knowledge Distillation:** Distillation using the teacher’s soft probabilities using forward KL divergence but without calibration.
* **Batch-Calibrated KD:** The teacher’s predictions are averaged across the entire training dataset to estimate class priors. Then we subtract it before computing distillation targets (Zhou et al., 2024).
* **Null Calibrated KD:** Our method as explained above.

While the above methods require no prior knowledge, there are a few alternatives that rely on validation statistics, known class frequencies, or excessive computation, making them impractical for real-world or streaming deployments:

* **Prior-Aware Distillation:** It uses the class distribution of the training set for calibration by incorporating it into the teacher's target (Liusie et al., 2023). This is infeasible, as ground truth is unknown in knowledge distillation.
* **Permutation-Based Calibration:** We permute the option labels provided and then average out teacher predictions to eliminate option bias. This is computationally expensive, as it requires multiple passes per sample through the teacher LLM.
* **Post-Hoc Output Calibration:** The student is trained with standard KD, and its logits are shifted at inference by subtracting its average validation output. This is impractical for a real-world scenario, where samples arrive one by one. It also degrades performance on rare minority classes.

---

## Experimental Setup

In order to prove this works, we set up a testing bench utilising this pipeline and ran it on 6 different multi-class classification tasks. The datasets used are publicly available.

* [Student Questions](https://huggingface.co/datasets/SetFit/student-question-categories) (STEM Subjects): Biology, Physics, Maths, Chemistry
* [TREC-6](https://huggingface.co/datasets/CogComp/trec) (Question Types): Abbreviation, Entity, Description, Human, Location, Numeric
* [App Reviews](https://huggingface.co/datasets/sealuzh/app_reviews) (Customer Reviews): 1 Star, 2 Stars, 3 Stars, 4 Stars, 5 Stars
* [Twitter Emotion](https://huggingface.co/datasets/dair-ai/emotion) (Social Affect): Sadness, Joy, Love, Anger, Fear, Surprise
* [TweetEval Emotion](https://huggingface.co/datasets/cardiffnlp/tweet_eval) (Social Affect): Anger, Joy, Optimism, Sadness
* [BBC News](https://huggingface.co/datasets/SetFit/bbc-news) (News Articles): Tech, Business, Sport, Entertainment, Politics

We have used a fixed-size subset from each dataset using stratified sampling to preserve natural class skew. We have selected a standardised split of 75:25 (1,200 training and 400 test) for training and testing.

For selecting the models, we have gone with:

* **Gemma-3-4B-IT:** We used this 4-billion-parameter instruction-tuned causal LLM as the teacher model, kept completely frozen in 16-bit precision.
* **RoBERTa-base:** This masked encoder model (125 million parameters) works as a lightweight student model. It consumes less memory and runs faster.

For capturing this quantitatively, we will use Macro-F1. This helps us to measure fairly across the minority and dominant classes alike. It gives equal weight to all K classes, so a model can’t simply score better by simply over predicting the majority class. We also report overall Accuracy as a secondary metric.

---

## Results

We ran our pipeline across all these datasets and captured the required metrics so as to provide an accurate representation of what works and what doesn’t.

Below is the table capturing Macro-F1 and Accuracy across all 6 datasets and all different methods.

| Architecture | Method | Student Questions ($K=4$) | TREC-6 ($K=6$) | App Reviews ($K=5$) | Twitter Emotion ($K=6$) | TweetEval Emotion ($K=4$) | BBC News ($K=5$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Teacher (Gemma-3-4B-IT)** | Base (Raw Zero-Shot) | 28.13% (33.75%) | 50.15% (52.25%) | 29.69% (48.25%) | 39.80% (53.00%) | 45.78% (57.25%) | 54.81% (56.00%) |
| **Teacher (Gemma-3-4B-IT)** | Null-Calibrated | 29.08% (33.75%) | 52.90% (54.25%) | 32.46% (49.25%) | 40.35% (52.00%) | 56.05% (65.00%) | 64.04% (63.25%) |
| **Student (RoBERTa-base)** | Standard KD | 22.86% (32.25%) | 62.28% (59.75%) | 30.25% (51.75%) | 29.78% (50.50%) | 34.16% (54.25%) | 55.56% (58.50%) |
| **Student (RoBERTa-base)** | Batch-Calibrated KD | 23.19% (32.00%) | 63.29% (61.00%) | 32.04% (52.25%) | 31.13% (50.00%) | 38.85% (56.50%) | 57.13% (60.25%) |
| **Student (RoBERTa-base)** | **Null Calibrated KD (Ours)** | **37.10% (39.25%)** | **64.71% (61.75%)** | **33.24% (53.25%)** | **34.46% (50.75%)** | **55.48% (66.50%)** | **71.40% (70.50%)** |

Apart from seeing substantial improvements in Macro-F1 and Accuracy across all datasets, we also notice that it prevents the collapse of certain classes and also prevents certain classes from getting overly boosted.

We can see that Standard KD completely collapses minority classes across multiple tasks (0 predictions for Politics on BBC News, Optimism on TweetEval, and Chemistry on Student Questions), whereas null-calibrated KD actively helps us avoid the same. The null probe also appears to isolate the teacher’s prompt bias without suppressing natural class distributions, avoiding the over-correction seen in batch-mean methods.

---

## Discussion

The bias in the teacher can have several sources. As the most common method of KD uses verbalizer options like A, B, C, etc., LLMs can have a favourite option even before they read the input. (Zheng et al. ,2024) call this token bias. They can also favour options based on their position in the prompt. (Zhao et al. ,2021) report a similar recency bias toward labels at the end of the prompt. When we tie these verbalizer options to the classes, we can safely assume that due to the attention mechanism, the model ingrains and relates the verbalizer option to the actual class.

All these effects mix in the teacher's output, and there is no single way to separate each one. The null probe aims to measure their combined effect for one fixed prompt and one fixed option order. As such, by subtracting the null logits, we aim to remove the combined prior bias before the student can learn it.

---

## Limitation

Even though this method works, there are a few limitations to it. If the teacher itself is very bad (near random), then this method cannot manufacture knowledge which was never there to begin with. We have limited the scope to multi-class classification tasks, but there may be cases where the task is sequence generation. In such cases, our method is not applicable. In certain cases where the teacher itself has an almost flat distribution (carries low bias), we might see minimal impact (or negative impact in certain cases). As such, developers’ discretion is advised. On simple, uniformly balanced tasks or binary tasks, standard KD is already sufficient, and null-calibrated KD provides minimal benefits.

---

## Conclusion

LLM teachers have built-in prompt and token biases that can affect knowledge distillation using soft targets. It causes smaller student models to learn the bias, and as a result, they sometimes completely collapse on minority classes.

A single offline probe with a blank prompt (N/A) can capture this bias. We subtract these null logits inside the distillation loss function before the student can learn from it.

Students show better performance, sometimes even better than the teacher. Macro-F1 increases by an average of ~10 pp on the tested datasets. This helps prevent class collapse or overboosting. No extra compute or human intervention is required as well.

---

## Credits

Code: [github.com/vishrutbhushan/calibrated-kd](https://github.com/vishrutbhushan/calibrated-kd)

### Team
* [Vishrut Bhushan](https://www.linkedin.com/in/vishrut-bhushan/) ([GitHub](https://github.com/vishrutbhushan)): Main Contributor
* [Ajay Badrinath](https://www.linkedin.com/in/ajay-badrinath/) ([GitHub](https://github.com/AjayBadrinath), [ORCID](https://orcid.org/0009-0007-4285-7095)): Main Contributor
* [Aakash S D](https://www.linkedin.com/in/aakashsd2000) ([GitHub](https://github.com/LoneCannibal)): Main Contributor
* [Pranjal Pandey](https://www.linkedin.com/in/pranjal-pandey-02a641239/) ([GitHub](https://github.com/ppranjal966)): Main Contributor
* [Dhruv Kumar](https://kudhru.github.io/): Faculty/Advisor
* [Aniket Upadhyay](https://www.linkedin.com/in/aniket-upadhyay-0b6327167/): Mentor

This post was written for the Agentic AI course at BITS Pilani.

---

## References

* Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL-HLT 2019*, 4171–4186.
* Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *NeurIPS Workshop on Deep Learning and Representation Learning*.
* Kullback, S., & Leibler, R. A. (1951). On information and sufficiency. *The Annals of Mathematical Statistics*, 22(1), 79–86.
* Liusie, A., Fathullah, Y., & Gales, M. J. F. (2023). Mitigating class imbalance in zero-shot prompt-based distillation via prior matching. *Findings of the Association for Computational Linguistics: IJCNLP-AACL 2023*.
* Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). RoBERTa: A robustly optimized BERT pretraining approach. *arXiv: 1907.11692*.
* Zhao, T. Z., Wallace, E., Feng, S., Klein, D., & Singh, S. (2021). Calibrate before use: Improving few-shot performance of language models. *Proceedings of the 38th International Conference on Machine Learning (ICML)*.
* Zheng, C., Zhou, H., Meng, F., Zhou, J., & Huang, M. (2024). Large language models are not robust multiple choice selectors. *International Conference on Learning Representations (ICLR)*.
* Zhou, H., Wan, X., Proleev, L., Bouchacourt, D., Kuo, Y.-L., & Cangea, C. (2024). Batch calibration: Rethinking test-time calibration for in-context learning and prompting. *International Conference on Learning Representations (ICLR)*.

---

## Appendix & Ablations

### Ablation 1: Cross-Architecture Triad (Skewed & Natural Benchmarks)

| Architecture Configuration | Teacher Model | Student Model & Mode | Benchmark Dataset Evaluated | Standard KD Macro-F1 (Acc) | CKD Macro-F1 (Acc) | Net Gain ($\Delta$ F1 pp) | Minority Recovery Status |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **1. Same-Size Distillation** | Gemma-3-4B-IT (4B) | Qwen3-4B LoRA (4B) | **AG News Skewed** (10% Sci/Tech) | — | — | — | — |
| **2. Cross-Family Distillation** | Qwen3-4B (4B Causal) | RoBERTa-base (125M Encoder) | **SST-2 Skewed** (Binary Sentiment) | — | — | — | — |
| **3. Small SLM Compression** | Gemma-3-4B-IT (4B) | Qwen2.5-0.5B LoRA (500M) | **TREC-6 Naturally Skewed** (1.6% Abbr) | — | — | — | — |

---

### Ablation 2: Multi-Task Distillation with a Unified Student Backbone

| Multi-Task Setup | Evaluated Tasks | Standard KD (Macro-F1) | CKD (Ours, Macro-F1) | Net Gain ($\Delta$ pp) |
| :--- | :--- | :---: | :---: | :---: |
| **Combo 1** | TweetEval Emotion + BBC News | — | — | — |
| **Combo 2** | BBC News + TREC-6 | — | — | — |
| **Combo 3** | Twitter Emotion + TREC-6 | — | — | — |
| **Combo 4** | TweetEval + BBC News + TREC-6 | — | — | — |
| **5-Task Unified** | TweetEval + BBC + PRAXIS + TwEmo + TREC | — | — | — |

---

### Ablation 3: Hard Label Fine-Tuning ($\text{argmax}$) vs. Soft Label Distillation (KL)

| Distillation Regime | Target Formulation | Loss Function | Macro-F1 (%) | Test Accuracy (%) | Minority Preds (Gold=250) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **1. Raw Hard Pseudo-Labels** | $\hat{y} = \arg\max(z_T)$ | Cross-Entropy | — | — | — |
| **2. Calibrated Hard Pseudo-Labels** | $\hat{y} = \arg\max(z_T - b_{\text{null}})$ | Cross-Entropy | — | — | — |
| **3. Standard Soft KD** | $p_T = \text{softmax}(z_T / \tau)$ | KL-Divergence | — | — | — |
| **4. Calibrated Soft KD (CKD, Ours)** | $p_T = \text{softmax}((z_T - b_{\text{null}}) / \tau)$ | KL-Divergence | — | — | — |

---

### Ablation 4: Permutation-Averaged Teacher Distillation vs. CKD (AG News Skewed)

| Benchmark Dataset | Evaluation Metric | Standard KD ($1\times$ Pass) | Permutation-Averaged KD ($4\times$ Passes) | Calibrated KD (CKD, $1\times$ Pass) |
| :--- | :--- | :---: | :---: | :---: |
| **AG News Skewed** | Test Accuracy (%) | — | — | — |
| | Macro-F1 (%) | — | — | — |
| | Worst-Class F1 ($F1_{\text{min}}$) | — | — | — |

---

### Ablation 5: Multi-Seed Statistical Significance Validation

| Distillation Regime | Seed 42 F1 (%) | Seed 43 F1 (%) | Seed 44 F1 (%) | Mean Macro-F1 $\pm$ Std (%) | Minority Preds (Seeds 42, 43, 44) | Mean Gain ($\Delta$ pp) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard KD** | 57.22% | 56.09% | 56.56% | $56.62\% \pm 0.46\%$ | 0, 0, 0 (Deterministic Collapse) | Baseline |
| **Calibrated KD (CKD, Ours)** | 64.22% | 64.70% | 65.53% | **$64.82\% \pm 0.54\%$** | **29, 36, 45 (Recovered)** | **+8.19 pp** |

---

### Ablation 6: Teacher Model Scaling (Gemma-3-12B-IT $\to$ RoBERTa-base)

| Dataset | $K$ | Teacher 12B Raw | Teacher 12B Null-Cal | Student Standard KD | Student Batch-Cal KD | Student CKD (Ours) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Student Questions** | 4 | 27.87% (31.75%) | 28.78% (33.25%) | 29.40% (32.50%) | **29.50%** (32.50%) | 29.12% (32.50%) |
| **TREC-6** | 6 | 65.52% (64.25%) | 63.62% (62.25%) | 63.70% (66.00%) | **69.19% (70.25%)** | 51.58% (64.00%) |
| **App Reviews** | 5 | 38.85% (62.25%) | 41.34% (63.00%) | 34.76% (62.00%) | **36.04%** (61.75%) | 35.64% (**63.25%**) |
| **Twitter Emotion** | 6 | 49.66% (56.00%) | 51.38% (58.25%) | 33.37% (52.75%) | 31.20% (51.25%) | **34.93% (53.00%)** |
| **TweetEval Emotion** | 4 | 72.95% (77.75%) | 72.55% (77.25%) | 72.81% (78.25%) | **73.98% (79.25%)** | 73.26% (**79.25%**) |
| **BBC News** | 5 | 89.48% (89.75%) | 89.28% (89.50%) | **93.91% (94.00%)** | 93.62% (93.75%) | 92.62% (92.75%) |
