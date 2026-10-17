# Midterm 1 — ANSWERS

Fingerprint: 12f03a24
GitHub username: darkhantursbekov
Session code: practice

## AI-use declaration

I used an AI assistant during this midterm, but strictly to check formulas
and code. With the assistant's help I verified the Wilson interval formula
and checked my Python scripts for recall@k, MRR@10, cost calculations and
the Part 3 questions. I also used it to outline the structure of the Part 2
memo. Every factual claim about pipelines H and L comes from my DATA.md and
doc_results.csv. Beyond that, I did not use AI to fabricate any measurement.
All computations were done with make_my_data.py and the code I ran in Colab.

---

## Part 1: Retrieval and cost decision

### 1a. Retrieval metrics

| Metric   | Config A | Config B |
|----------|----------|----------|
| recall@1 | 0.4500   | 0.3167   |
| recall@3 | 0.5833   | 0.6167   |
| recall@5 | 0.6333   | 0.6833   |
| MRR@10   | 0.5260   | 0.4685   |

### 1b. Recall@k with 95% Wilson intervals (n = 60)

| Candidate | hits/60 | recall@k | 95% CI |
|-----------|---------|----------|--------|
| (A, 3)    | 35      | 0.5833   | [0.4573, 0.6994] |
| (A, 5)    | 38      | 0.6333   | [0.5068, 0.7438] |
| (B, 3)    | 37      | 0.6167   | [0.4902, 0.7291] |
| (B, 5)    | 41      | 0.6833   | [0.5577, 0.7869] |

Reading: (A,3) vs (B,3) — intervals overlap (0.490 < 0.699), so no winner is
claimed. (A,5) vs (B,5) — intervals also overlap (0.558 < 0.744), so no winner
is claimed between A and B at these k values.

### 1c. Annual model cost

| Candidate | Annual cost | Within $11,000 budget? |
|-----------|-------------|------------------------|
| (A, 3)    | $12,209.69  | No                     |
| (A, 5)    | $15,887.25  | No                     |
| (B, 3)    | $8,492.38   | Yes                    |
| (B, 5)    | $9,691.74   | Yes                    |

Config A is eliminated by the budget constraint.

### 1d. Recommendation

Chosen candidate: (B, 5)

Config A is eliminated by the budget: both (A,3) at $12,210 and (A,5) at
$15,887 exceed the $11,000 annual limit. Of the two affordable candidates,
(B,5) has the higher recall@5 (0.683 vs 0.617), but the 95% CIs overlap
([0.558, 0.787] vs [0.490, 0.729]), so B@5 is not shown to beat B@3; I pick
(B,5) for its higher point estimate and remaining headroom under budget
($9,692, leaving about $1,300). Per-language recall@5 for (B,5): EN 0.650,
RU 0.650, KK 0.750. The worst-served language is RU at recall@3 (0.500), and
it ties EN at recall@5 (0.650); KK is served best (0.750).

### 1e. Next measurement

Add 30 more RU test questions and re-measure recall@3 for (B,3) and (B,5).
RU has the lowest point estimate and the current CI is wide at n=20; a larger
RU set would narrow the interval and could reverse the B@3 vs B@5 choice.

---

## Part 2: A document pipeline you can defend

### 2a. Pipeline comparison

| | Pipeline H | Pipeline L |
|---|---|---|
| Overall accuracy | 161/180 = 0.8944 | 151/180 = 0.8389 |
| Weakest (language, condition) | (KK, degraded) 24/30 = 0.8000, 95% CI [0.627, 0.905] | (KK, degraded) 19/30 = 0.6333, 95% CI [0.455, 0.781] |
| Annual cost | $18,823.05 | $32,400 (3 servers × $900 × 12) |
| Scanned pages per year with personal data leaving the country (H only) | 1,051,200 | 0 |

### 2b. Memo to the Head of Operations

**To:** Head of Operations
**Subject:** Choice of invoice-extraction pipeline

Overall accuracy looks close — H at 0.894 and L at 0.839 — but the average
hides the group that matters. Both pipelines are weakest on Kazakh degraded
scans: H at 0.800 [0.627, 0.905], L at 0.633 [0.455, 0.781]. The point
estimates differ substantially, though the intervals overlap, so the data
does not prove H is better on this cell; it does show L is the riskier
choice for Kazakh users on poor scans.

On personal data, the three Lecture 5.1 questions split the two designs.
Where does the scan go: under H it leaves Kazakhstan to a data centre abroad;
under L it stays on the bank's server in Kazakhstan. Who stores it: H stores
it at the vendor, L at the bank. For how long: L's retention is under our
control; H's is set by the vendor and we would need the DPA and the
subprocessor list to know. Since 32% of pages carry a name and an ИИН, under
H about 1,051,200 pages per year would cross the border. Kazakhstan's law
No. 94-V covers the ИИН, and a vendor's statement is not a compliance review.

Verifiable safeguards I would require: a signed DPA with audit rights, data
residency in Kazakhstan, access logging we can inspect, encryption at rest
and in transit, a documented deletion period, and an independent audit.

Weighing accuracy, cost and risk: L costs $32,400 a year against H's
$18,823, and L is weaker on Kazakh degraded scans, but L keeps personal data
in-country and under bank control. I recommend L, with the residual risk
that Kazakh degraded accuracy stays around 0.63 and must be improved before
scaling — for example by fine-tuning on Kazakh scans or adding a
pre-processing step. If H is chosen instead, the residual risk is legal:
1.05 million pages of ИИН data leave the country every year, and the bank
would carry that exposure unless a lawful transfer basis is established.

---

## Part 3: Explain your own numbers

### Q1 — Language premium

**Number:** token ratio KK/EN = 3990/1609 = 2.4798; bill ratio = $0.009990 /
$0.004418 = 2.26.

**Why (≤40 words):** The bill ratio is lower than the token ratio because
output tokens cost six times input (12 vs 2 per million), and EN uses a larger
absolute share of output tokens (120/1609 vs 201/3990), so EN's bill is
disproportionately expensive.

### Q2 — Chunk size and the embedding window

**Number:** the matching sentence starts at 0.85 × 640 × (263/256) = 559 e5
tokens.

**Why (≤40 words):** The sentence sits past e5's 512-token window, so e5
never encodes it and the chunk cannot be retrieved for that question. The
chunker raises no error; the tail is silently truncated.

### Q3 — Temperature and top-p

**Number:** after softmax at T=1, top-p=0.8 keeps Monday, Friday and Sunday
(cumulative 0.8432); P(Monday) after renormalisation = 53.6%.

**Why (≤40 words):** Temperature rescales all logits before softmax and
reshapes the whole distribution. Top-p truncates the tail and renormalises
the kept tokens, so it removes options rather than redistributing probability
mass smoothly.

### Q4 — Why models hallucinate

**Number:** lower bound on wrong answers over the applicable sets =
(375×0.15 + 219×0.37) / 594 = 137.28/594 = 23.1%.

**Why (≤40 words):** Multiplication and sequence questions are algorithmic —
the model can compute them, so a "seen once" bound does not apply. For
birthdays and account numbers the fact must be memorised; "never invent a
fact" cannot create knowledge the model lacks.

### Q5 — Cost of conversations

**Number:** total input tokens billed over 16 turns = 75 × (16×17/2) =
75 × 136 = 10,200 tokens. The first message is paid for 16 times.

**Why (≤40 words):** Turn 16 resends the first message along with all
history, so it has been billed 16 times. Prompt caching stores the shared
prefix, so later turns pay the cache-read price instead of full input — the
count stays 16 but the cost drops.
