# Joke Evaluation Pipeline

This note turns the draft evaluation idea in [main.tex](/Users/halaction/Workspace/humor-generation/docs/paper/main.tex) into a concrete pipeline.

## Design principles

1. Do not treat humor quality as a single scalar.
2. Use rubrics for dimensions that are relatively judgeable.
3. Treat funniness as a separate, comparative, human-grounded signal.
4. Calibrate LLM judges on a human-labeled subset before using them at scale.
5. Report disagreement, not just mean score, because humor is inherently controversial.

## What should change from the current draft

The current draft uses `constraint adherence / novelty / human-likeness / funniness`.

That set is not fully grounded:

- `constraint adherence` is good and should stay.
- `novelty` is supported by humor literature and should stay, but it should be operationalized as `originality`, not just "looks memorized."
- `human-likeness` should not be a core quality axis. It is useful as a diagnostic, but a joke can look human and still be bad, or look model-like and still be funny.
- `coherence` is too generic. Humor papers more often use `clarity` or `understandability` as an early gate.
- humor-specific work suggests at least one social or audience-facing axis, and at least one craft axis beyond relevance.

## Recommended rubric

Use a two-layer evaluation:

- `Layer A`: rubric axes for structured quality
- `Layer B`: human comparative funniness

### Layer A: structured rubric axes

Use six axes, but do not collapse them blindly into one score.

1. `Relevance / Constraint Adherence`
   - Does the joke satisfy the prompt or seed?
   - This matches `Relevance` in Oogiri-style evaluation.

2. `Clarity / Comprehensibility`
   - Can a competent reader understand what the joke is trying to say?
   - This is better than generic `coherence` for humor.

3. `Humor Intent and Payoff`
   - Is this recognizably a joke, and does it contain a payoff, twist, or comic mechanism?
   - This absorbs the "is this actually a joke?" check used in prior humor evaluation work.

4. `Originality`
   - Does the joke feel non-cliche, non-template, and non-copying?
   - This is the right place for novelty.

5. `Audience Fit / Relatability / Empathy`
   - Would the target audience likely relate to or emotionally connect with the joke?
   - Keep this axis because recent Oogiri work found human judgments and LLM judgments diverge strongly here.

6. `Safety / Appropriateness`
   - Is the joke non-offensive or at least acceptable for the target setting?
   - This should usually be a gate or penalty metric, not a quality reward.

### Axes to keep out of the main composite

- `Funniness`: keep separate from Layer A.
- `Human-likeness`: report only as an auxiliary diagnostic.

## Why funniness should not be a normal 1-5 rubric

Absolute funniness ratings are noisy for at least three reasons:

1. Individual taste varies heavily.
2. Calibration is unstable across annotators.
3. LLM judges systematically optimize for the wrong cues in humor.

So the pipeline should not ask either humans or LLMs to produce a single absolute "how funny is this?" number and then treat it as ground truth.

Instead:

- use human comparative judgments for funniness
- use rubric axes for diagnostic structure
- use disagreement as a first-class output

## Funniness pipeline

Use `Best-Worst Scaling` or pairwise comparison, not direct Likert rating.

### Recommended option: Best-Worst Scaling

For each prompt:

1. Collect one candidate joke from each system being compared.
2. Build 4-item comparison sets from jokes answering the same prompt.
3. Ask annotators:
   - Which joke is funniest?
   - Which joke is least funny?
4. Fit latent scores with a BWS-derived score or a Bradley-Terry-style model.

Why this is better:

- BWS is more reliable than rating scales for subjective annotation.
- Relative choice is easier than absolute scoring.
- It produces a stable ranking without pretending funniness is interval-scaled.

### Alternative option: pairwise funniness battles

If BWS is too expensive, use prompt-matched pairwise comparisons:

1. Show two jokes for the same prompt.
2. Ask `Which is funnier?`
3. Allow `tie / neither` only if necessary.
4. Fit Bradley-Terry or Elo-style scores.

## Human annotation protocol

Use a small but high-quality human subset to calibrate the pipeline.

### Sampling

Stratify by:

- prompt type
- topic
- joke mechanism if known
- model family
- checkpoint type
- output length

Also include:

- strong human references
- weak human references
- baseline model outputs
- MRVF outputs

### Pass structure

Run human annotation in two passes.

#### Pass 1: rubric pass

Annotators score:

- relevance
- clarity
- humor intent and payoff
- originality
- audience fit
- safety

Use 3 to 5 annotators per item.

#### Pass 2: funniness pass

Only items that pass minimum quality gates enter funniness comparison.

Recommended gates:

- clarity >= acceptable
- relevance >= acceptable
- humor intent >= acceptable
- safety not failed

Then annotate funniness comparatively with BWS or pairwise preference.

This avoids wasting funniness labels on obvious failures.

### Blindness and order control

Always:

- hide model identity
- randomize display order
- counterbalance order across judgments
- avoid mixing jokes from different prompts in the same comparison

## LLM-as-a-judge pipeline

Use LLM judges only after calibration.

### Judge setup

For each item, the judge should output:

- one score per rubric axis
- short rationale per axis
- confidence per axis

The judge prompt should:

- define each rubric level explicitly
- separate axes clearly
- forbid using one axis as a proxy for another
- remind the model that funniness is not the same as originality or relevance

### Calibration

On the human-labeled subset:

1. Run the LLM judge on all rubric axes.
2. Compare raw judge scores against human scores per axis.
3. Fit a lightweight calibrator from judge outputs to human labels.

Possible calibrators:

- ordinal regression
- isotonic regression
- small MLP as in LLM-Rubric-style calibration

Only retain axes whose judge-human agreement is acceptable.

Recommended acceptance rule:

- use an axis in large-scale automated evaluation only if agreement exceeds a predefined threshold, such as moderate or better weighted kappa or Spearman correlation

### Bias control

To reduce LLM-judge artifacts:

1. Use at least two strong judge models.
2. Repeat comparisons with swapped order when candidates are shown jointly.
3. Aggregate over multiple judge runs or temperatures only if needed for stability.
4. Inspect length effects and verbosity effects explicitly.
5. Send disputed or low-confidence cases back to humans.

## Novelty should not be judged only from text

If the real concern is memorization, rubric judgment alone is weak.

Add retrieval-based evidence:

1. Search the training corpus and reference corpus for near duplicates.
2. Compute lexical overlap and embedding-neighbor similarity.
3. Report:
   - exact match rate
   - near-duplicate rate
   - nearest-neighbor similarity
4. Provide top retrieved neighbors to the judge or human annotator when scoring originality on the calibration subset.

So `originality` becomes:

- human or judge impression of non-cliche novelty
- plus an external memorization diagnostic

## Proposed metrics

Do not report only one number.

Report at least four metric families.

### 1. Structured quality

For each rubric axis:

- mean calibrated score
- score distribution
- inter-annotator agreement
- judge-human agreement on the calibration subset

### 2. Comparative funniness

- BWS score or Bradley-Terry score
- win rate against baseline systems
- confidence intervals by prompt bootstrap

### 3. Controversy / disagreement

- variance or entropy of human funniness judgments
- variance across rubric annotators

This matters because a model that is sometimes very funny and sometimes terrible is different from a model that is consistently mediocre.

### 4. Memorization and safety diagnostics

- near-duplicate rate
- train-set retrieval similarity
- safety failure rate
- offensive or inappropriate rate

## Recommended aggregate reporting

Use two primary endpoints, not one:

1. `Primary endpoint A`: human comparative funniness
2. `Primary endpoint B`: calibrated rubric quality

Then report:

- originality diagnostics
- controversy
- safety
- human-likeness as auxiliary only

If you want one summary table column, use:

- `Rubric Quality Index`
- `Human Funniness Preference`

but do not combine them into a single scalar unless the paper explicitly motivates the weighting.

## Suggested 5-point rubric sketch

Use verbalized levels for each axis.

### Relevance / Constraint Adherence

- `1`: ignores the prompt or violates the core constraint
- `2`: loosely related but misses the intended constraint
- `3`: satisfies the prompt in a basic way
- `4`: clearly satisfies the prompt and uses the constraint well
- `5`: satisfies the prompt precisely and creatively

### Clarity / Comprehensibility

- `1`: hard to parse or unintelligible
- `2`: understandable only with effort
- `3`: understandable but awkward
- `4`: clear and easy to interpret
- `5`: immediately clear without losing subtlety

### Humor Intent and Payoff

- `1`: not recognizably a joke
- `2`: tries to be a joke but lacks a payoff
- `3`: contains a basic joke structure
- `4`: has a clear comic mechanism and payoff
- `5`: delivers a strong, well-executed payoff

### Originality

- `1`: cliche, copied, or highly template-like
- `2`: mostly familiar with little surprise
- `3`: somewhat fresh but conventional
- `4`: clearly original in framing or twist
- `5`: strikingly original and non-derivative

### Audience Fit / Relatability / Empathy

- `1`: alienating, tone-deaf, or mismatched to the target audience
- `2`: weak audience connection
- `3`: acceptable audience fit
- `4`: relatable and audience-aware
- `5`: strongly resonant and emotionally well-targeted

### Safety / Appropriateness

- `1`: clearly inappropriate or offensive
- `2`: borderline problematic
- `3`: acceptable with some risk
- `4`: appropriate for the target setting
- `5`: clearly safe and appropriate

## Recommended interpretation of Oogiri-style dimensions

The Oogiri paper uses:

- novelty
- clarity
- relevance
- intelligence
- empathy
- overall funniness

For this project, the best mapping is:

- `novelty` -> `originality`
- `clarity` -> `clarity`
- `relevance` -> `adherence`
- `intelligence` -> part of `humor intent and payoff` or a separate `cleverness` axis if you can afford more annotation
- `empathy` -> `audience fit / relatability`
- `overall funniness` -> comparative human outcome, not a standard rubric score

If annotation budget is high, split `Humor Intent and Payoff` into:

- `Comic Craft / Cleverness`
- `Jokehood / Intent`

Otherwise keep them together.

## Minimal viable version

If you need a simpler first implementation:

1. Human rubric on 500 to 1000 jokes with 3 annotators each.
2. Human BWS funniness on a prompt-matched subset.
3. LLM judge on all jokes for:
   - adherence
   - clarity
   - humor intent and payoff
   - originality
   - audience fit
4. Retrieval-based memorization diagnostic on all jokes.
5. Calibrate LLM judge on the human subset and use only validated axes.

## Bottom line

Your initial `novelty / adherence / coherence / funniness` proposal is directionally right, but it is incomplete.

The literature supports a better structure:

- keep `adherence`
- replace `coherence` with `clarity`
- keep `novelty`, but rename it `originality` and support it with retrieval diagnostics
- add `humor intent and payoff`
- add `audience fit / empathy`
- treat `safety` as a gate
- move `funniness` out of absolute rubrics and into human comparative evaluation
- keep `human-likeness` as auxiliary only

## Sources

- Hashemi et al. 2024, `LLM-Rubric: A Multidimensional, Calibrated Approach to Automated Evaluation of Natural Language Texts`:
  https://aclanthology.org/2024.acl-long.745/
- Gunjal et al. 2025, `Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains`:
  https://arxiv.org/abs/2507.17746
- Sakabe et al. 2025, `Assessing the Capabilities of LLMs in Humor: A Multi-dimensional Analysis of Oogiri Generation and Evaluation`:
  https://arxiv.org/abs/2511.09133
- Tikhonov and Shtykovskiy 2024, `Humor mechanics: Advancing humor generation with multistep reasoning`:
  https://computationalcreativity.net/iccc24/papers/ICCC24_paper_128.pdf
- Wang et al. 2023, `Large Language Models are not Fair Evaluators`:
  https://arxiv.org/abs/2305.17926
- Zheng et al. 2023, `Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena`:
  https://arxiv.org/abs/2306.05685
- Meaney et al. 2021, `SemEval 2021 Task 7: HaHackathon, Detecting and Rating Humor and Offense`:
  https://aclanthology.org/2021.semeval-1.9/
- Kiritchenko and Mohammad 2017, `Best-Worst Scaling More Reliable than Rating Scales`:
  https://aclanthology.org/P17-2074/
- Romanowski et al. 2025, `From Punchlines to Predictions: A Metric to Assess LLM Performance in Identifying Humor in Stand-Up Comedy`:
  https://aclanthology.org/2025.cmcl-1.6/
- Kim and Kim 2024, `Can Language Models Evaluate Human Written Text?`:
  https://arxiv.org/abs/2407.17022
