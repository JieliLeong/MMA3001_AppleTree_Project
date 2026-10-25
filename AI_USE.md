# AI use

The brief requires every material use of AI to be disclosed, checked and
reflected on. Sections 1-3 are a factual record. **Sections 4-7 must be written
by me (Jieli Leong) in my own words**; the prompts in italics are placeholders to
replace.

## 1. Tools

| Tool | Version | Used for |
|---|---|---|
| Claude (Anthropic) | Claude Opus 5.5, Claude app | Project planning, code and test drafting, Colab notebooks, troubleshooting and documentation |
| ChatGPT (OpenAI) | ChatGPT; model version not recorded| Explaining methods, reviewing figures and claims, drafting and revising report sections and the AI reflection|

## 2. What the AI produced

| Stage | AI contribution | What I did |
|---|---|---|
| Planning | Read the project brief and my Week 1-11 notes (OCR of the notebook PDFs); mapped course topics to the project; proposed the stage plan | Supplied the brief, notes and dataset; chose the apple-tree project; decided the core would be an ML method comparison against a baseline |
| 1 Data | Found the 8-bit depth encoding, 256 x 256 masks and the re-matched winter file names; wrote `data.py`, `scripts/audit_dataset.py` and tests | Ran the audit on the full dataset in Colab; checked the overlays and file counts |
| 1b Mask alignment | After I pointed out the mismatch, wrote `alignment.py`, `scripts/alignment_check.py`, tests and the notebook | Noticed that the truth outline misses the real trunk; ran the check on all 561 trees |
| 2 Hole filling | Wrote `holes.py` (Laplace solvers, interpolators), tests with manufactured solutions, and the experiment | Decided to keep this stage and judge it at the end |
| 3 Pixel models | Wrote features, baselines, metrics and the comparison experiment | Ran it; asked whether a baseline was needed and how many methods to compare |
| 4a-4d Shape method | After Stage 3 showed the pixel models only learn position, proposed predicting the scaffold as a shape; wrote `shapes.py`, `shape_pipeline.py`, the four scripts, tests and notebooks | Ran every notebook on the real data; asked for the work to be thorough; checked how validation works |
| Repository | README, this file's factual sections, `notebooks/run_all.ipynb`, set-up notebooks | Created the GitHub repository and token; pushed from Colab |

Approximate share of the code drafted by AI: *write your own estimate and reasoning*.

## 3. Errors and unhelpful suggestions from the AI (and how they were caught)

1. **First plan was outside the course.** The AI first suggested a U-Net (CNN); after
   reading the Week 1-11 notes it found no computer vision was taught and revised the plan.
2. **Hole filling added without asking.** The AI built Stage 2 before checking it fitted my
   plan; I questioned it, and kept it as an optional stage to judge later.
3. **Wrong repository name** in the first set-up instructions (`MMA3001-apple-scaffold`);
   the clone failed until a diagnostic listed the real name.
4. **Confusing hand-over:** a zip file to download and upload; replaced by notebooks that
   write the code inside Colab.
5. **File-name parser missed 104 winter files** with a second tree ID; caught by the
   audit's "unmatched" count and fixed with new tests.
6. **Over-optimistic prediction:** the AI expected the per-pixel models to beat the
   baseline; Stage 3 showed they did not.
7. **A test failed for the wrong reason:** the Monte Carlo band test used synthetic shapes
   that broke the band's normal-distribution assumption; the test was redesigned and the
   assumption documented.
8. **Timing bug:** Stage 4c counted the bark-map time twice (767 ms instead of ~490 ms per
   photo); noticed when checking the printed numbers, fixed in Stage 4d.
9. **Speed claim corrected:** the AI first estimated ~35 ms per photo, which assumed
   settings the rules did not choose (the final method takes ~520 ms; the fast variant ~90 ms).
10. **Mask alignment taken on trust.** The audit only overlaid the masks on summer photos,
    where the scaffold is hidden by leaves, and the AI treated that as confirmation that
    the masks fit. I noticed after the final test that the outline misses the visible
    trunk; the Stage 1b check over 561 trees found no reliable global correction. The
    cause (approximate annotation, image correspondence or coordinate mapping) is
    undetermined, so all scores are reported as agreement with the supplied annotations.

## 4. How I checked the AI's work

*I ran the supplied code in Google Colab and inspected the displayed results and figures. The final recorded test output reported 67 tests passing. I relied on AI explanations to interpret the tests and did not independently verify every calculation or read every line of code. My most important visual check was comparing the supplied scaffold outlines with the tree photographs. I noticed that some outlines did not follow the visible trunk and branches, and questioned the earlier interpretation of the results. An AI-assisted alignment investigation did not establish a reliable global correction. The documentation was therefore revised to acknowledge that the cause remains uncertain and that the scores measure agreement with the supplied annotations.*

## 5. Decisions that were mine

*I questioned whether the baseline was appropriate and whether the results were sufficient to support the project’s conclusions. I also raised the mismatch between the reference outlines and the visible trees rather than accepting the numerical scores alone. Many implementation and modelling choices were suggested by AI. My role included running the notebook, inspecting outputs, identifying confusing or questionable results, and requesting explanations and corrections. I do not claim that I independently designed all the methods.*

## 6. Effect on my understanding and workflow

*AI helped me produce and run code, troubleshoot errors, interpret figures and improve the documentation. However, the code developed faster than my understanding. Terms such as KNN, ridge regression, shape modes and Monte Carlo sampling were initially confusing, and I needed repeated explanations in simpler language. The explanations helped me distinguish the average-mask baseline, which predicts the same typical shape for every tree, from methods that use image information to adjust the prediction. I also learned that a higher Dice score means better agreement with the supplied mask, rather than a directly measured percentage of physical branches reconstructed correctly. I still need to strengthen my understanding of the mathematical details.*

## 7. Limitations

*My reliance on AI limited my ability to check every implementation detail independently. Passing software tests and obtaining plausible figures were useful checks, but they did not establish that the supplied annotations were correctly aligned or that the physical branches were accurately located. If repeating the project, I would inspect image–mask correspondence earlier, develop a smaller working example first, and explain each method in my own words before adding another. I would also keep the final test results unavailable until development choices were complete. In this project, the same 85 test trees were evaluated in Stage 3 and again after later development, which limits claims about an untouched final evaluation. AI also assisted with drafting this reflection. I am responsible for reviewing it and ensuring that it accurately describes my contribution and experience.*

The presentation and interview answers are my own; AI was not used for them.
