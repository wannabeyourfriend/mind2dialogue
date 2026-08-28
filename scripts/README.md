# scripts

Two kinds of script live here. The release pipeline turns generated
conversations into the artifacts we publish. The paper experiments answer a
question asked once during review and are not part of building the dataset.

Every script is standalone: run it with `python scripts/<name>.py --help`, and
read the module docstring at the top for what it does and why.

## release pipeline

| script | does |
|---|---|
| `assemble_final_release_dataset.py` | gathers the curated release dataset from the rollout artifacts |
| `deduplicate_and_reorder_jsonl.py` | removes duplicate training lines and fixes their order, conservatively enough for a release |
| `build_grounded_supervision_data.py` | teacher is guided by the private context but must justify its answer from the transcript alone; a blind checker drops whatever it cannot trace |
| `build_latent_supervision_data.py` | builds training data with genuine reasoning traces from the latent corpus |
| `generate_latent_corpus.py` | generates conversations whose latent state causes the dialogue instead of summarising it |
| `generate_multi_session_conversations.py` | generates conversations that span sessions, where carried state is the only source of the answer |
| `strip_comments_and_format_python.py` | strips comments, collapses blank lines and runs the formatter over the files you name |

## paper experiments

Ablations, controls and probes. Each one exists to make a specific claim
falsifiable.

| script | asks |
|---|---|
| `build_three_arm_supervision.py` | does the teacher need the true private context, or would an inferred one do? Arms: shared, inferred, blind |
| `build_three_arm_supervision_inferred_control.py` | the same, with questions written from the estimated context rather than the true one |
| `build_three_arm_supervision_paraphrase_control.py` | the same, with every question paraphrased, to separate wording from substance |
| `build_cross_session_questions.py` | in the regime where the answer lives in a previous session, how much is privileged state worth? |
| `build_cross_session_questions_transcript_control.py` | the same, giving the generator the previous transcript instead of the carried state |
| `build_supervision_ablation_mixtures.py` | builds the matched privileged and blind training mixtures, same items, same order |
| `relabel_conversations_with_blind_teacher.py` | regenerates every assistant turn without the state, holding everything else fixed |
| `relabel_questions_with_blind_teacher.py` | the same for the released question answering items, questions held fixed |
| `analyze_supervision_divergence.py` | how far does blind supervision actually diverge from privileged supervision? |
| `probe_state_predictive_power.py` | can a reader holding the state pick the user's real next turn more often than a blind one? |
| `probe_history_leakage.py` | is the multiple choice answer already recoverable from the replayed assistant turns? |
| `run_manipulation_check.py` | did the friction and guarded conditions actually produce a varying, under-reported state? |
| `evaluate_latent_corpus.py` | is the latent state non-trivial? |
| `evaluate_response_quality.py` | does privileged supervision produce better responses, not just better answers? |
| `evaluate_deal_breaker_deflection.py` | does the student avoid a deal-breaker the user never stated? |
| `evaluate_prefeval_implicit_preferences.py` | did the model learn to infer preferences, or did it just get generally better? |
| `evaluate_prefeval_arms.sh` | runs the PrefEval arms end to end |
| `collect_ablation_results.py` | assembles the privileged versus blind comparison table from the benchmark results |
| `plot_corpus_composition.py` | regenerates the corpus composition figure from `corpus_composition_data.json` |
