# Human validation of the automatic scores

30 scored tutor responses (seeded random draw from `experiment_results_rejudged.csv`) were labelled independently by two annotators.

- `annotatorA.csv`, `annotatorB.csv`: final labels. Scaffolding is judged on the student-visible text only; accuracy on visible text plus hidden reasoning.
- `annotatorB_round1_before_clarification.csv`: annotator B's first independent round. B judged the hidden reasoning of chain-of-thought responses as if the student saw it (scaffolding agreement with A was 57%, kappa 0.26). The instruction was clarified and B re-scored the sample.
- `table_validation.csv`: output of `python validate_judge.py experiment_results_rejudged.csv annotatorA.csv annotatorB.csv`.
