# RAG-code-Tutor

Code for the book chapter on pipeline optimization for RAG-based intelligent tutoring systems:
AST-aware versus fixed-size chunking, three prompting strategies (zero-shot, Socratic, chain-of-thought)
and an efficiency index (ITEI) for a Python code-tracing and debugging tutor.

## Files
| File | Purpose |
|---|---|
| `run_experiment.py` | Runs the experiment: chunking, hybrid retrieval, tutor calls, simulated student, judge, control arm |
| `analyze_results.py` | Builds the result tables (P, E, ITEI, paired effects, sensitivity) from the results CSV |
| `validate_judge.py` | Cohen's kappa between two annotators and agreement of the automatic scores with them |

## Setup
```
python -m pip install -r requirements.txt
```
Set the API keys in the same terminal you run the code from (never commit them):
```
# Windows PowerShell
$env:GEMINI_API_KEY="..."; $env:GROQ_API_KEY="..."; $env:CEREBRAS_API_KEY="..."
# macOS / Linux
export GEMINI_API_KEY=... GROQ_API_KEY=... CEREBRAS_API_KEY=...
```

## Run
```
python run_experiment.py --selftest                 # offline checks, no API calls
python run_experiment.py --retrieval-only           # retrieval table, no API calls
python run_experiment.py --list-models              # model IDs your keys can use
python run_experiment.py --quick                    # small smoke test
python run_experiment.py --repeats 3 --sleep 2 --simulator-model <model> --judge-model <model>
python analyze_results.py experiment_results.csv
python validate_judge.py experiment_results.csv annotatorA.csv annotatorB.csv
```
Use `--resume` to continue an interrupted run. `--mock` produces fake data to test the pipeline only;
`analyze_results.py` refuses mock data unless `--allow-mock` is given. Never report mock numbers.

## Outputs
`experiment_results.csv`, `*_metadata.json` (model IDs, retriever mode, seeds), `*_prompts_used.json`,
`*_human_validation_sample.csv`, and the three `table_*.csv` files from the analysis.

## Citation
Add the chapter reference here after publication.
