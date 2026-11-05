# Notes for AI assistants

Start with [`docs/status_and_plan.md`](docs/status_and_plan.md): what the project claims,
what is done, the numbers so far, known problems and the step-by-step plan. Raw results are
in [`results/`](results/).

Quick rules:

- Report only numbers that are in `results/` or a run folder; label guesses as hypotheses.
- Judge learned policies by the fixed-road soil-wetness (LI) sweeps, not by training-time numbers.
- Change one thing per training run; test changes with about 3M steps before any long run.
- Setup, training and evaluation commands are in `docs/status_and_plan.md` (section 7) and
  `docs/training.md`. Install with `pip install -e ".[train]"`.
- Update `docs/status_and_plan.md` whenever a step finishes.
