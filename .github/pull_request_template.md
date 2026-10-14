## What does this change?

<!-- One or two sentences. "Closes #123" links the issue it addresses. -->

## Type of change

- [ ] Bug fix
- [ ] New feature (material, phantom, artifact, algorithm, metric, …)
- [ ] **Changes simulated numbers** — explain below, regenerate the reference with
      `pytest tests/test_regression.py --update-reference`, and add a CHANGELOG entry
      under "Changes that alter simulated numbers"
- [ ] Documentation / tutorial
- [ ] Infrastructure (CI, packaging, tooling)

## Checklist

- [ ] `pytest` passes locally (the pre-push hook runs the fast tests for you)
- [ ] New behaviour has a test in `tests/`; a bug fix has a regression test
      (or removes the `xfail` marker that recorded the bug)
- [ ] Notebooks are committed without outputs (`pre-commit` does this for you)
- [ ] Docs (`docs/`) and `CHANGELOG.md` (`[Unreleased]`) updated for user-visible changes

## Physics / numerical impact

<!-- Which numbers change, by how much, and why. Delete if none. -->
