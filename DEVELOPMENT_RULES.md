# Development Rules

Permanent working instructions for Retro AI Lab. These expand on `CLAUDE.md` section 4 onward and `DESIGN.md` section 18. Where this file and `CLAUDE.md` differ, `CLAUDE.md` wins for procedure and `DESIGN.md` wins for product requirements.

## 1. The loop

Every task follows this sequence. No stage is skipped.

**PLAN → TEST PLAN → IMPLEMENT → TEST → FIX → RETEST → REGRESSION TEST → COMMIT → NEXT TASK**

1. **Plan.** Pick the next `Ready` task in `TASK_TRACKER.md`. Read the code it touches. Mark it `In progress`.
2. **Test plan.** Write the test plan in the task's entry (template below) *before* writing implementation code. Write the automated tests first where practical; they should fail.
3. **Implement.** Change only what the task scope allows. If you discover extra work, add a new task to the tracker instead of widening scope.
4. **Test, fix, retest.** Run the task's tests until 100% pass. Fix the code, not the assertion, unless the test itself is demonstrably wrong (record why in the completion report).
5. **Regression.** Run the full suite (`pytest`) plus lint and type checks. 100% must pass.
6. **Commit.** Commit on the task branch, update `TASK_TRACKER.md` and `CHANGELOG.md`, open a PR into `main`.
7. **Next.** Only after the PR is merged does the next task branch start, from the updated `main`.

## 2. One task at a time

- Exactly one task is `In progress` at any moment.
- A task should be completable in one sitting and touch one concern. If it has several independent features, split it.
- Never start a task on top of an untested or failing checkpoint.

## 3. Task template

Every task entry in `TASK_TRACKER.md` uses this shape:

```markdown
### T-NNN – Short title
- **Status:** Ready | In progress | Blocked | Done
- **Phase:** 1–8 (DESIGN.md §16)
- **Objective:** one sentence.
- **Scope:** what is in. **Out of scope:** what is not.
- **Acceptance criteria:**
  1. Observable, testable statement.
- **Expected changes:** files / modules.
- **Test plan:** unit / integration / UI / manual cases, including invalid input, edge cases, failure handling and regression.
- **Results:** total / passed / failed / excluded (with justification).
- **Commit:** hash. **PR:** link.
- **Completion report:** what was done, what was verified and how, known limitations.
```

## 4. Testing rules (summary; detail in `TESTING_STRATEGY.md`)

- 100% of mandatory tests pass, or the task is not done. 99% is a failure.
- Prohibited: skipping, disabling or `xfail`-ing a failing test; weakening assertions; catching and hiding errors; marking unfinished work done; proceeding with a known regression.
- An exclusion is allowed only when a test is genuinely inapplicable in an environment (for example, a real-emulator test on a CI runner with no emulator). It must be recorded in the task's results with a reason and a manual verification procedure that covers the gap.
- Tests check behaviour, not merely that a class exists.

## 5. Git workflow

- `main` is the approved checkpoint. Nothing lands on `main` without a passing PR.
- Branch per task: `task/T-NNN-short-slug`, created from the latest `main`.
- Commit messages: `T-NNN: imperative summary`.
- One PR per task. The PR body lists acceptance criteria, test results and any exclusions.
- Never rewrite history on `main`. Never force-push a branch someone else is using.

## 6. Code standards

- Python 3.11+, type hints throughout, `src/` layout, package `retro_ai_lab`.
- Format and lint with `ruff`; type check with `mypy` (strict on new modules).
- The UI layer never talks to an emulator or a model directly; it goes through services (see `ARCHITECTURE.md`).
- No network access at runtime unless a feature explicitly requires it (DESIGN.md §15). No execution of third-party scripts from imported projects.
- Child-facing text is plain English, no jargon, no hex. Advanced detail goes behind an "Advanced" view.

## 7. Stop conditions

Stop and ask the user (explain the issue, its impact and a recommended option) when:

- A requirement is ambiguous and a wrong guess would affect architecture.
- A major architectural decision has not been authorised (see the decision log in `ARCHITECTURE.md`; anything marked *Proposed* needs sign-off before tasks depend on it irreversibly).
- An external dependency prevents reliable testing.
- A critical security issue is found.
- The codebase contains an unexpected blocker.
- A task cannot meet its acceptance criteria, or fails repeatedly.

Never silently change project direction to get around a blocker.

## 8. Session handoff

At the end of every session, append to the "Session log" in `TASK_TRACKER.md`: tasks completed, commits, test results, current task and its exact state, and blockers. A new session resumes from the last merged checkpoint and never assumes an interrupted task passed.
