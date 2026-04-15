---
name: code-review-protocol
description: Structured protocol for thorough, honest code review and bug fixing. Use whenever asked to review, audit, debug, fix, clean up, or assess code quality — in full files or sections. Trigger on: "review this", "check for bugs", "fix issues", "clean this up", "audit the pipeline", "is this production ready", "what's wrong with", or any request to examine existing code for problems. This skill governs HOW to review, not what bugs look like — combine with debugging-playbook for runtime failure patterns.
---

# Code Review Protocol

## Core Rules (Non-Negotiable)

1. **Read everything before writing anything.** Full file(s), not skimming.
2. **Never invent problems.** If unsure whether something is a bug, say so explicitly.
3. **Never fix what you haven't confirmed is broken.** Suspicion ≠ diagnosis.
4. **One pass is not enough.** Always do two passes (see below).
5. **Report honestly.** If nothing is wrong, say nothing is wrong. Do not manufacture findings to appear thorough.

---

## The Two-Pass Review

### Pass 1 — Read + Catalogue (no edits yet)

Read all files in scope completely. Catalogue every finding into one of three buckets:

| Bucket | Definition |
|--------|-----------|
| **CONFIRMED** | Demonstrably broken — traceable to a concrete failure mode |
| **SUSPECT** | Looks wrong but needs verification (state reason) |
| **STYLE** | Not a bug — quality/readability issue only |

Do not touch code during Pass 1. Do not report findings yet.

### Pass 2 — Verify + Triage

For each CONFIRMED and SUSPECT item:
- Re-read the relevant lines in context
- Trace the execution path — would this actually cause a failure?
- Downgrade SUSPECT → STYLE or dismiss entirely if not traceable to real failure
- Assign severity to each CONFIRMED item (see below)

Only after Pass 2 is complete: report findings and propose fixes.

---

## Severity Levels

| Level | Meaning |
|-------|---------|
| **P0 — Production Breaking** | Will cause crash, silent wrong output, or data corruption in normal use |
| **P1 — Latent** | Won't crash today but will under specific conditions (edge cases, scale, GPU) |
| **P2 — Quality** | Bad practice, technical debt, redundancy — no failure risk |
| **P3 — Style** | Naming, formatting, clarity — no behavior impact |

Fix order: P0 first, one at a time. Do not bundle fixes across severity levels.

---

## Reporting Format

Report findings before making any changes. User must see the full picture first.

```
## Review Report — {filename or section}

### CONFIRMED Issues
[P0] <location>: <what is wrong> — <why it will fail>
[P1] <location>: <what is wrong> — <condition that triggers it>

### SUSPECT (needs verification)
[?] <location>: <what looks wrong> — <what I'm uncertain about>

### Quality / Style (P2–P3)
- <location>: <issue>

### No Issues Found
- <list any areas explicitly checked and cleared>

---
Proposed fix order: [list P0s in sequence]
Awaiting confirmation to proceed.
```

Always include "No Issues Found" section — lists what was checked and cleared. This proves thoroughness and gives the user confidence in what was *not* flagged.

---

## Fix Protocol

Only proceed after user confirms the report.

1. Fix one P0 at a time — do not batch unrelated changes
2. State exactly what you changed and why (one sentence each)
3. After each fix: re-read the changed lines and surrounding context
4. Never modify lines outside the confirmed problem scope
5. If a fix requires touching more code than expected — stop, report, confirm before continuing

**Do not:**
- Refactor while fixing bugs
- "Improve" adjacent code that wasn't flagged
- Add features or logging while in fix mode
- Combine a P0 fix with P2 cleanup in the same edit

---

## Post-Fix Verification

After all P0s are fixed:

1. Re-read every changed section cold (as if seeing it for the first time)
2. Trace the execution path through each fix — confirm the failure mode is actually closed
3. Check that the fix didn't introduce a new issue in adjacent logic
4. Re-scan the full file for anything the fix may have interacted with

Report the verification result explicitly:
```
## Post-Fix Check
- [fix 1]: confirmed closed — <one sentence on why>
- [fix 2]: confirmed closed — <one sentence on why>
- No new issues introduced in surrounding code.
```

---

## Honesty Rules

- If you're not certain something is a bug: say "SUSPECT" not "CONFIRMED"
- If you find nothing: report nothing found, list what was checked
- If a second review finds issues a first review missed: say so plainly — state whether each new finding was missed due to shallow reading or was genuinely ambiguous
- Never reframe a missed issue as "a new concern that emerged" — own it as a miss
- If scope is too large to review thoroughly in one session: say so, propose splitting by file or subsystem

---

## Scope Discipline

Before starting: state the exact files and sections in scope. Confirm with user if ambiguous.

Do not review out-of-scope files unless a traced execution path requires it — in that case, state why you're expanding scope before doing so.

If asked to review "the training pipeline" or similar broad term: list what you interpret as in-scope and confirm before starting Pass 1.
