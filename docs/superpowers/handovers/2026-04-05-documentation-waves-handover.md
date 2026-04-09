# Handover: Ecosystem Documentation Improvement (Waves 0-2)

Last updated: 2026-04-06

## Context

The research ecosystem (rltrain, samgria, toblox, xptrack, fractal-weight-spaces) has strong per-repo documentation but three systemic gaps identified through engineering team review:

1. **No ecosystem-level documentation** — each repo documents itself, nothing explains how they connect
2. **Mathematical "why" missing** — code mirrors the math, docs describe "what" not "why"
3. **Extension guides missing** — protocols exist (GradientTransform, Store, Reader, Hook, View), no user-facing guide on implementing against them

## Progress

| Phase | Card | Status | PR |
|-------|------|--------|----|
| **1** | Ecosystem guide (rltrain) | **Done** | DarkbyteAT/rltrain#26, merged to main |
| **1.5** | Docstring conversion (rltrain) | Todo | — |
| **1.5** | Docstring conversion (xptrack) | Todo | — |
| **1.5** | Docstring conversion (samgria) | Todo | — |
| **1.5** | Docstring conversion (toblox) | Todo | — |
| **2** | rltrain CLAUDE refresh | Todo | — |
| **2** | xptrack CLAUDE refresh | Todo | — |
| **2** | samgria CLAUDE refresh | Todo | — |
| **2** | toblox CLAUDE refresh | Todo | — |
| **2** | fws CLAUDE creation | Todo | — |
| **3** | samgria math formulations | Todo | — |
| **3** | toblox math formulations | Todo | — |
| **3** | rltrain composability guide | Todo | — |
| **4** | xptrack module READMEs | Todo | — |
| **4** | samgria extension guide | Todo | — |
| **4** | rltrain tracking guide | Todo | — |

## Design Values

All documentation must:
- **Mirror mathematical structure** — equations in collapsible blocks, citation comments in code
- **Cross-link with full paths** — never "see above", always relative links
- **Be scannable** — tables over prose, structured headings, collapsible detail
- **Work for humans and agents** — CLAUDE.md/AGENTS.md are agent entry points, READMEs are human entry points
- **Be additive not brittle** — new features (MAML, implicit params) should slot in without rewriting

## Boards and Card IDs

| Card | Board | Board ID | Link |
|---|---|---|---|
| ~~**W0** Ecosystem guide~~ | rltrain | `69cafa92b7759bad35d0801f` | https://trello.com/c/PKMYhOp7 |
| **DS** rltrain docstrings | rltrain | `69cafa92b7759bad35d0801f` | https://trello.com/c/nBaSb3rA |
| **DS** xptrack docstrings | xptrack | `69cb09a0f5627e88d4b651d2` | https://trello.com/c/k0BlLEMa |
| **DS** samgria docstrings | samgria | `69d2790d39ed1f3a6ddd37c8` | https://trello.com/c/0aLB7SHa |
| **DS** toblox docstrings | toblox | `69d2556094795017e80e996d` | https://trello.com/c/vtI8R6Ww |
| **W0** rltrain CLAUDE refresh | rltrain | `69cafa92b7759bad35d0801f` | https://trello.com/c/cxpMHScj |
| **W0** xptrack CLAUDE refresh | xptrack | `69cb09a0f5627e88d4b651d2` | https://trello.com/c/zZw4ekb4 |
| **W0** samgria CLAUDE refresh | samgria | `69d2790d39ed1f3a6ddd37c8` | https://trello.com/c/o3SGjjf2 |
| **W0** toblox CLAUDE refresh | toblox | `69d2556094795017e80e996d` | https://trello.com/c/ENUTUn2e |
| **W0** fws CLAUDE creation | fractal-weight-spaces | `69c97b7671b2621142b242cf` | https://trello.com/c/79ILWR1E |
| **W1** samgria math formulations | samgria | `69d2790d39ed1f3a6ddd37c8` | https://trello.com/c/d0v0cb6x |
| **W1** toblox math formulations | toblox | `69d2556094795017e80e996d` | https://trello.com/c/MsmMtF7N |
| **W1** rltrain composability guide | rltrain | `69cafa92b7759bad35d0801f` | https://trello.com/c/lNy1TNtE |
| **W2** xptrack module READMEs | xptrack | `69cb09a0f5627e88d4b651d2` | https://trello.com/c/J5ASeJYM |
| **W2** samgria extension guide | samgria | `69d2790d39ed1f3a6ddd37c8` | https://trello.com/c/n4stAK7B |
| **W2** rltrain tracking guide | rltrain | `69cafa92b7759bad35d0801f` | https://trello.com/c/Vt0ryuPL |

## Dependency Graph

```
[W0] Ecosystem guide (rltrain)                    ✅ DONE, merged
  │
  ├─ [DS] Docstring conversion (4-way parallel, no deps)
  │    ├─ rltrain   https://trello.com/c/nBaSb3rA
  │    ├─ xptrack   https://trello.com/c/k0BlLEMa
  │    ├─ samgria   https://trello.com/c/0aLB7SHa
  │    └─ toblox    https://trello.com/c/vtI8R6Ww
  │
  ├→ [W0] CLAUDE/AGENTS refresh (5-way parallel, after docstrings per repo)
  │    ├─ rltrain   (after rltrain docstrings merge)
  │    ├─ xptrack   (after xptrack docstrings merge)
  │    ├─ samgria   (after samgria docstrings merge)
  │    ├─ toblox    (after toblox docstrings merge)
  │    └─ fws       (no docstrings, unblocked now)
  │
  ├→ [W1] Math + Composability (3-way parallel, after CLAUDE refresh per repo)
  │    ├─ samgria math formulations   (after samgria CLAUDE refresh)
  │    ├─ toblox math formulations    (after toblox CLAUDE refresh)
  │    └─ rltrain composability guide (after rltrain CLAUDE refresh)
  │
  └→ [W2] Extension Guides (3-way parallel, after W1 per repo)
       ├─ xptrack module READMEs      (after xptrack CLAUDE refresh)
       ├─ samgria extension guide     (after samgria math formulations)
       └─ rltrain tracking guide      (after rltrain composability guide)
```

## Execution Plan

### Phase 1: Ecosystem Guide — DONE
Created `docs/ecosystem.md` with vision, repo table, Mermaid data flow, paper-to-repo mapping, protocol integration points, and getting-started instructions. Merged via DarkbyteAT/rltrain#26. Reviewed through 5 rounds of engineering team review (advocate, architect, contrarian, ML engineer, data scientist, data engineer, research intern).

### Phase 1.5: Docstring Conversion (4-way parallel)
Convert all docstrings from NumPy style to Google style across rltrain, samgria, toblox, and xptrack. Each repo gets its own card, branch, and PR. Must land before CLAUDE refresh so the refreshed docs reference "Google-style docstrings" as the convention.

Scope per repo: all `.py` files in source and tests. Update CONTRIBUTING.md with a docstring style guide.

### Phase 2: CLAUDE/AGENTS Refresh (5-way parallel)
Once docstrings merge per repo, dispatch parallel agents. Each refreshes CLAUDE.md and AGENTS.md with ecosystem links, current conventions (Google docstrings), and current state. The fws card creates CLAUDE.md from scratch (no docstrings to convert, so unblocked immediately).

### Phase 3: Math + Composability (3-way parallel)
Once the relevant W0 card merges per repo:
- samgria: collapsible math blocks, citation comments, comparison table
- toblox: orthogonal init rationale, D2RL citation, RFF bandwidth docs, enhanced catalogue
- rltrain: composability explanation, JSON config examples, Mermaid execution diagram

### Phase 4: Extension Guides (3-way parallel)
Once the relevant W1 card merges per repo:
- xptrack: store/README.md, ui/README.md, rl/README.md with protocol contracts and examples
- samgria: docs/extending.md with custom transform guide and FQN config examples
- rltrain: docs/experiment-tracking.md with backend comparison and MetricsLogger guide

## Deferred Work

### Ecosystem guide update for MetaOptimizer
When the `MetaOptimizer` protocol merges to samgria, update `docs/ecosystem.md` in the same PR or as part of the samgria extension guide (W2). Changes needed:
- New row in the Protocols table for `MetaOptimizer`
- Update samgria repo table entry to mention meta-learning support
- Possibly a new edge in the Mermaid diagram if it changes the data flow

### Ecosystem guide convention update
The ecosystem guide currently lists "NumPy-style docstrings" in the shared conventions table. This should be updated to "Google-style docstrings" when the docstring conversion cards land. Can be done as part of the rltrain CLAUDE refresh card since that card already touches rltrain documentation.

## Key Conventions

### Google-Style Docstrings (all new docstrings)
```python
"""Summary line.

Args:
    param: Description of param.

Returns:
    Description of return value.
"""
```

### Collapsible Math Pattern (Wave 1)
```markdown
Plain English explanation of the transform.

<details><summary>Mathematical formulation</summary>

$$\epsilon^* \approx \rho \frac{\nabla L(\theta)}{\|\nabla L(\theta)\|}$$

Reference: Foret et al. (2021), Algorithm 1.
</details>
```

### Citation Comment Pattern (Wave 1)
```python
# SAM perturbation: ε* ≈ ρ · ∇L/||∇L||, Foret et al. 2021 Eq. 1
adv_params = init_params + (self.rho * init_grad)
```

### Module README Pattern (Wave 2)
Follow rltrain's existing module READMEs: architecture overview, protocol contract table, "How to add your own X" section, link to tests as executable examples.

### CLAUDE.md Pattern (Wave 0)
```markdown
@AGENTS.md

## Project Context
[2-3 sentences]

## Scope
| In scope | Out of scope |
|----------|-------------|

## Ecosystem
Part of [research ecosystem](link). See docs/ecosystem.md.
- [repo]: [how this repo connects]

## Known Constraints
[honest status of anything incomplete]
```

## Content Guidelines

- **Don't describe features that might change** — reference the protocol contract, not the current implementation list. New transforms, views, or backends can be added without updating the guide.
- **Do describe the mathematical principles** — these are stable. SAM's perturbation equation won't change. Orthogonal init's rationale won't change.
- **Do describe the extension pattern** — "implement these methods, wire via FQN config" is stable regardless of what implementations exist.
- **Don't hardcode lists of implementations** — use "see `samgria/__init__.py` for current exports" rather than listing SAM, ASAM, LAMP explicitly in guides (new transforms like MAML will land in parallel sessions).

## Per-Card Execution

For each card:
1. Read the Trello card description and Definition of Done checklist
2. Check dependency card is in Done and merged to main
3. Create a branch, implement, PR
4. Run `/gemini review` and iterate until convergence (or engineering team review if Gemini quota exhausted)
5. Tick checklist items as completed
6. Move card Todo, Doing, Reviewing, Done
