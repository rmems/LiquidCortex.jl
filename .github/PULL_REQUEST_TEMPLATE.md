## Summary

<!-- What does this PR change, and why? -->

## Relationships

- **Closes** / **Does not close** <!-- issue number -->
- Related:
- PRs cannot have native blocked-by/sub-issue links

## Review checklist

### Code Quality

- [ ] No domain-specific code in core (market, mining, crypto, hardware telemetry)
- [ ] Generic interfaces (`AbstractVector`, `Real` kwargs) over concrete types where appropriate
- [ ] Input validation for user-facing constructors and functions
- [ ] SPDX license headers on new source files

### Testing

- [ ] New code has corresponding tests
- [ ] GPU tests properly gated by `LiquidCortex._cuda_available[]`
- [ ] No hardcoded dimensions — use configurable `n_in`/`n_out`

### CI/CD

- [ ] Julia 1.13 CI passes (CPU ubuntu-latest + GPU self-hosted when available)
- [ ] No Codacy warnings (SHA-pinned actions, no inline HTML in markdown)
- [ ] No unresolved bot review threads

### Documentation

- [ ] README / docs updated if public API changed
- [ ] Docstrings match actual function signatures
- [ ] `CHANGELOG.md` updated for user-facing changes
- [ ] `AGENTS.md` updated if build/test commands changed

### Breaking Changes

- [ ] Documented in PR description and `CHANGELOG.md`
- [ ] Version bump in `Project.toml` if applicable (`0.x` breaking → minor)
- [ ] Migration notes for downstream consumers

## Test plan

- [ ] `julia --project -e 'using Pkg; Pkg.test()'` (CPU; GPU gated)
