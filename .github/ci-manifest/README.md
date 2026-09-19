# CI-pinned Manifest (Julia 1.13)

Root `Manifest.toml` stays gitignored so library consumers resolve freely.
CI copies this file into the project before `Pkg.instantiate()` so the
self-hosted GPU runner and the ubuntu-latest smoke+coverage job in `ci.yml`
resolve the same 1.13 closure instead of floating against General. `#69`
folded the leftover `codecov.yml` workflow into that smoke job.

## Regenerate

Use Julia **1.13** (the CI channel):

```bash
julia --project=. -e 'using Pkg; Pkg.update()'
cp Manifest.toml .github/ci-manifest/Manifest.toml
```

`Pkg.instantiate()` with this Manifest must keep `julia_version = "1.13.0"`
(or whatever CI currently runs). A 1.10/1.11 Manifest will not satisfy the
1.13 jobs.

Dependabot and CompatHelper PRs run `Pkg.update()` in CI after copying this
file, so the new `[compat]` is actually resolved for that job. After merging
those PRs, regenerate and commit this Manifest or later jobs stay on the
previous lock.
