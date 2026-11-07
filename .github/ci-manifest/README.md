# CI-pinned Manifest (Julia 1.13.2)

Root `Manifest.toml` stays gitignored so library consumers resolve freely.
CI copies this file into the project before `Pkg.instantiate()` so the
ubuntu-latest and windows-latest smoke jobs, the Linux GPU runner, and the Windows self-hosted
runner resolve the same 1.13.2 closure instead of floating against General.
`#69` folded the leftover `codecov.yml` workflow into that smoke job.

## Regenerate

Use Julia **1.13.2** (the patch the Actions `1.13` channel installs):

```bash
cp .github/ci-manifest/Manifest.toml Manifest.toml
julia --project=. -e 'using Pkg; Pkg.resolve()'
cp Manifest.toml .github/ci-manifest/Manifest.toml
```

`Pkg.instantiate()` with this Manifest must keep `julia_version = "1.13.2"`.
A 1.10/1.11 Manifest will not satisfy the 1.13 jobs. Stdlib JLL versions in
this file follow the Julia patch that resolved them.

Dependabot and CompatHelper PRs run `Pkg.update()` in CI after copying this
file, so the new `[compat]` is actually resolved for that job. After merging
those PRs, regenerate and commit this Manifest or later jobs stay on the
previous lock.
