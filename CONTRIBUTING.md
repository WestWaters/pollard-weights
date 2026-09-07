# Contributing

Contributions are welcome — this project improves fastest when people run the
tools on hardware and models we don't have.

The most valuable contribution right now is **data**: run the harnesses in
`experiments/` on a MoE model, and open an issue with the routing-concentration
curve and your hardware profile. Second most valuable: `pollard-fit` results —
what you built, for what RAM budget, and how it ran.

## Onboarding a new architecture (help Pollard scale)

Found a model Pollard doesn't recognize yet (a new `model_type` / custom code)?
`pollard-onboard` does the audit for you and writes a PR-ready contribution — so
the next person's model of that family one-shots:

```bash
pollard-onboard --model <hf-repo-or-dir> --contribute
# -> prints the arch audit (tensor coverage, flags, verdict) and writes
#    onboarding/<model_type>.md, then shows the exact git + gh commands to PR it.
```

Fill in the per-lane "works?" rows after you actually build (and `pollard-verify`)
each lane, then open the PR. That contribution — even just the findings and which
lanes worked — is what expands Pollard's arch coverage. See
`notes/custom-arch-onboarding.md` for the full playbook.

## Code

For code: keep tools stdlib-only where possible, keep every measured claim
attached to something reproducible, and if you used an AI assistant heavily,
say so in the PR — assisted is fine, unreviewed is not (same policy as
llama.cpp). Apache-2.0 applies to all contributions.
