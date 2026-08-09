# Contributing

Contributions are welcome — this project improves fastest when people run the
tools on hardware and models we don't have.

The most valuable contribution right now is **data**: run the harnesses in
`experiments/` on a MoE model, and open an issue with the routing-concentration
curve and your hardware profile. Second most valuable: `pollard-fit` results —
what you built, for what RAM budget, and how it ran.

For code: keep tools stdlib-only where possible, keep every measured claim
attached to something reproducible, and if you used an AI assistant heavily,
say so in the PR — assisted is fine, unreviewed is not (same policy as
llama.cpp). Apache-2.0 applies to all contributions.
