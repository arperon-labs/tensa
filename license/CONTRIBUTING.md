# Contributing to TENSA

Thanks for considering a contribution. TENSA is a serious research codebase
and we welcome improvements to the analysis engine, narrative fingerprint
algorithms, documentation, and test coverage.

A few things to know before you start.

## 1. TENSA is dual-licensed

TENSA is released under **AGPL-3.0** for the open-source community and under
a **commercial license** for organizations that cannot accept AGPL
obligations. See [NOTICE](./NOTICE) for details.

For this dual-licensing model to work, Arperon s.r.o. must hold sufficient
rights in every line of code to relicense it. That's what the Contributor
License Agreement does. **Every contributor must accept the CLA before
their code can be merged.** See [CLA.md](./CLA.md).

If that's a dealbreaker for you, we understand, and we will not be offended
if you maintain your contribution as an external AGPL-3.0 fork. We will
gladly link to it from the README.

## 2. What we accept

**Happily:** bug fixes, documentation, new tests, new fingerprint axes with
supporting literature, performance improvements, Rust idiom cleanups,
language/locale support for the stylometry module, examples.

**With discussion first** (open an issue): new public API, new generation
architectures, changes to the `NarrativeFingerprint` schema, changes to
Phase 0–3 algorithms, changes to BSL/AGPL/commercial licensing.

**Not accepted:** contributions that carry copyleft obligations we cannot
relicense (GPL-only code from third parties, AGPL from other copyright
holders), contributions under "non-commercial" or "ethical" licenses,
contributions that introduce a hard dependency on closed-source services
we do not control.

## 3. Workflow

1. Open an issue describing the problem or feature. For small fixes this
   is optional.
2. Fork and create a branch: `git checkout -b feature/my-change`.
3. On your first PR, the CLA Assistant bot will post a comment. Reply with
   the acceptance statement.
4. Run `cargo test`, `cargo clippy --all-targets`, and `cargo fmt` before
   pushing. For Python/TypeScript glue code, run the language-specific
   linters listed in `CONTRIBUTING-dev.md`.
5. Open a PR against `main`. Describe what changed and why. Link any
   related issue.
6. A maintainer will review within roughly 5 working days.

## 4. Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/)
v2.1. Report unacceptable behavior to **info@arperon.com**.

## 5. Security

Do **not** file security issues in the public tracker. Email
**info@arperon.com** with a description, reproduction steps, and your
preferred disclosure window. We will respond within 72 hours.

## 6. Questions

- Bugs and features: GitHub Issues
- Licensing and commercial terms: **info@arperon.com**
- Research collaboration and EU grant consortia: **info@arperon.com**
- Everything else: **info@arperon.com**
