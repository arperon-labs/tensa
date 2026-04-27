# TENSA Dual-Licensing Template — README

This directory contains a ready-to-ship dual-licensing setup for the TENSA
open-source repository. It's designed to be dropped into the root of the
public `tensa` / `tensa-core` repo at Arperon.

## Files

| File | Purpose | Where it goes |
|---|---|---|
| `LICENSE` | AGPL-3.0 license text with Arperon header | repo root |
| `NOTICE` | Copyright + dual-license declaration | repo root |
| `CLA.md` | Contributor License Agreement | repo root |
| `CONTRIBUTING.md` | Contributor workflow + CLA requirement | repo root |
| `DUAL-LICENSING.md` | Plain-English explanation and FAQ | repo root |
| `COMMERCIAL-LICENSE-TERMS.md` | Standard commercial term sheet | repo root |
| `README-LICENSING-snippet.md` | Block to paste into your main README | (paste into README) |
| `finalize-license.sh` | One-time setup: fetches canonical AGPL text | run once |

## Quick start

```bash
# 1. Copy the files into your TENSA repo root
cp -r tensa-licensing/* /path/to/tensa/

# 2. Fill in the placeholders in NOTICE (IČO, DIČ, IČ DPH)
vi /path/to/tensa/NOTICE

# 3. Download the canonical AGPL-3.0 text into LICENSE
cd /path/to/tensa
chmod +x finalize-license.sh
./finalize-license.sh

# 4. Paste the README snippet into your main README
cat README-LICENSING-snippet.md   # copy the relevant section

# 5. Set up the CLA Assistant bot on GitHub
#    https://cla-assistant.io/
#    Point it at CLA.md in this repo.

# 6. Add SPDX headers to source files (see LICENSE for the template)
```

## Before you publish — checklist

- [ ] Slovak lawyer reviews `CLA.md` and `COMMERCIAL-LICENSE-TERMS.md`
      against Slovak Civil Code, GDPR, and the Act on Copyright
      (Autorský zákon č. 185/2015 Z. z.). Specifically confirm:
      - moral rights waiver wording (§ 18 ods. 8 Autorského zákona)
      - enforceability of Section 2.4 in CLA
      - VAT/DPH treatment of cross-border software licensing
- [ ] Arperon internal sign-off on the commercial pricing tiers.
      Tier 1 (€6,000) should cover a modest margin over support cost
      for indie studios; Tier 3 floor (€90,000) should be calibrated
      against what Strabag-class customers would pay for Zonograph.
- [ ] Decide: **AGPL-3.0** vs **AGPL-3.0 + exception for plugin API**.
      The plain AGPL copyleft reaches through linking, which can
      scare off users who only want to write plugins. Consider a
      written exception that says "linking with plugins via the public
      `tensa-plugin-api` crate does not create a derivative work."
- [ ] Register `licensing@arperon.com`, `security@arperon.com`,
      `research@arperon.com`, `conduct@arperon.com`, `hello@arperon.com`
      as working aliases before going public.
- [ ] Trademark: file Slovak/EU trademark application for "TENSA" and
      the Narrative Fingerprint radar design **before** the public
      release. The public release starts the clock on opposition and
      creates prior-use evidence you'll need if someone else tries to
      register the mark later.
- [ ] Set up **github.com/Arperon/tensa-core-commercial** (private)
      for maintaining the relicensed commercial branch. Nightly CI
      should mirror `main` from the public repo, strip AGPL file
      headers, insert commercial headers, and retag. This keeps the
      commercial codebase in lockstep with the public one without
      manual work.
- [ ] Decide how to handle patches from the private commercial
      branch. Default policy: **upstream them** to the public repo
      under AGPL so the codebase doesn't fork. Exceptions only where
      a patch contains customer-confidential logic.

## What this is **not**

This is a well-drafted template grounded in best practice from Harmony,
Sentry (FSL), and typical commercial-OSS dual-license setups. It is
**not a substitute for a Slovak IP lawyer**. Before relying on these
documents in any dispute, have a lawyer:

1. Review enforceability under Slovak and EU law.
2. Confirm the moral-rights waiver language is effective.
3. Adjust jurisdiction/arbitration clauses to Arperon's preference.
4. Review the commercial term sheet for tax and accounting implications
   (it's a template — real contracts with Strabag-class customers will
   have extensively negotiated variations).

Budget: one morning of a Trnava/Bratislava IP lawyer's time,
approximately €400–800.

## Evolution path

Once the public release is stable and the first few commercial licensees
have signed, you'll want to add:

- `TRADEMARK.md` — usage policy for "TENSA" and the Fingerprint mark
- `SECURITY.md` — disclosure policy with PGP key
- `GOVERNANCE.md` — who decides what in the project (currently: Arperon)
- A formal `CODE_OF_CONDUCT.md` based on Contributor Covenant 2.1
- Per-language CLA acceptance (if you want Slovak and Indonesian
  localized versions for FMK UCM and Indonesian partners respectively)
