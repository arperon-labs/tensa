# Dual Licensing of TENSA

TENSA is available under two licenses, and you choose the one that fits
your use case. You do **not** need permission from Arperon to use TENSA
under the AGPL — just follow the AGPL.

## Option 1 — GNU Affero General Public License v3.0 (free)

Use this if **any of the following** is true:

- You're doing research, academic work, or a personal project.
- You're building an open-source product and you're happy to release
  your whole product under AGPL-3.0.
- You're running TENSA strictly inside your own walls for internal use
  and you will never expose it to users outside your organization
  (including via a network).

Under AGPL-3.0 you can use, modify, and distribute TENSA for free,
including in commercial settings. The main obligation is that if you
**distribute** modified TENSA **or** make it available to users over
a network (SaaS, hosted API, web app), you must release the complete
corresponding source code of your product under AGPL-3.0 as well.

That's the "affero" clause, and it is the reason many companies cannot
adopt AGPL software: it would require open-sourcing their entire
product.

## Option 2 — Commercial license from Arperon s.r.o. (paid)

Use this if **any of the following** is true:

- You want to embed TENSA in a proprietary product (desktop, mobile,
  web) and ship it to customers without releasing your source code.
- You want to offer a hosted or SaaS product that uses TENSA, without
  releasing the service's source code under AGPL-3.0.
- Your legal/procurement team will not approve AGPL-3.0 software for
  any reason (this is common at Fortune 500 companies, defense
  contractors, and regulated industries).
- You need a written warranty, indemnification, priority support, or
  an SLA.

The commercial license removes all AGPL-3.0 source-disclosure obligations
and adds the commercial terms you need. See
[COMMERCIAL-LICENSE-TERMS.md](./COMMERCIAL-LICENSE-TERMS.md) for the
standard terms and pricing tiers, or write to
**info@arperon.com** for a quote.

## FAQ

### Can I try TENSA commercially before paying?

Yes. AGPL-3.0 does not distinguish between commercial and non-commercial
use. You can evaluate TENSA internally for as long as you like under
AGPL-3.0. A commercial license is only required when you distribute or
network-expose TENSA as part of a product whose source you don't want
to release.

### I'm building a closed-source app that calls TENSA as an external API
### I run myself. Do I need a commercial license?

If your users interact with TENSA over the network — even indirectly,
through your app — the AGPL section 13 network clause probably applies
to the TENSA deployment. The safest reading is: yes, you need a
commercial license for that deployment, **or** you must release your
app's source code under AGPL-3.0. When in doubt, email us.

### Can I contribute to TENSA and still use it commercially?

Yes. Contributing under the CLA doesn't change your options — you can
still obtain a commercial license if your product requires it.
Contributors who become commercial customers sometimes receive
discounts; ask.

### Does the commercial license include future versions?

Yes, within the term of your subscription. See the term sheet for
details.

### Does Arperon assert patents over TENSA?

Arperon may seek patent protection on specific downstream applications
(for example, the narrative-fingerprint visualization system or the
multi-voice generation pipeline). **The AGPL-3.0 grant and the
commercial license both include an explicit patent license** to
everything needed to use TENSA as distributed. Patents are not used
against users of TENSA operating within either license.

### What about trademarks?

**TENSA**, the TENSA logo, and the Narrative Fingerprint visual design
are trademarks of Arperon s.r.o. Neither the AGPL-3.0 grant nor the
commercial license grants any trademark rights. Forks must rename
themselves. See the `TRADEMARK.md` file (coming soon) for the trademark
usage policy.
