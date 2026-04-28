# TENSA Commercial License — Standard Terms

**Licensor:** Arperon s.r.o., Hlinícka 184/37, 919 26 Zavar, Slovak Republic
**Effective:** 2026-04-18 (version 1.0)
**Contact:** info@arperon.com

This document is a **term sheet**, not the commercial license contract
itself. It describes the standard terms Arperon will offer to commercial
licensees. The binding contract is a separate Licence Agreement which
incorporates these terms, customized per customer. Pricing and discounts
are indicative and subject to negotiation.

---

## 1. What the commercial license grants you

Upon full payment of the license fee, Arperon grants you a
**non-exclusive, non-transferable, worldwide license**, for the term of
your subscription, to:

- Use, reproduce, modify, and create derivative works of TENSA;
- Distribute TENSA and derivative works as part of **Your Product** in
  object-code or source form;
- Make TENSA and derivative works available over a network as part of
  Your Product (SaaS, hosted API, internal service);
- **Without the obligation** to release source code under AGPL-3.0 or any
  other license.

## 2. What you still cannot do

- Redistribute TENSA as a **standalone product** that competes with
  Arperon's commercial offerings (e.g. a hosted TENSA API priced per
  token or per call). Your Product must incorporate TENSA as a
  component delivering additional value.
- Use the TENSA trademark, logo, or Narrative Fingerprint visual
  design except as permitted by a separate trademark usage grant.
- Sub-license TENSA to third parties for use outside Your Product.
- Remove copyright or license notices from source files you distribute
  in source form.

## 3. Standard pricing tiers

Pricing is annual, in EUR, excluding DPH/VAT. All tiers include access
to new minor and major versions released during the subscription term.

### Tier 1 — Startup
**€6,000 / year**

- Companies with annual revenue < €2M **and** < 20 employees
- Up to **1 production deployment**
- Up to **5 developers** with source-tree access
- Community support via GitHub Issues, best-effort response
- No SLA, no indemnification
- Attribution required in product credits or About screen
- Most customers start here

### Tier 2 — Growth
**€24,000 / year** (indicative; negotiated)

- Companies with revenue < €50M
- Up to **3 production deployments**
- Unlimited internal developers
- Business-hours email support, 2 business-day response target
- Standard indemnification (see §5)
- Eligible for early-access builds and roadmap input
- Attribution optional

### Tier 3 — Enterprise
**from €90,000 / year** (quote-based)

- No revenue or deployment cap
- Unlimited developers and deployments globally
- Private Slack / Teams channel with maintainers
- 24-hour response SLA on P1, 72-hour on P2
- Extended indemnification, named-customer IP warranty
- Escrow option for source code (3rd-party escrow agent)
- Option to fund prioritized feature development
- Optional on-site training at customer expense
- Optional consulting engagement under separate SOW

### Other models (by negotiation)

- **Per-call / revenue-share** hybrid for platform partners who embed
  TENSA in very high-volume products
- **Perpetual buyout** (large one-time fee + 20%/year maintenance) for
  customers whose procurement cannot do subscriptions
- **Academic & non-profit** tier at symbolic cost (€500/year) where
  the customer releases any improvements back upstream under AGPL-3.0
- **OEM / white-label** grant for customers shipping TENSA-powered
  tools to their own enterprise customers

## 4. What's always included

- Clear, version-pinned license text ("TENSA vX.Y.Z under Arperon
  Commercial License v1.0, effective dates ...")
- Rights to build and maintain forks for your own production use
- Rights to use the dual-licensed code alongside AGPL versions of
  TENSA, without the AGPL obligations attaching to your private fork
- One courtesy re-issue per year if you restructure (M&A, spin-off)

## 5. Warranty and indemnification

Arperon warrants:

1. **IP ownership.** Arperon owns or has sufficient rights to license
   all parts of TENSA that it distributes under the commercial license.
2. **No known infringement.** At the time of licensing, Arperon is not
   aware of any claim that TENSA infringes a third party's IP rights.
3. **No malicious code.** TENSA as distributed by Arperon contains no
   deliberate back doors, malware, or data-exfiltration code.

Arperon will **defend and indemnify** the customer against third-party
claims that TENSA, as distributed by Arperon, infringes the third
party's copyright, trademark, or trade secret. Indemnification cap is
**1× annual fees** at Tier 1, **2× annual fees** at Tier 2, and
**negotiated** at Tier 3.

Customer obligations: prompt notice of claim, reasonable cooperation,
Arperon's right to control the defense, no settlement without
Arperon's consent.

**Excluded from indemnity:** customer's modifications, combinations of
TENSA with third-party code not distributed by Arperon, use outside
the licensed scope, use of superseded versions more than 12 months
after a fix was available.

## 6. Term and renewal

- Initial term: 12 months from the effective date
- Auto-renews for successive 12-month periods unless either party
  gives **60 days' written notice**
- Price increases on renewal capped at **CPI + 3%** unless the
  customer moves to a higher tier

## 7. Termination and survival

On termination or expiry:

- Customer **must stop distributing** new copies of Your Product that
  include TENSA components covered by the license.
- Existing end users of Your Product retain their rights in already-
  distributed copies (this protects your customers).
- Customer may continue to **operate** deployments shipped before
  termination for a 6-month wind-down.
- Customer may at any point choose to **continue under AGPL-3.0**
  instead; in that case the commercial restrictions lift but the AGPL
  obligations (including source disclosure for network services) apply.

Surviving clauses: confidentiality, accrued fees, indemnification for
pre-termination claims, §2 restrictions.

## 8. Payment, VAT, and invoicing

- Invoiced annually in advance, in EUR
- Net 30 days from invoice date
- DPH/VAT applied per Slovak law; EU B2B customers with valid VAT ID
  under reverse charge (OSS)
- Wire transfer or via Stripe/Paddle link, with bank details and
  payment instructions provided on each invoice
- Late fees: Euribor 12M + 4% per annum

## 9. Data, privacy, confidentiality

- Arperon does **not** receive customer data through this license.
- Optional telemetry in Your Product is disabled by default and is
  governed by a separate DPA if enabled.
- Each party will protect the other's Confidential Information with
  the same care as its own, for 5 years after termination.

## 10. Governing law and disputes

- Governing law: **Slovak Republic**
- Exclusive jurisdiction: Okresný súd Trnava, with right of appeal to
  Krajský súd Trnava
- By agreement, parties may substitute **VIAC (Vienna) arbitration**
  under the VIAC Rules, seat in Vienna, language English, one
  arbitrator, for any dispute exceeding €100,000

For customers outside the EU, Arperon will consider English law +
LCIA arbitration in London as an alternative on request.

---

## Quick quote

Fill in this short form and email it to **info@arperon.com**:

    Company:                 _______________________________________
    Registered country:      _______________________________________
    Annual revenue bracket:  < €2M  /  < €50M  /  > €50M
    Intended use:            (embedded / SaaS / internal / OEM / research)
    Expected user volume:    _______________________________________
    Preferred tier:          Startup / Growth / Enterprise / Custom
    Deployment regions:      _______________________________________
    Procurement constraints: (subscription / perpetual / other)
    Legal contact:           _______________________________________
    Commercial contact:      _______________________________________
    Target start date:       _______________________________________

We respond within 2 business days with a tailored quote and a draft
Licence Agreement.
