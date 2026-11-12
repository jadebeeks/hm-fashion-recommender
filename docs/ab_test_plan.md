# A/B test plan

Owner: Member 5. Offline results are an estimate; this plan explains how the company would confirm them live.

- **Hypothesis:** customers shown our recommendations buy more than customers shown random products.
- **Unit of randomisation:** customer (not session), so a customer always sees the same version.
- **Groups:** A = random (today), B = trending now, C = our hybrid.
- **Primary metric:** purchases from recommended items per customer per week.
- **Guardrail metrics:** return rate, catalogue coverage, share of sales from new items, page load time.
- **Sample size:** _calculate from baseline conversion and the smallest uplift worth detecting._
- **Duration:** at least two full weeks, to cover weekday/weekend cycles.
- **Exploration:** how a bandit (e.g. epsilon-greedy or Thompson sampling) gives new items some exposure.
- **Decision rule:** _what result means "roll out"?_
