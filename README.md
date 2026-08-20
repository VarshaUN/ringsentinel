# RingSentinel

**Graph-based fraud ring detector for chargeback/return abuse**

## The problem

Point-in-time fraud models score one transaction at a time. But real merchant
losses return abuse, chargeback fraud, coordinated account abuse , usually
involve **multiple accounts sharing infrastructure**: the same card, device,
or address, acting together. A model that only looks at one transaction at a
time structurally cannot see this pattern.

RingSentinel builds an identity graph from shared transaction attributes,
detects clusters (rings) of linked accounts using community detection, and
shows that ring-level features measurably improve fraud detection over a
transaction-only baseline with an honest, cost-aware precision/recall
evaluation on a held-out test set.

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)
(Kaggle) , 590,540 real, anonymized e-commerce transactions over a 182-day
window, with card, address, email-domain, and device identifiers, and a
ground-truth `isFraud` label (base rate 3.50%).

### Key EDA findings (shape the whole design)

| Field | Missing % | Role |
|---|---|---|
| `card1` | 0.0% | primary graph edge |
| `card3` | 0.3% | primary graph edge |
| `card5` | 0.7% | primary graph edge |
| `card2` | 1.5% | primary graph edge |
| `addr1`/`addr2` | 11.1% | primary graph edge |
| `P_emaildomain` | 16.0% | secondary edge |
| `DeviceType` | 76.2% | secondary edge (sparse) |
| `R_emaildomain` | 76.8% | secondary edge (sparse) |
| `DeviceInfo` | 79.9% | secondary edge (sparse) |

Device and email identifiers are missing for ~76-80% of transactions, so the
identity graph is built primarily on **card + address** combinations, with
device/email used as bonus edges where present rather than the backbone.
This is called out explicitly rather than glossed over, since it directly
affects recall on device-only fraud patterns.

Fraud base rate is 3.5% accuracy is not a meaningful metric here;
precision/recall/PR-AUC on a **time-based** split are used throughout.

## Scope note

This is a **detection/review tool**, not an automated decision system. Output
is a risk score and a human-readable evidence report intended for a human
reviewer to act on it never auto-blocks or auto-denies a transaction.
