# Split v1 report

Rule: per domain, units ranked by sha256(unit_id); locked_quota(n) = round(0.25 n), >= 1 when n >= 2; one-unit domains are tuning-only.

Overall: 81 of 316 units locked (25.6%); 98 of 367 datasets (26.7%).

| Domain | Units | Locked units | Locked share (units) | Datasets | Locked datasets |
|---|---|---|---|---|---|
| Agriculture-commodities | 7 | 2 | 29% | 9 | 2 |
| Air-quality | 5 | 1 | 20% | 6 | 1 |
| Aviation-tourism | 15 | 4 | 27% | 15 | 4 |
| Biology-growth | 10 | 3 | 30% | 10 | 3 |
| Buildings-heat | 19 | 5 | 26% | 19 | 5 |
| Cloud-servers | 19 | 5 | 26% | 22 | 8 |
| Crypto-markets | 5 | 1 | 20% | 5 | 1 |
| EV-parking | 8 | 2 | 25% | 8 | 2 |
| Ecology | 9 | 2 | 22% | 9 | 2 |
| Energy-gen | 16 | 4 | 25% | 17 | 4 |
| Energy-load | 16 | 4 | 25% | 23 | 4 |
| Epidemics-health | 17 | 4 | 24% | 21 | 6 |
| Events-entertainment | 10 | 3 | 30% | 10 | 3 |
| Government-stats | 2 | 1 | 50% | 2 | 1 |
| Housing-realestate | 7 | 2 | 29% | 7 | 2 |
| Hydrology | 7 | 2 | 29% | 14 | 3 |
| Industrial-PHM | 19 | 5 | 26% | 20 | 5 |
| Macro-finance | 7 | 2 | 29% | 10 | 5 |
| Oil-gas-fuels | 13 | 3 | 23% | 14 | 3 |
| Physics-reference | 1 | 0 | 0% | 1 | 0 |
| Physiology | 9 | 2 | 22% | 15 | 8 |
| Public-services | 6 | 2 | 33% | 6 | 2 |
| Retail-sales | 17 | 4 | 24% | 21 | 4 |
| Space-geo | 12 | 3 | 25% | 12 | 3 |
| Telecom-network | 5 | 1 | 20% | 5 | 1 |
| Transport-mobility | 22 | 6 | 27% | 24 | 8 |
| Utilities-waste | 1 | 0 | 0% | 1 | 0 |
| Utilities-water | 1 | 0 | 0% | 1 | 0 |
| Weather-climate | 21 | 5 | 24% | 25 | 5 |
| Web-attention | 10 | 3 | 30% | 15 | 3 |

Tuning-only domains (one unit): Physics-reference, Utilities-waste, Utilities-water.

By dataset the locked share can be lopsided where a large group falls in the locked set (2026-10-27: Physiology 8 of 15, Macro-finance 5 of 10). Accepted by the owner: the domain cap applies when scoring, so this changes weights, not leakage.
