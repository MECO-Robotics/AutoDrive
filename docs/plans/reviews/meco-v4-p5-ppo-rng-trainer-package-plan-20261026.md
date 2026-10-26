# Independent Pass 1 review: P5 PPO/RNG trainer package plan

- **Verdict:** ACCEPT (supersedes REVISE for exact predecessor `c2ed79b0c28a7e2860c0d74c10706dbcbbdfac68`)
- **Review date:** 2026-10-26
- **Reviewer identity:** `/root/closure_wave1_reviewer`
- **Plan author:** `plan_a_owner`
- **Reviewed plan commit:** `c8e65c542de1a8c3e44df8f8c6e5fa6d5924e6a7`
- **Reviewed file:** `docs/plans/meco-v4-p5-ppo-rng-trainer-package-plan-20261026.md`
- **Git blob:** `9a729ec2c48e5d316485d007b8207a6282c1ce29`
- **Plan file SHA-256:** `15c06a86bc4cca3ccc549c7ff50f7131e027a9007dd2d96554c06041503e0d53`
- **Implementation source base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`
- **Accepted overall staged-gate plan:** commit `8efec9378d1912f0839d7c6cd8e88467fceb1d5a`, file SHA-256 `57f20a1bcfe6f229429d402140de93a0ee1780372e21a9ac3344f81c72faca90`; ACCEPT record commit `b0b2962d299b1ecaab053b873ef2ae09e3c5f467`, record SHA-256 `54e68870b3b95fd91b92d3f934e44d5ec9bb63e1fb6d03f67e6d95f10d3abb4a`
- **Accepted package-plan dependencies:** P0 `72ee8fce1d4a955d100b97d47b58528374faddec` / SHA `16c1fbcd881bd77adf178c8d8bf5f17bd3cf3e988cde767fbbc7731c8ff088d9`, review `cb02ef8fd74ab4b65c4b0f86a1796eaad616e77d` / SHA `a8c5164730d62bd04c7a134e0d395a81ce8f76c1088db6e5f5a8e2b5bd62701d`; P1 `606ba5adce04afd84bfb372260aebde8fd4fcec6` / SHA `7dc5be35d3826a6047653021e97a941d6872a73a6997ba768e95753f693b0653`, review `77816edbd1923c018d4af774b2f1b3bae0f415b8`; P2 `fa15258776030ccb2ca5f9f637a38594075797d7` / SHA `123ff2f8741636eabc1aea4b876a59ac326fa81f4739a469ecc4271f4d6807ca`, review `cb02ef8fd74ab4b65c4b0f86a1796eaad616e77d` / SHA `10c1ac00586f9535b53905395afe34ca2fefe3cb062656ec38d2236a83db185c`.
- **Further accepted package-plan dependencies:** P3 `2b06304bde63cace9f59b6b25497c82250cf759e` / SHA `92a57003e63b0aad90c95f45b23c591106d03071675ff5f58ec39a8367777e60`, review `9790c4ad253074ff000c2ee98ae178aae58641cd` / record SHA `7f0ae276d1c48ffcd7a320440f8ff73d6b3022c01c2edc40faf0d23f20910b66`; P4 `c277571f1e286468445b912f8ce377279f4f4056` / SHA `b93e87b1fe1f18ef6aea99619f319cf0d91ffe0fa3c1d6747826f8e7cd754a0c`, review `1ca6c8f4e33f82a3c89ec76c2897336006a9f751`; P6 `ea51dd8c84ab4e59bf056fae8c47facaee0c1fbd` / SHA `60cb8e6126cea8d307a96caef6b3099e5db5e43efe41170df0cb4e3237462a5e`, review `f97a9ed025b892e75669b95eec4785d15f05afff` / record SHA `8bee77e4ce40dbc3be0058f61f08f992f539c8ed2a57945802263ae0eecc9710`.
- **Superseded review:** `cb02ef8fd74ab4b65c4b0f86a1796eaad616e77d`, P5 review record SHA `bc2c958c977a57aa8513b74d8e9aa65e07e8aa0fb07ccaf1c01aca470c6dc14d`.

## Review scope and evidence

The successor resolves the prior review's status-summary finding. Its ledger now pins the current accepted P0–P6 package-plan heads and corresponding review records, clearly identifies prior revised amendments, and distinguishes accepted plans from pending implementation/API SOURCE/API gates. The previously stale P4 status is corrected: its package plan is accepted while its implementation/API candidate remains pending. P3's exact review record is pinned by full commit and record SHA.

P5's source-authoring prerequisites correctly require accepted P0a human decisions/candidate, a P1 API SOURCE/API ACCEPT, P2/P3/P4 candidate interface SOURCE/API ACCEPTs, and exact pinned heads. Package-plan acceptance is not represented as API acceptance. Pre-G0 candidate authoring remains isolated, finite-allowlist and static-review only; no candidate integration/runtime use occurs before G0, bridge, and final acceptance. G0 retains the manual-stop, independent dual-WX9100 lease-release, authorization, and post-G0 bridge requirements. The live campaign no-contact boundary remains explicit.

The P5 scope remains bounded to trainer/counterfactual caller integration, RNG, atomic update, and five checkpoint manifests. It preserves the adopted five-track V4 contract and Philox details, fixed horizon 160, 12-tick macro cap, gamma 0.993, current counterfactual defaults (`candidate_k=16`, `prediction_mode="cv"`, `max_c_candidates=8`, `max_a_pairs=8`), and two-WX9100/no-CPU-or-P600 execution requirement. It retains seeded end-to-end simulator/replay evidence and current-dashboard restart/provenance checks. No source or runtime behavior is claimed as implemented or validated.

## Authority limits

This ACCEPT applies only to the exact P5 plan commit/blob/hash above. It accepts the planning artifact and dependency ledger only. It does not accept implementation/API heads, P0a human decisions, G0, tests, runtime, device use, integration, training, or qualification. No tests, runtime, GPU, dashboard, simulator, or campaign actions were performed.
