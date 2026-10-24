# Three-pass summary validation: Implementer summary

Commit 962a2f1814eec2a4d32f998a4e009ac2962cacc3 pins the trusted Git executable used to validate manifest commit objects. Its tree includes the schema-v2 dashboard summary reader, plan and manifest hash binding, safe in-repository summary paths, summary hashes, required roles, exact implementation/reviewer commit matching, and rendering of summary-only evidence. A poisoned PATH cannot substitute a fake Git executable. No simulator, training, held-out outcome, or qualification evidence is claimed.
