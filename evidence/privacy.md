# Public artifact privacy

The public source, website, both model packages, and release archives were inspected for personal contact information, private filesystem paths, geographic/timezone metadata, exact experiment dates, credentials, and unrelated account/runtime details.

The public copies omit personal commit email, original local timezone, exact run/deadline timestamps, account balances, and unrelated installed-package inventory. The source history is republished as a project snapshot using a GitHub no-reply identity. Git and ZIP timestamps are fixed packaging metadata, not experiment dates. Original records and history are preserved privately.

Model weights, frozen datasets, raw predictions, scores, scientific runtime durations, hardware descriptions, and upstream source revisions are preserved. Source hashes in original model provenance identify the training-era sources; current source changes remove historical deadlines and add privacy-aware packaging, without retraining or changing the learned weights.

The publication preparation path sanitizes provenance before generating its integrity manifest. Release archives use normalized file metadata. Two focused privacy tests cover metadata removal, preservation of scientific bytes, deterministic archives, and backup integrity; the existing publication boundary tests also pass.

Public account aliases and repository addresses remain necessary for access. Hosting providers generate their own publication times and account metadata. This cleanup cannot retract copies previously downloaded by third parties or guarantee removal from unrelated search-engine caches.
