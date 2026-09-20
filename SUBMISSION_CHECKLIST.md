# Curio Next Submission Checklist

- [x] Problem formulation
- [x] Synthetic dataset curated
- [x] Structured audio + QA format
- [x] Multiple question types: what, what-sound, presence, counting, temporal, why
- [x] Train/validation/test split by clip
- [x] Dataset construction methodology documented
- [x] Reproducible training script
- [x] Reproducible evaluation script
- [x] Audio event model
- [x] Environment model
- [x] Temporal context layer
- [x] Natural-language question routing
- [x] Quantitative test metrics
- [x] Breakdown by question type
- [x] Error-analysis output
- [x] CNN training loss curve
- [x] Technical report
- [x] CLI demo
- [x] Flask API endpoint

## Important caveat to mention during review

The reported 100% synthetic test performance is intentionally not presented as real-world performance. The benchmark is small and generated from a controlled acoustic vocabulary. The next research step is evaluation on real recordings and integration of a pretrained audio-language model such as CLAP/AST.
