# labeling_copilot

Pipeline alternatif untuk **event classification labeling** yang aman dipakai sebelum labeling skala besar, tanpa mengubah file lama.

## Yang diperbaiki dari pipeline lama
- Stable event ID berbasis hash fitur event (`stable_event_id`), bukan counter.
- Taxonomy label dipusatkan di satu file (`config/taxonomy.json`) + validasi strict.
- Split anti-leakage by `trip_id` (deterministik).
- QA report untuk kualitas label + quality report untuk timestamp/sampling raw.
- Run manifest untuk traceability (hash input + config version).

## Struktur
- `config/pipeline_config.json` - konfigurasi utama.
- `config/taxonomy.json` - kelas label dan alias.
- `run_labeling_copilot.py` - runner end-to-end.
- `ACTION_PLAN.md` - rencana implementasi yang dijalankan.

## Cara jalankan
Dari root repo:

```bash
python labeling_copilot/run_labeling_copilot.py --repo-root .
```

## Output
Semua output ditulis ke folder `output_copilot/`:
- `events_stable.csv`
- `ground_truth_stable.csv`
- `event_classification_dataset.csv`
- `trip_split_assignment.csv`
- `review_queue.csv`
- `raw_signal_quality_report.csv`
- `preprocessed_trip_summary.csv`
- `label_quality_report.json`
- `run_manifest.json`

Jika `preprocessing.enabled=true`, hasil resampling per trip juga ditulis ke:
- `output_copilot/preprocessed_raw/<trip_id>.csv`

## Catatan operasional
- Jika masih ada label di luar taxonomy, pipeline akan fail (mode strict).
- Tambahkan alias/kelas baru di `taxonomy.json` sebelum rerun.
