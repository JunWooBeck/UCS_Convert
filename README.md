CSV to UCS Converter

Converts dataset CSVs to the Universal Category System (UCS) format using rules defined in config.json.

Dependencies listed in requirements.txt

pip install -r requirements.txt

---Classification Logic---

1. Manual Mapping: Custom rules defined in config.json handle ambiguous tags (e.g., mapping "Domestic_sounds" -> "FOLEY"). 
2. Category Auto-Match: Direct string matching against UCS Category names.
3. SubCategory Auto-Match: Direct string matching against UCS SubCategory names.

---Configuration---

All settings are managed in config.json. 

Update the paths in this file before running.

dataset_sets: List of datasets to process.

input_csv: Path to source CSV.

audio_dir: Path to source audio folder.

output_classified_csv: Path for classified output.

output_unclassified_csv: Path for unclassified output.

output_unclassified_summary: Path for the failure report.

ucs_paths:

structure_csv: Path to the UCS v8.2.1 Full List.csv.

manual_mapping:

Defines custom rules to override auto-matching. This is the primary way to improve classification accuracy.

"manual_mapping": {
    "Gunshot_and_gunfire": {
        "Category": "WEAPONS",
        "SubCategory": "GUN"
    },
    "Speech": {
        "Category": "VOICE",
        "SubCategory": "SPEECH"
    }
}


---Run Conversion---

Generates _classified.csv, _unclassified.csv, and _unclassified_summary.csv for each set defined in config.json.

python UCS_Convert.py


Test Configuration (Dry Run)

Tests all settings and paths without writing any files.

python UCS_Convert.py --dry-run


Validate Existing Output

Checks already-generated _classified.csv files for invalid paths or unclassified rows.

python UCS_Convert.py --validate

