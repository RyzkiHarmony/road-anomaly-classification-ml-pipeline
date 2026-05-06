#!/usr/bin/env python3
"""
Empirical verification: Check if labeling.py output changes across runs.
"""
import os
import sys
import shutil
import subprocess
import hashlib
import csv
from pathlib import Path
from collections import defaultdict

def compute_sha256(filepath):
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def count_csv_rows(filepath):
    """Count rows in a CSV file (excluding header)."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            return sum(1 for _ in reader)
    except Exception as e:
        return f"Error: {e}"

def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _normalise_event_id(value):
    text = str(value).strip()
    try:
        # Normalize "12.0" -> "12" to avoid string mismatch artifacts.
        return str(int(float(text)))
    except (TypeError, ValueError):
        return text

def build_event_signature(row, time_decimals=3, latlon_decimals=5, peak_mag_decimals=1):
    """
    Build semantic event signature:
    trip_id + rounded_time + rounded_lat/lon + rounded_peak_mag.
    """
    trip_id = str(row.get("trip_id", "")).strip()
    time_s = _to_float(row.get("time_s"))
    lat = _to_float(row.get("lat"))
    lon = _to_float(row.get("lon"))
    peak_mag = _to_float(row.get("peak_mag"))

    if not trip_id or time_s is None or lat is None or lon is None or peak_mag is None:
        return None

    return (
        f"{trip_id}|"
        f"{round(time_s, time_decimals):.{time_decimals}f}|"
        f"{round(lat, latlon_decimals):.{latlon_decimals}f}|"
        f"{round(lon, latlon_decimals):.{latlon_decimals}f}|"
        f"{round(peak_mag, peak_mag_decimals):.{peak_mag_decimals}f}"
    )

def load_event_mappings(candidates_csv_path):
    """
    Load event_id <-> event_signature mappings from candidates_events.csv.
    Returns a dict with mapping and quality stats.
    """
    id_to_sig = {}
    sig_to_id = {}
    invalid_rows = 0
    duplicate_event_id_conflicts = 0
    duplicate_signature_conflicts = 0
    total_rows = 0

    if not os.path.exists(candidates_csv_path):
        return {
            "exists": False,
            "id_to_sig": id_to_sig,
            "sig_to_id": sig_to_id,
            "total_rows": 0,
            "invalid_rows": 0,
            "duplicate_event_id_conflicts": 0,
            "duplicate_signature_conflicts": 0,
        }

    with open(candidates_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"event_id", "trip_id", "time_s", "lat", "lon", "peak_mag"}
        if not required_cols.issubset(set(reader.fieldnames or [])):
            return {
                "exists": True,
                "id_to_sig": id_to_sig,
                "sig_to_id": sig_to_id,
                "total_rows": 0,
                "invalid_rows": -1,  # sentinel: missing required columns
                "duplicate_event_id_conflicts": 0,
                "duplicate_signature_conflicts": 0,
            }

        for row in reader:
            total_rows += 1
            event_id = _normalise_event_id(row.get("event_id"))
            signature = build_event_signature(row)
            if not event_id or signature is None:
                invalid_rows += 1
                continue

            if event_id in id_to_sig and id_to_sig[event_id] != signature:
                duplicate_event_id_conflicts += 1
            else:
                id_to_sig[event_id] = signature

            if signature in sig_to_id and sig_to_id[signature] != event_id:
                duplicate_signature_conflicts += 1
            else:
                sig_to_id[signature] = event_id

    return {
        "exists": True,
        "id_to_sig": id_to_sig,
        "sig_to_id": sig_to_id,
        "total_rows": total_rows,
        "invalid_rows": invalid_rows,
        "duplicate_event_id_conflicts": duplicate_event_id_conflicts,
        "duplicate_signature_conflicts": duplicate_signature_conflicts,
    }

def compare_mappings(base_map, other_map):
    """
    Compare run1 mapping against another run.
    """
    base_id_to_sig = base_map["id_to_sig"]
    other_id_to_sig = other_map["id_to_sig"]
    base_sig_to_id = base_map["sig_to_id"]
    other_sig_to_id = other_map["sig_to_id"]

    base_ids = set(base_id_to_sig.keys())
    other_ids = set(other_id_to_sig.keys())
    common_ids = base_ids & other_ids

    id_to_sig_mismatch = sum(
        1 for event_id in common_ids if base_id_to_sig[event_id] != other_id_to_sig[event_id]
    )
    id_missing_in_other = len(base_ids - other_ids)
    id_new_in_other = len(other_ids - base_ids)

    base_sigs = set(base_sig_to_id.keys())
    other_sigs = set(other_sig_to_id.keys())
    common_sigs = base_sigs & other_sigs

    sig_to_id_mismatch = sum(
        1 for signature in common_sigs if base_sig_to_id[signature] != other_sig_to_id[signature]
    )
    sig_missing_in_other = len(base_sigs - other_sigs)
    sig_new_in_other = len(other_sigs - base_sigs)

    return {
        "common_ids": len(common_ids),
        "id_to_sig_mismatch": id_to_sig_mismatch,
        "id_missing_in_other": id_missing_in_other,
        "id_new_in_other": id_new_in_other,
        "common_signatures": len(common_sigs),
        "sig_to_id_mismatch": sig_to_id_mismatch,
        "sig_missing_in_other": sig_missing_in_other,
        "sig_new_in_other": sig_new_in_other,
        "is_consistent": (
            id_to_sig_mismatch == 0
            and id_missing_in_other == 0
            and id_new_in_other == 0
            and sig_to_id_mismatch == 0
            and sig_missing_in_other == 0
            and sig_new_in_other == 0
            and other_map["invalid_rows"] in (0,)
            and other_map["duplicate_event_id_conflicts"] == 0
            and other_map["duplicate_signature_conflicts"] == 0
        ),
    }

def main():
    repo_root = r"D:\VSCode Data\Road Detection - Project Skripsi\ml_pipelines"
    os.chdir(repo_root)
    print(f"Working directory: {os.getcwd()}\n")
    
    # Setup temp directory
    tmp_dir = ".tmp_labeling_compare"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
        print(f"Deleted existing {tmp_dir}")
    os.makedirs(tmp_dir)
    print(f"Created fresh {tmp_dir}\n")
    
    files_to_copy = [
        "candidates_events.csv",
        "windows_features.csv",
        "labeling_high_conf.csv",
        "labeling_candidate.csv",
        "labeling_normal.csv"
    ]
    
    hashes = defaultdict(dict)  # file -> run -> hash
    row_counts = defaultdict(dict)  # file -> run -> row_count
    file_sizes = defaultdict(dict)  # file -> run -> size
    
    # Run 5 times
    for run in range(1, 6):
        print(f"\n{'='*60}")
        print(f"RUN {run}")
        print(f"{'='*60}")
        
        # Execute labeling script
        print("Executing: python labeling\\labeling.py")
        result = subprocess.run([sys.executable, "labeling\\labeling.py"], 
                              capture_output=True, text=True)
        
        # Show last few lines of output
        output_lines = result.stdout.split('\n')
        print('\n'.join(output_lines[-5:]))
        
        if result.returncode != 0:
            print(f"WARNING: Script exited with code {result.returncode}")
            if result.stderr:
                print(f"STDERR: {result.stderr[:500]}")
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        labeling_dir = os.path.join(base_dir, "01_labeling_pipeline")
        copilot_dir = os.path.join(base_dir, "02_dataset_pipeline")
        
        # Create run directory
        run_dir = os.path.join(tmp_dir, f"run{run}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Copy and hash files
        print(f"\nCopying files to {run_dir}:")
        for filename in files_to_copy:
            src_path = os.path.join("labeling", "out", filename)
            if os.path.exists(src_path):
                dest_path = os.path.join(run_dir, filename)
                shutil.copy2(src_path, dest_path)
                
                # Compute hash and row count
                file_hash = compute_sha256(dest_path)
                file_size = os.path.getsize(dest_path)
                row_count = count_csv_rows(dest_path)
                
                hashes[filename][run] = file_hash
                file_sizes[filename][run] = file_size
                row_counts[filename][run] = row_count
                
                print(f"  ✓ {filename:<30} | Size: {file_size:>10} | Rows: {row_count:>6} | Hash: {file_hash[:16]}...")
            else:
                print(f"  ✗ {filename:<30} | NOT FOUND")
        
        print(f"Run {run} complete\n")
    
    # Analysis and comparison
    print(f"\n{'='*80}")
    print("ANALYSIS: RUN1 vs RUN2-5")
    print(f"{'='*80}\n")
    
    comparison_results = {}
    
    for filename in files_to_copy:
        print(f"\n{filename}:")
        print(f"{'-'*70}")
        
        run1_hash = hashes[filename].get(1)
        if not run1_hash:
            print("  SKIPPED (not found in run1)")
            comparison_results[filename] = "SKIPPED"
            continue
        
        all_identical = True
        changes = []
        
        for run in range(2, 6):
            run_hash = hashes[filename].get(run)
            if not run_hash:
                print(f"  Run {run}: NOT FOUND")
                all_identical = False
                continue
            
            is_same = run_hash == run1_hash
            status = "✓ IDENTICAL" if is_same else "✗ CHANGED"
            
            if not is_same:
                all_identical = False
                size_run1 = file_sizes[filename].get(1, 0)
                size_run = file_sizes[filename].get(run, 0)
                rows_run1 = row_counts[filename].get(1, 0)
                rows_run = row_counts[filename].get(run, 0)
                
                change_info = f"Run{run}: Size {size_run1}→{size_run} | Rows {rows_run1}→{rows_run}"
                changes.append(change_info)
                print(f"  Run {run}: {status} | {change_info}")
            else:
                print(f"  Run {run}: {status}")
        
        conclusion = "STABLE" if all_identical else "UNSTABLE"
        comparison_results[filename] = conclusion
        print(f"  CONCLUSION: {conclusion}")
    
    # Summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}\n")
    
    print(f"{'File':<35} | {'Status':<10} | {'Hashes (Run1..5)':<50}")
    print(f"{'-'*80}")
    
    for filename in files_to_copy:
        status = comparison_results.get(filename, "UNKNOWN")
        hash_str = ""
        for run in range(1, 6):
            h = hashes[filename].get(run, "N/A")
            hash_str += h[:8] if h != "N/A" else "N/A "
            if run < 5:
                hash_str += " | "
        
        print(f"{filename:<35} | {status:<10} | {hash_str}")

    # Semantic consistency check for event_id <-> event_signature
    print(f"\n\n{'='*80}")
    print("SEMANTIC CONSISTENCY CHECK (event_id <-> event_signature)")
    print(f"{'='*80}\n")

    run_mappings = {}
    for run in range(1, 6):
        candidates_path = os.path.join(tmp_dir, f"run{run}", "candidates_events.csv")
        run_mappings[run] = load_event_mappings(candidates_path)

    run1_map = run_mappings[1]
    semantic_results = {}
    semantic_blocked = False

    if not run1_map["exists"]:
        semantic_blocked = True
        print("Run1 candidates_events.csv tidak ditemukan. Semantic check dilewati.")
    elif run1_map["invalid_rows"] == -1:
        semantic_blocked = True
        print("Run1 candidates_events.csv tidak punya kolom wajib semantic check.")
        print("Wajib: event_id, trip_id, time_s, lat, lon, peak_mag")
    else:
        print("Run1 mapping quality:")
        print(f"  total_rows                    : {run1_map['total_rows']}")
        print(f"  invalid_rows                  : {run1_map['invalid_rows']}")
        print(f"  duplicate_event_id_conflicts  : {run1_map['duplicate_event_id_conflicts']}")
        print(f"  duplicate_signature_conflicts : {run1_map['duplicate_signature_conflicts']}")
        print(f"  unique_event_ids              : {len(run1_map['id_to_sig'])}")
        print(f"  unique_signatures             : {len(run1_map['sig_to_id'])}")

        if (
            run1_map["invalid_rows"] != 0
            or run1_map["duplicate_event_id_conflicts"] != 0
            or run1_map["duplicate_signature_conflicts"] != 0
        ):
            print("\nWARNING: Run1 sendiri tidak clean. Konsistensi lintas-run tetap dihitung,")
            print("namun kualitas mapping dasar sudah bermasalah.")

        print(f"\n{'Run':<6} | {'Consistency':<12} | {'id->sig mismatch':<16} | {'sig->id mismatch':<16} | {'id missing/new':<14} | {'sig missing/new':<14}")
        print(f"{'-'*95}")

        for run in range(2, 6):
            other_map = run_mappings[run]
            if not other_map["exists"] or other_map["invalid_rows"] == -1:
                semantic_results[run] = {"is_consistent": False, "reason": "missing_or_invalid_columns"}
                print(f"{run:<6} | {'BLOCKED':<12} | {'N/A':<16} | {'N/A':<16} | {'N/A':<14} | {'N/A':<14}")
                continue

            comp = compare_mappings(run1_map, other_map)
            semantic_results[run] = comp
            consistency = "CONSISTENT" if comp["is_consistent"] else "INCONSISTENT"
            id_missing_new = f"{comp['id_missing_in_other']}/{comp['id_new_in_other']}"
            sig_missing_new = f"{comp['sig_missing_in_other']}/{comp['sig_new_in_other']}"
            print(
                f"{run:<6} | {consistency:<12} | {comp['id_to_sig_mismatch']:<16} | "
                f"{comp['sig_to_id_mismatch']:<16} | {id_missing_new:<14} | {sig_missing_new:<14}"
            )

            if (
                other_map["invalid_rows"] != 0
                or other_map["duplicate_event_id_conflicts"] != 0
                or other_map["duplicate_signature_conflicts"] != 0
            ):
                print(
                    f"       NOTE run{run}: invalid_rows={other_map['invalid_rows']}, "
                    f"dup_event_id_conflicts={other_map['duplicate_event_id_conflicts']}, "
                    f"dup_signature_conflicts={other_map['duplicate_signature_conflicts']}"
                )

    # Final summary
    print(f"\n\n{'='*80}")
    print("EMPIRICAL VERIFICATION RESULT")
    print(f"{'='*80}\n")
    
    stable_count = sum(1 for v in comparison_results.values() if v == "STABLE")
    unstable_count = sum(1 for v in comparison_results.values() if v == "UNSTABLE")
    
    print(f"Files analyzed: {len(comparison_results)}")
    print(f"Stable (identical across runs): {stable_count}")
    print(f"Unstable (changed across runs): {unstable_count}")
    
    if unstable_count > 0:
        print(f"\n⚠️  OUTPUT IS NON-DETERMINISTIC")
        print(f"   The following files change across runs:")
        for filename, status in comparison_results.items():
            if status == "UNSTABLE":
                print(f"   - {filename}")
    else:
        print(f"\n✓ OUTPUT IS DETERMINISTIC")
        print(f"  All files produce identical output across all runs.")

    if semantic_blocked:
        print("\n⚠️  SEMANTIC CONSISTENCY: NOT EVALUATED")
    else:
        semantic_consistent_runs = sum(
            1 for run in range(2, 6) if semantic_results.get(run, {}).get("is_consistent")
        )
        if semantic_consistent_runs == 4:
            print("✓ SEMANTIC CONSISTENCY: PASS")
            print("  event_id <-> event_signature mapping konsisten pada run2..run5 terhadap run1.")
        else:
            print("⚠️  SEMANTIC CONSISTENCY: FAIL")
            print("  Ada run dengan mismatch event_id <-> event_signature atau missing/new mapping.")
    
    print(f"\nTemp folder preserved at: {os.path.join(os.getcwd(), tmp_dir)}")
    print("You can inspect individual runs at: .tmp_labeling_compare/runN/")

if __name__ == "__main__":
    main()
