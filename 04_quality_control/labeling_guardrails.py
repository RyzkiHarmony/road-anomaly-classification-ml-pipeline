import os
import pandas as pd
import numpy as np

def load_data(candidates_path, ground_truth_path=None):
    """
    Memuat data kandidat event dan (opsional) label ground truth.
    Jika ground truth ada di file terpisah, akan digabung (merge) berdasarkan event_id.
    Jika file candidates sudah memiliki kolom 'label', maka ground_truth_path bisa diabaikan.
    """
    if not os.path.exists(candidates_path):
        raise FileNotFoundError(f"File kandidat tidak ditemukan: {candidates_path}")
        
    df = pd.read_csv(candidates_path)
    
    if ground_truth_path and os.path.exists(ground_truth_path):
        gt_df = pd.read_csv(ground_truth_path)
        # Asumsi ada kolom 'event_id' untuk merge
        if 'event_id' in gt_df.columns and 'event_id' in df.columns:
            df = df.merge(gt_df[['event_id', 'label']], on='event_id', how='left', suffixes=('', '_gt'))
            if 'label_gt' in df.columns:
                df['label'] = df['label_gt']
                df = df.drop(columns=['label_gt'])
        else:
            print("Peringatan: 'event_id' tidak ditemukan. Gagal menggabungkan ground truth.")
    
    # Pastikan ada kolom label
    if 'label' not in df.columns:
        print("Peringatan: Kolom 'label' tidak ditemukan di data. Guardrail akan diabaikan untuk baris tanpa label.")
        df['label'] = 'Unknown'
        
    return df

def _check_pothole(row):
    """
    Guardrail untuk label 'Pothole'.
    Rule: Jika gyro sangat dominan, asymmetry sangat tinggi, num_peaks_gyro tinggi, 
    dan pola temporal cocok dengan maneuver -> REVIEW
    """
    reasons = []
    
    # Toleransi fallback jika kolom tidak ada
    accel_to_gyro = row.get('accel_to_gyro_ratio', 10.0)
    asym_score = row.get('asymmetry_score', 0.0)
    peaks_gyro = row.get('num_peaks_gyro', 0)
    
    # Logika konservatif
    is_gyro_dominant = (pd.notna(accel_to_gyro) and accel_to_gyro < 0.5)
    is_asym_high = (pd.notna(asym_score) and asym_score > 0.8)
    is_peaks_gyro_high = (pd.notna(peaks_gyro) and peaks_gyro >= 3)
    
    if is_gyro_dominant and is_asym_high and is_peaks_gyro_high:
        reasons.append("Gyro dominan, asimetri tinggi, dan banyak peak gyro (karakteristik Maneuver)")
        
    if reasons:
        return 'REVIEW', " | ".join(reasons)
    return 'PASS', ""

def _check_non_event(row):
    """
    Guardrail untuk label 'Non-Event'.
    Flags events labeled as Non-Event that show strong anomaly characteristics,
    suggesting they might actually be Pothole or Speed Bump.
    """
    reasons = []

    peak_accel = row.get('peak_mag_g', 0.0)
    peaks_accel_count = row.get('num_peaks_accel', 0)
    peak_gyro = row.get('peak_gyro_mag', 0.0)
    duration = row.get('local_duration', 1.0)
    accel_to_gyro = row.get('accel_to_gyro_ratio', 0.0)

    # Cek 1: Accel sangat tinggi + multi-peak + gyro aktif → kemungkinan anomali nyata
    is_high_accel = (pd.notna(peak_accel) and peak_accel > 1.5)
    is_multi_peak = (pd.notna(peaks_accel_count) and peaks_accel_count >= 2)
    is_gyro_active = (pd.notna(peak_gyro) and peak_gyro > 50.0)

    if is_high_accel and is_multi_peak and is_gyro_active:
        reasons.append("Accel sangat tinggi, multi-peak, dan gyro aktif (indikasi kuat ada anomali)")

    # Cek 2: Accel dominan + durasi pendek → kemungkinan pothole yang terlewat
    is_accel_dominant = (pd.notna(accel_to_gyro) and accel_to_gyro > 2.0)
    is_duration_short = (pd.notna(duration) and duration < 0.6)

    if is_accel_dominant and is_multi_peak and is_duration_short:
        reasons.append("Accel dominan, multi-peak, dan durasi pendek (kemungkinan Pothole)")

    if reasons:
        return 'REVIEW', " | ".join(reasons)
    return 'PASS', ""

def _check_speed_bump(row):
    """
    Guardrail untuk label 'Speed Bump'.
    Rule: Jika hanya satu peak dominan dan durasi terlalu pendek -> WARN/REVIEW
    """
    reasons = []

    peaks_accel = row.get('num_peaks_accel', 2)
    duration = row.get('local_duration', 1.0)

    is_single_peak = (pd.notna(peaks_accel) and peaks_accel <= 1)
    is_duration_short = (pd.notna(duration) and duration < 0.3)

    if is_single_peak and is_duration_short:
        reasons.append("Hanya 1 peak dan durasi sangat pendek (mungkin noise atau pothole kecil)")
        return 'WARN', " | ".join(reasons)

    return 'PASS', ""

def evaluate_guardrails(df):
    """
    Mengevaluasi label manual berdasarkan rule guardrail untuk menjaga konsistensi.
    Tidak mengubah label asli, hanya menambahkan flag dan alasan.
    """
    results = []
    
    for idx, row in df.iterrows():
        label = str(row.get('label', '')).strip().title()
        
        status = 'PASS'
        reason = ''
        
        if label == 'Pothole':
            status, reason = _check_pothole(row)
        elif label == 'Speed Bump':
            status, reason = _check_speed_bump(row)
        elif label in ('Non-Event', 'Non-event'):
            status, reason = _check_non_event(row)
            
        results.append({'guardrail_status': status, 'guardrail_reason': reason})
        
    res_df = pd.DataFrame(results, index=df.index)
    return pd.concat([df, res_df], axis=1)

def summarize_results(df):
    """
    Menghitung jumlah event berdasarkan status guardrail.
    """
    summary = df['guardrail_status'].value_counts().to_dict()
    # Pastikan key yang standar ada
    for k in ['PASS', 'WARN', 'REVIEW']:
        if k not in summary:
            summary[k] = 0
            
    return summary

def save_report(df, output_report_path, output_summary_path):
    """
    Menyimpan laporan evaluasi dan ringkasan ke file terpisah.
    """
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_summary_path), exist_ok=True)
    
    # Pilih kolom penting untuk disave agar tidak terlalu besar
    # Sesuaikan dengan kolom minimal yang diminta
    cols_to_keep = ['event_id', 'trip_id', 'label', 'guardrail_status', 'guardrail_reason', 'score', 'priority']
    
    # Tambahkan fitur-fitur penting yang dipakai di rule jika tersedia di dataset
    feature_cols = [
        'peak_mag_g', 'peak_gyro_mag', 'speed_mean', 'num_peaks_accel', 'num_peaks_gyro',
        'asymmetry_score', 'accel_to_gyro_ratio', 'local_duration'
    ]
    
    available_cols = [c for c in cols_to_keep + feature_cols if c in df.columns]
    
    # Hanya simpan baris yang bukan PASS ke file flags terpisah (opsional, tapi berguna)
    flags_df = df[df['guardrail_status'].isin(['WARN', 'REVIEW'])]
    
    # Save CSV penuh
    df[available_cols].to_csv(output_report_path, index=False)
    
    # Save CSV khusus flag
    flags_path = output_report_path.replace('_report.csv', '_flags.csv')
    flags_df[available_cols].to_csv(flags_path, index=False)
    
    # Generate Markdown Summary
    summary = summarize_results(df)
    
    md_content = f"# Labeling Guardrail Summary\n\n"
    md_content += f"**Total Events Evaluated**: {len(df)}\n\n"
    md_content += f"## Status Counts\n"
    md_content += f"- **PASS**: {summary.get('PASS', 0)}\n"
    md_content += f"- **WARN**: {summary.get('WARN', 0)}\n"
    md_content += f"- **REVIEW**: {summary.get('REVIEW', 0)}\n\n"
    
    if len(flags_df) > 0:
        md_content += f"## Top Flags to Review\n"
        md_content += "Beberapa event yang perlu dicek ulang:\n"
        
        # Ambil sampel 10 teratas berdasarkan priority/score jika ada, atau random
        sort_cols = [c for c in ['priority', 'score'] if c in flags_df.columns]
        if sort_cols:
            sample_flags = flags_df.sort_values(by=sort_cols, ascending=False).head(10)
        else:
            sample_flags = flags_df.head(10)
            
        md_content += "| Event ID | Label | Status | Reason |\n"
        md_content += "|----------|-------|--------|--------|\n"
        for _, row in sample_flags.iterrows():
            eid = row.get('event_id', 'N/A')
            lbl = row.get('label', 'N/A')
            st = row.get('guardrail_status', 'N/A')
            rsn = row.get('guardrail_reason', 'N/A')
            md_content += f"| {eid} | {lbl} | {st} | {rsn} |\n"
            
    with open(output_summary_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
        
    print(f"[OK] Laporan Guardrail berhasil disimpan ke: {output_report_path}")
    print(f"[OK] Ringkasan Markdown disimpan ke: {output_summary_path}")
    print(f"[OK] Event yang di-flag (WARN/REVIEW) disimpan ke: {flags_path}")
    print(f"[INFO] Hasil: PASS={summary.get('PASS', 0)}, WARN={summary.get('WARN', 0)}, REVIEW={summary.get('REVIEW', 0)}")

def run_guardrails(candidates_csv, ground_truth_csv=None, output_dir="out"):
    """
    Fungsi utama untuk menjalankan pipeline guardrail.
    """
    print("Memuat data...")
    try:
        df = load_data(candidates_csv, ground_truth_csv)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
        
    print("Mengevaluasi guardrails...")
    df_evaluated = evaluate_guardrails(df)
    
    report_csv = os.path.join(output_dir, "labeling_guardrail_report.csv")
    summary_md = os.path.join(output_dir, "labeling_guardrail_summary.md")
    
    print("Menyimpan hasil...")
    save_report(df_evaluated, report_csv, summary_md)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Labeling Guardrails for Road Anomaly")
    parser.add_argument('--candidates', type=str, default='labeling/out/candidates_events.csv', help='Path to candidates_events.csv')
    parser.add_argument('--ground_truth', type=str, default='labeling/out/ground_truth_labels.csv', help='Path to ground_truth_labels.csv (optional if labels are in candidates)')
    parser.add_argument('--out_dir', type=str, default='04_quality_control/out', help='Output directory for reports')
    
    args = parser.parse_args()
    
    run_guardrails(args.candidates, args.ground_truth, args.out_dir)
