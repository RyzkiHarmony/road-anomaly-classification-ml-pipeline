import os
import re
import optuna
import optuna.visualization.matplotlib as ovm
import matplotlib.pyplot as plt

def main():
    # Setup paths
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_path = os.path.join(_PROJECT_ROOT, "xgboost_optuna.log")
    out_dir = os.path.join(_PROJECT_ROOT, "evaluation", "reports", "xgboost", "optuna")
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(log_path):
        print(f"Error: Log file {log_path} not found!")
        return
        
    print("Membaca dan mem-parsing log Optuna XGBoost...")
    
    # We will reconstruct the Optuna study by parsing the log file
    study = optuna.create_study(direction="maximize", study_name="Reconstructed_XGBoost_Study")
    
    # Regex to parse line like:
    # [I 2026-07-10 16:33:58,907] Trial 49 finished with value: 0.6286407791639692 and parameters: {'n_estimators': 292, 'max_depth': 5, ...}. Best is trial 33 with value: 0.64502779988039.
    # Note: Log might have line breaks depending on how it was captured. We'll join everything into one string first.
    with open(log_path, 'r', encoding='utf-16le') as f:
        content = f.read()
        
    # Standardize newlines and remove some wrapping artifacts
    content = content.replace('\n', ' ')
    
    trial_pattern = re.compile(
        r"Trial (\d+) finished with value: ([\d.]+) and parameters: (\{.*?\})\."
    )
    
    matches = trial_pattern.findall(content)
    if not matches:
        # Try UTF-8 fallback
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().replace('\n', ' ')
        matches = trial_pattern.findall(content)
        
    if not matches:
        print("Gagal menemukan riwayat trial dalam file log. Pastikan format log benar.")
        return
        
    for match in matches:
        trial_id = int(match[0])
        value = float(match[1])
        # safely evaluate the dictionary string
        params_str = match[2]
        # Replace python dict formatting with json friendly format if needed, or use eval
        try:
            params = eval(params_str)
            
            # Create a completed trial and add it to the study
            trial = optuna.trial.create_trial(
                params=params,
                distributions={k: optuna.distributions.FloatDistribution(-1000, 1000) if isinstance(v, float) else optuna.distributions.IntDistribution(0, 1000) for k,v in params.items()},
                value=value,
            )
            study.add_trial(trial)
        except Exception as e:
            print(f"Gagal memparsing trial {trial_id}: {e}")
            
    print(f"Berhasil merekonstruksi {len(study.trials)} trials ke dalam memory!")
    
    # Set matplotlib style to match the requested academic/clean style
    plt.style.use('default')
    plt.rcParams.update({
        'axes.grid': True,
        'grid.alpha': 0.7,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0,
        'font.size': 11
    })
    
    # 1. Optimization History (Custom Line Plot like the uploaded image)
    print("Mencetak plot Optimization History (Custom Style)...")
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    trials = sorted(trials, key=lambda t: t.number)
    
    trial_numbers = [t.number for t in trials]
    trial_values = [t.value for t in trials]
    
    # Calculate best value so far
    best_values = []
    current_best = float('-inf')
    for val in trial_values:
        if val > current_best:
            current_best = val
        best_values.append(current_best)
        
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(trial_numbers, trial_values, label='Trial Value (PR-AUC)', marker='.', linestyle='', alpha=0.5, color='#1f77b4')
    ax1.plot(trial_numbers, best_values, label='Best Value So Far', linewidth=2, color='#ff7f0e')
    
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Objective Value (PR-AUC)')
    ax1.set_title('Optimization History')
    ax1.legend(loc='lower right' if best_values[-1] > 0.5 else 'upper left')
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, "xgboost_optimization_history.png"), dpi=150)
    plt.close(fig1)
    
    # 2. Hyperparameter Importances (Custom Bar Plot)
    print("Mencetak plot Hyperparameter Importances (Custom Style)...")
    try:
        importances = optuna.importance.get_param_importances(study)
        params = list(importances.keys())
        vals = list(importances.values())
        
        # Sort for horizontal bar chart
        params.reverse()
        vals.reverse()
        
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        bars = ax2.barh(params, vals, color='#1f77b4')
        ax2.set_xlabel('Importance')
        ax2.set_title('Hyperparameter Importances')
        fig2.tight_layout()
        fig2.savefig(os.path.join(out_dir, "xgboost_param_importances.png"), dpi=150)
        plt.close(fig2)
    except Exception as e:
        print(f"Gagal mencetak param importances: {e}")
        
    # 3. Parallel Coordinate Plot (Optuna default, but with grid styling applied)
    print("Mencetak plot Parallel Coordinate...")
    fig3 = ovm.plot_parallel_coordinate(study)
    plt.tight_layout()
    fig3.figure.savefig(os.path.join(out_dir, "xgboost_parallel_coordinate.png"), dpi=150)
    plt.close(fig3.figure)
    
    # 4. Contour Plot (Heatmap 2D)
    print("Mencetak plot Contour (Heatmap 2D)...")
    try:
        importances = optuna.importance.get_param_importances(study)
        top_params = list(importances.keys())[:3]
        
        ovm.plot_contour(study, params=top_params)
        # We don't use tight_layout here as it conflicts with optuna contour subplots
        plt.gcf().savefig(os.path.join(out_dir, "xgboost_contour_plot.png"), dpi=150)
        plt.close(plt.gcf())
    except Exception as e:
        print(f"Gagal mencetak contour plot: {e}")
        
    # 5. Slice Plot
    print("Mencetak plot Slice...")
    try:
        ovm.plot_slice(study, params=top_params)
        # We don't use tight_layout here as it conflicts with optuna slice subplots
        plt.gcf().savefig(os.path.join(out_dir, "xgboost_slice_plot.png"), dpi=150)
        plt.close(plt.gcf())
    except Exception as e:
        print(f"Gagal mencetak slice plot: {e}")

    print(f"Semua grafik PNG telah berhasil disimpan di: {out_dir}")

if __name__ == "__main__":
    main()
