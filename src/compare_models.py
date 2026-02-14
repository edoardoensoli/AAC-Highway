"""
Script per confrontare le performance di modelli RL addestrati su highway-env.
Confronta modelli su episodi deterministici per garantire confronti equi.

Uso:
    python compare_models.py --models dqn ppo --episodes 10 --render --seed 42
    python compare_models.py --models dqn --episodes 100 --no-render
"""

import argparse
import gymnasium
import highway_env
import torch
from stable_baselines3 import DQN, PPO
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any

from metrics_tracker import HighwayMetrics


# Mappa dei modelli disponibili
MODEL_REGISTRY = {
    'dqn': {
        'class': DQN,
        'path': 'highway_dqn/model.zip',
        'name': 'DQN',
        'color': 'blue'
    },
    'ppo': {
        'class': PPO,
        'path': 'highway_ppo/v1/model_1000k.zip',
        'name': 'PPO',
        'color': 'green'
    },
    'dqn_plr': {
        'class': DQN,
        'path': '../plr_dqn/plr/runs/dqn_v2/model_final.pt',
        'name': 'DQN_PLR',
        'color': 'green'
    }
    
}

# Configurazione environment standard
ENV_CONFIG = {
     "lanes_count": 3,
    "vehicles_count": 15,
    "vehicles_density": 0.8,
    "duration": 60,                    # 60 secondi per episodio
    "simulation_frequency": 30,
    "other_vehicles_type": "aggressive_vehicle.AggressiveIDMVehicle",
}


def load_model(model_key: str, device: str = 'auto'):
    """
    Carica un modello dal registry.
    
    Args:
        model_key: Chiave del modello nel registry (es: 'dqn', 'ppo')
        device: Device per il modello ('cuda', 'mps', 'cpu', 'auto')
    
    Returns:
        Modello caricato
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Modello '{model_key}' non trovato. Disponibili: {list(MODEL_REGISTRY.keys())}")
    
    config = MODEL_REGISTRY[model_key]
    model_path = Path(__file__).parent / config['path']
    
    if not model_path.exists():
        raise FileNotFoundError(f"Modello non trovato: {model_path}")
    
    print(f"Caricamento {config['name']} da: {model_path}")
    model = config['class'].load(str(model_path), device=device)
    
    return model


def evaluate_model_with_render(
    model,
    model_name: str,
    env,
    n_episodes: int,
    seed: int,
    metrics: HighwayMetrics,
    ax=None,
    img_obj=None
) -> Dict[str, float]:
    """
    Valuta un modello con rendering visivo.
    
    Args:
        model: Modello da valutare
        model_name: Nome del modello (per display)
        env: Environment gymnasium
        n_episodes: Numero di episodi
        seed: Seed per determinismo
        metrics: Tracker delle metriche
        ax: Axes matplotlib (opzionale, per rendering parallelo)
        img_obj: Oggetto immagine matplotlib (opzionale)
    
    Returns:
        Dizionario con le metriche calcolate
    """
    for ep_num in range(n_episodes):
        obs, info = env.reset(seed=seed + ep_num)
        metrics.start_episode(env)
        
        done = truncated = False
        step = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            metrics.step(env, action, reward, done, truncated, info)
            
            # Rendering
            if ax is not None and img_obj is not None:
                frame = env.render()
                img_obj.set_data(frame)
                ax.set_title(f'{model_name} | Ep {ep_num+1}/{n_episodes} | Step {step} | R: {reward:.2f}',
                            fontsize=10, fontweight='bold')
                plt.pause(0.01)
            
            step += 1
        
        metrics.end_episode()
        print(f"  {model_name} - Episodio {ep_num + 1}/{n_episodes} completato")
    
    return metrics.compute()


def evaluate_model_no_render(
    model,
    model_name: str,
    env,
    n_episodes: int,
    seed: int,
    metrics: HighwayMetrics
) -> Dict[str, float]:
    """
    Valuta un modello senza rendering (più veloce).
    
    Args:
        model: Modello da valutare
        model_name: Nome del modello (per display)
        env: Environment gymnasium
        n_episodes: Numero di episodi
        seed: Seed per determinismo
        metrics: Tracker delle metriche
    
    Returns:
        Dizionario con le metriche calcolate
    """
    for ep_num in range(n_episodes):
        obs, info = env.reset(seed=seed + ep_num)
        metrics.start_episode(env)
        
        done = truncated = False
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            metrics.step(env, action, reward, done, truncated, info)
        
        metrics.end_episode()
        
        # Progress report ogni 10%
        if (ep_num + 1) % max(1, n_episodes // 10) == 0:
            print(f"  {model_name} - Progresso: {ep_num + 1}/{n_episodes} episodi")
    
    return metrics.compute()


def print_comparison_table(results: Dict[str, Dict[str, float]], models: List[str]):
    """
    Stampa una tabella di confronto tra i modelli.
    
    Args:
        results: Dizionario {model_key: {metric: value}}
        models: Lista dei modelli confrontati
    """
    print("\n" + "="*100)
    print("CONFRONTO MODELLI")
    print("="*100)
    
    # Ottieni tutte le metriche disponibili
    all_metrics = set()
    for model_results in results.values():
        all_metrics.update(model_results.keys())
    
    # Ordine preferito
    metric_order = [
        'survival_rate', 'collision_rate', 'avg_reward',
        'cars_overtaken', 'total_cars_overtaken',
        'avg_speed', 'max_speed', 'distance_traveled',
        'avg_episode_length', 'lane_changes',
        'min_ttc', 'near_miss_rate'
    ]
    
    # Filtra solo le metriche presenti
    metrics_to_show = [m for m in metric_order if m in all_metrics]
    
    # Header
    header = f"{'Metrica':<30}"
    for model_key in models:
        model_name = MODEL_REGISTRY[model_key]['name']
        header += f"{model_name:>15}"
    header += f"{'Migliore':>15}"
    print(header)
    print("-"*100)
    
    # Righe per ogni metrica
    for metric in metrics_to_show:
        label = metric.replace('_', ' ').title()
        row = f"{label:<30}"
        
        values = {}
        for model_key in models:
            if metric in results[model_key]:
                value = results[model_key][metric]
                values[model_key] = value
                
                # Formattazione
                if value == float('inf') or value is None:
                    row += f"{'N/A':>15}"
                elif 'rate' in metric:
                    row += f"{value:>14.1f}%"
                elif 'speed' in metric or metric == 'distance_traveled':
                    row += f"{value:>14.1f}"
                elif 'total' in metric:
                    row += f"{int(value):>15}"
                else:
                    row += f"{value:>15.2f}"
            else:
                row += f"{'N/A':>15}"
        
        # Determina il migliore (dipende dalla metrica)
        if values:
            if metric in ['collision_rate', 'near_miss_rate']:
                # Più basso è meglio
                best = min(values, key=values.get)
            elif metric == 'min_ttc':
                # Più alto è meglio (più tempo prima della collisione)
                valid = {k: v for k, v in values.items() if v != float('inf')}
                best = max(valid, key=valid.get) if valid else None
            else:
                # Più alto è meglio
                best = max(values, key=values.get)
            
            if best:
                best_name = MODEL_REGISTRY[best]['name']
                row += f"{best_name:>15}"
            else:
                row += f"{'N/A':>15}"
        
        print(row)
    
    print("="*100 + "\n")


def plot_comparison_charts(results: Dict[str, Dict[str, float]], models: List[str], output_path: Path):
    """
    Crea grafici di confronto tra i modelli.
    
    Args:
        results: Dizionario {model_key: {metric: value}}
        models: Lista dei modelli confrontati
        output_path: Path dove salvare i grafici
    """
    # Metriche chiave da visualizzare
    key_metrics = [
        ('survival_rate', 'Survival Rate (%)', False),
        ('avg_reward', 'Average Reward', False),
        ('cars_overtaken', 'Cars Overtaken (avg)', False),
        ('avg_speed', 'Average Speed (m/s)', False),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    fig.suptitle('Confronto Performance Modelli', fontsize=16, fontweight='bold')
    
    for idx, (metric, title, invert) in enumerate(key_metrics):
        ax = axes[idx]
        
        # Raccogli dati
        model_names = []
        values = []
        colors = []
        
        for model_key in models:
            if metric in results[model_key]:
                value = results[model_key][metric]
                if value != float('inf') and value is not None:
                    model_names.append(MODEL_REGISTRY[model_key]['name'])
                    values.append(value)
                    colors.append(MODEL_REGISTRY[model_key]['color'])
        
        if values:
            bars = ax.bar(model_names, values, color=colors, alpha=0.7, edgecolor='black')
            ax.set_title(title, fontweight='bold')
            ax.set_ylabel(title)
            ax.grid(axis='y', alpha=0.3)
            
            # Aggiungi valori sopra le barre
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}',
                       ha='center', va='bottom', fontweight='bold')
            
            # Evidenzia il migliore
            if invert:
                best_idx = values.index(min(values))
            else:
                best_idx = values.index(max(values))
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Grafici salvati in: {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Confronta performance di modelli RL su highway-env',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Confronta DQN e PPO su 10 episodi con rendering
  python compare_models.py --models dqn ppo --episodes 10 --render
  
  # Confronta solo DQN su 100 episodi senza rendering
  python compare_models.py --models dqn --episodes 100 --no-render
  
  # Usa seed specifico
  python compare_models.py --models dqn ppo --episodes 50 --seed 12345
        """
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help=f'Modelli da confrontare. Disponibili: {list(MODEL_REGISTRY.keys())}'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help='Numero di episodi per valutazione (default: 10)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed per determinismo (default: 42)'
    )
    
    parser.add_argument(
        '--render',
        action='store_true',
        help='Mostra visualizzazione in tempo reale'
    )
    
    parser.add_argument(
        '--no-render',
        action='store_true',
        help='Disabilita visualizzazione (più veloce)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Mostra informazioni dettagliate durante la valutazione'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'mps', 'cpu'],
        help='Device per i modelli (default: auto)'
    )
    
    args = parser.parse_args()
    
    # Determina rendering
    if args.no_render:
        render = False
    elif args.render:
        render = True
    else:
        # Default: render se pochi episodi
        render = args.episodes <= 10
    
    # Device detection
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = "cuda"
            print("Device: CUDA GPU")
        elif torch.backends.mps.is_available():
            device = "mps"
            print("Device: Apple Silicon MPS")
        else:
            device = "cpu"
            print("Device: CPU")
    else:
        device = args.device
        print(f"Device: {device}")
    
    print("\n" + "="*80)
    print("CONFRONTO MODELLI RL - HIGHWAY-ENV")
    print("="*80)
    print(f"Modelli: {[MODEL_REGISTRY[m]['name'] for m in args.models]}")
    print(f"Episodi per modello: {args.episodes}")
    print(f"Seed: {args.seed}")
    print(f"Rendering: {'Sì' if render else 'No'}")
    print(f"Verbose: {'Sì' if args.verbose else 'No'}")
    print("="*80 + "\n")
    
    # Carica tutti i modelli
    models = {}
    for model_key in args.models:
        try:
            models[model_key] = load_model(model_key, device=device)
            print(f"✓ {MODEL_REGISTRY[model_key]['name']} caricato con successo")
        except Exception as e:
            print(f"✗ Errore caricamento {model_key}: {e}")
            return 1
    
    print()
    
    # Crea environment per ogni modello
    envs = {}
    for model_key in args.models:
        envs[model_key] = gymnasium.make(
            "highway-v0",
            config=ENV_CONFIG,
            render_mode='rgb_array' if render else None
        )
    
    # Prepara metriche
    all_metrics = HighwayMetrics.AVAILABLE_METRICS
    results = {}
    
    # Valutazione con rendering parallelo
    if render:
        print("Avvio valutazione con rendering parallelo...\n")
        
        n_models = len(args.models)
        fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 6))
        if n_models == 1:
            axes = [axes]
        
        fig.suptitle(f'Confronto Modelli in Tempo Reale (Seed={args.seed})', 
                     fontsize=14, fontweight='bold')
        
        # Setup axes
        img_objs = {}
        for idx, model_key in enumerate(args.models):
            ax = axes[idx]
            model_name = MODEL_REGISTRY[model_key]['name']
            color = MODEL_REGISTRY[model_key]['color']
            
            ax.set_title(f'{model_name}', fontsize=12, color=color, fontweight='bold')
            ax.axis('off')
            
            # Render iniziale
            envs[model_key].reset(seed=args.seed)
            frame = envs[model_key].render()
            img_objs[model_key] = ax.imshow(frame)
            
            # Bordo colorato
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                      fill=False, edgecolor=color, linewidth=4))
        
        plt.ion()
        plt.tight_layout()
        plt.show()
        
        # Valuta ogni modello
        for model_key in args.models:
            print(f"\nValutazione {MODEL_REGISTRY[model_key]['name']}...")
            print("-" * 60)
            
            tracker = HighwayMetrics(metrics=all_metrics, verbose=args.verbose)
            
            idx = args.models.index(model_key)
            results[model_key] = evaluate_model_with_render(
                models[model_key],
                MODEL_REGISTRY[model_key]['name'],
                envs[model_key],
                args.episodes,
                args.seed,
                tracker,
                ax=axes[idx],
                img_obj=img_objs[model_key]
            )
        
        plt.ioff()
        plt.close()
    
    # Valutazione senza rendering (più veloce)
    else:
        print("Avvio valutazione (no render)...\n")
        
        for model_key in args.models:
            print(f"\nValutazione {MODEL_REGISTRY[model_key]['name']}...")
            print("-" * 60)
            
            tracker = HighwayMetrics(metrics=all_metrics, verbose=args.verbose)
            results[model_key] = evaluate_model_no_render(
                models[model_key],
                MODEL_REGISTRY[model_key]['name'],
                envs[model_key],
                args.episodes,
                args.seed,
                tracker
            )
            
            # Stampa report individuale
            tracker.print_report(f"{MODEL_REGISTRY[model_key]['name']} - Performance Report")
    
    # Cleanup environments
    for env in envs.values():
        env.close()
    
    # Tabella di confronto
    print_comparison_table(results, args.models)
    
    # Grafici di confronto
    if len(args.models) > 1:
        repo_root = Path(__file__).resolve().parents[1]
        logs_dir = repo_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = logs_dir / f"comparison_{timestamp}.png"
        
        plot_comparison_charts(results, args.models, plot_path)
    
    # Salva risultati JSON
    repo_root = Path(__file__).resolve().parents[1]
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = logs_dir / f"comparison_{timestamp}.json"
    
    output_data = {
        'timestamp': timestamp,
        'seed': args.seed,
        'n_episodes': args.episodes,
        'models': args.models,
        'config': ENV_CONFIG,
        'results': results
    }
    
    with open(json_path, 'w') as f:
        # Gestisci inf per JSON
        def convert_inf(obj):
            if isinstance(obj, float) and obj == float('inf'):
                return None
            return obj
        
        json.dump(output_data, f, indent=2, default=convert_inf)
    
    print(f"Risultati salvati in: {json_path}\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
