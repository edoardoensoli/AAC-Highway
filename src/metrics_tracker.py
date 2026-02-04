"""
Sistema di metriche per valutare performance dei modelli RL in highway-env.
Utilizza env.unwrapped per accedere ai dati reali dei veicoli.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
import json
from datetime import datetime
from pathlib import Path


@dataclass
class EpisodeData:
    """Dati raccolti durante un singolo episodio."""
    # Stato finale
    crashed: bool = False
    truncated: bool = False
    
    # Contatori
    total_reward: float = 0.0
    steps: int = 0
    cars_overtaken: int = 0
    lane_changes: int = 0
    
    # Velocità
    speeds: List[float] = field(default_factory=list)
    
    # Distanza
    start_x: Optional[float] = None
    end_x: float = 0.0
    
    # Sicurezza
    min_ttc: float = float('inf')  # Time To Collision minimo
    near_miss_count: int = 0  # Situazioni di quasi-incidente
    
    # Tracciamento interno (non metriche finali)
    _previous_lane: Optional[int] = None
    _overtaken_ids: Set[int] = field(default_factory=set)
    _vehicle_positions: Dict[int, float] = field(default_factory=dict)


class HighwayMetrics:
    """
    Tracker di metriche per highway-env.
    
    Metriche disponibili:
    - collision_rate: % di episodi terminati con collisione
    - survival_rate: % di episodi completati senza crash
    - avg_reward: ricompensa media per episodio
    - avg_speed: velocità media (m/s)
    - max_speed: velocità massima raggiunta (m/s)
    - cars_overtaken: numero medio di auto superate per episodio
    - total_cars_overtaken: numero totale di auto superate
    - avg_episode_length: durata media degli episodi (steps)
    - lane_changes: numero medio di cambi corsia per episodio
    - distance_traveled: distanza media percorsa (m)
    - min_ttc: Time To Collision minimo medio (s) - metrica di sicurezza
    - near_miss_rate: % di episodi con quasi-incidenti
    """
    
    AVAILABLE_METRICS = {
        'collision_rate',
        'survival_rate', 
        'avg_reward',
        'avg_speed',
        'max_speed',
        'cars_overtaken',
        'total_cars_overtaken',
        'avg_episode_length',
        'lane_changes',
        'distance_traveled',
        'min_ttc',
        'near_miss_rate',
    }
    
    def __init__(
        self,
        metrics: Optional[Set[str]] = None,
        overtake_threshold: float = 3.0,   # metri per considerare un sorpasso (abbassato per catturare più sorpassi)
        near_miss_distance: float = 5.0,   # metri per quasi-incidente
        ttc_threshold: float = 2.0,        # secondi per TTC critico
        verbose: bool = False
    ):
        """
        Args:
            metrics: Set di metriche da calcolare. None = tutte.
            overtake_threshold: Distanza (m) per rilevare un sorpasso
            near_miss_distance: Distanza (m) per considerare un quasi-incidente
            ttc_threshold: Soglia TTC (s) per situazioni critiche
            verbose: Stampa debug durante la valutazione
        """
        if metrics is None:
            self.active_metrics = self.AVAILABLE_METRICS.copy()
        else:
            invalid = metrics - self.AVAILABLE_METRICS
            if invalid:
                raise ValueError(f"Metriche non valide: {invalid}. Disponibili: {self.AVAILABLE_METRICS}")
            self.active_metrics = metrics
        
        self.overtake_threshold = overtake_threshold
        self.near_miss_distance = near_miss_distance
        self.ttc_threshold = ttc_threshold
        self.verbose = verbose
        
        self.episodes: List[EpisodeData] = []
        self.current: Optional[EpisodeData] = None
    
    def start_episode(self, env: Any):
        """Inizia un nuovo episodio."""
        self.current = EpisodeData()
        
        # Salva posizione iniziale
        try:
            ego = env.unwrapped.vehicle
            self.current.start_x = float(ego.position[0])
        except:
            self.current.start_x = 0.0
    
    def step(self, env: Any, action: int, reward: float, done: bool, truncated: bool, info: Dict):
        """
        Aggiorna le metriche dopo ogni step.
        
        Args:
            env: L'environment gymnasium (verrà usato env.unwrapped)
            action: Azione eseguita
            reward: Ricompensa ottenuta
            done: Episodio terminato
            truncated: Episodio troncato
            info: Info dall'environment
        """
        if self.current is None:
            return
        
        ep = self.current
        ep.steps += 1
        ep.total_reward += reward
        
        # Leggi stato da info (sempre disponibile)
        ep.crashed = info.get('crashed', False)
        ep.truncated = truncated
        
        # Accedi ai dati reali tramite env.unwrapped
        try:
            u = env.unwrapped
            ego = u.vehicle
            ego_x = float(ego.position[0])
            ego_y = float(ego.position[1])
            ego_speed = float(ego.speed)
            ego_vx = float(ego.velocity[0]) if hasattr(ego, 'velocity') else ego_speed
            
            # Aggiorna posizione finale
            ep.end_x = ego_x
            
            # Velocità
            ep.speeds.append(ego_speed)
            
            # Lane changes
            lane_idx = ego.lane_index
            if isinstance(lane_idx, tuple):
                current_lane = int(lane_idx[-1])
            else:
                current_lane = 0
            
            if ep._previous_lane is not None and current_lane != ep._previous_lane:
                ep.lane_changes += 1
            ep._previous_lane = current_lane
            
            # Analizza altri veicoli
            for v in u.road.vehicles:
                if v is ego:
                    continue
                
                vid = id(v)
                v_x = float(v.position[0])
                v_y = float(v.position[1])
                v_vx = float(v.velocity[0]) if hasattr(v, 'velocity') else float(v.speed)
                
                # Distanza relativa
                rel_x = v_x - ego_x
                rel_y = v_y - ego_y
                distance = np.sqrt(rel_x**2 + rel_y**2)
                
                # Near miss detection
                if distance < self.near_miss_distance and abs(rel_y) < 2.0:
                    ep.near_miss_count += 1
                
                # Time To Collision (TTC) per veicoli davanti nella stessa corsia
                if 0 < rel_x < 50 and abs(rel_y) < 2.0:  # Davanti, stessa corsia approssimativa
                    relative_speed = ego_vx - v_vx
                    if relative_speed > 0.1:  # Ci stiamo avvicinando
                        ttc = rel_x / relative_speed
                        if ttc < ep.min_ttc:
                            ep.min_ttc = ttc
                
                # Rilevamento sorpassi MIGLIORATO
                # Tracciamo la posizione MASSIMA (più avanti) che ogni veicolo ha raggiunto
                # rispetto a noi. Se era davanti e ora è dietro = sorpasso.
                # Questo funziona su TUTTE le corsie.
                
                prev_data = ep._vehicle_positions.get(vid)
                
                if prev_data is None:
                    # Prima volta che vediamo questo veicolo
                    # Salviamo: (posizione_attuale, max_posizione_vista, già_sorpassato)
                    ep._vehicle_positions[vid] = {
                        'current': rel_x,
                        'max_ahead': rel_x if rel_x > 0 else 0,
                        'was_ahead': rel_x > self.overtake_threshold
                    }
                else:
                    was_ahead = prev_data['was_ahead']
                    max_ahead = prev_data['max_ahead']
                    
                    # Aggiorna max_ahead se il veicolo è andato più avanti
                    if rel_x > max_ahead:
                        max_ahead = rel_x
                    
                    # Se non l'abbiamo già contato, era davanti, e ora è dietro = SORPASSO
                    if was_ahead and rel_x < -self.overtake_threshold:
                        if vid not in ep._overtaken_ids:
                            ep._overtaken_ids.add(vid)
                            ep.cars_overtaken += 1
                            
                            # Info sulla corsia del veicolo sorpassato
                            v_lane = getattr(v, 'lane_index', None)
                            v_lane_str = f" (corsia {v_lane[-1]})" if isinstance(v_lane, tuple) else ""
                            
                            if self.verbose:
                                print(f"  [Step {ep.steps}] SORPASSO!{v_lane_str} max_ahead: {max_ahead:.1f}m -> now: {rel_x:.1f}m (Totale: {ep.cars_overtaken})")
                    
                    # Se il veicolo va davanti alla soglia, segna che era davanti
                    if rel_x > self.overtake_threshold:
                        was_ahead = True
                    
                    ep._vehicle_positions[vid] = {
                        'current': rel_x,
                        'max_ahead': max_ahead,
                        'was_ahead': was_ahead
                    }
                
        except Exception as e:
            if self.verbose:
                print(f"  [Warning] Impossibile leggere env.unwrapped: {e}")
    
    def end_episode(self):
        """Termina l'episodio corrente e salva i dati."""
        if self.current is not None:
            self.episodes.append(self.current)
            
            if self.verbose:
                ep = self.current
                status = "CRASH" if ep.crashed else ("TRONCATO" if ep.truncated else "OK")
                print(f"Episodio {len(self.episodes)}: {status} | "
                      f"Steps: {ep.steps} | Reward: {ep.total_reward:.1f} | "
                      f"Sorpassi: {ep.cars_overtaken}")
            
            self.current = None
    
    def compute(self) -> Dict[str, float]:
        """
        Calcola tutte le metriche attive.
        
        Returns:
            Dizionario con i valori delle metriche
        """
        if not self.episodes:
            return {m: 0.0 for m in self.active_metrics}
        
        n = len(self.episodes)
        results = {}
        
        # Collision rate
        if 'collision_rate' in self.active_metrics:
            crashes = sum(1 for ep in self.episodes if ep.crashed)
            results['collision_rate'] = (crashes / n) * 100
        
        # Survival rate (opposto di collision rate)
        if 'survival_rate' in self.active_metrics:
            survived = sum(1 for ep in self.episodes if not ep.crashed)
            results['survival_rate'] = (survived / n) * 100
        
        # Average reward
        if 'avg_reward' in self.active_metrics:
            results['avg_reward'] = np.mean([ep.total_reward for ep in self.episodes])
        
        # Average speed
        if 'avg_speed' in self.active_metrics:
            all_speeds = [s for ep in self.episodes for s in ep.speeds]
            results['avg_speed'] = np.mean(all_speeds) if all_speeds else 0.0
        
        # Max speed
        if 'max_speed' in self.active_metrics:
            max_speeds = [max(ep.speeds) if ep.speeds else 0 for ep in self.episodes]
            results['max_speed'] = max(max_speeds) if max_speeds else 0.0
        
        # Cars overtaken (media per episodio)
        if 'cars_overtaken' in self.active_metrics:
            results['cars_overtaken'] = np.mean([ep.cars_overtaken for ep in self.episodes])
        
        # Total cars overtaken
        if 'total_cars_overtaken' in self.active_metrics:
            results['total_cars_overtaken'] = sum(ep.cars_overtaken for ep in self.episodes)
        
        # Average episode length
        if 'avg_episode_length' in self.active_metrics:
            results['avg_episode_length'] = np.mean([ep.steps for ep in self.episodes])
        
        # Lane changes (media)
        if 'lane_changes' in self.active_metrics:
            results['lane_changes'] = np.mean([ep.lane_changes for ep in self.episodes])
        
        # Distance traveled (media)
        if 'distance_traveled' in self.active_metrics:
            distances = []
            for ep in self.episodes:
                if ep.start_x is not None:
                    distances.append(ep.end_x - ep.start_x)
            results['distance_traveled'] = np.mean(distances) if distances else 0.0
        
        # Min TTC (media dei minimi per episodio, escludendo inf)
        if 'min_ttc' in self.active_metrics:
            ttcs = [ep.min_ttc for ep in self.episodes if ep.min_ttc < float('inf')]
            results['min_ttc'] = np.mean(ttcs) if ttcs else float('inf')
        
        # Near miss rate
        if 'near_miss_rate' in self.active_metrics:
            near_misses = sum(1 for ep in self.episodes if ep.near_miss_count > 0)
            results['near_miss_rate'] = (near_misses / n) * 100
        
        return results
    
    def print_report(self, title: str = "Performance Report"):
        """Stampa un report formattato delle metriche."""
        metrics = self.compute()
        
        print(f"\n{'='*60}")
        print(f"{title:^60}")
        print(f"{'='*60}")
        print(f"Episodi valutati: {len(self.episodes)}")
        print(f"{'-'*60}")
        
        # Ordine personalizzato per leggibilità
        order = [
            'collision_rate', 'survival_rate', 'avg_reward',
            'cars_overtaken', 'total_cars_overtaken',
            'avg_speed', 'max_speed', 'distance_traveled',
            'avg_episode_length', 'lane_changes',
            'min_ttc', 'near_miss_rate'
        ]
        
        for key in order:
            if key not in metrics:
                continue
            value = metrics[key]
            label = key.replace('_', ' ').title()
            
            # Formattazione specifica per tipo
            if 'rate' in key:
                print(f"  {label:.<40} {value:>12.1f}%")
            elif 'speed' in key:
                print(f"  {label:.<40} {value:>12.1f} m/s")
            elif key == 'distance_traveled':
                print(f"  {label:.<40} {value:>12.1f} m")
            elif key == 'min_ttc':
                if value == float('inf'):
                    print(f"  {label:.<40} {'N/A':>12}")
                else:
                    print(f"  {label:.<40} {value:>12.2f} s")
            elif 'total' in key:
                print(f"  {label:.<40} {int(value):>12}")
            else:
                print(f"  {label:.<40} {value:>12.2f}")
        
        print(f"{'='*60}\n")
    
    def save_json(self, filepath: str):
        """Salva le metriche in formato JSON."""
        metrics = self.compute()
        
        # Gestisci inf per JSON
        for k, v in metrics.items():
            if v == float('inf'):
                metrics[k] = None
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'n_episodes': len(self.episodes),
            'metrics': list(self.active_metrics),
            'results': metrics,
            'episodes': [
                {
                    'crashed': ep.crashed,
                    'steps': ep.steps,
                    'reward': ep.total_reward,
                    'cars_overtaken': ep.cars_overtaken,
                    'avg_speed': np.mean(ep.speeds) if ep.speeds else 0,
                }
                for ep in self.episodes
            ]
        }
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Risultati salvati in: {path}")
        return path
    
    def reset(self):
        """Reset completo del tracker."""
        self.episodes.clear()
        self.current = None


def evaluate(
    model,
    env,
    n_episodes: int = 100,
    metrics: Optional[Set[str]] = None,
    render: bool = False,
    verbose: bool = False,
    seed: Optional[int] = None
) -> Dict[str, float]:
    """
    Valuta un modello RL su highway-env.
    
    Args:
        model: Modello con metodo .predict(obs)
        env: Environment gymnasium
        n_episodes: Numero di episodi
        metrics: Metriche da calcolare (None = tutte)
        render: Renderizza l'environment
        verbose: Stampa debug
    
    Returns:
        Dizionario con le metriche calcolate
    """
    tracker = HighwayMetrics(metrics=metrics, verbose=verbose)
    
    for ep_num in range(n_episodes):
        if seed is not None:
            obs, info = env.reset(seed=seed + ep_num)
        else:
            obs, info = env.reset()
            
        tracker.start_episode(env)
        
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            tracker.step(env, action, reward, done, truncated, info)
            
            if render:
                env.render()
        
        tracker.end_episode()
        
        # Progress
        if (ep_num + 1) % max(1, n_episodes // 10) == 0:
            print(f"Progresso: {ep_num + 1}/{n_episodes} episodi")
    
    tracker.print_report()
    return tracker.compute()
