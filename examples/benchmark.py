"""
MiniTorch Comprehensive Benchmark
==================================
Benchmark riguroso para medir el rendimiento de diferentes componentes de MiniTorch.
Incluye entrenamiento de red neuronal, inferencia, backpropagation y operaciones de tensor.

Configuración centralizada para facilitar la comparación entre versiones.
"""

import numpy as np
import time
import statistics
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

from src.tensor import Tensor
import src.nn as nn
import src.nn.functional as F
from src.utils.data import Dataset, Dataloader, random_split


# ============================================================================
# CONFIGURACIÓN DEL BENCHMARK
# ============================================================================

@dataclass
class BenchmarkConfig:
    """Configuración centralizada del benchmark"""
    
    # Configuración del dataset
    n_samples: int = 20000  # Más pesado
    input_dim: int = 50  # Más features
    output_dim: int = 10  # Más clases
    train_split: float = 0.8
    
    # Configuración de la red neuronal
    hidden_layers: List[int] = field(default_factory=lambda: [256, 128, 64, 32])  # Más capas y neuronas
    learning_rate: float = 0.001
    batch_size: int = 64
    
    # Configuración del entrenamiento
    epochs: int = 50  # Más epochs
    warmup_epochs: int = 3  # Epochs para estabilizar antes de medir
    
    # Regularización
    l1_lambda: float = 0.0001  # Factor de regularización L1
    
    # Configuración del benchmark
    n_iterations: int = 3  # Repeticiones del benchmark completo
    measure_per_epoch: bool = True  # Medir tiempo por epoch
    
    # Semilla para reproducibilidad
    random_seed: int = 42
    
    def __post_init__(self):
        """Establece la semilla aleatoria"""
        np.random.seed(self.random_seed)


# ============================================================================
# GENERACIÓN DE DATOS
# ============================================================================

def generate_synthetic_data(config: BenchmarkConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera datos sintéticos para clasificación multiclase.
    
    Args:
        config: Configuración del benchmark
        
    Returns:
        X: Features de entrada (n_samples, input_dim)
        y: Etiquetas de clase (n_samples,)
    """
    X = np.random.randn(config.n_samples, config.input_dim).astype(np.float32)
    
    # Generar etiquetas basadas en combinaciones lineales de features
    weights = np.random.randn(config.input_dim, config.output_dim)
    logits = X @ weights
    y = np.argmax(logits, axis=1)
    
    return X, y


# ============================================================================
# DEFINICIÓN DEL MODELO
# ============================================================================

def create_model(config: BenchmarkConfig) -> nn.Sequential:
    """
    Crea una red neuronal feedforward con las especificaciones del config.
    
    Args:
        config: Configuración del benchmark
        
    Returns:
        Modelo Sequential
    """
    layers = []
    
    # Capa de entrada
    prev_dim = config.input_dim
    
    # Capas ocultas con activaciones
    for hidden_dim in config.hidden_layers:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        prev_dim = hidden_dim
    
    # Capa de salida
    layers.append(nn.Linear(prev_dim, config.output_dim))
    
    return nn.Sequential(*layers)


# ============================================================================
# FUNCIONES DE MEDICIÓN
# ============================================================================

class Timer:
    """Context manager para medir tiempo de ejecución"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed = 0.0
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start


class BenchmarkResults:
    """Almacena y procesa resultados del benchmark"""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.metrics: Dict[str, List[float]] = {}
    
    def add_timing(self, phase: str, duration: float):
        """Añade una medición de tiempo"""
        if phase not in self.timings:
            self.timings[phase] = []
        self.timings[phase].append(duration)
    
    def add_metric(self, name: str, value: float):
        """Añade una métrica (accuracy, loss, etc.)"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_statistics(self, phase: str) -> Dict[str, float]:
        """Calcula estadísticas para una fase"""
        if phase not in self.timings or len(self.timings[phase]) == 0:
            return {}
        
        times = self.timings[phase]
        return {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'stdev': statistics.stdev(times) if len(times) > 1 else 0.0,
            'min': min(times),
            'max': max(times)
        }
    
    def print_summary(self):
        """Imprime un resumen completo de los resultados"""
        print("\n" + "="*80)
        print("RESUMEN DEL BENCHMARK")
        print("="*80)
        
        # Tiempos de ejecución
        print("\n📊 TIEMPOS DE EJECUCIÓN (segundos)")
        print("-"*80)
        
        phases = [
            'training_data_loading',
            'training_forward',
            'training_regularization',
            'training_backward',
            'training_optimization',
            'training_total'
        ]
        
        for phase in phases:
            if phase in self.timings:
                stats = self.get_statistics(phase)
                if stats:
                    print(f"\n{phase.replace('_', ' ').title()}:")
                    print(f"  Media:    {stats['mean']:.4f}s ± {stats['stdev']:.4f}s")
                    print(f"  Mediana:  {stats['median']:.4f}s")
                    print(f"  Rango:    [{stats['min']:.4f}s - {stats['max']:.4f}s]")
        
        # Métricas de rendimiento
        if self.metrics:
            print("\n" + "-"*80)
            print("📈 MÉTRICAS DE RENDIMIENTO")
            print("-"*80)
            
            for metric_name, values in self.metrics.items():
                if values:
                    mean_val = statistics.mean(values)
                    std_val = statistics.stdev(values) if len(values) > 1 else 0.0
                    print(f"\n{metric_name.replace('_', ' ').title()}:")
                    print(f"  Media: {mean_val:.4f} ± {std_val:.4f}")
        
        print("\n" + "="*80)


# ============================================================================
# FUNCIONES DE ENTRENAMIENTO Y EVALUACIÓN
# ============================================================================

def train_epoch(model: nn.Sequential, 
                dataloader: Dataloader,
                optimizer,
                l1_lambda: float = 0.0,
                measure_phases: bool = False) -> Tuple[float, Dict[str, float]]:
    """
    Entrena el modelo por un epoch.
    
    Args:
        model: Modelo a entrenar
        dataloader: Dataloader con datos de entrenamiento
        optimizer: Optimizador
        l1_lambda: Factor de regularización L1
        measure_phases: Si True, mide el tiempo de cada fase
        
    Returns:
        loss_promedio: Loss promedio del epoch
        phase_times: Diccionario con tiempos de cada fase (si measure_phases=True)
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    phase_times = {
        'data_loading': 0.0,
        'forward': 0.0,
        'regularization': 0.0,
        'backward': 0.0,
        'optimization': 0.0
    }
    
    for data, labels in dataloader:
        # Data loading (convertir a tensor si es necesario)
        if measure_phases:
            with Timer() as t:
                # Simular operaciones de preparación de datos
                if not isinstance(data, Tensor):
                    data = Tensor(data, requires_grad=True)
            phase_times['data_loading'] += t.elapsed
        
        # Forward pass
        if measure_phases:
            with Timer() as t:
                output = model(data)
                loss = F.cross_entropy(output, labels)
            phase_times['forward'] += t.elapsed
        else:
            output = model(data)
            loss = F.cross_entropy(output, labels)
        
        # Regularización L1
        if measure_phases:
            with Timer() as t:
                if l1_lambda > 0:
                    loss = F.l1(loss, model, l1_lambda)
            phase_times['regularization'] += t.elapsed
        else:
            if l1_lambda > 0:
                loss = F.l1(loss, model, l1_lambda)
        
        # Backward pass
        if measure_phases:
            with Timer() as t:
                loss.backward()
            phase_times['backward'] += t.elapsed
        else:
            loss.backward()
        
        # Optimization step
        if measure_phases:
            with Timer() as t:
                optimizer.step()
                optimizer.zero_grad()
            phase_times['optimization'] += t.elapsed
        else:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.data
        n_batches += 1
    
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    return avg_loss, phase_times


def evaluate_model(model: nn.Sequential, dataloader: Dataloader) -> Tuple[float, float]:
    """
    Evalúa el modelo en un conjunto de datos.
    
    Args:
        model: Modelo a evaluar
        dataloader: Dataloader con datos de evaluación
        
    Returns:
        accuracy: Precisión del modelo
        avg_loss: Loss promedio
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    n_batches = 0
    
    for data, labels in dataloader:
        output = model(data)
        predictions = np.argmax(output.data, axis=1)
        
        # Convertir labels a numpy si es necesario
        if isinstance(labels, Tensor):
            labels_np = labels.data
        else:
            labels_np = labels
        
        correct += np.sum(predictions == labels_np)
        total += len(labels_np)
        
        loss = F.cross_entropy(output, labels)
        total_loss += loss.data
        n_batches += 1
    
    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    
    return accuracy, avg_loss


# ============================================================================
# BENCHMARK PRINCIPAL
# ============================================================================

def run_single_benchmark(config: BenchmarkConfig, 
                        iteration: int,
                        results: BenchmarkResults) -> None:
    """
    Ejecuta una iteración completa del benchmark.
    
    Args:
        config: Configuración del benchmark
        iteration: Número de iteración actual
        results: Objeto para almacenar resultados
    """
    print(f"\n{'='*80}")
    print(f"ITERACIÓN {iteration + 1}/{config.n_iterations}")
    print(f"{'='*80}")
    
    # Preparación (sin medir)
    X, y = generate_synthetic_data(config)
    dataset = Dataset(X, y, to_tensor=True)
    train_data, test_data = random_split(
        dataset, 
        (config.train_split, 1 - config.train_split)
    )
    train_dataloader = Dataloader(
        train_data, 
        batch_size=config.batch_size,
        shuffle=True
    )
    test_dataloader = Dataloader(
        test_data,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    model = create_model(config)
    model_params = model.parameters()
    optimizer = nn.Adam(model_params, lr=config.learning_rate)
    
    print(f"\n📋 Configuración de la iteración:")
    print(f"   - Samples entrenamiento: {len(train_data)}")
    print(f"   - Samples prueba: {len(test_data)}")
    print(f"   - Arquitectura: {config.input_dim} -> {' -> '.join(map(str, config.hidden_layers))} -> {config.output_dim}")
    print(f"   - Parámetros totales: {sum(p.data.size for p in model_params)}")
    print(f"   - Regularización L1: λ={config.l1_lambda}")
    
    # === ENTRENAMIENTO (ÚNICO FOCO DE MEDICIÓN) ===
    print(f"\n🏋️  Entrenando modelo ({config.epochs} epochs)...")
    
    # Warmup
    print(f"   Warmup: {config.warmup_epochs} epochs...")
    for epoch in range(config.warmup_epochs):
        train_epoch(model, train_dataloader, optimizer, l1_lambda=config.l1_lambda, measure_phases=False)
    
    # Entrenamiento medido
    print(f"   Midiendo rendimiento...")
    total_data_loading = 0.0
    total_forward = 0.0
    total_regularization = 0.0
    total_backward = 0.0
    total_optimization = 0.0
    
    with Timer() as t_total:
        for epoch in range(config.epochs):
            with Timer() as t_epoch:
                avg_loss, phase_times = train_epoch(
                    model, 
                    train_dataloader, 
                    optimizer,
                    l1_lambda=config.l1_lambda,
                    measure_phases=config.measure_per_epoch
                )
            
            total_data_loading += phase_times['data_loading']
            total_forward += phase_times['forward']
            total_regularization += phase_times['regularization']
            total_backward += phase_times['backward']
            total_optimization += phase_times['optimization']
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch + 1}/{config.epochs} - Loss: {avg_loss:.4f} - Tiempo: {t_epoch.elapsed:.4f}s")
    
    results.add_timing('training_total', t_total.elapsed)
    results.add_timing('training_data_loading', total_data_loading)
    results.add_timing('training_forward', total_forward)
    results.add_timing('training_regularization', total_regularization)
    results.add_timing('training_backward', total_backward)
    results.add_timing('training_optimization', total_optimization)
    
    print(f"\n   ✓ Entrenamiento completado en {t_total.elapsed:.4f}s")
    
    # Evaluación rápida (sin medir en detalle)
    test_accuracy, test_loss = evaluate_model(model, test_dataloader)
    results.add_metric('test_accuracy', test_accuracy)
    results.add_metric('test_loss', test_loss)
    print(f"   ✓ Accuracy final: {test_accuracy:.4f}")


def run_benchmark(config: BenchmarkConfig = None) -> BenchmarkResults:
    """
    Ejecuta el benchmark completo con múltiples iteraciones.
    
    Args:
        config: Configuración del benchmark (usa default si es None)
        
    Returns:
        BenchmarkResults con todos los resultados
    """
    if config is None:
        config = BenchmarkConfig()
    
    print("="*80)
    print("MINITORCH BENCHMARK - ANÁLISIS DE RENDIMIENTO")
    print("="*80)
    print("\n📋 CONFIGURACIÓN:")
    print(f"   - Dataset: {config.n_samples} samples ({config.input_dim} features, {config.output_dim} clases)")
    print(f"   - Arquitectura: {len(config.hidden_layers)} capas ocultas {config.hidden_layers}")
    print(f"   - Entrenamiento: {config.epochs} epochs, batch size {config.batch_size}")
    print(f"   - Regularización L1: λ={config.l1_lambda}")
    print(f"   - Iteraciones del benchmark: {config.n_iterations}")
    print(f"   - Semilla aleatoria: {config.random_seed}")
    print("\n⚡ Fases medidas: Data Loading | Forward | Regularization | Backward | Optimization")
    
    results = BenchmarkResults()
    
    # Ejecutar múltiples iteraciones
    for i in range(config.n_iterations):
        run_single_benchmark(config, i, results)
    
    # Mostrar resumen
    results.print_summary()
    
    return results


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    # Crear configuración personalizada (opcional)
    config = BenchmarkConfig(
        n_samples=20000,
        input_dim=50,
        output_dim=10,
        hidden_layers=[256, 128, 64, 32],
        learning_rate=0.001,
        batch_size=64,
        epochs=50,
        warmup_epochs=3,
        l1_lambda=0.0001,
        n_iterations=3,
        measure_per_epoch=True,
        random_seed=42
    )
    
    # Ejecutar benchmark
    results = run_benchmark(config)
    
    print("\n✅ Benchmark completado exitosamente!")
