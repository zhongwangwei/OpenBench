# -*- coding: utf-8 -*-
"""
Parallel Processing Engine for OpenBench

This module provides enhanced parallel processing capabilities with
intelligent task scheduling, resource management, and progress tracking.

Author: Zhongwang Wei  
Version: 1.0
Date: July 2025
"""

import os
import sys
import time
import logging
import multiprocessing as mp
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import threading
import queue
import psutil
from tqdm import tqdm
import numpy as np

# Import dependencies
try:
    from joblib import Parallel, delayed, parallel_backend
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False
    Parallel = None
    delayed = None

try:
    import dask
    from dask import delayed as dask_delayed
    from dask.distributed import Client, as_completed as dask_as_completed
    _HAS_DASK = True
except ImportError:
    _HAS_DASK = False
    dask = None
    Client = None

try:
    from openbench.util.Mod_Exceptions import ParallelProcessingError, error_handler
    from openbench.util.Mod_LoggingSystem import get_logging_manager, performance_logged
    _HAS_DEPENDENCIES = True
except ImportError:
    _HAS_DEPENDENCIES = False
    ParallelProcessingError = Exception
    def error_handler(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def performance_logged(operation=None):
        def decorator(func):
            return func
        return decorator


class TaskResult:
    """Container for task execution results."""
    
    def __init__(self, task_id: str, success: bool, result: Any = None, 
                 error: Optional[Exception] = None, duration: float = 0.0):
        """
        Initialize task result.
        
        Args:
            task_id: Unique task identifier
            success: Whether task completed successfully
            result: Task result (if successful)
            error: Exception (if failed)
            duration: Execution duration in seconds
        """
        self.task_id = task_id
        self.success = success
        self.result = result
        self.error = error
        self.duration = duration
        self.timestamp = time.time()


class ResourceMonitor:
    """Monitor system resources for intelligent scheduling."""
    
    def __init__(self):
        """Initialize resource monitor."""
        self.cpu_count = mp.cpu_count()
        self.memory_total = psutil.virtual_memory().total
        self._lock = threading.Lock()
    
    def get_available_resources(self) -> Dict[str, Any]:
        """Get current available system resources."""
        with self._lock:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            return {
                'cpu_available': max(1, int(self.cpu_count * (100 - cpu_percent) / 100)),
                'cpu_percent_used': cpu_percent,
                'memory_available_gb': memory.available / (1024**3),
                'memory_percent_used': memory.percent,
                'load_average': os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0
            }
    
    def can_schedule_task(self, cpu_required: int = 1, 
                         memory_required_gb: float = 1.0) -> bool:
        """
        Check if resources are available for task.
        
        Args:
            cpu_required: Number of CPUs required
            memory_required_gb: Memory required in GB
            
        Returns:
            True if resources available
        """
        resources = self.get_available_resources()
        
        return (resources['cpu_available'] >= cpu_required and 
                resources['memory_available_gb'] >= memory_required_gb and
                resources['cpu_percent_used'] < 90)
    
    def get_optimal_workers(self, max_workers: Optional[int] = None) -> int:
        """
        Determine optimal number of workers based on resources.
        
        Args:
            max_workers: Maximum workers to consider
            
        Returns:
            Optimal number of workers
        """
        resources = self.get_available_resources()
        
        # Base on available CPUs and memory
        cpu_based = resources['cpu_available']
        memory_based = int(resources['memory_available_gb'] / 0.5)  # 0.5GB per worker
        
        optimal = min(cpu_based, memory_based)
        
        if max_workers:
            optimal = min(optimal, max_workers)
        
        return max(1, optimal)


class ParallelEngine:
    """Main parallel processing engine with multiple backends."""
    
    def __init__(self, backend: str = 'auto', max_workers: Optional[int] = None,
                 show_progress: bool = True):
        """
        Initialize parallel engine.
        
        Args:
            backend: Backend to use ('auto', 'joblib', 'dask', 'concurrent', 'threading')
            max_workers: Maximum number of workers
            show_progress: Whether to show progress bars
        """
        self.backend = self._select_backend(backend)
        self.max_workers = max_workers
        self.show_progress = show_progress
        self.resource_monitor = ResourceMonitor()
        
        # Task tracking
        self.task_queue = queue.Queue()
        self.results = {}
        self.active_tasks = set()
        self._lock = threading.Lock()
        
        # Initialize backend
        self._initialize_backend()
        
        logging.info(f"Initialized ParallelEngine with backend: {self.backend}")
    
    def _select_backend(self, backend: str) -> str:
        """Select appropriate backend based on availability."""
        if backend == 'auto':
            # Prefer joblib over dask due to compatibility issues
            if _HAS_JOBLIB:
                return 'joblib'
            elif _HAS_DASK:
                return 'dask'
            else:
                return 'concurrent'
        
        # Validate requested backend
        if backend == 'dask' and not _HAS_DASK:
            logging.warning("Dask not available, falling back to joblib")
            return 'joblib' if _HAS_JOBLIB else 'concurrent'
        
        if backend == 'joblib' and not _HAS_JOBLIB:
            logging.warning("Joblib not available, falling back to concurrent")
            return 'concurrent'
        
        return backend
    
    def _initialize_backend(self):
        """Initialize the selected backend."""
        if self.backend == 'dask':
            # Initialize Dask client
            self.dask_client = Client(
                n_workers=self.max_workers or self.resource_monitor.get_optimal_workers(),
                threads_per_worker=1,
                memory_limit='auto',
                silence_logs=logging.WARNING
            )
            logging.info(f"Dask client initialized: {self.dask_client}")
        else:
            self.dask_client = None
    
    @error_handler(reraise=True)
    @performance_logged("parallel_map")
    def map(self, func: Callable, items: List[Any], 
            chunk_size: Optional[int] = None,
            task_name: str = "Processing") -> List[Any]:
        """
        Parallel map operation with progress tracking.
        
        Args:
            func: Function to apply
            items: Items to process
            chunk_size: Optional chunk size for batching
            task_name: Name for progress bar
            
        Returns:
            List of results in order
        """
        if not items:
            return []
        
        # Determine optimal workers
        n_workers = self.max_workers or self.resource_monitor.get_optimal_workers()
        n_workers = min(n_workers, len(items))
        
        logging.info(f"Starting parallel {task_name} with {n_workers} workers")
        
        # Select implementation based on backend
        if self.backend == 'dask':
            return self._map_dask(func, items, n_workers, task_name)
        elif self.backend == 'joblib':
            return self._map_joblib(func, items, n_workers, task_name)
        elif self.backend == 'threading':
            return self._map_threading(func, items, n_workers, task_name)
        else:
            return self._map_concurrent(func, items, n_workers, task_name)
    
    def _map_dask(self, func: Callable, items: List[Any], 
                  n_workers: int, task_name: str) -> List[Any]:
        """Map using Dask backend."""
        # Create delayed tasks
        tasks = [dask_delayed(func)(item) for item in items]
        
        # Submit and track progress
        futures = self.dask_client.compute(tasks)
        
        if self.show_progress:
            results = []
            with tqdm(total=len(items), desc=task_name) as pbar:
                for future in dask_as_completed(futures):
                    results.append(future.result())
                    pbar.update(1)
        else:
            results = [future.result() for future in futures]
        
        return results
    
    def _map_joblib(self, func: Callable, items: List[Any], 
                    n_workers: int, task_name: str) -> List[Any]:
        """Map using Joblib backend."""
        with parallel_backend('loky', n_jobs=n_workers):
            if self.show_progress:
                results = Parallel()(
                    delayed(func)(item) 
                    for item in tqdm(items, desc=task_name)
                )
            else:
                results = Parallel()(delayed(func)(item) for item in items)
        
        return results
    
    def _map_concurrent(self, func: Callable, items: List[Any], 
                       n_workers: int, task_name: str) -> List[Any]:
        """Map using concurrent.futures backend."""
        results = [None] * len(items)
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(func, item): i 
                for i, item in enumerate(items)
            }
            
            # Track progress
            if self.show_progress:
                with tqdm(total=len(items), desc=task_name) as pbar:
                    for future in as_completed(future_to_index):
                        index = future_to_index[future]
                        try:
                            results[index] = future.result()
                        except Exception as e:
                            logging.error(f"Task {index} failed: {e}")
                            raise
                        pbar.update(1)
            else:
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    results[index] = future.result()
        
        return results
    
    def _map_threading(self, func: Callable, items: List[Any], 
                      n_workers: int, task_name: str) -> List[Any]:
        """Map using threading backend (for I/O bound tasks)."""
        results = [None] * len(items)
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(func, item): i 
                for i, item in enumerate(items)
            }
            
            # Track progress
            if self.show_progress:
                with tqdm(total=len(items), desc=task_name) as pbar:
                    for future in as_completed(future_to_index):
                        index = future_to_index[future]
                        results[index] = future.result()
                        pbar.update(1)
            else:
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    results[index] = future.result()
        
        return results
    
    @error_handler(reraise=True)
    def map_reduce(self, map_func: Callable, reduce_func: Callable,
                   items: List[Any], initial: Any = None,
                   task_name: str = "MapReduce") -> Any:
        """
        Parallel map-reduce operation.
        
        Args:
            map_func: Function to map over items
            reduce_func: Function to reduce results
            items: Items to process
            initial: Initial value for reduction
            task_name: Name for progress tracking
            
        Returns:
            Reduced result
        """
        # Parallel map
        mapped_results = self.map(map_func, items, task_name=f"{task_name} (Map)")
        
        # Reduce (could be parallelized for associative operations)
        if initial is not None:
            result = initial
            for item in mapped_results:
                result = reduce_func(result, item)
        else:
            result = mapped_results[0]
            for item in mapped_results[1:]:
                result = reduce_func(result, item)
        
        return result
    
    def submit_batch(self, tasks: List[Tuple[Callable, Tuple, Dict]],
                    task_name: str = "Batch Processing") -> List[TaskResult]:
        """
        Submit batch of heterogeneous tasks.
        
        Args:
            tasks: List of (function, args, kwargs) tuples
            task_name: Name for progress tracking
            
        Returns:
            List of TaskResult objects
        """
        results = []
        n_workers = self.max_workers or self.resource_monitor.get_optimal_workers()
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            futures = []
            for i, (func, args, kwargs) in enumerate(tasks):
                task_id = f"{task_name}_{i}"
                future = executor.submit(self._execute_task, task_id, func, args, kwargs)
                futures.append((future, task_id))
            
            # Collect results
            if self.show_progress:
                with tqdm(total=len(tasks), desc=task_name) as pbar:
                    for future, task_id in futures:
                        result = future.result()
                        results.append(result)
                        pbar.update(1)
            else:
                for future, task_id in futures:
                    result = future.result()
                    results.append(result)
        
        return results
    
    def _execute_task(self, task_id: str, func: Callable, 
                     args: Tuple, kwargs: Dict) -> TaskResult:
        """Execute a single task and return result."""
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            return TaskResult(task_id, True, result, None, duration)
        except Exception as e:
            duration = time.time() - start_time
            logging.error(f"Task {task_id} failed: {e}")
            return TaskResult(task_id, False, None, e, duration)
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get current resource usage summary."""
        resources = self.resource_monitor.get_available_resources()
        
        summary = {
            'backend': self.backend,
            'max_workers': self.max_workers,
            'optimal_workers': self.resource_monitor.get_optimal_workers(),
            'resources': resources
        }
        
        if self.backend == 'dask' and self.dask_client:
            summary['dask_info'] = {
                'workers': len(self.dask_client.scheduler_info()['workers']),
                'dashboard': self.dask_client.dashboard_link
            }
        
        return summary
    
    def shutdown(self):
        """Shutdown parallel engine and cleanup resources."""
        if self.backend == 'dask' and self.dask_client:
            self.dask_client.close()
            logging.info("Dask client closed")


# Global engine instance
_parallel_engine = None


def get_parallel_engine(backend: str = 'auto', 
                       max_workers: Optional[int] = None) -> ParallelEngine:
    """
    Get or create global parallel engine.
    
    Args:
        backend: Backend to use
        max_workers: Maximum workers
        
    Returns:
        ParallelEngine instance
    """
    global _parallel_engine
    
    if _parallel_engine is None:
        _parallel_engine = ParallelEngine(backend, max_workers)
    
    return _parallel_engine


# Convenience functions
@error_handler(reraise=True)
def parallel_map(func: Callable, items: List[Any], 
                 max_workers: Optional[int] = None,
                 backend: str = 'auto',
                 show_progress: bool = True,
                 task_name: str = "Processing") -> List[Any]:
    """
    Convenience function for parallel map.
    
    Args:
        func: Function to apply
        items: Items to process
        max_workers: Maximum workers
        backend: Backend to use
        show_progress: Show progress bar
        task_name: Name for progress
        
    Returns:
        List of results
    """
    engine = ParallelEngine(backend, max_workers, show_progress)
    try:
        return engine.map(func, items, task_name=task_name)
    finally:
        engine.shutdown()


def parallel_decorator(max_workers: Optional[int] = None,
                      backend: str = 'auto',
                      show_progress: bool = False):
    """
    Decorator for automatic parallelization of functions.
    
    Args:
        max_workers: Maximum workers
        backend: Backend to use
        show_progress: Show progress
    """
    def decorator(func):
        def wrapper(items, *args, **kwargs):
            # Create partial function with additional arguments
            partial_func = partial(func, *args, **kwargs)
            
            # Run in parallel
            return parallel_map(
                partial_func, 
                items, 
                max_workers=max_workers,
                backend=backend,
                show_progress=show_progress,
                task_name=func.__name__
            )
        
        return wrapper
    return decorator


# Example usage patterns
if __name__ == "__main__":
    # Example 1: Simple parallel map
    def process_item(x):
        time.sleep(0.1)  # Simulate work
        return x ** 2
    
    items = list(range(10))
    results = parallel_map(process_item, items, task_name="Squaring numbers")
    print(f"Results: {results}")
    
    # Example 2: Using decorator
    @parallel_decorator(max_workers=4, show_progress=True)
    def expensive_computation(x, multiplier=2):
        time.sleep(0.1)
        return x * multiplier
    
    results = expensive_computation(items, multiplier=3)
    print(f"Decorated results: {results}")
    
    # Example 3: Map-reduce
    engine = get_parallel_engine()
    total = engine.map_reduce(
        lambda x: x ** 2,  # Square each item
        lambda a, b: a + b,  # Sum results
        items,
        initial=0
    )
    print(f"Sum of squares: {total}")
    
    # Show resource summary
    print("\nResource Summary:")
    print(engine.get_resource_summary())