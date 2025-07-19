#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenBench Simple Usage Examples

This script demonstrates how to use the new unified OpenBench API.

Author: Zhongwang Wei
Date: July 2025
"""

import sys
import os

# Add openbench directory to path
openbench_dir = os.path.join(os.path.dirname(__file__), '..', 'openbench')
sys.path.insert(0, openbench_dir)

from openbench import OpenBench
import numpy as np
import xarray as xr


def example_1_from_config():
    """Example 1: Create OpenBench instance from configuration file."""
    print("=== Example 1: From Configuration File ===")
    
    try:
        # Create OpenBench instance from config
        ob = OpenBench.from_config('../nml/main-Debug.json')
        
        print(f"✓ OpenBench instance created: {ob}")
        print(f"✓ Available engines: {ob.get_available_engines()}")
        print(f"✓ Available metrics: {ob.get_available_metrics()}")
        
        # Get system info
        info = ob.get_system_info()
        print(f"✓ System status: {info['modules_available']} modules, {info['initialized']} initialized")
        
        return ob
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def example_2_simple_evaluation():
    """Example 2: Simple evaluation with test data."""
    print("\n=== Example 2: Simple Evaluation ===")
    
    try:
        # Create test datasets
        time = np.arange('2004-01-01', '2005-01-01', dtype='datetime64[M]')
        lat = np.linspace(40, 50, 5)
        lon = np.linspace(-10, 10, 5)
        
        # Simulation data
        sim_data = np.random.random((len(time), len(lat), len(lon))) * 100
        simulation = xr.Dataset({
            'LE': (['time', 'lat', 'lon'], sim_data)
        }, coords={'time': time, 'lat': lat, 'lon': lon})
        
        # Reference data (simulation + noise)
        ref_data = sim_data + np.random.normal(0, 5, sim_data.shape)
        reference = xr.Dataset({
            'LE': (['time', 'lat', 'lon'], ref_data)
        }, coords={'time': time, 'lat': lat, 'lon': lon})
        
        # Create OpenBench instance
        ob = OpenBench()
        
        # Run evaluation
        results = ob.run(
            simulation_data=simulation,
            reference_data=reference,
            metrics=['bias', 'RMSE', 'correlation'],
            engine_type='modular'
        )
        
        print("✓ Evaluation completed successfully!")
        print("Results:")
        for metric, data in results['results']['metrics'].items():
            value = data['value']
            if isinstance(value, float) and not np.isnan(value):
                print(f"  - {metric}: {value:.4f}")
        
        return results
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def example_3_context_manager():
    """Example 3: Using context manager for resource cleanup."""
    print("\n=== Example 3: Context Manager Usage ===")
    
    try:
        with OpenBench.from_config('../nml/main-Debug.json') as ob:
            print(f"✓ OpenBench instance in context: {ob}")
            
            # Check configuration
            config = ob.get_config()
            print(f"✓ Configuration sections: {list(config.keys())}")
            
            # Validate configuration
            validation = ob.validate_config()
            print(f"✓ Config validation: {validation['valid']}")
            
            return True
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def example_4_api_service():
    """Example 4: Create API service (without starting)."""
    print("\n=== Example 4: API Service Creation ===")
    
    try:
        # Create OpenBench instance
        ob = OpenBench.from_config('../nml/main-Debug.json')
        
        # Create API service
        api_service = ob.create_api_service(
            host='127.0.0.1',
            port=8080,
            max_concurrent_tasks=5
        )
        
        print("✓ API service created successfully!")
        print(f"✓ Service configured for: {api_service.config['host']}:{api_service.config['port']}")
        print("✓ API endpoints ready (call start_api_service() to run)")
        
        return api_service
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def example_5_convenience_function():
    """Example 5: Using convenience function for quick evaluation."""
    print("\n=== Example 5: Convenience Function ===")
    
    try:
        from openbench import run_evaluation
        
        # Run evaluation with minimal setup
        results = run_evaluation(
            config_path='../nml/main-Debug.json',
            engine_type='modular'
        )
        
        print("✓ Quick evaluation completed!")
        print(f"✓ Results type: {results.get('evaluation_type', 'unknown')}")
        
        return results
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def main():
    """Run all examples."""
    print("OpenBench Unified API Examples")
    print("=" * 40)
    
    # Run examples
    ob = example_1_from_config()
    example_2_simple_evaluation()
    example_3_context_manager()
    example_4_api_service()
    example_5_convenience_function()
    
    print("\n" + "=" * 40)
    print("✓ All examples completed!")
    
    if ob:
        print(f"\nFinal system info: {ob.get_system_info()}")


if __name__ == "__main__":
    main()