# -*- coding: utf-8 -*-
"""
API Service Module for OpenBench

This module provides standardized REST API interfaces for external services,
enabling remote access to OpenBench evaluation capabilities.

Author: OpenBench Contributors  
Version: 1.0
Date: July 2025
"""

import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

# Optional web server import
try:
    import uvicorn
    _HAS_UVICORN = True
except ImportError:
    _HAS_UVICORN = False

# FastAPI imports with fallbacks
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, FileResponse
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel, Field, validator
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False
    FastAPI = object
    HTTPException = Exception
    BaseModel = object
    HTTPBearer = object
    HTTPAuthorizationCredentials = object
    BackgroundTasks = object
    FileResponse = object
    def Field(*args, **kwargs):
        return None

# Import OpenBench modules
try:
    from config.manager import ConfigManager
    from Mod_Exceptions import APIError, ValidationError, error_handler
    from Mod_EvaluationEngine import create_evaluation_engine, evaluate_datasets
    from Mod_OutputManager import ModularOutputManager
    from Mod_LoggingSystem import get_logging_manager
    from Mod_ParallelEngine import ParallelEngine
    from Mod_CacheSystem import get_cache_manager
    _HAS_OPENBENCH_MODULES = True
except ImportError:
    _HAS_OPENBENCH_MODULES = False
    ConfigManager = object
    APIError = Exception
    ValidationError = Exception
    def error_handler(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Import xarray for data handling
try:
    import xarray as xr
    import numpy as np
    import pandas as pd
    _HAS_DATA_MODULES = True
except ImportError:
    _HAS_DATA_MODULES = False


# API Models
class EvaluationRequest(BaseModel if _HAS_FASTAPI else object):
    """Request model for evaluation."""
    
    simulation_path: str = Field(..., description="Path to simulation data")
    reference_path: str = Field(..., description="Path to reference data") 
    metrics: List[str] = Field(..., description="List of metrics to calculate")
    evaluation_type: str = Field(default="modular", description="Type of evaluation engine")
    output_format: str = Field(default="json", description="Output format")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Additional configuration")


class EvaluationResponse(BaseModel if _HAS_FASTAPI else object):
    """Response model for evaluation."""
    
    task_id: str = Field(..., description="Task identifier")
    status: str = Field(..., description="Task status")
    results: Optional[Dict[str, Any]] = Field(default=None, description="Evaluation results")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    created_at: str = Field(..., description="Task creation time")
    completed_at: Optional[str] = Field(default=None, description="Task completion time")


class ConfigurationRequest(BaseModel if _HAS_FASTAPI else object):
    """Request model for configuration."""
    
    config_data: Dict[str, Any] = Field(..., description="Configuration data")
    config_type: str = Field(default="json", description="Configuration format")


class StatusResponse(BaseModel if _HAS_FASTAPI else object):
    """Response model for system status."""
    
    system_status: str = Field(..., description="System status")
    active_tasks: int = Field(..., description="Number of active tasks")
    completed_tasks: int = Field(..., description="Number of completed tasks")
    failed_tasks: int = Field(..., description="Number of failed tasks")
    system_info: Dict[str, Any] = Field(..., description="System information")


class APIService:
    """OpenBench API Service for external access."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize API service."""
        self.config_path = config_path
        self.app = None
        self.tasks = {}
        self.task_counter = 0
        
        # Initialize components
        if _HAS_OPENBENCH_MODULES:
            self.config_manager = ConfigManager()
            self.output_manager = ModularOutputManager()
            self.parallel_engine = ParallelEngine()
            
            # Load configuration
            if config_path and os.path.exists(config_path):
                self.config = self.config_manager.load_config(config_path)
            else:
                self.config = self._get_default_config()
            
            # Setup logging
            self.logger = get_logging_manager().get_logger("APIService")
        else:
            self.config = self._get_default_config()
            self.logger = logging.getLogger("APIService")
        
        # Setup cache if available
        if _HAS_OPENBENCH_MODULES:
            try:
                self.cache_manager = get_cache_manager()
            except:
                self.cache_manager = None
        else:
            self.cache_manager = None
        
        # Security
        self.security = HTTPBearer() if _HAS_FASTAPI else None
        self.api_keys = self.config.get('api_keys', [])
        
        # Initialize FastAPI app
        if _HAS_FASTAPI:
            self._create_app()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default API configuration."""
        return {
            'host': '0.0.0.0',
            'port': 8000,
            'reload': False,
            'workers': 1,
            'max_upload_size': 100 * 1024 * 1024,  # 100MB
            'rate_limit': 100,  # requests per minute
            'enable_cors': True,
            'cors_origins': ['*'],
            'api_keys': [],
            'enable_auth': False,
            'result_ttl': 3600,  # 1 hour
            'max_concurrent_tasks': 10
        }
    
    def _create_app(self):
        """Create FastAPI application."""
        if not _HAS_FASTAPI:
            raise ImportError("FastAPI is required for API service")
        
        self.app = FastAPI(
            title="OpenBench API",
            description="Land Surface Model Benchmarking API",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # CORS middleware
        if self.config.get('enable_cors', True):
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.get('cors_origins', ['*']),
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register API routes."""
        
        @self.app.get("/")
        async def root():
            """Root endpoint."""
            return {"message": "OpenBench API", "version": "2.0.0"}
        
        @self.app.get("/status", response_model=StatusResponse)
        async def get_status():
            """Get system status."""
            return await self._get_system_status()
        
        @self.app.post("/evaluate", response_model=EvaluationResponse)
        async def create_evaluation(
            request: EvaluationRequest,
            background_tasks: BackgroundTasks,
            credentials: HTTPAuthorizationCredentials = Depends(self.security) if self.config.get('enable_auth') else None
        ):
            """Create new evaluation task."""
            if self.config.get('enable_auth') and not self._validate_auth(credentials):
                raise HTTPException(status_code=401, detail="Invalid authentication")
            
            return await self._create_evaluation_task(request, background_tasks)
        
        @self.app.get("/evaluate/{task_id}", response_model=EvaluationResponse)
        async def get_evaluation(task_id: str):
            """Get evaluation task status and results."""
            return await self._get_evaluation_task(task_id)
        
        @self.app.delete("/evaluate/{task_id}")
        async def delete_evaluation(task_id: str):
            """Delete evaluation task."""
            return await self._delete_evaluation_task(task_id)
        
        @self.app.get("/evaluate/{task_id}/download")
        async def download_results(task_id: str):
            """Download evaluation results."""
            return await self._download_results(task_id)
        
        @self.app.post("/config/validate")
        async def validate_config(request: ConfigurationRequest):
            """Validate configuration."""
            return await self._validate_configuration(request)
        
        @self.app.get("/metrics")
        async def get_available_metrics():
            """Get list of available metrics."""
            return await self._get_available_metrics()
        
        @self.app.get("/engines")
        async def get_available_engines():
            """Get list of available evaluation engines."""
            return await self._get_available_engines()
    
    def _validate_auth(self, credentials: HTTPAuthorizationCredentials) -> bool:
        """Validate API authentication."""
        if not self.api_keys:
            return True  # No auth required if no keys configured
        return credentials.credentials in self.api_keys
    
    async def _get_system_status(self) -> StatusResponse:
        """Get current system status."""
        active_tasks = sum(1 for task in self.tasks.values() if task['status'] == 'running')
        completed_tasks = sum(1 for task in self.tasks.values() if task['status'] == 'completed')
        failed_tasks = sum(1 for task in self.tasks.values() if task['status'] == 'failed')
        
        # System information
        try:
            import psutil
            system_info = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            }
        except ImportError:
            system_info = {'message': 'System monitoring not available'}
        
        return StatusResponse(
            system_status="healthy",
            active_tasks=active_tasks,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            system_info=system_info
        )
    
    async def _create_evaluation_task(
        self, 
        request: EvaluationRequest, 
        background_tasks: BackgroundTasks
    ) -> EvaluationResponse:
        """Create new evaluation task."""
        
        # Check concurrent task limit
        active_tasks = sum(1 for task in self.tasks.values() if task['status'] == 'running')
        if active_tasks >= self.config.get('max_concurrent_tasks', 10):
            raise HTTPException(status_code=429, detail="Too many concurrent tasks")
        
        # Generate task ID
        self.task_counter += 1
        task_id = f"eval_{self.task_counter}_{int(datetime.now().timestamp())}"
        
        # Create task record
        task_record = {
            'id': task_id,
            'status': 'pending',
            'request': request.dict(),
            'created_at': datetime.now().isoformat(),
            'completed_at': None,
            'results': None,
            'error': None
        }
        
        self.tasks[task_id] = task_record
        
        # Schedule background task
        background_tasks.add_task(self._execute_evaluation, task_id, request)
        
        return EvaluationResponse(
            task_id=task_id,
            status='pending',
            created_at=task_record['created_at']
        )
    
    async def _execute_evaluation(self, task_id: str, request: EvaluationRequest):
        """Execute evaluation in background."""
        try:
            # Update status
            self.tasks[task_id]['status'] = 'running'
            
            if not _HAS_OPENBENCH_MODULES or not _HAS_DATA_MODULES:
                raise Exception("Required modules not available")
            
            # Load datasets
            try:
                simulation = xr.open_dataset(request.simulation_path)
                reference = xr.open_dataset(request.reference_path)
            except Exception as e:
                raise Exception(f"Failed to load datasets: {e}")
            
            # Create evaluation engine
            engine = create_evaluation_engine(
                request.evaluation_type,
                **(request.config or {})
            )
            
            # Perform evaluation
            results = engine.evaluate(simulation, reference, request.metrics)
            
            # Save results if requested
            if request.output_format != 'memory':
                output_path = f"./output/api_results/{task_id}.{request.output_format}"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                if request.output_format == 'json':
                    with open(output_path, 'w') as f:
                        json.dump(results, f, indent=2, default=str)
                elif request.output_format == 'csv':
                    # Convert to CSV format
                    metrics_data = []
                    for metric_name, metric_data in results['metrics'].items():
                        metrics_data.append({
                            'metric': metric_name,
                            'value': metric_data['value'],
                            'description': metric_data['info']['description'],
                            'unit': metric_data['info']['unit']
                        })
                    df = pd.DataFrame(metrics_data)
                    df.to_csv(output_path, index=False)
                
                results['output_path'] = output_path
            
            # Update task with results
            self.tasks[task_id].update({
                'status': 'completed',
                'results': results,
                'completed_at': datetime.now().isoformat()
            })
            
            self.logger.info(f"Evaluation task {task_id} completed successfully")
            
        except Exception as e:
            # Update task with error
            self.tasks[task_id].update({
                'status': 'failed',
                'error': str(e),
                'completed_at': datetime.now().isoformat()
            })
            
            self.logger.error(f"Evaluation task {task_id} failed: {e}")
    
    async def _get_evaluation_task(self, task_id: str) -> EvaluationResponse:
        """Get evaluation task status and results."""
        if task_id not in self.tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task = self.tasks[task_id]
        
        return EvaluationResponse(
            task_id=task_id,
            status=task['status'],
            results=task['results'],
            error=task['error'],
            created_at=task['created_at'],
            completed_at=task['completed_at']
        )
    
    async def _delete_evaluation_task(self, task_id: str):
        """Delete evaluation task."""
        if task_id not in self.tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Clean up output files if they exist
        task = self.tasks[task_id]
        if task.get('results') and 'output_path' in task['results']:
            output_path = task['results']['output_path']
            if os.path.exists(output_path):
                os.remove(output_path)
        
        del self.tasks[task_id]
        
        return {"message": f"Task {task_id} deleted successfully"}
    
    async def _download_results(self, task_id: str):
        """Download evaluation results."""
        if task_id not in self.tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task = self.tasks[task_id]
        
        if task['status'] != 'completed':
            raise HTTPException(status_code=400, detail="Task not completed")
        
        if not task.get('results') or 'output_path' not in task['results']:
            raise HTTPException(status_code=404, detail="Results file not found")
        
        output_path = task['results']['output_path']
        
        if not os.path.exists(output_path):
            raise HTTPException(status_code=404, detail="Results file not found")
        
        return FileResponse(
            output_path,
            filename=f"{task_id}_results.{output_path.split('.')[-1]}",
            media_type='application/octet-stream'
        )
    
    async def _validate_configuration(self, request: ConfigurationRequest):
        """Validate configuration data."""
        try:
            if _HAS_OPENBENCH_MODULES:
                # Use ConfigManager for validation
                self.config_manager.validate_config(request.config_data)
            
            return {"valid": True, "message": "Configuration is valid"}
        
        except Exception as e:
            return {"valid": False, "message": str(e)}
    
    async def _get_available_metrics(self):
        """Get list of available metrics."""
        if _HAS_OPENBENCH_MODULES:
            engine = create_evaluation_engine()
            metrics = engine.get_supported_metrics()
            
            metric_info = {}
            for metric in metrics:
                try:
                    metric_info[metric] = engine.get_metric_info(metric)
                except:
                    metric_info[metric] = {"description": "No description available"}
            
            return {"metrics": metric_info}
        else:
            return {"metrics": {"error": "OpenBench modules not available"}}
    
    async def _get_available_engines(self):
        """Get list of available evaluation engines."""
        engines = [
            {
                "name": "modular",
                "description": "General purpose modular evaluation engine",
                "supported_data_types": ["gridded", "station", "time_series"]
            },
            {
                "name": "grid", 
                "description": "Specialized engine for gridded data evaluation",
                "supported_data_types": ["gridded"]
            },
            {
                "name": "station",
                "description": "Specialized engine for station data evaluation", 
                "supported_data_types": ["station", "point"]
            }
        ]
        
        return {"engines": engines}
    
    def run(self, **kwargs):
        """Run the API service."""
        if not _HAS_FASTAPI:
            raise ImportError("FastAPI is required to run API service")
        
        if not _HAS_UVICORN:
            raise ImportError("uvicorn is required to run API service")
        
        # Merge config with kwargs
        config = {**self.config, **kwargs}
        
        self.logger.info(f"Starting OpenBench API service on {config['host']}:{config['port']}")
        
        uvicorn.run(
            self.app,
            host=config['host'],
            port=config['port'],
            reload=config.get('reload', False),
            workers=config.get('workers', 1)
        )


def create_api_service(config_path: Optional[str] = None) -> APIService:
    """Create API service instance."""
    return APIService(config_path)


# CLI interface for running API service
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenBench API Service")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Create and run service
    service = create_api_service(args.config)
    service.run(
        host=args.host,
        port=args.port,
        reload=args.reload
    )