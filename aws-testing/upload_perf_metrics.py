"""
Step Time Metrics Uploader for Performance Tests

Parses step_time_metrics.json and uploads performance data to Scuba database.

Author: @yoamol
Created: 2025
Last Modified: 2025
Version: 2.0

Usage:
    python upload_perf_metrics2.py <vs_event_id> <json_file_path>
"""

import os
import json
import sys
import re
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from kaena_scuba_data_api import ScubaHeartbeat

@dataclass
class TestResult:
    """Data class representing a single test result."""
    test_name: str
    model_type: str
    status: str
    duration: Optional[float]
    error_message: Optional[str] = None
    error_category: Optional[str] = None
    error_categories: Optional[List[str]] = None
    user: Optional[str] = None

def parse_test_config():
    """Extract test configuration from environment variables."""
    model_arch = os.environ.get("MODEL_ARCH", "unknown")
    fsdp = os.environ.get("FSDP", "0")
    model = os.environ.get("MODEL", "0") 
    pipeline = os.environ.get("PIPELINE", "0")
    grad_accum = os.environ.get("GRAD_ACCUM", "1")
    num_nodes = os.environ.get("NUM_NODES", "1")
    
    return {
        "model_arch": model_arch,
        "fsdp": int(fsdp),
        "tensor_parallel": int(model),
        "pipeline_parallel": int(pipeline),
        "grad_accumulation": int(grad_accum),
        "num_nodes": int(num_nodes)
    }

def parse_test_parameters():
    """Extract test parameters from environment variables."""
    return {
        'batch_size': os.environ.get('BATCH_SIZE', '1'),
        'sequence_length': os.environ.get('SEQUENCE_LENGTH', '0'),
        'hidden_size': os.environ.get('HIDDEN_SIZE', '0')
    }

def parse_model_info(model_type: str) -> Dict:
    """Parse model type string into components."""
    info = {}
    
    # Extract model type (12B, deepseek-v3, etc.) - everything before _MoE
    type_match = re.search(r'^([^_]+)_MoE', model_type)
    if type_match:
        info['model_type'] = type_match.group(1)
    
    # Extract experts and hidden size
    experts_match = re.search(r'e(\d+)', model_type)
    if experts_match:
        info['num_experts'] = int(experts_match.group(1))
    
    hidden_match = re.search(r'h(\d+)', model_type)
    if hidden_match:
        info['hidden_size'] = int(hidden_match.group(1))
    
    return info

def generate_performance_summary(result: TestResult, training_metrics: Dict, config: Dict, test_params: Dict) -> Dict:
    """Generate comprehensive performance summary.
    
    Creates detailed statistics and analysis of performance metrics,
    including latency distribution, throughput, and configuration details.
    
    Args:
        result: TestResult object containing test information
        training_metrics: Performance metrics from step_time_metrics.json
        config: Test configuration dictionary
        test_params: Test parameters dictionary
        
    Returns:
        Dictionary containing comprehensive performance summary:
            - test_info: Test metadata
            - performance_metrics: Latency and throughput statistics
            - configuration: Model and test configuration
            - timestamp: When the summary was generated
    """
    if not training_metrics:
        return {}
    
    # Performance metrics analysis
    latency_metrics = {
        "avg_latency_ms": training_metrics.get("latency_ms_avg", 0),
        "p50_latency_ms": training_metrics.get("latency_ms_p50", 0),
        "p90_latency_ms": training_metrics.get("latency_ms_p90", 0),
        "p95_latency_ms": training_metrics.get("latency_ms_p95", 0),
        "p99_latency_ms": training_metrics.get("latency_ms_p99", 0),
        "p100_latency_ms": training_metrics.get("latency_ms_p100", 0)
    }
    
    # Calculate throughput if available
    throughput_metrics = {}
    if training_metrics.get("step_count", 0) > 0 and training_metrics.get("latency_ms_avg", 0) > 0:
        steps_per_second = 1000.0 / training_metrics["latency_ms_avg"]
        throughput_metrics["steps_per_second"] = steps_per_second
    
    # Configuration summary
    config_summary = {
        "model_architecture": config["model_arch"],
        "parallelization": {
            "fsdp": config["fsdp"],
            "tensor_parallel": config["tensor_parallel"],
            "pipeline_parallel": config["pipeline_parallel"],
            "grad_accumulation": config["grad_accumulation"],
            "num_nodes": config["num_nodes"]
        },
        "test_parameters": {
            "batch_size": int(test_params.get('batch_size', 1)),
            "sequence_length": int(test_params.get('sequence_length', 0)),
            "hidden_size": int(test_params.get('hidden_size', 0))
        }
    }
    
    return {
        "test_info": {
            "test_name": result.test_name,
            "model_type": result.model_type,
            "status": result.status,
            "duration_seconds": result.duration or 0.0,
            "user": result.user or "unknown"
        },
        "performance_metrics": {
            "latency": latency_metrics,
            "throughput": throughput_metrics,
            "total_steps": training_metrics.get("step_count", 0)
        },
        "configuration": config_summary,
        "timestamp": datetime.now().isoformat()
    }

def upload_metrics(vs_event_id, json_file_path, dry_run=False):
    """Upload step time metrics to Scuba."""
    try:
        # Initialize Scuba client
        try:
            scuba = ScubaHeartbeat(dry_run=dry_run)
            scuba_available = True
        except Exception as e:
            print(f"Warning: Scuba not available ({e}). Running in offline mode.")
            scuba = None
            scuba_available = False
        
        # Load metrics
        with open(json_file_path, 'r') as f:
            metrics = json.load(f)
        
        training_metrics = metrics.get("training_step_time", {})
        if not training_metrics:
            print("No training_step_time found in metrics")
            return False
        
        # Parse test configuration
        config = parse_test_config()
        test_params = parse_test_parameters()
        
        # Create TestResult object
        model_type = f"{config['model_arch']}_fsdp{config['fsdp']}_tp{config['tensor_parallel']}"
        test_name = f"perf_{config['model_arch']}_fsdp{config['fsdp']}_tp{config['tensor_parallel']}"
        
        result = TestResult(
            test_name=test_name,
            model_type=model_type,
            status="PASS",
            duration=training_metrics.get("latency_ms_avg", 0) / 1000.0,  # Convert ms to seconds
            error_message=None,
            error_category=None,
            error_categories=None
        )
        
        # Parse model info
        model_info = parse_model_info(result.model_type)
        
        # Prepare data for Scuba
        data = {
            "cfg": {
                "TestDef": {
                    "ModelName": result.model_type,
                    "Identifier": "Performance_Dashboard",
                    "TestName": result.test_name,
                    "ModelType": model_info.get('model_type', config["model_arch"]),
                    # "InferenceOrTraining": "Training",
                    # "SequenceLength": int(test_params.get('sequence_length', 0)),
                    # "HiddenSize": model_info.get('hidden_size', int(test_params.get('hidden_size', 0))),
                    # "Custom": {
                    #     "TotalSteps": training_metrics.get("step_count", 0),
                    #     "FSDP": config["fsdp"],
                    #     "TensorParallel": config["tensor_parallel"],
                    #     "PipelineParallel": config["pipeline_parallel"],
                    #     "GradAccumulation": config["grad_accumulation"],
                    #     "NumNodes": config["num_nodes"],
                    #     "NumExperts": model_info.get('num_experts', 0)
                    # }
                },
                "TestOpt": {"BatchSize": int(test_params.get('batch_size', 1))},
                "Hardware": {"InstanceType": "trn2.48xlarge"},
                "Software": {"VersionSet": vs_event_id}
            },
            "kpi": {
                "E2ETime": result.duration or 0.0,
                "LatencyP50": training_metrics.get("latency_ms_p50", 0),
                "LatencyP90": training_metrics.get("latency_ms_p90", 0),
                "LatencyP95": training_metrics.get("latency_ms_p95", 0),
                "LatencyP99": training_metrics.get("latency_ms_p99", 0),
                "LatencyP100": training_metrics.get("latency_ms_p100", 0),
                "LatencyAverage": training_metrics.get("latency_ms_avg", 0),
                "TestPassFail": result.status,
                "TestFailMessage": result.error_message or ""
            },
            "misc": {
                "User": "yoamol",
                "ModelCategory": result.model_type,
                "Timestamp": datetime.now().isoformat()
            },
            "artifact": {}
        }
        
        # Upload to Scuba
        if scuba_available:
            success = scuba.post_metrics(data, use_dev_table=True, dry_run=dry_run)
            
            if success:
                print(f"✓ {result.test_name} ({result.status}) - {result.model_type}")
                return True
            else:
                print(f"✗ Failed: {result.test_name}")
                return False
        else:
            print(f"{result.test_name} ({result.status}) - {result.model_type} [OFFLINE MODE]")
            return True
            
    except Exception as e:
        print(f"Error uploading metrics: {e}")
        return False

def main():
    if len(sys.argv) != 3:
        print("Usage: python upload_step_metrics.py <vs_event_id> <json_file_path>")
        sys.exit(1)
    
    vs_event_id = sys.argv[1]
    json_file_path = sys.argv[2]
    
    if not os.path.exists(json_file_path):
        print(f"Error: File {json_file_path} not found")
        sys.exit(1)
    
    # Load metrics for summary
    try:
        with open(json_file_path, 'r') as f:
            metrics = json.load(f)
        
        training_metrics = metrics.get("training_step_time", {})
        if training_metrics:
            config = parse_test_config()
            test_params = parse_test_parameters()
            
            # Create TestResult for summary
            model_type = f"{config['model_arch']}_fsdp{config['fsdp']}_tp{config['tensor_parallel']}"
            test_name = f"perf_{config['model_arch']}_fsdp{config['fsdp']}_tp{config['tensor_parallel']}"
            
            result = TestResult(
                test_name=test_name,
                model_type=model_type,
                status="PASS",
                duration=training_metrics.get("latency_ms_avg", 0) / 1000.0,
                error_message=None,
                error_category=None,
                error_categories=None,
                user=os.environ.get("USER", "unknown")
            )
            
            # Generate and print summary
            summary = generate_performance_summary(result, training_metrics, config, test_params)
            
            print("\n" + "="*60)
            print("PERFORMANCE SUMMARY")
            print("="*60)
            
            if summary:
                print(f"User: {summary['test_info']['user']}")
                print(f"Test: {summary['test_info']['test_name']}")
                print(f"Model: {summary['test_info']['model_type']}")
                print(f"Status: {summary['test_info']['status']}")
                print(f"Duration: {summary['test_info']['duration_seconds']:.3f}s")
                
                print("\nPerformance Metrics:")
                latency = summary['performance_metrics']['latency']
                print(f"  Average Latency: {latency['avg_latency_ms']:.2f}ms")
                print(f"  P50 Latency: {latency['p50_latency_ms']:.2f}ms")
                print(f"  P90 Latency: {latency['p90_latency_ms']:.2f}ms")
                print(f"  P95 Latency: {latency['p95_latency_ms']:.2f}ms")
                print(f"  P99 Latency: {latency['p99_latency_ms']:.2f}ms")
                
                if summary['performance_metrics']['throughput']:
                    throughput = summary['performance_metrics']['throughput']
                    print(f"  Throughput: {throughput['steps_per_second']:.2f} steps/sec")
                
                print(f"  Total Steps: {summary['performance_metrics']['total_steps']}")
                
                print("\nConfiguration:")
                config_info = summary['configuration']
                print(f"  Model Architecture: {config_info['model_architecture']}")
                print(f"  Batch Size: {config_info['test_parameters']['batch_size']}")
                print(f"  Sequence Length: {config_info['test_parameters']['sequence_length']}")
                print(f"  FSDP: {config_info['parallelization']['fsdp']}")
                print(f"  Tensor Parallel: {config_info['parallelization']['tensor_parallel']}")
                print(f"  Pipeline Parallel: {config_info['parallelization']['pipeline_parallel']}")
                print(f"  Nodes: {config_info['parallelization']['num_nodes']}")
            
            print("\n" + "="*60)
            print("UPLOADING TO SCUBA")
            print("="*60)
            
    except Exception as e:
        print(f"Error loading metrics for summary: {e}")
    
    success = upload_metrics(vs_event_id, json_file_path)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()