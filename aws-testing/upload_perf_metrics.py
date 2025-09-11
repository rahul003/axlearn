"""
Step Time Metrics Uploader for Performance Tests

Parses step_time_metrics.json and uploads performance data to Scuba database.

Author: @yoamol
Created: 2025
Last Modified: 2025
Version: 1.0

Usage:
    python upload_perf_metrics.py <vs_event_id> <json_file_path>
"""

import os
import json
import sys
from datetime import datetime
from kaena_scuba_data_api import ScubaHeartbeat

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

def upload_metrics(vs_event_id, json_file_path):
    """Upload step time metrics to Scuba."""
    try:
        # Load metrics
        with open(json_file_path, 'r') as f:
            metrics = json.load(f)
        
        training_metrics = metrics.get("training_step_time", {})
        if not training_metrics:
            print("No training_step_time found in metrics")
            return False
        
        # Parse test configuration
        config = parse_test_config()
        
        # Initialize Scuba client
        scuba = ScubaHeartbeat(dry_run=False)
        
        # Prepare data for Scuba
        data = {
            "cfg": {
                "TestDef": {
                    "ModelName": config["model_arch"],
                    "Identifier": "Performance_Dashboard",
                    "TestName": f"perf_{config['model_arch']}_fsdp{config['fsdp']}_tp{config['tensor_parallel']}",
                    "ModelType": config["model_arch"],
                    "InferenceOrTraining": "Training",
                    "Custom": {
                        "TotalSteps": training_metrics.get("step_count", 0),
                        "FSDP": config["fsdp"],
                        "TensorParallel": config["tensor_parallel"],
                        "PipelineParallel": config["pipeline_parallel"],
                        "GradAccumulation": config["grad_accumulation"],
                        "NumNodes": config["num_nodes"]
                    }
                },
                "Hardware": {"InstanceType": "trn2.48xlarge"},
                "Software": {"VersionSet": vs_event_id}
            },
            "kpi": {
                "LatencyBenchMarkLatencyP50": training_metrics.get("latency_ms_p50", 0),
                "LatencyBenchMarkLatencyP90": training_metrics.get("latency_ms_p90", 0),
                "LatencyBenchMarkLatencyP95": training_metrics.get("latency_ms_p95", 0),
                "LatencyBenchMarkLatencyP99": training_metrics.get("latency_ms_p99", 0),
                "LatencyBenchMarkLatencyP100": training_metrics.get("latency_ms_p100", 0),
                "LatencyAverage": training_metrics.get("latency_ms_avg", 0),
                "TestPassFail": "PASS"
            },
            "misc": {
                "User": os.environ.get("USER", "unknown"),
                "Timestamp": datetime.now().isoformat(),
                "JobID": os.environ.get("JOB_ID", "unknown")
            },
            "artifact": {}
        }
        
        # Upload to Scuba
        success = scuba.post_metrics(data, use_dev_table=True, dry_run=False)
        
        if success:
            print(f"Uploaded metrics for {config['model_arch']} to Scuba")
            return True
        else:
            print(f"Failed to upload metrics")
            return False
            
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
    
    success = upload_metrics(vs_event_id, json_file_path)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()