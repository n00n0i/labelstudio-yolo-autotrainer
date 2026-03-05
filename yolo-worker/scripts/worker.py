#!/usr/bin/env python3
"""
YOLO Training Worker
Consumes training jobs from Redis queue
"""

import os
import json
import time
import redis
from datetime import datetime
from ultralytics import YOLO

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
DATASET_DIR = os.getenv("DATASET_DIR", "/datasets")
MODELS_DIR = os.getenv("MODELS_DIR", "/models")

# Connect to Redis
r = redis.from_url(REDIS_URL)

def process_training_job(job_data: dict):
    """Process a training job"""
    
    job_id = job_data["job_id"]
    project_name = job_data["project_name"]
    dataset_path = job_data["dataset_path"]
    config = job_data.get("config", {})
    
    print(f"[{datetime.now()}] Starting job: {job_id}")
    
    try:
        # Update status
        update_job_status(job_id, "running", 0)
        
        # Load model
        model_type = config.get("model_type", "yolov8")
        model_size = config.get("model_size", "n")
        model_name = f"{model_type}{model_size}"
        
        print(f"[{job_id}] Loading model: {model_name}")
        model = YOLO(f"{model_name}.pt")
        
        # Training parameters
        epochs = config.get("epochs", 100)
        batch_size = config.get("batch_size", 16)
        image_size = config.get("image_size", 640)
        
        # Train with callback for progress
        def on_train_epoch_end(trainer):
            progress = (trainer.epoch + 1) / epochs * 100
            update_job_status(job_id, "running", progress, {
                "current_epoch": trainer.epoch + 1,
                "total_epochs": epochs,
                "loss": trainer.loss,
                "metrics": trainer.metrics
            })
        
        # Add callback
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        
        # Start training
        print(f"[{job_id}] Training started: {epochs} epochs")
        results = model.train(
            data=f"{dataset_path}/data.yaml",
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size,
            project=MODELS_DIR,
            name=f"{project_name}_{job_id}",
            save=True,
            device=0
        )
        
        # Export to ONNX
        print(f"[{job_id}] Exporting to ONNX...")
        model.export(format='onnx')
        
        # Training complete
        model_path = f"{MODELS_DIR}/{project_name}_{job_id}"
        update_job_status(job_id, "completed", 100, {
            "model_path": model_path,
            "best_map": results.results_dict.get('metrics/mAP50-95(B)', 0),
            "epochs_trained": epochs
        })
        
        print(f"[{job_id}] Training completed successfully!")
        
    except Exception as e:
        print(f"[{job_id}] Training failed: {e}")
        update_job_status(job_id, "failed", 0, {"error": str(e)})
        raise


def update_job_status(job_id: str, status: str, progress: float, details: dict = None):
    """Update job status in Redis"""
    data = {
        "status": status,
        "progress": progress,
        "updated_at": datetime.now().isoformat()
    }
    if details:
        data["details"] = details
    
    r.hset(f"training_job:{job_id}", mapping={
        "data": json.dumps(data)
    })
    
    # Publish update
    r.publish(f"job_updates:{job_id}", json.dumps(data))


def main():
    """Main worker loop"""
    print("YOLO Training Worker started")
    print(f"Waiting for jobs from: {REDIS_URL}")
    
    while True:
        try:
            # Blocking pop from queue
            result = r.blpop("training_queue", timeout=5)
            
            if result:
                _, job_json = result
                job_data = json.loads(job_json)
                
                try:
                    process_training_job(job_data)
                except Exception as e:
                    print(f"Error processing job: {e}")
                    # Job failed, could retry or notify
            
            else:
                # No jobs, sleep briefly
                time.sleep(1)
                
        except Exception as e:
            print(f"Worker error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
