#!/usr/bin/env python3
"""
Label Studio Webhook Handler
Receives annotation events and triggers YOLO training
"""

import os
import json
import asyncio
import subprocess
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel
import httpx
import yaml

app = FastAPI(title="Label Studio YOLO Auto-Trainer")

# Configuration
LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
LABEL_STUDIO_TOKEN = os.getenv("LABEL_STUDIO_TOKEN", "")
TRAINING_THRESHOLD = int(os.getenv("TRAINING_THRESHOLD", "100"))  # Min annotations to trigger
DATASET_DIR = os.getenv("DATASET_DIR", "./datasets")
MODELS_DIR = os.getenv("MODELS_DIR", "./models")

# Training state
active_trainings: Dict[int, dict] = {}


class LabelStudioWebhook(BaseModel):
    """Label Studio webhook payload"""
    action: str  # ANNOTATION_CREATED, ANNOTATION_UPDATED, etc.
    project: dict
    task: dict
    annotation: Optional[dict] = None


class TrainingConfig(BaseModel):
    """YOLO training configuration"""
    project_id: int
    project_name: str
    model_type: str = "yolov8"
    model_size: str = "n"  # n, s, m, l, x
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    auto_deploy: bool = False


class TrainingStatus(BaseModel):
    """Training job status"""
    job_id: str
    project_id: int
    status: str  # pending, running, completed, failed
    progress: float
    current_epoch: int
    total_epochs: int
    metrics: Optional[dict] = None
    model_path: Optional[str] = None
    started_at: str
    completed_at: Optional[str] = None


@app.post("/webhook/labelstudio")
async def handle_labelstudio_webhook(
    payload: LabelStudioWebhook,
    background_tasks: BackgroundTasks
):
    """
    Handle Label Studio webhook events
    """
    project_id = payload.project.get("id")
    project_name = payload.project.get("title", f"project_{project_id}")
    
    # Only process annotation events
    if payload.action not in ["ANNOTATION_CREATED", "ANNOTATION_UPDATED"]:
        return {"status": "ignored", "action": payload.action}
    
    # Check annotation count
    annotation_count = await get_annotation_count(project_id)
    
    print(f"[{datetime.now()}] Project {project_name}: {annotation_count} annotations")
    
    # Check if training should trigger
    if annotation_count >= TRAINING_THRESHOLD:
        # Check if already training
        if project_id in active_trainings:
            return {
                "status": "skipped",
                "reason": "Training already in progress",
                "annotation_count": annotation_count
            }
        
        # Trigger training
        config = TrainingConfig(
            project_id=project_id,
            project_name=project_name
        )
        
        background_tasks.add_task(auto_train_yolo, config)
        
        return {
            "status": "training_triggered",
            "annotation_count": annotation_count,
            "threshold": TRAINING_THRESHOLD,
            "project": project_name
        }
    
    return {
        "status": "counting",
        "annotation_count": annotation_count,
        "threshold": TRAINING_THRESHOLD
    }


async def get_annotation_count(project_id: int) -> int:
    """Get total annotation count for a project"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{LABEL_STUDIO_URL}/api/projects/{project_id}",
            headers={"Authorization": f"Token {LABEL_STUDIO_TOKEN}"}
        )
        response.raise_for_status()
        data = response.json()
        return data.get("num_annotations", 0)


async def export_yolo_dataset(project_id: int, project_name: str) -> str:
    """Export Label Studio project to YOLO format"""
    
    dataset_path = f"{DATASET_DIR}/{project_name}_{project_id}"
    os.makedirs(dataset_path, exist_ok=True)
    
    # Export from Label Studio
    async with httpx.AsyncClient() as client:
        # Get export in YOLO format
        response = await client.get(
            f"{LABEL_STUDIO_URL}/api/projects/{project_id}/export",
            params={"exportType": "YOLO"},
            headers={"Authorization": f"Token {LABEL_STUDIO_TOKEN}"}
        )
        response.raise_for_status()
        
        # Save export
        export_file = f"{dataset_path}/export.zip"
        with open(export_file, "wb") as f:
            f.write(response.content)
        
        # Extract
        subprocess.run(["unzip", "-o", export_file, "-d", dataset_path], check=True)
        
        # Create data.yaml for YOLO
        await create_yolo_config(project_id, project_name, dataset_path)
        
    return dataset_path


async def create_yolo_config(project_id: int, project_name: str, dataset_path: str):
    """Create YOLO data.yaml configuration"""
    
    # Get labels from Label Studio
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{LABEL_STUDIO_URL}/api/projects/{project_id}",
            headers={"Authorization": f"Token {LABEL_STUDIO_TOKEN}"}
        )
        response.raise_for_status()
        project_data = response.json()
        
        # Extract labels from label config
        labels = extract_labels_from_config(project_data.get("label_config", ""))
    
    # Create data.yaml
    data_config = {
        "path": dataset_path,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(labels),
        "names": {i: name for i, name in enumerate(labels)}
    }
    
    with open(f"{dataset_path}/data.yaml", "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    return data_config


def extract_labels_from_config(label_config: str) -> List[str]:
    """Extract label names from Label Studio config XML"""
    import xml.etree.ElementTree as ET
    
    try:
        root = ET.fromstring(label_config)
        labels = []
        for label in root.findall(".//Label"):
            value = label.get("value")
            if value:
                labels.append(value)
        return labels
    except:
        # Fallback: return default labels
        return ["object"]


async def auto_train_yolo(config: TrainingConfig):
    """Auto-train YOLO model"""
    
    job_id = f"{config.project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    project_id = config.project_id
    
    # Register training
    active_trainings[project_id] = {
        "job_id": job_id,
        "status": "starting",
        "started_at": datetime.now().isoformat()
    }
    
    try:
        # Step 1: Export dataset
        print(f"[{job_id}] Exporting dataset...")
        active_trainings[project_id]["status"] = "exporting"
        dataset_path = await export_yolo_dataset(project_id, config.project_name)
        
        # Step 2: Start training
        print(f"[{job_id}] Starting YOLO training...")
        active_trainings[project_id]["status"] = "training"
        
        model_name = f"{config.model_type}{config.model_size}"
        model_output = f"{MODELS_DIR}/{config.project_name}_{job_id}"
        
        # Run YOLO training
        training_script = f"""
from ultralytics import YOLO

# Load model
model = YOLO('{model_name}.pt')

# Train
results = model.train(
    data='{dataset_path}/data.yaml',
    epochs={config.epochs},
    imgsz={config.image_size},
    batch={config.batch_size},
    project='{MODELS_DIR}',
    name='{config.project_name}_{job_id}',
    save=True,
    device=0
)

# Export
model.export(format='onnx')
print(f"Model saved to: {model_output}")
"""
        
        # Write training script
        script_path = f"/tmp/train_{job_id}.py"
        with open(script_path, "w") as f:
            f.write(training_script)
        
        # Run training
        process = subprocess.Popen(
            ["python3", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Monitor training
        while process.poll() is None:
            await asyncio.sleep(10)
            # Update progress (in real implementation, parse logs)
            active_trainings[project_id]["progress"] = 50  # Placeholder
        
        # Training complete
        if process.returncode == 0:
            active_trainings[project_id]["status"] = "completed"
            active_trainings[project_id]["model_path"] = model_output
            
            # Auto-deploy if enabled
            if config.auto_deploy:
                await deploy_model(model_output, config.project_name)
        else:
            active_trainings[project_id]["status"] = "failed"
            stderr = process.stderr.read()
            active_trainings[project_id]["error"] = stderr
            
    except Exception as e:
        active_trainings[project_id]["status"] = "failed"
        active_trainings[project_id]["error"] = str(e)
        print(f"[{job_id}] Training failed: {e}")
    
    finally:
        active_trainings[project_id]["completed_at"] = datetime.now().isoformat()


async def deploy_model(model_path: str, project_name: str):
    """Deploy trained model to inference service"""
    print(f"Deploying model: {model_path}")
    # Implementation depends on your inference service
    pass


@app.get("/training/status/{project_id}")
async def get_training_status(project_id: int):
    """Get training status for a project"""
    if project_id not in active_trainings:
        raise HTTPException(status_code=404, detail="No active training for this project")
    
    return active_trainings[project_id]


@app.get("/training/status")
async def list_all_trainings():
    """List all training jobs"""
    return active_trainings


@app.post("/training/trigger/{project_id}")
async def manual_trigger_training(
    project_id: int,
    config: Optional[TrainingConfig] = None,
    background_tasks: BackgroundTasks = None
):
    """Manually trigger training for a project"""
    
    if project_id in active_trainings:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    if config is None:
        config = TrainingConfig(
            project_id=project_id,
            project_name=f"project_{project_id}"
        )
    
    background_tasks.add_task(auto_train_yolo, config)
    
    return {"status": "training_triggered", "project_id": project_id}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_trainings": len(active_trainings),
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
