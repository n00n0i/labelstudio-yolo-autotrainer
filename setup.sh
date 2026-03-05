#!/bin/bash
# Label Studio + YOLO Auto-Training Setup

set -e

echo "========================================"
echo "  Label Studio + YOLO Auto-Trainer"
echo "========================================"
echo ""

# Check prerequisites
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found"
    exit 1
fi

# Create directories
mkdir -p datasets models labelstudio-data

# Generate token
LABEL_STUDIO_TOKEN=$(openssl rand -hex 32)

# Create .env
cat > .env << EOF
LABEL_STUDIO_TOKEN=$LABEL_STUDIO_TOKEN
TRAINING_THRESHOLD=100
EOF

echo "✅ Environment configured"

# Start services
echo "🚀 Starting services..."
docker-compose up -d

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "Access URLs:"
echo "  Label Studio:  http://localhost:8080"
echo "  Auto-Trainer:  http://localhost:8000"
echo "  Model Registry: http://localhost:8081"
echo ""
echo "Default Credentials:"
echo "  Label Studio:  admin@example.com / admin"
echo ""
echo "Next Steps:"
echo "  1. Login to Label Studio"
echo "  2. Create a project with bounding box labels"
echo "  3. Start annotating (auto-training at 100 annotations)"
echo "  4. Check training status at http://localhost:8000/training/status"
echo ""
echo "Webhook configured automatically!"
echo "========================================"
