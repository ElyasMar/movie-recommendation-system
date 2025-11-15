#!/bin/bash
# Docker entrypoint script to run both Flask API and Streamlit UI

set -e

echo "=========================================="
echo "Movie Recommendation System"
echo "=========================================="
echo ""

# Check if models exist
if [ ! -f "models/content_based_model.pkl" ]; then
    echo "Warning: Models not found!"
    echo "Please run 'python src/train.py' to train models first."
    echo ""
fi

# Start Flask API in background
echo "Starting Flask API on port 5000..."
python src/api/app.py &
FLASK_PID=$!

# Wait a moment for Flask to start
sleep 3

# Start Streamlit UI
echo "Starting Streamlit UI on port 8501..."
streamlit run src/streamlit_app.py &
STREAMLIT_PID=$!

echo ""
echo "=========================================="
echo "Services started successfully!"
echo "=========================================="
echo "Flask API: http://localhost:5000"
echo "Streamlit UI: http://localhost:8501"
echo "=========================================="
echo ""

# Wait for both processes
wait $FLASK_PID $STREAMLIT_PID