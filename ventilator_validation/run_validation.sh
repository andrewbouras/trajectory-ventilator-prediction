#!/bin/bash
# Quick launcher for validation runs

echo "======================================"
echo "VitalDB Validation with Checkpoints"
echo "======================================"
echo ""
echo "Select validation option:"
echo ""
echo "1) 50 cases   (~1 hour, conservative)"
echo "2) 100 cases  (~2 hours, RECOMMENDED)"
echo "3) 200 cases  (~4 hours, strong)"
echo "4) 500 cases  (~10 hours, comprehensive)"
echo "5) ALL cases  (~5 days, maximum)"
echo "6) Resume previous run"
echo ""
read -p "Enter choice [1-6]: " choice

case $choice in
    1)
        echo "Running 50-case validation..."
        python3 validate_with_checkpoints.py 50
        ;;
    2)
        echo "Running 100-case validation..."
        python3 validate_with_checkpoints.py 100
        ;;
    3)
        echo "Running 200-case validation..."
        python3 validate_with_checkpoints.py 200
        ;;
    4)
        echo "Running 500-case validation..."
        echo "TIP: Run in background with nohup for long runs"
        read -p "Run in background? [y/N]: " bg
        if [[ $bg =~ ^[Yy]$ ]]; then
            nohup python3 validate_with_checkpoints.py 500 > validation.log 2>&1 &
            echo "Running in background. PID: $!"
            echo "Monitor with: tail -f validation.log"
        else
            python3 validate_with_checkpoints.py 500
        fi
        ;;
    5)
        echo "Running ALL cases (6,020)..."
        echo "This will take ~5 days. Strongly recommend background mode."
        read -p "Run in background? [Y/n]: " bg
        if [[ ! $bg =~ ^[Nn]$ ]]; then
            nohup python3 validate_with_checkpoints.py > validation_full.log 2>&1 &
            echo "Running in background. PID: $!"
            echo "Monitor with: tail -f validation_full.log"
        else
            python3 validate_with_checkpoints.py
        fi
        ;;
    6)
        echo "Resuming previous run..."
        if [ -f validation_checkpoints/progress.json ]; then
            python3 validate_with_checkpoints.py
        else
            echo "No previous run found!"
        fi
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "Done! Check validation_checkpoints/ for results."

