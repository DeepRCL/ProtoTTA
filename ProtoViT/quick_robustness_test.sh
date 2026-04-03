#!/bin/bash
# Quick robustness testing script for ProtoViT
# This script provides convenient shortcuts for common robustness evaluation tasks

set -e  # Exit on error

# Default values
MODEL="./saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth"
GPUID="0"
MODE="all"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  generate     Generate corrupted CUB-C dataset"
    echo "  evaluate     Evaluate model on all corruptions"
    echo "  quick        Quick test on 3 corruptions"
    echo "  single       Test single corruption"
    echo "  compare      Compare different models"
    echo ""
    echo "Options:"
    echo "  --model PATH        Path to model file"
    echo "  --gpuid ID          GPU ID (default: 0)"
    echo "  --mode MODE         Evaluation mode: normal,tent,proto,fisher,all (default: all)"
    echo ""
    echo "Examples:"
    echo "  $0 generate"
    echo "  $0 evaluate --model saved_models/best.pth"
    echo "  $0 quick --mode normal,tent"
    echo "  $0 single gaussian_noise 3"
}

# Parse arguments
COMMAND=$1
shift || true

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --gpuid)
            GPUID="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

case $COMMAND in
    generate)
        echo -e "${BLUE}=== Generating CUB-200-C Dataset ===${NC}"
        echo "This will take ~1-2 hours and use ~50-100GB of disk space"
        echo ""
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python create_cub_c.py \
                --input_dir ./datasets/cub200_cropped/test_cropped/ \
                --output_dir ./datasets/cub200_c/ \
                --corruption all \
                --severity 1 2 3 4 5
            echo -e "${GREEN}✓ Dataset generation complete!${NC}"
        fi
        ;;
    
    evaluate)
        echo -e "${BLUE}=== Full Robustness Evaluation ===${NC}"
        echo "Model: $MODEL"
        echo "Mode: $MODE (normal, tent, proto, loss, fisher)"
        echo "GPU: $GPUID"
        echo ""
        
        OUTPUT="./results/robustness_$(date +%Y%m%d_%H%M%S).json"
        mkdir -p ./results
        
        python evaluate_robustness.py \
            --model "$MODEL" \
            --data_dir ./datasets/cub200_c/ \
            --mode "$MODE" \
            --gpuid "$GPUID" \
            --eval_clean \
            --output "$OUTPUT"
        
        echo -e "${GREEN}✓ Evaluation complete!${NC}"
        echo "Results saved to: $OUTPUT"
        ;;
    
    quick)
        echo -e "${BLUE}=== Quick Robustness Test ===${NC}"
        echo "Testing 3 representative corruptions (noise, blur, weather)"
        echo "Model: $MODEL"
        echo "Mode: $MODE"
        echo ""
        
        OUTPUT="./results/quick_test_$(date +%Y%m%d_%H%M%S).json"
        mkdir -p ./results
        
        python evaluate_robustness.py \
            --model "$MODEL" \
            --corruptions gaussian_noise defocus_blur fog \
            --severities 3 5 \
            --mode "$MODE" \
            --gpuid "$GPUID" \
            --eval_clean \
            --output "$OUTPUT"
        
        echo -e "${GREEN}✓ Quick test complete!${NC}"
        echo "Results saved to: $OUTPUT"
        ;;
    
    single)
        CORRUPTION=$EXTRA_ARGS
        if [ -z "$CORRUPTION" ]; then
            echo -e "${RED}Error: Please specify corruption type${NC}"
            echo "Usage: $0 single <corruption_type> [severity]"
            echo "Example: $0 single gaussian_noise 3"
            exit 1
        fi
        
        echo -e "${BLUE}=== Testing Single Corruption ===${NC}"
        echo "Corruption: $CORRUPTION"
        echo "Model: $MODEL"
        echo "Mode: $MODE"
        echo ""
        
        python run_inference.py \
            -model "$MODEL" \
            -corruption $CORRUPTION \
            -gpuid "$GPUID" \
            -mode "$MODE"
        ;;
    
    compare)
        echo -e "${BLUE}=== Comparing Multiple Models ===${NC}"
        echo "This will evaluate all finetuned models in saved_models/"
        echo ""
        
        mkdir -p ./results/comparison
        
        for model_file in ./saved_models/deit_small_patch16_224/exp1/*finetuned*.pth; do
            if [ -f "$model_file" ]; then
                echo -e "${GREEN}Evaluating: $(basename $model_file)${NC}"
                
                OUTPUT="./results/comparison/$(basename $model_file .pth)_robustness.json"
                
                python evaluate_robustness.py \
                    --model "$model_file" \
                    --data_dir ./datasets/cub200_c/ \
                    --mode "$MODE" \
                    --gpuid "$GPUID" \
                    --eval_clean \
                    --output "$OUTPUT" \
                    --corruptions gaussian_noise shot_noise contrast brightness defocus_blur
                
                echo ""
            fi
        done
        
        echo -e "${GREEN}✓ Comparison complete!${NC}"
        echo "Results saved to: ./results/comparison/"
        ;;
    
    *)
        echo -e "${RED}Error: Unknown command '$COMMAND'${NC}"
        echo ""
        print_usage
        exit 1
        ;;
esac

