#!/bin/bash
# usage: bash run_train_pepbench_physics.sh <GPU> <PORT> <WDB> <MODE> [PHYSICS_FLAG] [RESUME]
# PHYSICS_FLAG: 1 for physics-enhanced (default), 0 for baseline
# RESUME: 1 to resume from latest checkpoint (default), 0 to start fresh

########## Parse arguments from wrapper script ##########
GPU=$1
PORT=$2
WDB=$3
MODE=$4
PHYSICS_FLAG=${5:-1}  # Default to physics enabled
RESUME=${6:-1}  # Default to resume if checkpoint exists
BATCH_SIZE=${7:-16}
EPOCHS=${8:-700}
########## setup project directory ##########
CODE_DIR=`pwd`
echo "Locate the project folder at ${CODE_DIR}"
cd ${CODE_DIR}

########## Set hardcoded pepbench configs ##########
NAME=$WDB
AECONFIG=*PATH*/train_codesign.yaml
LDMCONFIG=*PATH*/ldm/train_codesign.yaml
LATENT_DIST_CONFIG=*PATH*/ldm/setup_latent_guidance.yaml
TEST_CONFIG=*PATH*/test_codesign.yaml

echo "Mode: $MODE, [train AE] / [train LDM] / [Generate] / [Evalulation]"
if [ "$PHYSICS_FLAG" = "1" ]; then
    echo "Physics Enhancement: ENABLED"
    NAME="${NAME}"
else
    echo "Physics Enhancement: DISABLED"
    NAME="${NAME}"
fi

TRAIN_AE_FLAG=${MODE:0:1}
TRAIN_LDM_FLAG=${MODE:1:1}
GENERATE_FLAG=${MODE:2:1}
EVAL_FLAG=${MODE:3:1}

AE_SAVE_DIR=./exps/$NAME/AE
LDM_SAVE_DIR=./exps/$NAME/LDM
OUTLOG=./exps/$NAME/output.log

# Function to get the highest existing version_* directory (fallback: version_0)
get_latest_version() {
    local base_dir=$1
    local highest=-1
    local dir
    for dir in "$base_dir"/version_*; do
        [ -d "$dir" ] || continue
        local name="${dir##*/}"
        local num="${name#version_}"
        if [[ "$num" =~ ^[0-9]+$ ]]; then
            if (( num > highest )); then
                highest=$num
            fi
        fi
    done
    if (( highest >= 0 )); then
        echo "version_${highest}"
    else
        echo "version_0"
    fi
}

# Function to get the latest checkpoint from a directory
get_latest_checkpoint() {
    local ckpt_dir=$1
    local version=$(get_latest_version "$ckpt_dir")
    if [ -f "${ckpt_dir}/${version}/checkpoint/topk_map.txt" ]; then
        # Get the best checkpoint (first line in topk_map.txt)
        cat "${ckpt_dir}/${version}/checkpoint/topk_map.txt" | head -n 1 | awk -F " " '{print $2}'
    else
        echo ""
    fi
}

if [[ ! -e ./exps/$NAME ]]; then
    mkdir -p ./exps/$NAME
elif [[ -e $AE_SAVE_DIR ]] && [ "$TRAIN_AE_FLAG" = "1" ] && [ "$RESUME" = "0" ]; then
    echo "Directory ${AE_SAVE_DIR} exists! But training flag is 1 and RESUME is 0!"
    echo "To start fresh training, remove the existing directory or set a different experiment name."
    echo "To resume training, set RESUME=1 (or omit it for default)."
    exit 1;
elif [[ -e $LDM_SAVE_DIR ]] && [ "$TRAIN_LDM_FLAG" = "1" ] && [ "$RESUME" = "0" ]; then
    echo "Directory ${LDM_SAVE_DIR} exists! But training flag is 1 and RESUME is 0!"
    echo "To start fresh training, remove the existing directory or set a different experiment name."
    echo "To resume training, set RESUME=1 (or omit it for default)."
    exit 1;
fi

######### train autoencoder ##########
echo "Training Autoencoder with config $AECONFIG:" > $OUTLOG
cat $AECONFIG >> $OUTLOG
if [ "$TRAIN_AE_FLAG" = "1" ]; then
    # Check if we should resume from existing checkpoint
    AE_RESUME_CKPT=""
    AE_VERSION=$(get_latest_version "$AE_SAVE_DIR")
    if [ "$RESUME" = "1" ] && [ -f "${AE_SAVE_DIR}/${AE_VERSION}/checkpoint/topk_map.txt" ]; then
        AE_RESUME_CKPT=$(get_latest_checkpoint "$AE_SAVE_DIR")
        echo "Resuming AE training from checkpoint: ${AE_RESUME_CKPT}" >> $OUTLOG
        GPU="$GPU" PORT="$PORT" bash scripts/train.sh $AECONFIG --trainer.config.save_dir=$AE_SAVE_DIR --dataloader.batch_size=$BATCH_SIZE --load_ckpt=$AE_RESUME_CKPT
    else
        echo "Starting fresh AE training" >> $OUTLOG
        GPU="$GPU" PORT="$PORT" bash scripts/train.sh $AECONFIG --trainer.config.save_dir=$AE_SAVE_DIR --dataloader.batch_size=$BATCH_SIZE
    fi
fi

######### train ldm ##########
echo "Training LDM with config $LDMCONFIG:" >> $OUTLOG
echo "Physics Enhancement: $([ "$PHYSICS_FLAG" = "1" ] && echo "ENABLED" || echo "DISABLED")" >> $OUTLOG
cat $LDMCONFIG >> $OUTLOG

# Get autoencoder checkpoint
AE_VERSION=$(get_latest_version "$AE_SAVE_DIR")
if [ "$TRAIN_AE_FLAG" = "1" ] || [ -f "${AE_SAVE_DIR}/${AE_VERSION}/checkpoint/topk_map.txt" ]; then
    AE_CKPT=`cat ${AE_SAVE_DIR}/${AE_VERSION}/checkpoint/topk_map.txt | head -n 1 | awk -F " " '{print $2}'`
else
    # Use existing autoencoder if available
    EXISTING_AE_VERSION=$(get_latest_version "./exps/${WDB}/AE")
    if [ -f "./exps/${WDB}/AE/${EXISTING_AE_VERSION}/checkpoint/topk_map.txt" ]; then
        AE_CKPT=`cat ./exps/${WDB}/AE/${EXISTING_AE_VERSION}/checkpoint/topk_map.txt | head -n 1 | awk -F " " '{print $2}'`
        echo "Using existing autoencoder from ./exps/${WDB}/AE/"
    else
        echo "Error: No autoencoder checkpoint found!"
        echo "Either train autoencoder first (set first digit of MODE to 1) or ensure existing checkpoint exists"
        exit 1
    fi
fi
echo "Using Autoencoder checkpoint: ${AE_CKPT}" >> $OUTLOG

if [ "$TRAIN_LDM_FLAG" = "1" ]; then
    # Check if we should resume from existing checkpoint
    LDM_RESUME_CKPT=""
    RESUME_ARG=""
    LDM_VERSION=$(get_latest_version "$LDM_SAVE_DIR")
    if [ "$RESUME" = "1" ] && [ -f "${LDM_SAVE_DIR}/${LDM_VERSION}/checkpoint/topk_map.txt" ]; then
        LDM_RESUME_CKPT=$(get_latest_checkpoint "$LDM_SAVE_DIR")
        echo "Resuming LDM training from checkpoint: ${LDM_RESUME_CKPT}" >> $OUTLOG
        RESUME_ARG="--load_ckpt=$LDM_RESUME_CKPT"
    else
        echo "Starting fresh LDM training" >> $OUTLOG
    fi
    
    if [ "$PHYSICS_FLAG" = "1" ]; then
        # Train with physics enhancement (default config already has physics enabled)
        GPU="$GPU" PORT="$PORT" bash scripts/train.sh $LDMCONFIG --trainer.config.save_dir=$LDM_SAVE_DIR --model.autoencoder_ckpt=$AE_CKPT $RESUME_ARG --trainer.config.max_epoch=$EPOCHS
    else
        # Train baseline model (disable physics)
        GPU="$GPU" PORT="$PORT" bash scripts/train.sh $LDMCONFIG --trainer.config.save_dir=$LDM_SAVE_DIR --model.autoencoder_ckpt=$AE_CKPT --model.use_physics_loss=false $RESUME_ARG --trainer.config.max_epoch=$EPOCHS
    fi
fi

########## get latent distance ##########
LDM_VERSION=$(get_latest_version "$LDM_SAVE_DIR")
LDM_CKPT=`cat ${LDM_SAVE_DIR}/${LDM_VERSION}/checkpoint/topk_map.txt | head -n 1 | awk -F " " '{print $2}'`
echo "Get distances in latent space" >> $OUTLOG
python setup_latent_guidance.py --config ${LATENT_DIST_CONFIG} --model_config ${LDM_SAVE_DIR}/${LDM_VERSION}/train_config.yaml --ckpt ${LDM_CKPT} --gpu ${GPU:0:1} 

########## generate ##########
echo "Generate results Using LDM checkpoint: ${LDM_CKPT} gpu ${GPU:0:1}" 
if [ "$GENERATE_FLAG" = "1" ]; then
    CUDA_VISIBLE_DEVICES=$1 python generate.py --config $TEST_CONFIG --ckpt $LDM_CKPT --gpu 0
fi

########## cal metrics ##########
if [ "$EVAL_FLAG" = "1" ]; then
    echo "Evaluation:" >> $OUTLOG
    python cal_metrics.py --results ${LDM_SAVE_DIR}/${LDM_VERSION}/results/results.jsonl 
fi 

python -m evaluation.dG.run --results ${LDM_SAVE_DIR}/${LDM_VERSION}/results/results.jsonl --n_sample 40

echo "Training completed!"
echo "Physics Enhancement: $([ "$PHYSICS_FLAG" = "1" ] && echo "ENABLED" || echo "DISABLED")"
echo "Resume Mode: $([ "$RESUME" = "1" ] && echo "ENABLED" || echo "DISABLED")"
echo "Results saved in: ${LDM_SAVE_DIR}" 