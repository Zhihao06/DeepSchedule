
export NCCL_DEBUG=WARN
export ENABLE_NSYS=1
export ENABLE_NCU=0
export MODE=0
export ENABLE_TRAVERSE=0
export LAUNCH_TYPE="sync" # sync or sched
export NUM_SPLITS="" # split with ",", like "256,256"
export NSYS_FILE="deepfuse_seq_sm_mode${MODE}_launch${LAUNCH_TYPE}_splits${NUM_SPLITS}"
export NCU_FILE="deepfuse_seq_sm_mode${MODE}_ncu"
export PROC=8
export LD_LIBRARY_PATH=/mnt/data/nas/zhihao/zhl-sglang/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH # your torch lib path

NSYS="
    nsys profile -o $NSYS_FILE --force-overwrite=true
"
NCU="
    ncu -o $NCU_FILE --force-overwrite --replay-mode app-range --target-processes all
"

if [ $ENABLE_NSYS -eq 1 ]; then
    $NSYS mpirun -np $PROC --allow-run-as-root --output-filename output_log --merge-stderr-to-stdout -H localhost:$PROC build/fuse_app $MODE $ENABLE_TRAVERSE $LAUNCH_TYPE $NUM_SPLITS
elif [ $ENABLE_NCU -eq 1 ]; then
    $NCU mpirun -np $PROC --allow-run-as-root --output-filename output_log --merge-stderr-to-stdout -H localhost:$PROC build/fuse_app $MODE $ENABLE_TRAVERSE $LAUNCH_TYPE $NUM_SPLITS
else
    mpirun -np $PROC --allow-run-as-root --output-filename output_log --merge-stderr-to-stdout -H localhost:$PROC build/fuse_app $MODE $ENABLE_TRAVERSE $LAUNCH_TYPE $NUM_SPLITS
fi
