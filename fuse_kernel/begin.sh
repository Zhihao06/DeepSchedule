
export NCCL_DEBUG=WARN
export ENABLE_NSYS=1
export ENABLE_NCU=0
export MODE=0
export ENABLE_TRAVERSE=0
export NSYS_FILE="deepfuse_seq_sm_mode$MODE"
export NCU_FILE="deepfuse_seq_sm_mode${MODE}_ncu"
export PROC=8

NSYS="
    nsys profile -o $NSYS_FILE --force-overwrite=true
"
NCU="
    ncu -o $NCU_FILE --force-overwrite --replay-mode app-range --target-processes all
"

if [ $ENABLE_NSYS -eq 1 ]; then
    $NSYS mpirun -np $PROC --allow-run-as-root --output-filename output_log --merge-stderr-to-stdout -H localhost:$PROC build/fuse_app $MODE $ENABLE_TRAVERSE
elif [ $ENABLE_NCU -eq 1 ]; then
    $NCU mpirun -np $PROC --allow-run-as-root --output-filename output_log --merge-stderr-to-stdout -H localhost:$PROC build/fuse_app $MODE $ENABLE_TRAVERSE
else
    mpirun -np $PROC --allow-run-as-root --output-filename output_log --merge-stderr-to-stdout -H localhost:$PROC build/fuse_app $MODE $ENABLE_TRAVERSE
fi
