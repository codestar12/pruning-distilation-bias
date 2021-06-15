for SPARSITY in 0.45 0.60 0.75 0.90
do

    python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
            --distill kd \
            --model_s resnet8x4 \
            -r 0.1 \
            -a 0.9 \
            -b 0 \
            --trial 1 \
            --target_sparsity $SPARSITY
    
done