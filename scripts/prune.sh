eval "$(conda shell.bash hook)"
conda activate RepDist

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$1

# default="$1"
# base=4
# trial=$(($default + $base))
trial=$1
STRAT="struct"
SPARSITY=$2

# for SPARSITY in 0.30 0.45 0.60 0.75 0.90
# do
#     python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
#     --distill crd --model_s resnet8x4 \
#     -a 0 -b 0.8 \
#     --trial $trial \
#     --target_sparsity $SPARSITY \
#     --strat $STRAT \
#     --batch_size 256 \
#     --learning_rate .05
# done

# python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
# --distill crd --model_s resnet8x4 \
# -a 0 -b 0.8 \
# --trial $trial \
# --target_sparsity $SPARSITY \
# --strat $STRAT \
# --batch_size 256 \
# --learning_rate .05

# python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
# --distill crd --model_s resnet8x4 \
# -a 0 -b 0.8 \
# --trial $trial \
# --target_sparsity $SPARSITY \
# --strat $STRAT \
# --batch_size 256 \
# --learning_rate .05


# for SPARSITY in 0.30 .45 0.60 0.75 0.90
# do
#         python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
#         --distill kd --model_s resnet8x4 \
#         -r 0.1 \
#         -a 0.9 \
#         -b 0 \
#         --trial $trial \
#         --target_sparsity $SPARSITY \
#         --strat $STRAT \
#         --batch_size 512 \
#         --learning_rate .05    
# done

# python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
# --distill kd --model_s resnet8x4 \
# -r 0.1 \
# -a 0.9 \
# -b 0 \
# --trial $trial \
# --target_sparsity $SPARSITY \
# --strat $STRAT \
# --batch_size 512 \
# --learning_rate .05 \
# --bias True

# for SPARSITY in 0.30 0.45 0.60 0.75 0.90
# do
#     python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
#     --batch_size 256 \
#     --distill attention \
#     -a 0 \
#     -b 1000 \
#     --trial $trial \
#     --target_sparsity $SPARSITY \
#     --strat $STRAT \
#     --learning_rate .005
# done

# python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
# --batch_size 256 \
# --distill attention \
# -a 0 \
# -b 1000 \
# --trial $trial \
# --target_sparsity $SPARSITY \
# --strat $STRAT \
# --bias True \
# --learning_rate .005

# python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
# --batch_size 256 \
# --distill attention \
# -a 1 \
# -b 1000 \
# --trial $trial \
# --target_sparsity $SPARSITY \
# --strat $STRAT \
# --bias True \
# --learning_rate .005

# python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
# --batch_size 256 \
# --distill attention \
# -a 1 \
# -b 1000 \
# --trial $trial \
# --target_sparsity $SPARSITY \
# --strat $STRAT \
# --learning_rate .005

# for SPARSITY in 0.30 0.45 0.60 0.75 0.90
# do
#     python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
#     --distill pkt \
#     --model_s resnet8x4 \
#     -a 0 \
#     -b 30000 \
#     --trial $trial \
#     --learning_rate .005 \
#     --target_sparsity $SPARSITY \
#     --strat $STRAT \
#     --batch_size 256
# done

# python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
# --distill pkt \
# --model_s resnet8x4 \
# -a 0 \
# -b 30000 \
# --trial $trial \
# --learning_rate .005 \
# --target_sparsity $SPARSITY \
# --strat $STRAT \
# --bias True \
# --batch_size 256

# python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
# --distill pkt \
# --model_s resnet8x4 \
# -a 1 \
# -b 30000 \
# --trial 0 \
# --learning_rate .005 \
# --target_sparsity $SPARSITY \
# --strat $STRAT \
# --bias True \
# --batch_size 256

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
--distill pkt \
--model_s resnet8x4 \
-a 1 \
-b 30000 \
--trial $trial \
--learning_rate .005 \
--target_sparsity $SPARSITY \
--strat $STRAT \
--batch_size 256

# for SPARSITY in 0.30 0.45 0.60 0.75 0.90
# do
#     python train_teacher.py \
#     --batch_size 512 \
#     --learning_rate .05 \
#     --model resnet32x4 \
#     --trial $trial \
#     --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
#     --target_sparsity $SPARSITY \
#     --strat struct
# done

# python train_teacher.py \
# --batch_size 512 \
# --learning_rate .05 \
# --model resnet32x4 \
# --trial $trial \
# --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
# --target_sparsity $SPARSITY \
# --bias True \
# --strat struct

# for SPARSITY in 0.30 0.45 0.60 0.75 0.90
# do
#     python train_student.py \
#     --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
#     --distill similarity \
#     --model_s resnet8x4 \
#     -a 0 -b 3000 \
#     --trial $trial \
#     --batch_size 512 \
#     --target_sparsity $SPARSITY \
#     --strat struct --learning_rate 0.05
# done

# python train_student.py \
# --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
# --distill similarity \
# --model_s resnet8x4 \
# -a 0 -b 3000 \
# --trial $trial \
# --batch_size 512 \
# --target_sparsity $SPARSITY \
# --strat struct --learning_rate 0.05 \
# --bias True

# python train_student.py \
# --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
# --distill similarity \
# --model_s resnet8x4 \
# -a 1 -b 3000 \
# --trial 0 \
# --batch_size 512 \
# --target_sparsity $SPARSITY \
# --strat struct --learning_rate 0.05 \
# --bias True

# python train_student.py \
# --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
# --distill similarity \
# --model_s resnet8x4 \
# -a 1 -b 3000 \
# --trial $trial \
# --batch_size 512 \
# --target_sparsity $SPARSITY \
# --strat struct --learning_rate 0.05 \
#SPARSITY=0.30
# normalized learning rate (in awk)
# test out "our method with differnt alpha/beta mix"
# for SPARSITY in 0.30 0.45 0.60 0.75 #0.90
# do
#     python train_student.py \
#     --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
#     --distill all \
#     --model_s resnet8x4 \
#     -r 1 \
#     -a 0 -b 1 \
#     --trial $trial \
#     --batch_size 512 \
#     --target_sparsity $SPARSITY \
#     --strat struct --learning_rate 0.05
# done

# python train_student.py \
# --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
# --distill all \
# --model_s resnet8x4 \
# -r 1 \
# -a 0 -b 1 \
# --trial $trial \
# --batch_size 512 \
# --target_sparsity $SPARSITY \
# --strat struct --learning_rate 0.05 \
# --bias True

# python train_student.py \
# --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
# --distill all \
# --model_s resnet8x4 \
# -r 1 \
# -a 0 -b 1 \
# --trial 0 \
# --batch_size 512 \
# --target_sparsity $SPARSITY \
# --strat struct --learning_rate 0.05 \

# python train_student.py \
# --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
# --distill all \
# --model_s resnet8x4 \
# -r 1 \
# -a 1 -b 1 \
# --trial 0 \
# --batch_size 512 \
# --target_sparsity $SPARSITY \
# --strat struct --learning_rate 0.05 \
# --bias True

python train_student.py \
--path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
--distill all \
--model_s resnet8x4 \
-r 1 \
-a 1 -b 1 \
--trial $trial \
--batch_size 512 \
--target_sparsity $SPARSITY \
--strat struct --learning_rate 0.05 \
# #for SPARSITY in 0.30 0.45 0.60 0.75 0.90

# # test out "our method with differnt alpha/beta mix"
# # for LR in 0.05 0.005
# # do
# #     python train_student.py \
# #     --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
# #     --distill all \
# #     --model_s resnet8x4 \
# #     -r $2 \
# #     -a 0 -b 1 \
# #     --trial $1 \
# #     --batch_size 512 \
# #     --target_sparsity $SPARSITY \
# #     --strat struct --learning_rate $LR 
# # done