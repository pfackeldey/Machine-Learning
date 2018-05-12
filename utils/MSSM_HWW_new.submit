#!/usr/bin/env zsh
#BSUB -J MSSM.py.job
#BSUB -W 48:00
#BSUB -M 14000
#BSUB -n 11
#BSUB -o /home/mf278754/master/output/out.out
#BSUB -e /home/mf278754/master/output/out.err
#BSUB -a 'openmp'
#BSUB -gpu 'num=2'
#BSUB -R pascal
#BSUB -R gpu
#BSUB -P phys3b
###source /home/phys3b/Envs/keras_tf/bin/activate
source /home/phys3b/Envs/keras_tf_sharedUsers/bin/activate
export CUDA_VISIBLE_DEVICES=`/home/phys3b/etc/check_gpu.py 2`
if [ '$CUDA_VISIBLE_DEVICES' = '-1' ];
then
           echo '##### GPUs busy. Restart job later.' exit 1
else
           echo 'Found free GPU devices :'
           echo 'CUDA_VISIBLE_DEVICES =  $CUDA_VISIBLE_DEVICES '
fi
nvidia-smi

python /home/mf278754/master/Machine-Learning/training/train.py /home/mf278754/master/Machine-Learning/tasks/analysis/MSSM_HWW.yaml
