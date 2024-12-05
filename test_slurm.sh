#!/bin/bash
#SBATCH -J val # 指定作业名
#SBATCH -n 1 # 指定核心数量，不需要更改
#SBATCH -N 1 # 指定node的数量，不需要更改
#SBATCH -t 1-10:00 # 运行总时间，天数-小时数-分钟， D-HH:MM，最长时间限制为一
#SBATCH -p normal # 提交到哪一个分区（指定队列），不需要更改
#SBATCH --mem-per-cpu=4G # 每个CPU分配的内存，总内存=CPU数*该值，总共124G内存
#SBATCH --cpus-per-task=1 # 需要使用多少CPU，总共有24个CPU。建议使用12个CPU。
#SBATCH -o slurm.output2 # 把输出结果STDOUT保存在哪一个文件
#SBATCH -e slurm.error2 # 把报错结果STDERR保存在哪一个文件
#SBATCH --gres=gpu:1 # 需要使用多少个GPU，总共有4个GPU。建议使用1个GPU。

source /home/liyunfei/SeqFakeFormer/venu/bin/activate
CUDA_VISIBLE_DEVICES=0 python test.py \
    --log_name  test_lr2\
    --cfg  "/home/liyunfei/SeqFakeFormer/code/SeqDeepFake-master_2/configs/r34.json"\
    --data_dir "/home/liyunfei/SeqFakeFormer/dataset"\
    --dataset_name  'facial_components'\
    --test_type  'adaptive'\
    --world_size 1 \
    --rank 0 \
    --launcher pytorch \
    --results_dir "/home/liyunfei/SeqFakeFormer/code/SeqDeepFake-master_2/results"