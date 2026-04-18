for dataset in terra_incognita ; do # Options: terra_incognita, PACS, VLCS, office-home 
    for bs in 16; do 
        for lr in 3e-4; do
            for seed in 42; do
                for num_steps in 1 6 12; do
                    for test_env in 0 1 2 3; do
                        python3 -m torch.distributed.launch \
                            --nproc_per_node=1 \
                            --master_port 11773 main.py \
                            --cfg ./configs/vmambav2v_tiny_224.yaml \
                            --batch-size ${bs} \
                            --use-checkpoint \
                            --data-path /path/to/dataset/ \
                            --dataset ${dataset} \
                            --output ./outputs/output${seed}-test \
                            --num_steps ${num_steps} \
                            --test_envs ${test_env} \
                            --lr ${lr} \
                            --seed ${seed} \
                            # --eval /path/to/CrossFlowDG_ckpt/
                    done
                done
            done
        done
    done
done
