for seed in 0 1 2 3 4; do
    python train.py --scale .01 \
        --normalize \
        --warmup_epochs 10 \
        --num_nets 5 \
        --tradeoff .1 \
        --seed $seed
done
