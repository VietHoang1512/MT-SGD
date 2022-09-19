export PYTHONHASHSEED=0

# example

for num_nets in 3 5 7 9 15; do
    python train.py --dset multi_fashion_and_mnist \
        --output_scale 1 \
        --latent_scale 1 \
        --lr 1e-3 \
        --normalize \
        --n_epochs 100 \
        --batch_size 256 \
        --output_tradeoff .25 \
        --latent_tradeoff .0015 \
        --warmup_epoch 1 \
        --seed 0 \
        --num_nets $num_nets
done
