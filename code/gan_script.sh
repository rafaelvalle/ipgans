#!/bin/bash -x

# declare gans to be used
declare -a gans=("dcgan" "lsgan" "wgan" "wgan-gp")
declare -a thresholds=("0.5" "0.0")
declare -a activations=("sigmoid" "scaled_tanh" "linear")
declare -a optimizers=("adam" "rmsprop")


# loop over gans and run gans
for gan in "${gans[@]}"; do
  for threshold in "${thresholds[@]}"; do
    for activation in "${activations[@]}"; do
      for optimizer in "${optimizers[@]}"; do
        if [ "$gan" == "wgan-gp" ]; then
          ./gan_mnist.py "$gan" --optimizer $optimizer --threshold ${threshold} --activation $activation --eta_decay 
          ./gan_mnist.py "$gan" --optimizer $optimizer --threshold ${threshold} --activation $activation
        else
          ./gan_mnist.py "$gan" --optimizer $optimizer --do_batch_norm --threshold ${threshold} --activation $activation --eta_decay 
          ./gan_mnist.py "$gan" --optimizer $optimizer --do_batch_norm --threshold ${threshold} --activation $activation
        fi
      done
    done
  done
done
