#!/bin/bash -x

# declare gans to be used
declare -a gans=("dcgan" "lsgan" "wgan" "wgan-gp")
declare -a thresholds=("0.0" "0.5")
declare -a activations=("sigmoid" "scaled_tanh" "linear")
declare -a optimizers=("adam" "rmsprop")
declare -a noises=("uniform" "normal")


# loop over gans and run gans
for gan in "${gans[@]}"; do
  for threshold in "${thresholds[@]}"; do
    for activation in "${activations[@]}"; do
      for optimizer in "${optimizers[@]}"; do
        for noise in "${noises[@]}"; do
          if [ "$gan" == "wgan-gp" ]; then
            ./gan_mnist.py "$gan" --optimizer $optimizer --threshold ${threshold} --activation $activation --noise_type $noise --eta_decay 
            ./gan_mnist.py "$gan" --optimizer $optimizer --threshold ${threshold} --activation $activation --noise_type $noise
          else
            ./gan_mnist.py "$gan" --optimizer $optimizer --do_batch_norm --threshold ${threshold} --activation $activation --noise_type $noise --eta_decay 
            ./gan_mnist.py "$gan" --optimizer $optimizer --do_batch_norm --threshold ${threshold} --activation $activation --noise_type $noise
          fi
        done
      done
    done
  done
done
