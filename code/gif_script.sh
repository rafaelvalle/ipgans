#!/bin/bash 

folder_path=$1
declare -a gans=("dcgan" "lsgan" "wgan" "wgan-gp")
declare -a bns=("True" "False")
declare -a etas=("True" "False")
declare -a opts=("adam" "rmsprop")
declare -a acts=("sigmoid" "scaled_tanh" "linear")
declare -a threshs=("0.0" "0.5")
declare -a noises=("uniform" "normal")

for gan in "${gans[@]}"; do
  for bn in "${bns[@]}"; do
    for eta in "${etas[@]}"; do
      for opt in "${opts[@]}"; do
        for act in "${acts[@]}"; do
          for thresh in "${threshs[@]}"; do
            for noise in "${noises[@]}"; do
              if [ -e ${folder_path}${gan}_mnist_hist_epoch_0_non_lin_${act}_opt_${opt}_bn_${bn}_etadecay_${eta}_thresh_${thresh}_noise_${noise}.png ]; then 
                echo ${folder_path}${gan}_mnist_hist_epoch_0_non_lin_${act}_opt_${opt}_bn_${bn}_etadecay_${eta}_thresh_${thresh}_noise_${noise}.png found
                convert \
                  -resize 50% -delay 10 -loop 0 \
                  $(for i in $(seq 0 1 199); do 
                    echo ${folder_path}${gan}_mnist_hist_epoch_${i}_non_lin_${act}_opt_${opt}_bn_${bn}_etadecay_${eta}_thresh_${thresh}_noise_${noise}.png; done) \
                  ${folder_path}${gan}_mnist_hist_non_lin_${act}_opt_${opt}_bn_${bn}_etadecay_${eta}_thresh_${thresh}_noise_${noise}.gif
              else
                echo ${folder_path}${gan}_mnist_hist_epoch_0_non_lin_${act}_opt_${opt}_bn_${bn}_etadecay_${eta}_thresh_${thresh}_noise_${noise}.png missing 
              fi
            done
          done
        done
      done
    done
  done
done
