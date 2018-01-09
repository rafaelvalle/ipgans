#!/bin/bash 

folder_path=$1
declare -a gans=("dcgan" "lsgan" "wgan" "wgan-gp")
declare -a bns=("True" "False")
declare -a etas=("True" "False")
declare -a opts=("adam" "rmsprop")
declare -a acts=("sigmoid" "scaled_tanh" "linear")
declare -a threshs=("0.0" "0.5")

for gan in "${gans[@]}"; do
  for bn in "${bns[@]}"; do
    for eta in "${etas[@]}"; do
      for opt in "${opts[@]}"; do
        for act in "${acts[@]}"; do
          for thresh in "${threshs[@]}"; do
            if [ -e ${folder_path}${gan}_mnist_hist_epoch_0_non_lin_${act}_opt_${opt}_bn_${bn}_etadecay_${eta}_thresh_${thresh}.png ]; then 
              echo ${folder_path}${gan}_mnist_hist_epoch_0_non_lin_${act}_opt_${opt}_bn_${bn}_etadecay_${eta}_thresh_${thresh}.png found
              convert \
                -resize 50% -delay 10 -loop 0 \
                $(for i in $(seq 0 1 199); do 
                  echo ${folder_path}${gan}_mnist_hist_epoch_${i}_non_lin_${act}_opt_${opt}_bn_${bn}_etadecay_${eta}_thresh_${thresh}.png; done) \
                ${folder_path}${gan}_mnist_hist_non_lin_${act}_opt_${opt}_bn_${bn}_etadecay_${eta}_thresh_${thresh}.gif
            else
              echo ${folder_path}${gan}_mnist_hist_epoch_0_non_lin_${act}_opt_${opt}_bn_${bn}_etadecay_${eta}_thresh_${thresh}.png missing 
            fi
          done
        done
      done
    done
  done
done
