#!/bin/bash                                                                                                                                                                                        

# Parameters   
recover=false                                                                                                                                                                                  
factor="randomness"
batch_size=2000
dataset="cifar10" # {cifar100,tinyimagenet}
architecture="Resnet18"
cycles=13
init_set_seed_order=0

# Filter configs
filter="${architecture}_${dataset}_I1000_B${batch_size}_c${cycles}_initseed${init_set_seed_order}"  
echo "filter: $filter"                                                                                                                                                                               
cfgs=$(ls -LR ./conf/$factor | tr ' ' '\n' | grep -P "$filter")
for cfg in $cfgs; do   
    # Paths
    method=${cfg%%_*}                                                                                                                                                                                     
    cfg_path="./conf/$factor/$cfg"
    log_dir="./experiments_paper/log/$factor"
    results_dir="./experiments_paper/results_txt/$factor/$dataset/$method"
    log_path="$log_dir/$(echo $cfg | sed 's/\.yml/.out/')"
    mkdir -p $log_dir

    # Dispatch slurm job                                                                                                                                                                           
    echo "Config: $cfg"

    # cmd="python3 main.py -c $cfg_path -warm 1" &
    if $recover; then
        state_path="$results_dir/$(echo $cfg | sed 's/\.yml/_latest_state.pkl/')"
        echo "state_path: $state_path"
        log_path="$results_dir/$(echo $cfg | sed 's/\.yml/_out.txt/')"

        python3 al_trainer.py -c $cfg_path -s $state_path >$log_path &
        echo "Recover from $state_path"
    else
        python3 al_trainer.py -c $cfg_path >$log_path &
    fi
    sleep 2
done