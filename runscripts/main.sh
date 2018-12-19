if [ -d source ]; then
        cd source
        git pull
else
        git clone https://github.com/AIJoris/DP-DialogueGAN source
        cd source
        # git fetch
        # git checkout pretrain-gen
fi

cd ../
sbatch job.sh
