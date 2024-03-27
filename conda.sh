# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/davide.cavicchini/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/davide.cavicchini/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/davide.cavicchini/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/davide.cavicchini/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/anaconda3/lib