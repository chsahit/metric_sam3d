notes on install
- `git submodules update --init --recursive`
- follow sam3d-objects docs
- follow scenecomplete docs
- run setup_envs_properly.sh to fix some issues with the scaling script

This repo assumes the conda env can be found under /home/$USER/miniconda3, if this is not the case update the pipeline shell script accordingly where `LD_LIBRARY_PATH` is redefined

TODO: 
- serve as an endpoint
- add flag for right cuda device
- multiple masks
- compute masks with SAM3
- compute masks with GPT + SAM{2, 3}
- "cheap" endpoint using built in pointmap
