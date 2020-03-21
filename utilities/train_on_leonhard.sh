#!/bin/bash

# Put this file into the same directory where your training.py file is located.
# Execute by typing './train_on_leonhard.sh'.
# If you get a message: 'permission denied: ./train_on_leonhard.sh' then
# run 'chmod +x train_on_leonhard.sh' to give execution rights to the file, and
# try again: './train_on_leonhard.sh'.

# If you only want to start a short job (that gets scheduled faster) to test
# something, add the 't' option:
# './train_on_leonhard.sh t'.

if [[ $1 = "t" ]]; then
  echo "Deploying test job..."
  bsub -n 2 -W 4:00 -o log_test -R "rusage[mem=2048, ngpus_excl_p=1]" python training.py
else
  echo "Deploying long running job!"
  bsub -n 4 -W 24:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python training.py
fi