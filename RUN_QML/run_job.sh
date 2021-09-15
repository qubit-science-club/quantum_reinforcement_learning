#!/bin/bash

NAME=`echo "QML"`

PBS="#!/bin/bash\n\
#PBS -N ${NAME}\n\
#PBS -l walltime=72:00:00\n\
#PBS -l select=1:ncpus=8:mem=4000MB\n\
#PBS -l software=QML_DQN_FROZEN_LAKE.py\n\
#PBS -m n\n\
cd \$PBS_O_WORKDIR\n\

python3 QML_DQN_FROZEN_LAKE.py"

# Echo the string PBS to the function qsub, which submits it as a cluster job for you
# A small delay is included to avoid overloading the submission process

echo -e ${PBS} | qsub
#echo %{$PBS}
sleep 0.5
echo "done."
