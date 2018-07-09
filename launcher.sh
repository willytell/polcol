# Con este fichero lanzamos un trabajo que nos permite ver que valor contendrá en el entorno
# CUDA_VISIBLE_DEVICES al solicitar la cantidad de gráficas que queramos.

#SBATCH -n 4 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -D /tmp # working directory
#SBATCH -t 0-00:05 # Runtime in D-HH:MM
#SBATCH -p dcc # Partition to submit to
#SBATCH --mem 2048 # 2GB solicitados.
#SBATCH --gres gpu:Pascal:1 # Para pedir Pascales MAX 8
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err # File to which STDERR will be written


sbatch -D /home/csanchez/polcol -t 5-10:05 --mem 8192 --gres=gpu:Pascal:1 ./run.sh
