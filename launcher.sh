# Con este fichero lanzamos un trabajo que nos permite ver que valor contendrá en el entorno
# CUDA_VISIBLE_DEVICES al solicitar la cantidad de gráficas que queramos.

#for e in `seq 6 9`; do
#for e in `seq 3 5`; do
#for e in `seq 0 2 `; do
#   for k in `seq 0 8`; do
#      sbatch ./run-cluster.sh $e $k
#   done
#done

for k in `seq 0 8 `; do
   for e in `seq 0 9`; do
      sbatch ./run-cluster.sh $e $k
   done
done

#sbatch ./run-cluster.sh 0 0
#sbatch ./run-cluster.sh 1 0
#sbatch ./run-cluster.sh 3 4
#sbatch ./run-cluster.sh 7 4
#sbatch ./run-cluster.sh 9 4
#sbatch ./run-cluster.sh 9 7

# test
#sbatch ./run-cluster.sh 0 6
#sbatch ./run-cluster.sh 1 4
#sbatch ./run-cluster.sh 2 4
#sbatch ./run-cluster.sh 3 4
#sbatch ./run-cluster.sh 4 6
#sbatch ./run-cluster.sh 5 0
#sbatch ./run-cluster.sh 6 0
#sbatch ./run-cluster.sh 7 0
#sbatch ./run-cluster.sh 8 4
#sbatch ./run-cluster.sh 9 4
