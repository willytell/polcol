# Con este fichero lanzamos un trabajo que nos permite ver que valor contendrá en el entorno
# CUDA_VISIBLE_DEVICES al solicitar la cantidad de gráficas que queramos.

for k in `seq 0 8 `; do
   for e in `seq 0 9`; do
      sbatch ./run-cluster.sh $e $k
   done
done

#sbatch ./run-cluster.sh 0 2
#sbatch ./run-cluster.sh 2 3
#sbatch ./run-cluster.sh 3 3
#sbatch ./run-cluster.sh 8 0
#sbatch ./run-cluster.sh 9 4
#sbatch ./run-cluster.sh 9 7

# test
#sbatch ./run-cluster.sh 0 5
#sbatch ./run-cluster.sh 1 4
#sbatch ./run-cluster.sh 2 4
#sbatch ./run-cluster.sh 3 0
#sbatch ./run-cluster.sh 4 0
#sbatch ./run-cluster.sh 5 0
#sbatch ./run-cluster.sh 6 5
#sbatch ./run-cluster.sh 7 5
#sbatch ./run-cluster.sh 8 6
#sbatch ./run-cluster.sh 9 6
