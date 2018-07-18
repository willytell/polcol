# Con este fichero lanzamos un trabajo que nos permite ver que valor contendrá en el entorno
# CUDA_VISIBLE_DEVICES al solicitar la cantidad de gráficas que queramos.

#for e in `seq 3 9`; do
#   for k in `seq 0 8`; do
#      sbatch ./run-cluster.sh $e $k
#   done
#done

#sbatch ./run-cluster.sh 1 6
#sbatch ./run-cluster.sh 2 0

sbatch ./run-cluster.sh 5 5
sbatch ./run-cluster.sh 9 8
#
#sbatch ./run1.sh 1 0
#sbatch ./run1.sh 1 1
#sbatch ./run1.sh 1 2
#sbatch ./run1.sh 1 3
#sbatch ./run1.sh 1 4
#
#sbatch ./run2.sh 2 0
#sbatch ./run2.sh 2 1
#sbatch ./run2.sh 2 2
#sbatch ./run2.sh 2 3
#sbatch ./run2.sh 2 4
#
#sbatch ./run3.sh 3 0
#sbatch ./run3.sh 3 1
#sbatch ./run3.sh 3 2
#sbatch ./run3.sh 3 3
#sbatch ./run3.sh 3 4
#
#sbatch ./run4.sh 4 0
#sbatch ./run4.sh 4 1
#sbatch ./run4.sh 4 2
#sbatch ./run4.sh 4 3
#sbatch ./run4.sh 4 4
#
#sbatch ./run5.sh 5 0
#sbatch ./run5.sh 5 1
#sbatch ./run5.sh 5 2
#sbatch ./run5.sh 5 3
#sbatch ./run5.sh 5 4

#sbatch ./run6.sh 6 0
#sbatch ./run6.sh 6 1
#sbatch ./run6.sh 6 2
#sbatch ./run6.sh 6 3
#sbatch ./run6.sh 6 4

#sbatch ./run7.sh 7 0
#sbatch ./run7.sh 7 1
#sbatch ./run7.sh 7 2
#sbatch ./run7.sh 7 3
#sbatch ./run7.sh 7 4

#sbatch ./run8.sh 8 0
#sbatch ./run8.sh 8 1
#sbatch ./run8.sh 8 2
#sbatch ./run8.sh 8 3
#sbatch ./run8.sh 8 4

#sbatch ./run9.sh 9 0
#sbatch ./run9.sh 9 1
#sbatch ./run9.sh 9 2
#sbatch ./run9.sh 9 3
#sbatch ./run9.sh 9 4

