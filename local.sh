mkdir -p ../tests/tests-local
mkdir -p ../results/results-local

for i in $(seq 1 1 100); do
    for k in 16 32 64 128 512 1024; do 
      # Gerar arquivo de entra\da para a multiplicação de matriz
      echo -e "$k\n$i" > ../tests/tests-local/$i.in
      # # Executar a multiplicação de matriz de forma serial e adicionar os resultados em serial.out
      # ./serial ../tests/tests-local/$i.in >> ../results/results-local/serial-$k.out

      # for p in 2 4; do
      #   # Executar a multiplicação de matriz usando MPI  e adicionar os resultados em parallel.out
      #   mpirun -np $p ./parallel ../tests/tests-local/$i.in >> ../results/results-local/parallel-$p-$k.out

      #   # Executar a multiplicação de matriz usando MPI + BLAS e adicionar os resultados em parallel_blas.out
      #   mpirun -np $p ./parallel_blas ../tests/tests-local/$i.in >> ../results/results-local/blas-$p-$k.out
    # done
  done
done