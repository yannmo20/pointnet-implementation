CUDA_ROOT=/usr/local/cuda
TF_ROOT=/home/moellya/miniconda3/envs/tf14/lib/python3.6/site-packages/tensorflow

#/bin/bash
/usr/local/cuda/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.2
# g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I /home/moellya/miniconda3/envs/tf12/lib/python3.6/site-packages/tensorflow/include -I /usr/local/cuda/include -lcudart -L /usr/local/cuda/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.4
# g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I /home/moellya/miniconda3/envs/tf/lib/python3.6/site-packages/tensorflow/include -I /home/moellya/miniconda3/envs/tf/lib/python3.6/site-packages/tensorflow/include/external/nsync/public -I /home/moellya/miniconda3/envs/tf/lib/python3.6/site-packages/tensorflow/core -I /usr/local/cuda/include -lcudart -L /usr/local/cuda/lib64/ -L/home/moellya/miniconda3/envs/tf/lib/python3.6/site-packages/tensorflow/ -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0




# from zswang666:
g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I ${TF_ROOT}/include -I ${CUDA_ROOT}/include -I ${TF_ROOT}/include/external/nsync/public -lcudart -L ${CUDA_ROOT}/lib64/ -L ${TF_ROOT} -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0