#sudo ./../../bin/april-mlp.cuda.mkl test_cuda.lua
cd ../../
#lua -l formiga build_debug.lua
lua -l formiga build_cuda_and_mkl_release.lua
cd PFC_adrian_test/digitos
sudo ./../../bin/april-mlp.cuda.mkl test_cuda.lua
#./compare_files.sh red_blas.net red_cuda.net
