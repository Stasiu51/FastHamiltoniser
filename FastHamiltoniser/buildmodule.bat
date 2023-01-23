cd "C:\Users\Stasiu Wolanski\Projects\FastHamiltoniser\FastHamiltoniser\"
del /s /Q build
del /s /Q dist
del /Q CUDA\build\*
nvcc -c -odir CUDA\build\ CUDA\source\cuda_source.cu CUDA\source\error_utils.cpp
lib /out:CUDA\build\cuda_lib.lib CUDA\build\*
..\venv\Scripts\python.exe setup.py build
echo Finished build!