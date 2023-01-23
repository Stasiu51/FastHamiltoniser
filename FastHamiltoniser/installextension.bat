cd "C:\Users\Stasiu Wolanski\Projects\FastHamiltoniser\FastHamiltoniser\"
..\venv\Scripts\python.exe setup.py install
copy CUDA\cudart64_12.dll ..\venv\lib\site-packages\FastHamiltoniser-1.0-py3.11-win-amd64.egg
echo Finished Install!