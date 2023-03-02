# Python Agent for Infinite Mario

Steps to get running:
* Download this repo.
* Open the project in Eclipse via File -> Open Projects from File System...
* Run the server from Eclipse by specifying '-server on' in the program arguments. One issue I had is that the default working directory (in the second screenshot below) was somehow set incorrectly. Overriding it by clicking *Other* and specifying the "classes" subdirectory solved the problem.

<img src="doc/run_config_1.png" width="800" height="377" />
<img src="doc/run_config_2.png" width="800" height="508" />

* Open a terminal in the 'src' directory.
* Activate a Python 2 conda environment if necessary.
* ```python python/competition/ipymario.py ```
* This should launch the custom 'MichaelAgent' I've created that randomly runs left and right.
