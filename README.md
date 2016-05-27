# Metagrobolizer
Chess AI with UCI compatibility.

Usage:
Install PyInstaller for Python 3.x (2.x won't work for obscure I/O threading reasons)  
Run "pyinstaller --onefile metagrobolizer.py" to create the executable  
Load the executable as an engine into your favourite UCI-compatible chess GUI, such as Arena or Tarrasch.

Changelog:
2016-05-18: recursive to iterative minimax: 44s to 37s for 3 moves at depth 3.  
2016-05-19: alpha beta pruning: 37s to 19s for 3 moves at depth 3.  
2016-05-24: replace deepcopy with custom copy constructor: 19s to 5s for 3 moves depth 3, 64s for 3 moves at depth 4.