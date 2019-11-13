import numpy as np

sizes = [(2, 2), (2, 3), (2, 4), (3, 3), (2, 5), (3, 4), (4, 4), (3, 6), (4, 5), (3, 7), (3, 8),
            (5, 5), (3, 9), (4, 7), (5, 6), (4, 8), (5, 7), (6, 6), (6, 7), (7, 7), (7, 8), (8, 8)]

Js = [-2, -1, -0.5, 0, 0.5, 1, 2]

fi = open('tasks','w')

for J in Js:
	for s in sizes:
		nX = s[0]
		nY = s[1]

		name = str(J) + '_' + str(nX) + '_' + str(nY) + '.periodic'
	
		fi.write('activate base; cd /mnt/home/ajermyn/Software/PyTNR; export PYTHONPATH=.; python TNR/Examples/ising2DJ.py ' + str(J) + ' ' + str(nX) + ' ' + str(nY) + ' > LOGS/ ' + name + '.log 2>&1 \n')		

		name = str(J) + '_' + str(nX) + '_' + str(nY) + '.open'
	
		fi.write('activate base; cd /mnt/home/ajermyn/Software/PyTNR; export PYTHONPATH=.; python TNR/Examples/ising2DJopen.py ' + str(J) + ' ' + str(nX) + ' ' + str(nY) + ' > LOGS/ ' + name + '.log 2>&1 \n')

# Sweep
Js = np.linspace(-3, 3, num=35, endpoint=True)
nX = 7
nY = 7
for J in Js:
		name = str(J) + '_' + str(nX) + '_' + str(nY) + '.periodic_sweep'
	
		fi.write('activate base; cd /mnt/home/ajermyn/Software/PyTNR; export PYTHONPATH=.; python TNR/Examples/ising2DJ.py ' + str(J) + ' ' + str(nX) + ' ' + str(nY) + ' > LOGS/ ' + name + '.log 2>&1 \n')		
