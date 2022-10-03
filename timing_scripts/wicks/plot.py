import matplotlib.pyplot as plt
import pandas as pd

header = ['ndets', 'nex', 'time']
cpu = pd.read_csv('cpu_time.dat', sep=r'\s+', header=None)
cpu.columns = header
gpu = pd.read_csv('gpu_time.dat', sep=r'\s+', header=None)
gpu.columns = header

cpu = cpu.groupby('nex')
gpu = gpu.groupby('nex')

for group in [1, 2, 3, 4]:
    c = cpu.get_group(group)
    g = gpu.get_group(group)
    print(c.ndets, g.time, c.time)
    plt.plot(c.ndets, c['time'] / g['time'], label=f'nex = {group}',
            linewidth=0, marker='o')

plt.xscale('log')

plt.legend()
plt.xlabel('ndets')
plt.ylabel('cpu / gpu')
plt.savefig('ovlp_comparison.png', bbox_inches='tight')
