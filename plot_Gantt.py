import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib
matplotlib.use('TkAgg')

# Define the jobs and their operations: (machine, duration)
jobs = {
    1: [('M1', 2), ('M2', 1)],
    2: [('M1', 3), ('M2', 4)]
}

# Define the two sequences as lists of (job, op_index) tuples
sequences = [
    [(1,1), (1,2), (2,1), (2,2)],
    [(2,1), (1,1), (2,2), (1,2)]
]

# Colors for jobs
colors = {1: 'skyblue', 2: 'lightgreen'}

# Function to compute schedule
def compute_schedule(seq, jobs):
    machine_free = {'M1': 0, 'M2': 0}
    job_last_end = {1: 0, 2: 0}
    starts = {}
    ends = {}
    
    for op in seq:
        job, op_idx = op
        machine, duration = jobs[job][op_idx - 1]
        prev_end = job_last_end[job]
        start = max(prev_end, machine_free[machine])
        end = start + duration
        starts[op] = start
        ends[op] = end
        machine_free[machine] = end
        job_last_end[job] = end
    
    makespan = max(ends.values())
    return starts, ends, makespan

# Function to plot Gantt chart
def plot_gantt(seq, starts, makespan, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    machines = ['M1', 'M2']
    ax.set_yticks([1, 2])
    ax.set_yticklabels(machines)
    ax.set_ylim(0.5, 2.5)
    ax.set_xlim(0, makespan + 1)
    ax.set_xlabel('Time')
    ax.set_title(title)
    ax.grid(True)
    
    legend_elements = []
    for job in [1, 2]:
        legend_elements.append(Patch(facecolor=colors[job], label=f'Job {job}'))
    
    for op in seq:
        job, op_idx = op
        machine, duration = jobs[job][op_idx - 1]
        start = starts[op]
        y_pos = machines.index(machine) + 1
        ax.barh(y_pos, duration, left=start, height=0.4, color=colors[job], edgecolor='black')
        ax.text(start + duration / 2, y_pos, f'o{job},{op_idx}', ha='center', va='center', color='black')
    
    ax.legend(handles=legend_elements, loc='upper right')
    plt.show()

# Compute and plot for each sequence
for i, seq in enumerate(sequences, 1):
    starts, ends, makespan = compute_schedule(seq, jobs)
    plot_gantt(seq, starts, makespan, '')