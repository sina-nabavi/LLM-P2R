import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
# Sample data
architectures = {
    'T5': {
        'names': ['T5nbase', 'T5 large', 'T5 x-large', 'T5 xx-large'],
        'sizes': [60000000, 220000000, 770000000 ,3000000000],
        'scores': [-11.27, -14.50, -14.51, -15.37]
    },
    'BERT': {
        # 'names': ['BERT base', 'BERT large'],
        # 'sizes': ['110M', '340M'],
        'names': ['BERT large'],
        'sizes': [340000000],
        'scores': [-14.86]
    },

    'RoBERTa': {
        'names': ['RoBERTa base', 'RoBERTa large'],
        'sizes': [125000000, 355000000],
        'scores': [-5.78, -13.46]
    },
    
    'MiniLM L12 v1': {
        'names': ['MiniLM L12 v1'],
        'sizes': [33000000],
        'scores': [-11.15]
    },

    'GTE large v1.5': {
        'names': ['GTE large v1.5'],
        'sizes': [335000000],
        'scores': [-18.57]
    },

    'MXBAI large v1': {
        'names': ['MXBAI large v1'],
        'sizes': [335000000],
        'scores': [-18.02]
    },
}

# Define colors for each architecture
colors = {
    'T5': '#66c2a5',          # Light green
    'BERT': '#fc8d62',        # Light orange
    'RoBERTa': '#8da0cb',       # Light blue
    'MiniLM L12 v1': '#e78ac3',   # Light pink
    'GTE large v1.5': '#990147', # Light red
    'MXBAI large v1': '#ab65f8', # Light purple
}

plt.figure(figsize=(17.6, 10))
plt.grid(True)
ax = plt.gca()

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20  # Sets the default font size
plt.rcParams['axes.labelweight'] = 'bold'  # Sets bold labels
plt.rcParams['axes.labelsize'] = 22  # Sets the font size for labels
plt.rcParams['lines.linewidth'] = 5  # Sets the default line width
plt.rcParams['legend.fontsize'] = 20  # Sets the legend font size
ax.xaxis.set_tick_params(labelsize=25)
ax.yaxis.set_tick_params(labelsize=25)

for spine in ax.spines.values():  # Change border color to match grid color
    spine.set_edgecolor('#b0b0b0')
# Define the x-axis bounds, starting slightly earlier
xmin, xmax = 5.5, 11.5  # Extend further to the left

for model, data in architectures.items():
    sizes_log = np.log10(data['sizes'])
    scores = data['scores']
    
    if len(sizes_log) > 1:
        # Plot the actual line without markers
        line, = plt.plot(sizes_log, scores, linestyle='-', color=colors[model], label=f'{model}')
        
        # Plot the empty markers on top of the lines
        marker = plt.scatter(sizes_log, scores, edgecolors=colors[model], facecolors='w', s=100, linewidths=2, zorder=3)
        
        # Extend the line to the left without markers
        plt.plot([xmin, sizes_log[0]], [scores[0], scores[0]], linestyle='-', color=colors[model])
        
        # Extend the line to the right without markers
        plt.plot([sizes_log[-1], xmax], [scores[-1], scores[-1]], linestyle='-', color=colors[model])
    else:
        # For single point, extend a straight line across the entire x-axis range
        line, = plt.plot([xmin, xmax], [scores[0], scores[0]], linestyle='--', color=colors[model], label=f'{model}')
        marker = plt.scatter(sizes_log, scores, facecolors='w', edgecolors=colors[model], s=100, linewidths=2, zorder=3)

    # Combine both line and marker in the legend
    legend_elements = [
        Line2D([0], [0], color=colors['T5'], lw=5, linestyle='-', marker='o', markersize=20, markerfacecolor='none', label='T5'),
        Line2D([0], [0], color=colors['RoBERTa'], lw=5, linestyle='-', marker='o', markersize=20, markerfacecolor='none', label='RoBERTa'),
        Line2D([0], [0], color=colors['BERT'], lw=5, linestyle='--', marker='o', markersize=20, markerfacecolor='none', label='BERT'),
        Line2D([0], [0], color=colors['MiniLM L12 v1'], lw=5, linestyle='--', marker='o', markersize=20, markerfacecolor='none', label='MiniLM L12 v1'),
        Line2D([0], [0], color=colors['GTE large v1.5'], lw=5, linestyle='--', marker='o', markersize=20, markerfacecolor='none', label='GTE large v1.5'),
        Line2D([0], [0], color=colors['MXBAI large v1'], lw=5, linestyle='--', marker='o', markersize=20, markerfacecolor='none', label='MXBAI large v1'),
    ]

    # Use these custom legend entries in your plot
    # (Assuming the rest of your plot code remains unchanged)
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(-0.3, 1))


plt.tick_params(left=False, bottom=False)
# Adding labels and title
plt.xlabel('Model size: Log(number of parameters)', fontsize=25)
plt.ylabel('$\\Delta m\\%$', fontsize=25)

# Configure grid settings
plt.grid(True)
plt.xlim(xmin, xmax)

# Configure x-axis to have coarser grid
ax = plt.gca()
ax.xaxis.set_major_locator(plt.MaxNLocator(2))  # Set max number of ticks on the x-axis to 3

# Configure y-axis to have finer grid
ax.yaxis.set_major_locator(plt.MaxNLocator(5))  # Set max number of ticks on the y-axis to 10
plt.title('Scaling Experiment', fontweight='bold')

# Save plot as PDF
plt.tight_layout()
plt.savefig("scaling_experiment_results.pdf", format='pdf', bbox_inches='tight', pad_inches=0)
# Configure grid settings

# Show the plot
plt.show()

