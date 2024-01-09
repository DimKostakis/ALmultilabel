import matplotlib.pyplot as plt

# Define the query strategies
query_strategies = ['MMC', 'Bin-Min', 'RandS']

# Define the values for the y-axis (percentages)
y_values = [54, 37, 10]

# Create the bar graph
plt.bar(query_strategies, y_values)

# Set the labels for the x-axis and y-axis
plt.xlabel('Query Strategies')
plt.ylabel('Percentage')

# Set the title of the graph
plt.title('Comparison of Query Strategies for Recall-macro %increase')
plt.savefig('bar_graph.png', dpi=300, bbox_inches='tight')
# Display the graph
plt.show()