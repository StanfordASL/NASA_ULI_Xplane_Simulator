import seaborn as sns
import matplotlib.pyplot as plt
import pandas

sns.set_theme(style="ticks", palette="pastel")

###############################
csv_fname = 'results/results.csv'

df = pandas.read_csv(csv_fname)

x_var = 'Condition' 
y_var = 'Error'
hue_var = 'Train/Test'

title_str = 'Tiny Taxinet Pre-trained Model'

# Draw a nested boxplot to show bills by day and time
sns.boxplot(x = x_var, y = y_var,
            hue = hue_var, data = df)
plt.title(title_str)
plt.savefig('results/taxinet_barplot.pdf')
plt.close()
