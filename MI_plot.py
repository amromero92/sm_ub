#Make a pretty matrix plot from mutual information data

import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
import sys 
rc('text', usetex=True)


output_name = "output.pdf"
input_name  = "mutual_info.dat"

sp_len = 6         #Dimension of configuration space. Only of one fluid (protons or neutrons)
subshells = [4]    #List with subshells dimensions in order of energy (For example, in the p-shell: [4], in the sd-shell: [6,8]


MI_data = pd.read_csv(input_name, sep=' ')
orbs1, orbs2, MI = MI_data["orb1"], MI_data["orb2"], MI_data["S12"]
data_to_plot = MI_data.pivot("orb1", "orb2", "S12")

#Width of lines marking different subshells
lw = 1.
lwadd = .5

#Ticks colorbar
cticks = list(np.arange(.0,.07+.01,.01))



###SINGLE PLOT CASE
fig, ax = plt.subplots()


ax = sns.heatmap(data_to_plot, cmap="Blues",linecolor='white',linewidths=lw,cbar_kws={'ticks': cticks, 'label': r'$S_{ij}$'}, ax=ax)

ax.figure.axes[-1].yaxis.label.set_size(18)


ax.invert_yaxis()
ax.vlines(sp_len, *ax.get_ylim(), ls="solid", lw=lw+lwadd, color="black")
ax.hlines(sp_len, *ax.get_xlim(), ls="solid", lw=lw+lwadd, color="black")
for _, spine in ax.spines.items():
    spine.set_visible(True)
for i in subshells:
 ax.vlines(i, *ax.get_ylim(), ls="--", lw=lw+lwadd, color="black")
 ax.hlines(i, *ax.get_xlim(), ls="--", lw=lw+lwadd, color="black")
 ax.vlines(i+sp_len, *ax.get_ylim(), ls="--", lw=lw+lwadd, color="black")
 ax.hlines(i+sp_len, *ax.get_xlim(), ls="--", lw=lw+lwadd, color="black")
ax.tick_params(axis="y",direction="in",right=1,left=1)
ax.tick_params(axis="x",direction="in",bottom=1,top=1)
ax.set_xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5])
ax.set_yticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5])
ax.set_xticklabels([r'$0$', r'$1$',r'$2$',r'$3$',r'$4$',r'$5$',r'$6$',r'$7$',r'$8$',r'$9$',r'$10$',r'$11$'])
ax.set_yticklabels([r'$0$', r'$1$',r'$2$',r'$3$',r'$4$',r'$5$',r'$6$',r'$7$',r'$8$',r'$9$',r'$10$',r'$11$'])
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(12)
ax.set(xlabel=None)
ax.set(ylabel=None)
ax.set_aspect('equal')
transx = ax.get_xaxis_transform()
transy = ax.get_yaxis_transform()
#Annotations
#Subshells
ax.plot([0.1,3.9],[-.08,-.08], color="k", transform=transx, clip_on=False)
ax.annotate(r'$0p_{3/2}$', xy=(2, -.1), xycoords=transx, ha="center", va="top",fontsize=12)
ax.plot([4.1,5.9],[-.08,-.08], color="k", transform=transx, clip_on=False)
ax.annotate(r'$0p_{1/2}$', xy=(5., -.1), xycoords=transx, ha="center", va="top",fontsize=12)

ax.plot([6.1,9.9],[-.08,-.08], color="k", transform=transx, clip_on=False)
ax.annotate(r'$0p_{3/2}$', xy=(8, -.1), xycoords=transx, ha="center", va="top",fontsize=12)
ax.plot([10.1,11.9],[-.08,-.08], color="k", transform=transx, clip_on=False)
ax.annotate(r'$0p_{1/2}$', xy=(11., -.1), xycoords=transx, ha="center", va="top",fontsize=12)

ax.plot([-.08,-.08], [0.1,3.9], color="k", transform=transy, clip_on=False)
ax.annotate(r'$0p_{3/2}$', xy=(-.1, 2.), xycoords=transy, va="center", ha="right",rotation=90,fontsize=12)
ax.plot([-.08,-.08], [4.1,5.9], color="k", transform=transy, clip_on=False)
ax.annotate(r'$0p_{1/2}$', xy=(-.1, 5.), xycoords=transy, va="center", ha="right",rotation=90,fontsize=12)

ax.plot([-.08,-.08], [6.1,9.9],color="k", transform=transy, clip_on=False)
ax.annotate(r'$0p_{3/2}$', xy=(-.1, 8.), xycoords=transy, va="center", ha="right",rotation=90,fontsize=12)
ax.plot([-.08,-.08], [10.1,11.9], color="k", transform=transy, clip_on=False)
ax.annotate(r'$0p_{1/2}$', xy=(-.1, 11.), xycoords=transy, va="center", ha="right",rotation=90,fontsize=12)

#Protons and neutrons
ax.plot([0.1,5.9],[-.18,-.18], color="k", transform=transx, clip_on=False)
ax.plot([6.1,11.9],[-.18,-.18], color="k", transform=transx, clip_on=False)
ax.annotate(r'$\rm{protons}$', xy=(3, -.2), xycoords=transx, ha="center", va="top",fontsize=12)
ax.annotate(r'$\rm{neutrons}$', xy=(9, -.2), xycoords=transx, ha="center", va="top",fontsize=12)

ax.plot([-.18,-.18], [0.1,5.9], color="k", transform=transy, clip_on=False)
ax.plot([-.18,-.18],[6.1,11.9], color="k", transform=transy, clip_on=False)
ax.annotate(r'$\rm{protons}$', xy=(-.2, 3), xycoords=transy, va="center", ha="right",rotation=90,fontsize=12)
ax.annotate(r'$\rm{neutrons}$', xy=(-.2, 9), xycoords=transy, va="center", ha="right",rotation=90,fontsize=12)



### END SINGLE PLOT CASE


plt.savefig("8Be_MI_singles.pdf", format='pdf',bbox_inches='tight')



