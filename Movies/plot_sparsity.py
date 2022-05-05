# encoding=utf-8
import matplotlib
#matplotlib.rcParams['backend'] = 'SVG'
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from pylab import *                                 
#mpl.rcParams['font.sans-serif'] = ['SimHei']
#simsun = FontProperties(fname=r'C:\Windows\Fonts\simsun.ttc', size=10) # simsun
roman = FontProperties(fname=r'C:\Windows\Fonts\times.ttf', size=15) # Times new roman
mpl.rcParams['font.sans-serif'] = ['SimSun']
#fontcn = {'family': 'SimSun','size': 10} # 1pt = 4/3px
fonten = {'family':'Times New Roman','size': 15}



total_width, n = 0.8, 2  
width = total_width / n 


# auto label
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.- 0.2, 1.03*height, '%s' % int(height))

  
name_list = ['Movies','Cars']  
num_list = [256,128]  
num_list2 = [7349/500, 1041/30]
x =list(range(len(num_list)))  
"""  
autolabel(plt.bar(x, num_list, label='positive spikes',fc = 'y'))
autolabel(plt.bar(x, num_list1, bottom=num_list, label='negative spikes', tick_label = name_list))
autolabel(plt.bar(x, num_list2, label='spiking neurons', fc = 'r'))
"""
autolabel(plt.bar(x, num_list, width=width, label='neurons',fc = 'peachpuff', hatch = '///'))
for i in range(len(x)):  
    x[i] = x[i] + width  
autolabel(plt.bar(x, num_list2, width=width, label='spikes',tick_label = name_list,fc = 'olivedrab', hatch = '///'))

plt.legend(loc='upper right', prop=fonten)  
#text(60, -0.01, u'Di', style='italic', fontdict=fonten) 
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(" ", fontproperties=roman) 
plt.ylabel("Number", fontproperties=roman) 
plt.title("Spiking sparsity", fontproperties=roman) 
  
plt.show() 

#plt.savefig('srnn_spike_neuron_plot.svg',format='svg')

