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

# auto label
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.- 0.2, 1.03*height, '%s' % int(height))


name_list = ['Movies','Cars']  
num_list = [2830277,1326721]  
x =list(range(len(num_list)))  

autolabel(plt.bar(x, num_list, color='blueviolet', tick_label=name_list, hatch = '///'))
#plt.legend(prop=fonten)  

plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(" ", fontproperties=roman) 
plt.ylabel("Number", fontproperties=roman) 
plt.title("Synaptic operations", fontproperties=roman) 

plt.show()

#plt.savefig('srnn_sop_plot.svg',format='svg')
