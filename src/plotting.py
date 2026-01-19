import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-white')
#mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.size'] = 8
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['axes.prop_cycle'] = plt.cycler(color=[
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    '#bcbd22', '#17becf'
])

blue        = '#1f77b4'  # Ocean blue
lightblue   = '#87ceeb'  # Sky blue
darkblue    = '#003366'  # Deep sea / midnight

green       = '#2ca02c'  # Leaf green
lightgreen  = '#90ee90'  # Fresh leaves
darkgreen   = '#006400'  # Forest green

red         = '#d62728'  # Rose red
darkred     = '#8b0000'  # Lava red / blood
pink        = '#ff6f61'  # Coral pink

orange      = '#ff7f0e'  # Sunset orange
lightorange = '#ffb347'  # Soft peach

yellow      = '#ffd700'  # Sun yellow
gold        = '#ffcc00'  # Wheat field

brown       = '#8b4513'  # Earth brown
lightbrown  = '#a0522d'  # Tree bark
darkbrown   = '#5c3317'  # Rich soil

gray        = '#7f7f7f'  # Mountain gray
lightgray   = '#d3d3d3'  # Fog/cloud
darkgray    = '#404040'  # Stone

black       = '#1c1c1c'  # Night black
white       = '#f7f7f7'  # Cloud white
beige       = '#f5deb3'  # Sand

teal        = '#17becf'  # River water
purple      = '#9467bd'  # Lavender / wildflower
lightpurple = '#d8bfd8'  # Thistle

# For convenience, organize them in a dictionary
nature_palette = {
    "blue": blue,
    "lightblue": lightblue,
    "darkblue": darkblue,
    "green": green,
    "lightgreen": lightgreen,
    "darkgreen": darkgreen,
    "red": red,
    "darkred": darkred,
    "pink": pink,
    "orange": orange,
    "lightorange": lightorange,
    "yellow": yellow,
    "gold": gold,
    "brown": brown,
    "lightbrown": lightbrown,
    "darkbrown": darkbrown,
    "gray": gray,
    "lightgray": lightgray,
    "darkgray": darkgray,
    "black": black,
    "white": white,
    "beige": beige,
    "teal": teal,
    "purple": purple,
    "lightpurple": lightpurple
}