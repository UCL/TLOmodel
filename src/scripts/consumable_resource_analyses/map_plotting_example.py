from pathlib import Path

import pandas as pd
import shapefile as shp
from matplotlib import pyplot as plt

# Example plotting script for administrative district boundaries in Malawi
# More details available in the cookbook: https://github.com/UCL/TLOmodel/wiki/Cookbook#plotting-maps

# assume that the present working directory is the top level TLOmodel directory
resourcefilepath = Path("./resources")
outputfilepath = Path("./outputs")

# read in the shape file for district level maps
sf = shp.Reader(shp=open(resourcefilepath / 'ResourceFile_mwi_admbnda_adm2_nso_20181016.shp', 'rb'),
                dbf=open(resourcefilepath / 'ResourceFile_mwi_admbnda_adm2_nso_20181016.dbf', 'rb'),
                shx=open(resourcefilepath / 'ResourceFile_mwi_admbnda_adm2_nso_20181016.shx', 'rb'))

# create a figure
plt.figure()

# loop through the parts in the shape file
for shape in sf.shapeRecords():
    for i in range(len(shape.shape.parts)):
        i_start = shape.shape.parts[i]
        if i == len(shape.shape.parts) - 1:
            i_end = len(shape.shape.points)
        else:
            i_end = shape.shape.parts[i + 1]
        x = [i[0] for i in shape.shape.points[i_start:i_end]]
        y = [i[1] for i in shape.shape.points[i_start:i_end]]
        plt.plot(x, y, color='k', linewidth=0.5)

# remove figure axes and set the aspect ratio to equal so that the map isn't stretched
plt.axis('off')
plt.gca().set_aspect('equal')

# example of how to add data to the map using a colour map:
paracetamol_df = pd.read_csv(resourcefilepath / 'ResourceFile_Example_Paracetamol_DataFrame.csv')
stock_out_days = paracetamol_df['Stock Out Days']
eastings = paracetamol_df['Eastings']
northings = paracetamol_df['Northings']
cm = plt.cm.get_cmap('Purples')
sc = plt.scatter(eastings, northings, c=stock_out_days, cmap=cm, s=4)
plt.colorbar(sc, fraction=0.01, pad=0.01, label="Stock Out Days")

# give the figure a title
plt.title("Paracetamol Stock Out Days Example")

# save the figure
plt.savefig(outputfilepath / 'Map_Paracetamol_Stock_Out_Days_Example.png', bbox_inches="tight", dpi=600)

# display the figure in PyCharm's Plots window
plt.show()
