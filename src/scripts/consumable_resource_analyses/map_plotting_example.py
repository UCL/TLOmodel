import matplotlib.pyplot as plt
import shapefile as shp

# Example plotting script for administrative district boundaries in Malawi
# More details available in the cookbook: https://github.com/UCL/TLOmodel/wiki/Cookbook#plotting-maps

# read in the shape file
sf = shp.Reader('../../../resources/ResourceFile_mwi_admbnda_adm2_nso_20181016.shp')
# note: at the moment this script uses it's directory as the working directory

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
        plt.plot(x, y, color='k', linewidth=0.1)

# remove figure axes and set the aspect ratio to equal so that the map isn't stretched
plt.axis('off')
plt.gca().set_aspect('equal')

# example of how to add data to the map using a colour map:

# paracetamol_df = pd.read_csv('paracetamol_df.csv')
# stock_out_days = paracetamol_df['Stock Out Days']
# eastings = paracetamol_df['Eastings']
# northings = paracetamol_df['Northings']
# cm = plt.cm.get_cmap('Purples')
# sc = plt.scatter(eastings, northings, c=stock_out_days, cmap=cm, s=4)
# plt.colorbar(sc, fraction=0.01, pad=0.01, label="Stock Out Days")

# give the figure a title
plt.title("Plot Title")

# display the figure in PyCharm's Plots window
plt.show()
