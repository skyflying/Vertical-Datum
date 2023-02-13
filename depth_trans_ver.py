import os

from tkinter.constants import END
import numpy as np
from scipy.interpolate import griddata
# GUI development: using tkinter
import tkinter as tk
import tkinter.filedialog as fdialog
import tkinter.messagebox as tkMessageBox
import tkinter.ttk as ttk
from PIL import Image, ImageTk


# The title of graphical user interface (GUI)
GUI_title = 'Vertical Datum Transformation'
# size of window: '600x500'
GUI_size = '600x500'
# Names of two tabs
Tab1 = 'Information'
Tab2 = 'Transformation'

# Information: Tab 1
# subtitle
Input_text = 'Input Surface'
Output_text = 'Output Surface'

# Different surfaces for transformation
Surface = ['Mean Sea Surface (MSS)',
           'Highest Astronomical Tide (HAT)',
           'Mean High Water (MHW)',
           'Mean Low Water (MLW)',	### 'Lowest Astronomical Tide (LAT)',
           'Lowest Astronomical Tide (LAT)',	### 'Lowest Low Water (LLW)',
           'Indian Spring Low Water (ISLW)',
           'Geoid',
           'Ellipsoid']	### 'Seabed Ellipsoidal Height (EL)']
# brief name
Surface_nickname = ['MSS', 'HAT', 'MHW', 'MLW',	### Surface_nickname = ['MSS', 'HAT', 'MHW', 'LAT',
                    'LAT', 'ISIW', 'Geoid', 'EL']	### 'LLW', 'ISIW', 'Geoid', '    EL    ']
#
Surface_file = ['file/MSS.xyz',	### Surface_file = ['file/NCTU_mss_surface.xyz',
                'file/HAT.xyz',
                'file/MHW.xyz',
                'file/MLW.xyz',	### 'file/LAT.xyz',
                'file/LAT.xyz',	### 'file/LLW.xyz',
                'file/ISLW.xyz',
                'file/geoid.xyz']
# Figure
Fig1 = 'file/fig1.png'	### ISLW LAT switch positions


# Information: Tab 2
# subtitle
Single_text = 'Single point'
File_text = 'Import a file'
# buttons
Single_trans_btn = 'Transform >'
Import_file_btn = 'Browse'
Output_dir_btn = 'Browse'
File_trans_btn = 'Transform'
Close_btn = 'Close'



def Read_llv(filename):
    lon = []
    lat = []
    value = []
    with open(filename, 'r') as f:
        data = f.readlines()
    for i in range(len(data)):
        info = data[i].split()
        lon.append(float(info[0]))
        lat.append(float(info[1]))
        value.append(float(info[2]))
    return(lon, lat, value)

# Write result: lon/lat/value/new value


def Write_llvn(filename, lon, lat, value1, value2):
    with open(filename, 'w') as g:
        for i in range(len(lon)):
            # g.write(str(lon[i])+'\t'+str(lat[i])+'\t'+str(value1[i])+'\t'+str(value2[i])+'\n')
            # g.write(str(lon[i])+'\t'+str(lat[i])+'\t')
            g.write('%11.7f %10.7f %8.3f %8.3f\n' % (lon[i], lat[i], value1[i], value2[i]))	


# for Impoting a file --> check if the point is within the range or not

# Revised by Chong-You Wang, 2021-06-16.
def check_range(lon, lat, value):
    lon = np.array(lon)
    lat = np.array(lat)
    value = np.array(value)
    index = np.where((lon <= 125) & (lon >= 118) & (lat <= 27) & (lat >= 21))	
    lon_range = lon[index]
    lat_range = lat[index]
    value_range = value[index]
    outrange_index = np.setdiff1d(range(len(lon)), index)
    return(lon_range, lat_range, value_range, outrange_index)



def Transform(input_index, output_index, input_lon, input_lat, input_depth):
    print(Surface[input_index], '-> ', Surface[output_index])

    lonlat_input = np.column_stack((input_lon, input_lat))
    input_depth = np.array(input_depth)
    del input_lon, input_lat

    # Read the initial depth from input surface
    if (input_index == 7) | ((output_index >=1) & (output_index <=5) & (input_index == 0)) :
        initial_depth = np.zeros(input_depth.size)
    else:
        # Read initial surface
        [loni, lati, value_ini] = Read_llv(Surface_file[input_index])
        lonlat_ini = np.column_stack((loni, lati))
        value_ini = np.array(value_ini)
        del loni, lati
        index_model = np.where(((lonlat_input[:, 0].max() + 0.5) >= lonlat_ini[:, 0]) & ((lonlat_input[:, 0].min() - 0.5) <= lonlat_ini[:, 0]) & ((lonlat_input[:, 1].max() + 0.5) >= lonlat_ini[:, 1]) & ((lonlat_input[:, 1].min() - 0.5) <= lonlat_ini[:, 1]))   # Optimized by Chong You. 2021.
        lonlat_ini, value_ini = [lonlat_ini[index_model], value_ini[index_model]]
        initial_depth=griddata(lonlat_ini, value_ini,lonlat_input, method='cubic') #input index 1~6
        if (input_index >= 1) & (input_index <= 5) & (output_index >5) :
            # Read MSS surface to transform reference to ell.
            [lonp, latp, value_mss] = Read_llv(Surface_file[0])
            lonlat_mss = np.column_stack((lonp, latp))
            value_mss = np.array(value_mss)
            del lonp, latp
            lonlat_mss, value_mss = [ lonlat_mss[index_model], value_mss[index_model]]
            mss=griddata(lonlat_mss, value_mss,lonlat_input, method='cubic')
            initial_depth=initial_depth + mss

    # Read the final depth from output surface
    if (output_index == 7)  | ((input_index >=1) & (input_index <=5) & (output_index == 0)):
        final_depth=np.zeros(input_depth.size)
    else:
        [lonO, latO, value_O] = Read_llv(Surface_file[output_index])
        lonlat_O = np.column_stack((lonO, latO))
        value_O = np.array(value_O)
        del lonO, latO

        index_model = np.where(((lonlat_input[:, 0].max() + 0.5) >= lonlat_O[:, 0]) & ((lonlat_input[:, 0].min() - 0.5) <= lonlat_O[:, 0]) & ((lonlat_input[:, 1].max() + 0.5) >= lonlat_O[:, 1]) & ((lonlat_input[:, 1].min() - 0.5) <= lonlat_O[:, 1]))   # Optimized by Chong You. 2021.
        lonlat_O, value_O = [lonlat_O[index_model], value_O[index_model]]
        final_depth=griddata(lonlat_O, value_O,lonlat_input, method='cubic')

        if (output_index >= 1) & (output_index <= 5) & (input_index >5) :
            # Read MSS surface to transform reference to ell.
            [lonp, latp, value_mss] = Read_llv(Surface_file[0])
            lonlat_mss = np.column_stack((lonp, latp))
            value_mss = np.array(value_mss)
            del lonp, latp
            lonlat_mss, value_mss = [ lonlat_mss[index_model], value_mss[index_model]]
            mss=griddata(lonlat_mss, value_mss,lonlat_input, method='cubic')
            final_depth=final_depth + mss


    # Calculation
    #if input_index != 7 and output_index != 7:
            # 各垂直基準面互轉
    new_depth=input_depth + final_depth - initial_depth

    #elif input_index == 7 and output_index != 7:
            # 由海床橢球高算水深
        #new_depth=final_depth - input_depth

    #elif input_index != 7 and output_index == 7:
            # 由水深算海床橢球高
        #new_depth=initial_depth - input_depth
    return(new_depth, initial_depth, final_depth)

# Main execution
# 0. Save selected information in tab 1
# 1. Check the input information in tab 2
# 2. Transformation --> call function "Transform"
# 3. Display


def Execute(case):
    input_index=input_index_tmp.get()    # input surface
    output_index=output_index_tmp.get()  # output surface
    # print(Surface[input_index],'===>',Surface[output_index])

    # Step 1. Check the range and format
    check_condition=False
    # <<<<< single point >>>>> save variables, check if it is in the range
    if case == 0:
        # this can just use when it is StringVar
        if not input_lon_tmp.get() or not input_lat_tmp.get() or not input_depth_tmp.get():
            tkMessageBox.showinfo('Error',
                                  'Please fill in all the information:\nLongitude, Latitude and Depth.')
        else:
            try:
                float(input_lon_tmp.get()) and float(
                    input_lon_tmp.get()) and float(input_depth_tmp.get())
            except ValueError:
                tkMessageBox.showinfo('Error', 'Please type in float format.')
            else:
                # get the variables from entries
                input_lon=float(input_lon_tmp.get())
                input_lat=float(input_lat_tmp.get())
                input_depth=float(input_depth_tmp.get())
                # check if it's out of the range
                if input_lon > 125 or input_lon < 118 or input_lat > 27 or input_lat < 21:
                    tkMessageBox.showinfo('Error',
                                          'The point must be inside the range:\n 118~125E, 21~27N.')
                else:
                    check_condition=True
    # <<<<< file import >>>>> save variables, read file, check if they are in the range
    elif case == 1:
        # Save variables
        if not (input_path_tmp.get()):
            tkMessageBox.showinfo('Error', 'Please select the input file')
        elif not (output_dir_tmp.get()):
            tkMessageBox.showinfo(
                'Error', 'Please choose a directory for output file')
        elif not (output_name_tmp.get()):
            tkMessageBox.showinfo('Error', 'Please name the output file')
        else:
            # get the variables from entries
            input_path=input_path_tmp.get()
            output_dir=output_dir_tmp.get()
            output_name=output_name_tmp.get()
            # Read the file
            [input_lon0, input_lat0, input_depth0]=Read_llv(input_path)
            # check if all the points are within the range
            [input_lon, input_lat, input_depth, outrange_i]=check_range(
                input_lon0, input_lat0, input_depth0)

            if len(outrange_i) == 1:
                tkMessageBox.showinfo('Warning',
                                      'There is '+str(len(outrange_i))+' point outside the range:\n 118~125E, 21~27N.')
            elif len(outrange_i) > 1:
                tkMessageBox.showinfo('Warning',
                                      'There are '+str(len(outrange_i))+' points outside the range:\n 118~125E, 21~27N.')
            check_condition=True

    # Step 2. Transform
    while check_condition:  # Execute step 2 after step 1
        [new_depth, initial_depth, final_depth]=Transform(
            input_index, output_index, input_lon, input_lat, input_depth)

        # Step 3. Display
        # '({:^10})'.format('Hello')
        if case == 0:
            label=tk.Label(page2, text='({:^7})'.format(
                Surface_nickname[input_index]))
            label.grid(row=12, column=6, columnspan=1, sticky='W')
            label=tk.Label(page2, text='New value')
            label.grid(row=13, column=3)
            label=tk.Label(
                page2, text='({:^7})'.format(Surface_nickname[output_index]))
            label.grid(row=14, column=6, columnspan=1, sticky='W')
            label=tk.Label(page2, text='%.4f' % new_depth)
            label.grid(row=14, column=3)

        elif case == 1:
            PWD=os.getcwd()
            os.chdir(output_dir)
            if len(outrange_i) > 0:
                for i in range(len(outrange_i)):
                    new_depth=np.insert(new_depth, outrange_i[i], np.nan)

                Write_llvn(output_name, input_lon0,
                           input_lat0, input_depth0, new_depth)
            else:
                Write_llvn(output_name, input_lon,
                           input_lat, input_depth, new_depth)
            os.chdir(PWD)
            label=tk.Label(page2, text=' OK! ')
            label.grid(row=24, column=7)
        check_condition=False


####################################################
############### Create GUI Interface ###############
####################################################
main=tk.Tk()
main.title(GUI_title)
main.geometry(GUI_size)

# gives weight to the cells in the grid
rows=0
while rows < 50:
    main.rowconfigure(rows, weight=1)
    main.columnconfigure(rows, weight=1)
    rows += 1

# Defines and places the notebook widget
nb=ttk.Notebook(main)
nb.grid(row=1, column=0, columnspan=50, rowspan=49, sticky='NESW')

####################################################
###################### Tab 1. ######################
####################################################
# Adds tab 1 of the notebook
page1=ttk.Frame(nb)
nb.add(page1, text=Tab1)

# labels for input and output titles
label=tk.Label(page1, text=Input_text,
                 font="Helvetica 12 bold")
label.grid(row=0, column=0, sticky='nw')

label=tk.Label(page1, text=Output_text,
                 font="Helvetica 12 bold")
label.grid(row=0, column=2, sticky='nw')

# Suface selection
# set the variables for input/output index (tmp)
input_index_tmp=tk.IntVar()
output_index_tmp=tk.IntVar()
# set the initial selection of the two buttons
input_index_tmp.set(0)
output_index_tmp.set(1)

# check the two buttons for not making the same selection


def CHECK1(IN, OUT):
    if IN == OUT:
        output_index_tmp.set((OUT+1) % len(Surface))


def CHECK2(IN, OUT):
    if IN == OUT:
        input_index_tmp.set((IN+1) % len(Surface))


# create radio buttons
for i in range(len(Surface)):
    rb1=tk.Radiobutton(page1, text=Surface[i], variable=input_index_tmp, value=i,
                         command=lambda *args: CHECK1(input_index_tmp.get(), output_index_tmp.get()))
    rb1.grid(row=i+1, column=0, sticky='nw')
    rb2=tk.Radiobutton(page1, text=Surface[i], variable=output_index_tmp, value=i,
                         command=lambda *args: CHECK2(input_index_tmp.get(), output_index_tmp.get()))
    rb2.grid(row=i+1, column=2, sticky='nw')

# Figure
pil_image=Image.open(Fig1)

width=450
ratio=float(width)/pil_image.size[0]
height=int(pil_image.size[1]*ratio)

pil_image_resized=pil_image.resize((width, height), Image.BILINEAR)
tk_image=ImageTk.PhotoImage(pil_image_resized)
label=tk.Label(page1, image=tk_image)
label.grid(row=20, column=0, columnspan=3)

####################################################
###################### Tab 2. ######################
####################################################
page2=ttk.Frame(nb)
nb.add(page2, text=Tab2)


# Refinements
label=tk.Label(page2, text='Range', font="Helvetica 12 bold")
label.config(fg='#000000')
label.grid(row=5, column=0, sticky='nw')
label=tk.Label(page2, text='''     ---------  27 N ---------
    |                                        |
    118 E                                  125 E
    |                                        |
    ---------  21 N ---------
                   ''')
label.grid(row=6, column=0, columnspan=5, rowspan=2, sticky='nw')

# <<<<< Single point >>>>>
# labels for subtitle
label=tk.Label(page2, text=Single_text,
                 font="Helvetica 12 bold")
label.grid(row=10, column=0, sticky='nw')

# labels for single point
label=tk.Label(page2, text='Longitude')
label.grid(row=11, column=1)
label=tk.Label(page2, text='Latitude')
label.grid(row=11, column=2)

label=tk.Label(page2, text='Input value')
label.grid(row=11, column=3)

# textboxes for single lon/lat/depths
input_lon_tmp=tk.StringVar()
input_lat_tmp=tk.StringVar()
input_depth_tmp=tk.StringVar()
edTxt1=tk.Entry(page2, textvariable=input_lon_tmp,
                  width=13, justify='center', borderwidth=2)
edTxt1.grid(row=12, column=1)
edTxt2=tk.Entry(page2, textvariable=input_lat_tmp,
                  width=13, justify='center', borderwidth=2)
edTxt2.grid(row=12, column=2)
edTxt3=tk.Entry(page2, textvariable=input_depth_tmp,
                  width=13, justify='center', borderwidth=2)
edTxt3.grid(row=12, column=3)

# <<<<< Import a file >>>>>
label=tk.Label(page2, text=File_text, font="Helvetica 12 bold")
label.grid(row=20, column=0, sticky='nw')
# labels for subtitle
label=tk.Label(page2, text='Input file')
label.grid(row=22, column=0)
label=tk.Label(page2, text='Output directory')
label.grid(row=23, column=0)
label=tk.Label(page2, text='Output file')
label.grid(row=24, column=1, sticky='e')

# entries for path
input_path_tmp=tk.StringVar()
output_dir_tmp=tk.StringVar()
output_name_tmp=tk.StringVar()
edTxt4=tk.Entry(page2, textvariable=input_path_tmp, width=45, borderwidth=2)
edTxt4.grid(row=22, column=1, columnspan=4)
edTxt5=tk.Entry(page2, textvariable=output_dir_tmp, width=45, borderwidth=2)
edTxt5.grid(row=23, column=1, columnspan=4)
edTxt6=tk.Entry(page2, textvariable=output_name_tmp,
                  justify='right', width=29, borderwidth=2)
edTxt6.grid(row=24, column=2, columnspan=2)

# bottons for browsing file/directory


def browse_file():
    dir=fdialog.askopenfilename(
        initialdir=os.getcwd(),
        title="Select a file")
    repeatedly_click(edTxt4,dir)


def browse_dir():
    dir=fdialog.askdirectory(
        initialdir=os.getcwd(),
        title="Select a directory")
    repeatedly_click(edTxt5,dir)

# Revised by Chong-You Wang, 2021-06-16.
def repeatedly_click(Ety,new):
    Ety.delete(0,END)
    Ety.insert(0,new)



btn1=tk.Button(page2, text=Import_file_btn,    # change browse file and dir.
                 command=lambda *args: browse_file())
btn2=tk.Button(page2, text=Output_dir_btn,
                 command=lambda *args: browse_dir())
btn1.grid(row=22, column=6)
btn2.grid(row=23, column=6)

# button for Single point Transform
btn=tk.Button(page2, text=Single_trans_btn, command=lambda *args: Execute(0))
btn.grid(row=14, column=2)

# button for File import Transform
btn=tk.Button(page2, text=File_trans_btn, command=lambda *args: Execute(1))
btn.grid(row=24, column=6)


# close button
def Done():
    # close the GUI interface
    main.quit()


btn=tk.Button(page2, text=Close_btn, bg='#c0c0c0',
                command=lambda *args: Done())
btn.grid(row=25, column=7)

# <<<<< note text >>>>>
note1="註1 : 水深值(垂直基準面至海床垂直距離)坐標軸向下為正"
note2="註2 : 海床橢球高(橢球面至海床垂直距離)坐標軸向上為正"
note3="註3 : Ellipsoid在海域上指的是海床橢球高，在陸域則是代表該點的橢球高"	### note3="註3 : EL 代表海床橢球高"
note4="註4 : 內陸橢球高進行正高轉換時，輸出值為負代表該點在geoid之上，"	### 新增
note5="         例如: 輸出值為-5公尺代表該點正高為5公尺"	### 新增

label=tk.Label(page2, text=note1)
label.grid(row=26, column=0, columnspan=5, rowspan=1, sticky='nw')

label=tk.Label(page2, text=note2)
label.grid(row=27, column=0, columnspan=5, rowspan=1, sticky='nw')

label=tk.Label(page2, text=note3)
label.grid(row=28, column=0, columnspan=5, rowspan=1, sticky='nw')

label=tk.Label(page2, text=note4)	### 新增
label.grid(row=29, column=0, columnspan=5, rowspan=1, sticky='nw')	### 新增

label=tk.Label(page2, text=note5)	### 新增
label.grid(row=30, column=0, columnspan=5, rowspan=1, sticky='nw')	### 新增

main.mainloop()
