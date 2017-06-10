"""
Tutorial followed from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
"""
import numpy as np # linear algebra
np.set_printoptions(threshold=np.inf)
import dicom
import os
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

from skimage import measure, morphology, segmentation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import skimage
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import feature
from skimage.segmentation import clear_border
from skimage import data

import scipy.misc
from subprocess import check_output

# Some constants

HU_MIN = 0
HU_MAX = 1424
# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness
    return slices

# converts to Hounsfield Unit (HU), which is a measure of radiodensity used in CT scans
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    # print(image[np.shape(image)[0]/2, np.shape(image)[1]/2, :])
    image = image.astype(np.int16)
    # print('-'*100)
    # print(image[np.shape(image)[0]/2, np.shape(image)[1]/2, :])
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        # print slope, intercept
        if slope != 1:
            image[slice_number] = image[slice_number].astype(np.float64)*slope
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
        image[slice_number] += 1024
    return np.array(image, dtype=np.int16)

# displays an example of one patient's scan
def showOneExample(data, plot=False, image=False):
    first_patient = load_scan(data)
    first_patient_pixels = np.asarray(get_pixels_hu(first_patient))

    if (not plot and not image):
        return first_patient, first_patient_pixels
    if plot:
        fig = plt.figure()
        fig.suptitle('Histogram frequencies from different locations for one patient')
        fig.subplots_adjust(hspace=0.5)
        a = fig.add_subplot(2, 2, 1)
        a.set_title("Scan from 20/128 pixels", fontsize=8)
        plt.xlabel("Hounsfield Units (HU)")
        plt.ylabel("Frequency")
        plt.hist(first_patient_pixels.flatten(), bins=20, color='c')
        b = fig.add_subplot(2, 2, 2)
        b.set_title("Scan from 50/128 pixels", fontsize=8)
        plt.xlabel("Hounsfield Units (HU)")
        plt.ylabel("Frequency")
        plt.hist(first_patient_pixels.flatten(), bins=50, color='c')
        c = fig.add_subplot(2, 2, 3)
        c.set_title("Scan from 80/128 pixels", fontsize=8)
        plt.xlabel("Hounsfield Units (HU)")
        plt.ylabel("Frequency")
        plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
        d = fig.add_subplot(2, 2, 4)
        d.set_title("Scan from 110/128 pixels", fontsize=8)
        plt.xlabel("Hounsfield Units (HU)")
        plt.ylabel("Frequency")
        plt.hist(first_patient_pixels.flatten(), bins=110, color='c')
        plt.show()
    if image:
        fig = plt.figure()
        fig.suptitle('Scans from different locations for one patient')
        fig.subplots_adjust(hspace=0.5)
        a = fig.add_subplot(2, 2, 1)
        a.set_xlabel("Scan from 20/128 pixels")
        plt.imshow(first_patient_pixels[20], cmap=plt.cm.gray)
        b = fig.add_subplot(2, 2, 2)
        b.set_xlabel("Scan from 50/128 pixels")
        plt.imshow(first_patient_pixels[50], cmap=plt.cm.gray)
        c = fig.add_subplot(2, 2, 3)
        c.set_xlabel("Scan from 80/128 pixels")
        plt.imshow(first_patient_pixels[80], cmap=plt.cm.gray)
        d = fig.add_subplot(2, 2, 4)
        d.set_xlabel("Scan from 110/128 pixels")
        plt.imshow(first_patient_pixels[110], cmap=plt.cm.gray)
        plt.show()
    return first_patient, first_patient_pixels

# resamples scans to isotropic resolution set by new_spacing
def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing

# 3d plot the image
# def plot_3d(image, threshold=-300, show=False):
#     # Position the scan upright,
#     # so the head of the patient would be at the top facing the camera
#     p = image.transpose(2, 1, 0)

#     verts, faces = measure.marching_cubes(p, threshold)
#     # verts, faces = measure.marching_cubes(p, None)
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')

#     # Fancy indexing: `verts[faces]` to generate a collection of triangles
#     mesh = Poly3DCollection(verts[faces], alpha=0.70)
#     face_color = [0.45, 0.45, 0.75]
#     mesh.set_facecolor(face_color)
#     ax.add_collection3d(mesh)

#     ax.set_xlim(0, p.shape[0])
#     ax.set_ylim(0, p.shape[1])
#     ax.set_zlim(0, p.shape[2])

#     plt.show()

def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    p = p[:,:,::-1]
    
    verts, faces, _, _ = measure.marching_cubes_lewiner(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()



def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

# def segment_lung_mask(image, fill_lung_structures=True):
#     # not actually binary, but 1 and 2.
#     # 0 is treated as background, which we do not want
#     binary_image = np.array(image > -320, dtype=np.int8) + 1
#     labels = measure.label(binary_image)

#     # Pick the pixel in the very corner to determine which label is air.
#     #   Improvement: Pick multiple background labels from around the patient
#     #   More resistant to "trays" on which the patient lays cutting the air
#     #   around the person in half
#     background_label1 = labels[0, 0, 0]
#     background_label2 = labels[0, 0, -1]
#     background_label3 = labels[0, -1, 0]
#     background_label4 = labels[0, -1, -1]
#     background_label5 = labels[-1, 0, 0]
#     background_label6 = labels[-1, 0, -1]
#     background_label7 = labels[-1, -1, 0]
#     background_label8 = labels[-1, -1, -1]

#     # Fill the air around the person
#     binary_image[background_label1 == labels] = 2
#     binary_image[background_label2 == labels] = 2
#     binary_image[background_label3 == labels] = 2
#     binary_image[background_label4 == labels] = 2
#     binary_image[background_label5 == labels] = 2
#     binary_image[background_label6 == labels] = 2
#     binary_image[background_label7 == labels] = 2
#     binary_image[background_label8 == labels] = 2

#     # Method of filling the lung structures (that is superior to something like
#     # morphological closing)
#     if fill_lung_structures:
#         # For every slice we determine the largest solid structure
#         for i, axial_slice in enumerate(binary_image):
#             axial_slice = axial_slice - 1
#             labeling = measure.label(axial_slice)
#             l_max = largest_label_volume(labeling, bg=0)

#             if l_max is not None:  # This slice contains some lung
#                 binary_image[i][labeling != l_max] = 1

#     binary_image -= 1  # Make the image actual binary
#     binary_image = 1 - binary_image  # Invert it, lungs are now 1

#     # Remove other air pockets insided body
#     labels = measure.label(binary_image, background=0)
#     l_max = largest_label_volume(labels, bg=0)
#     if l_max is not None:  # There are air pockets
#         binary_image[labels != l_max] = 0

#     return binary_image

def zero_center(image):
    image = image - PIXEL_MEAN
    return image

# # gives a run through of the preprocessing tutorial
# def testScans():
#     slices = load_scan(INPUT_FOLDER + patients[0])
#     hu_slices = get_pixels_hu(slices)
#     first_patient, first_patient_pixels = showOneExample(INPUT_FOLDER + patients[0])
#     pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1, 1, 1])
#     # print("Shape before resampling\t", first_patient_pixels.shape) # (128, 512, 512)
#     # print("Shape after resampling\t", pix_resampled.shape)   # (320, 347, 347)
#     # plot_3d(pix_resampled, 400, show=True)
#     segmented_lungs = segment_lung_mask(pix_resampled, False)
#     segmented_lungs_fill = segment_lung_mask(pix_resampled, True)
#     show3D = False
#     if show3D:
#         # segmented_lungs has no air.
#         plot_3d(segmented_lungs, 0)
#         # plot_3d(segmented_lungs_fill, 0)
#         # plot_3d(segmented_lungs_fill - segmented_lungs, 0)


def normalize(image):
    MIN_BOUND = float(HU_MIN)
    MAX_BOUND = float(HU_MAX)
    print(np.max(image), ' is max of image in normalize')
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return np.float32(image)

def get_segmented_lungs(im, plot=False):
    
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image. 
    '''
    binary = im < 604
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone) 
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone) 
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       label_image[coordinates[0], coordinates[1]] = 0
    # print label_image[71]
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndimage.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = HU_MIN
    if plot == True:
        plots[7].axis('off')
        plots[7].imshow(im, cmap=plt.cm.bone) 
        
    return im

def clip_edges(image):
    x_min = 0
    x_max = image.shape[0]
    y_min = 0
    y_max = image.shape[1]
    z_min = 0
    z_max = image.shape[2]
    for x in range(0, image.shape[0]):
        if np.all(image[x,:,:] < 0.0001):
            continue
        else:
            x_min = max(0,x-1)
            break
    for x in range(image.shape[0]-1, -1, -1):       
        if np.all(image[x,:,:] < 0.0001):
            continue
        else:
            x_max = min(x+2,image.shape[0])
            break
    image = image[x_min:x_max, :, :]

    for y in range(0, image.shape[1]):
        if np.all(image[:,y,:] < 0.0001):
            continue
        else:
            y_min = max(0,y-1)
            break
    for y in range(image.shape[1]-1, -1, -1):        
        if np.all(image[:,y,:] < 0.0001):
            continue
        else:
            y_max = min(y+2,image.shape[1])
            break
    image = image[:, y_min:y_max, :]

    for z in range(0, image.shape[2]):
        if np.all(image[:,:,z] < 0.0001):
            continue
        else:
            z_min = max(0,z-1)
            break
    for z in range(image.shape[2]-1, -1, -1):        
        if np.all(image[:,:,z] < 0.00001):
            continue
        else:
            z_max = min(z+2,image.shape[2])
            break
    image = image[:, :, z_min:z_max]
    return image



def preprocessPatient(patient):
    first_patient = load_scan(patient)
    pix_resampled = np.asarray(get_pixels_hu(first_patient))
    # print pix_resampled[70]
    # plt.imshow(pix_resampled[200], cmap='gray')
    # plt.show()
    # sys.exit()
    # test_segmented, test_lungfilter, test_outline, test_watershed, test_sobel_gradient, test_marker_internal, test_marker_external, test_marker_watershed = seperate_lungs(first_patient_pixels[65])
    # test_segmented = get_segmented_lungs(first_patient_pixels[65])
    # pix_resampled, _ = resample(first_patient_pixels, first_patient, [1, 1, 1])
    # print pix_resampled[70]
    # sys.exit()
    segmented_ct_scan = np.asarray([get_segmented_lungs(slice) for slice in pix_resampled])
    print ("Segmented Lung")
    # plt.ihow()
    # segmented_ct_scan[segmented_ct_scan < 604] = HU_MIN
    # print segmented_ct_scan[70]
    # plt.imshow(segmented_ct_scan[71], cmap='gray')
    # plt.show()
    # plt.s
    # plot_3d(segmented_ct_scan, 604)
    # sys.exit()
    print segmented_ct_scan.shape
    # plt.imshow(test_segmented, cmap='gray')
    # plt.show()
    # print ("Watershed Image")
    # plt.imshow(test_watershed, cmap='gray')
    # plt.show()
    # print ("Outline after reinclusion")
    # plt.imshow(test_outline, cmap='gray')
    # plt.show()
    # print ("Lungfilter after clot.imsing")
    # plt.imshow(test_lungfilter, cmap='gray')
    # plt.show()
    
    # plt.show()
    # plt.imshow(segmented_ct_scan[65], cmap='gray')
    # plt.show()

    # print segmented_ct_scan.shape
    # print segmented_ct_scan.min()
    # print segmented_ct_scan.max()
    # plot_3d(segmented_ct_scan, -400)
    '''
    selem = ball(2)
    binary = binary_closing(segmented_ct_scan, selem)

    label_scan = label(binary)
    # print np.sum(label_scan)
    # print binary[70]
    # print np.all(label_scan)

    areas = [r.area for r in regionprops(label_scan)]
    areas.sort()
    # print len(areas)
    components = 4

    for r in regionprops(label_scan):
        max_x, max_y, max_z = 0, 0, 0
        min_x, min_y, min_z = 1000, 1000, 1000
        
        for c in r.coords:
            max_z = max(c[0], max_z)
            max_y = max(c[1], max_y)
            max_x = max(c[2], max_x)
            
            min_z = min(c[0], min_z)
            min_y = min(c[1], min_y)
            min_x = min(c[2], min_x)
        if (min_z == max_z or min_y == max_y or min_x == max_x or r.area > areas[-components-1]):
            for c in r.coords:
                segmented_ct_scan[c[0], c[1], c[2]] = HU_MIN
        else:
            index = (max((max_x - min_x), (max_y - min_y), (max_z - min_z))) / (min((max_x - min_x), (max_y - min_y) , (max_z - min_z)))
    
    # print segmented_ct_scan[70]
    segmented_ct_scan = clip_edges(segmented_ct_scan)
    segmented_ct_scan = normalize(segmented_ct_scan)
    

    # print segmented_ct_scan.shape
    print segmented_ct_scan.shape
    # print segmented_ct_scan.min()
    # print segmented_ct_scan.max()
    # plot_3d(segmented_ct_scan, (604.-HU_MIN)/(HU_MAX - HU_MIN))
    # sys.exit()
    # pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1, 1, 1])
    # norm_lung_data = normalize(pix_resampled)
    # segmented_lungs_fill = segment_lung_mask(pix_resampled, True)
    # norm_lung_data = norm_lung_data * segmented_lungs_fill
    '''

    return segmented_ct_scan

# makes a directory
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def preprocessAll():
    i = 0
    for STAGE in range(1,3):
        path = '/media/ninja2/Seagate Backup Plus Drive/kaggle-lung-masks/%s-' % STAGE
        INPUT_FOLDER = '/media/ninja2/Seagate Backup Plus Drive/CS231N-project/stage%s/' % STAGE
        patients = os.listdir(INPUT_FOLDER)
        patients.sort()
        # ensure_dir(path)
        # n_slices = 64.0
        # x_dim = 128.0
        # y_dim = 128.0
        for p in range(len(patients)): #range(len(patients)):
            i += 1
            if i <= 1180:
                continue
            # if p != 402: continue
            print p
            print patients[p]
            # if p == 1137: continue
            x = preprocessPatient(INPUT_FOLDER + patients[p])
            # x_resample = x.astype(np.float16)
            x_resample = np.float16(ndimage.zoom(x, (0.5, 0.5, 0.5), order=0))
            # plt.imshow(x_resample[30].astype(np.float32), cmap='gray')
            # plt.show()
            # sys.exit()
            print x_resample.shape
            np.save(path + patients[p], x_resample)
            print('wrote patient' + path +patients[p])

if __name__ == "__main__":
    # testScans()
    preprocessAll()