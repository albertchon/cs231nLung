"""
Tutorial followed from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
"""
import numpy as np # linear algebra
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Some constants
INPUT_FOLDER = '/media/ninja2/Seagate Backup Plus Drive/CS231N-project/stage1/'
PIXEL_MEAN = 0.25
# INPUT_FOLDER = '/Users/albertchon/Desktop/cs231nProject/cs231nLung/patient1/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()

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
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)
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

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing

# 3d plot the image
def plot_3d(image, threshold=-300, show=False):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces = measure.marching_cubes(p, threshold)
    # verts, faces = measure.marching_cubes(p, None)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
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

def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8) + 1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label1 = labels[0, 0, 0]
    background_label2 = labels[0, 0, -1]
    background_label3 = labels[0, -1, 0]
    background_label4 = labels[0, -1, -1]
    background_label5 = labels[-1, 0, 0]
    background_label6 = labels[-1, 0, -1]
    background_label7 = labels[-1, -1, 0]
    background_label8 = labels[-1, -1, -1]

    # Fill the air around the person
    binary_image[background_label1 == labels] = 2
    binary_image[background_label2 == labels] = 2
    binary_image[background_label3 == labels] = 2
    binary_image[background_label4 == labels] = 2
    binary_image[background_label5 == labels] = 2
    binary_image[background_label6 == labels] = 2
    binary_image[background_label7 == labels] = 2
    binary_image[background_label8 == labels] = 2

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image

def zero_center(image):
    image = image - PIXEL_MEAN
    return image

# gives a run through of the preprocessing tutorial
def testScans():
    slices = load_scan(INPUT_FOLDER + patients[0])
    hu_slices = get_pixels_hu(slices)
    first_patient, first_patient_pixels = showOneExample(INPUT_FOLDER + patients[0])
    pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1, 1, 1])
    # print("Shape before resampling\t", first_patient_pixels.shape) # (128, 512, 512)
    # print("Shape after resampling\t", pix_resampled.shape)   # (320, 347, 347)
    # plot_3d(pix_resampled, 400, show=True)
    segmented_lungs = segment_lung_mask(pix_resampled, False)
    segmented_lungs_fill = segment_lung_mask(pix_resampled, True)
    show3D = False
    if show3D:
        # segmented_lungs has no air.
        plot_3d(segmented_lungs, 0)
        # plot_3d(segmented_lungs_fill, 0)
        # plot_3d(segmented_lungs_fill - segmented_lungs, 0)


def normalize(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    print(np.max(image), ' is max of image in normalize')
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return np.float32(image)


def preprocessPatient(patient):
    first_patient = load_scan(patient)
    first_patient_pixels = np.asarray(get_pixels_hu(first_patient))
    pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1, 1, 1])
    norm_lung_data = normalize(pix_resampled)
    segmented_lungs_fill = segment_lung_mask(pix_resampled, True)
    norm_lung_data = norm_lung_data * segmented_lungs_fill

    return norm_lung_data

# makes a directory
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def preprocessAll():
    path = './stage1-npy-3d/'
    ensure_dir(path)
    n_slices = 64.0
    x_dim = 128.0
    y_dim = 128.0
    for p in range(len(patients)): #range(len(patients)):
    	if p == 1137:
    		continue
        x = preprocessPatient(INPUT_FOLDER + patients[p])
        x_resample = np.float16(scipy.ndimage.zoom(x, (n_slices / x.shape[0], x_dim / x.shape[1], y_dim / x.shape[2]), order=0))
        np.save(path + patients[p], x_resample)
        print('wrote patient' + path +patients[p])

if __name__ == "__main__":
    # testScans()
    preprocessAll()