import os
import re
from collections import defaultdict
from PIL import Image 



def generate_text_file(path_to_files, file_name):
	"""
	Generates a text file with pairs of rainy images from adjacent images in time
	path_to_files [input]: path to the files whose structure will be written to the text file
	file_name [input] : file name which containts all noisy pairs
	return : none - file written to provided path
	"""
	#create a dict of lists
	dict_files = defaultdict(list)
	for root, dirs, files in os.walk(path_to_files):
		# go over each file name and split them to find out pairs
		for file in files:
			#speficy more than one delimeter in this case underscore and dash
			# only work on the ones contain 5 string pieces in total
			if len(re.split('[-_.]', file)) == 5:
				(root, time, x, y, png) = re.split('[-_.]', file)
				# create a key for each file name - use underscores to be able to decode
				key = root + '_' + x  + '_'+ y
				# create a dictionary consisting of lists
				dict_files[key].append(int(time))


	# open the file to be written
	txt_file = open(file_name,"w") 
	# traverse each key value pair in default dict
	for key,value in dict_files.items():
		# sort the frames in ascending order
		value.sort()
		# len of each list
		len_value = len(value)
		# decrypt the key into root file name, x, and y
		(root, x, y) = re.split("[_]", key)
		for i in range(1,len_value):
			# if they are adjacent we have found a pair!
			if value[i]-value[i-1] ==1:
				# file name for the pair
				str2 = path_to_files + '/' + str(root) + '/' + str(root) + '-' + str(value[i]) + '/' + str(root) +  \
					'-' + str(value[i]) + '_' + str(x) + '_' + str(y) + '.png'
				str1 = path_to_files + '/' + str(root) + '/' + str(root) + '-' + str(value[i-1]) + '/' + str(root) +  \
					'-' + str(value[i-1]) + '_' + str(x) + '_' + str(y) + '.png'
				txt_file.write("%s \t %s \n" %(str1,str2))

	#close the file
	txt_file.close()


def crop_images(gt, cropping_size, input_dir, output_dir):
	"""
	This routine crops and saves given pair of images consisting of gt (desnowed) and syn (snowed) images
	gt [input]: dir. to ground-truth image (desnowed) of random size 
	syn [input]: dir. to synthesized image (snowed) of above of the same size
	size [input]: cropping size of each image
	output_dir [input]: directory to which cropped images will be written
	return none
	"""
	# input directory of gt
	input_dir_gt = input_dir +'/gt/' + gt
	input_dir_syn = input_dir + '/synthetic/' + gt
	# open images in rgb mode
	img_gt, img_syn = Image.open(input_dir_gt, mode='r'), Image.open(input_dir_syn, mode='r')
	# detect origin of each image (center)
	center_x, center_y = int(img_gt.size[0]/2.0),int(img_gt.size[1]/2.0)
	dx, dy = int(cropping_size[0]/2.0), int(cropping_size[1]/2.0)
	# create a 4-tuple for cropping: origin is center in Cartesian move in x-y with dx,dy
	# convention: x_min, y_min, x_max, y_max
	crop_box = (center_x-dx,center_y-dy,center_x+dx,center_y+dy)
	# crop gt and syn
	img_gt_cropped = img_gt.crop(crop_box)
	img_syn_cropped = img_syn.crop(crop_box)
	# save the images
	img_gt_cropped.save(output_dir+'/gt/'+gt, format = img_gt.format)
	img_syn_cropped.save(output_dir+'/syn/'+gt, format = img_syn.format)

def process_DesnowNet(input_dir, output_dir, file_name, size):
        """
        This routine goes over each gt and syn image pairs to crop & and save to the specified dir.
        input_dir [input]: Input directory of images to be processed
        output_dir [input]: Output directory of images to which cropped images will be saved
	size [input]: tuple of dimensions
        """
	# input directions for gt and synthetic images
	# create output dir for gt
	if not os.path.isdir(output_dir+'/gt'):
		os.makedirs(output_dir+'/gt')
	#create output dir for syn
	if not os.path.isdir(output_dir+'/syn'):
		os.makedirs(output_dir+'/syn')
	#open the file
	txt_file = open(file_name,"w") 
	# walk over gt images and pair with corresponding synthethic ones
        for root, dirs, files in os.walk(input_dir+'/gt'):
		for img_gt in files:
			# call crop function to "crop" and "save"
			crop_images(img_gt, size, input_dir, output_dir)
			img_gt_dir = output_dir+'/gt/'+ img_gt
			img_syn_dir = output_dir +'/syn/' + img_gt
			txt_file.write("%s \t %s \n" %(img_gt_dir,img_syn_dir))
	#close the file
	txt_file.close()
		
	
def main():
	# SPECIFY the absolute path to the RAIN data
	path_to_files = '/home/mda/CSCI2470/real_world_rain_dataset_CVPR19/real_world'
	# SPECIFY the name of TEXT file for RAIN data
	file_name = '/real_world_noisy_pairs.txt'
	file_name = path_to_files + file_name
	# call the generate_text_file routine to create text file for noisy images
	generate_text_file(path_to_files, file_name)
	# SPEFICY ABSOLUTE INPUT PATH for DESNOW data - Note that we have 3 sets L-M-S
	input_dir = '/home/mda/CSCI2470/desnow/media/jdway/GameSSD/overlapping/test/Snow100K-M'
	# Cropped images and text file be written to input_dir + 'cropped'
	output_dir = input_dir + '/cropped'
	# SPECIFY the name of TEXT file for DESNOW data
	file_name_desnow = '/desnow_pairs.txt'
	file_name_desnow = output_dir + file_name_desnow
	# SPECIFY of CROPPING size
	# tuple of crop dimensions
	size = (256,256)
	# call processor function to crop & save images
	process_DesnowNet(input_dir,output_dir, file_name_desnow, size)
     
       
if __name__ == '__main__':
	main()



