import os
import re
from collections import defaultdict


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
				str2 = '/real_world' + '/' + str(root) + '/' + str(root) + '-' + str(value[i]) + '/' + str(root) +  \
					'-' + str(value[i]) + '_' + str(x) + '_' + str(y) + '.png'
				str1 = '/real_world' + '/' + str(root) + '/' + str(root) + '-' + str(value[i-1]) + '/' + str(root) +  \
					'-' + str(value[i-1]) + '_' + str(x) + '_' + str(y) + '.png'
				txt_file.write("%s \t %s \n" %(str1,str2))

	#close the file
	txt_file.close()


def main():
    	# specify the absolute path to the 
	path_to_files = '/home/mda/CSCI2470/real_world_rain_dataset_CVPR19/real_world'
	file_name = 'real_world_noisy_pairs.txt'
	# call the generate_text_file routine to create text file for noisy images
	generate_text_file(path_to_files, file_name)



if __name__ == '__main__':
    main()



