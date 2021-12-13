import glob
import os
'''
Find the last edited file in a dictionary
'''

def newest_file(path):
    files = os.listdir(path)
    if not files:
    	print("No file in: ", path)
    	checkpoint_path = ''
    	return checkpoint_path
    
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)
   

def all_subdirs_of(path):
  result = []
  for d in os.listdir(path):
    bd = os.path.join(path, d)
    if os.path.isdir(bd): result.append(bd)
  return result
  
def load_checkpoint_path(path):
	checkpoint_path = path
	no_folder_left = False

	while no_folder_left is False:
		all_subdirs = all_subdirs_of(checkpoint_path)
		if not all_subdirs:
			no_folder_left = True
			break
			
		latest_subdir_path = max(all_subdirs, key=os.path.getmtime)
		checkpoint_path = latest_subdir_path

	checkpoint_path = newest_file(checkpoint_path)
	
	if not checkpoint_path:
		checkpoint_path = ''
		return checkpoint_path

	#Checkpoint path has to be given without extension in model.restore() in test.py:
	checkpoint_path = checkpoint_path.replace('.tune_metadata', '')

	print(checkpoint_path)
	return checkpoint_path
