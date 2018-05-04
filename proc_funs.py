

def no_proc():
    def for_proc_fun(img_path):
        with open(img_path, 'rb') as f:
            img_bytes = f.read()
        return img_bytes
    
    def rev_proc_fun (rec_img_path, img_bytes):
        with open(rec_img_path, 'wb') as f:
            f.write(img_bytes)
    
    return (for_proc_fun, rev_proc_fun)


def grayscale_proc():
    
    def for_proc_fun(img_path):
        marconi_im = scipy.ndimage.imread(img_path, mode='L')
        shape = marconi_im.shape
        marc_vec = np.ndarray.flatten(marconi_im)
        return (np.ndarray.tobytes(marc_vec), shape)
    
    def rev_proc_fun (rec_img_path, img_bytes, shape):
        rec_vec = np.frombuffer(img_bytes, dtype=np.uint8)
        rows, cols = shape
        rec_mtx = np.reshape(rec_vec, [rows, cols])
        scipy.misc.imsave(rec_img_path, rec_mtx)
    return (for_proc_fun, rev_proc_fun)


def color_proc():

	def for_proc_fun(img_path):
		test_img = scipy.ndimage.imread(img_path)
		channels = [test_img[:,:,0], test_img[:,:,1], test_img[:,:,2]]
		byte_list = []
    	shapes = []
		for channel in channels:
			curr_shape = channel.shape
			shapes.append(curr_shape)
			flattened_channel = np.reshape(channel, (curr_shape[0]*curr_shape[1]))
			curr_bytes = flattened_channel.tobytes()
			byte_list.append(curr_bytes)
		all_bytes = b''.join(byte_list)
		return (all_bytes, shapes)


	def rev_proc_fun(rec_img_path, img_bytes, shapes):
		flattend_channels = np.frombuffer(bytes_str, dtype = np.uint8)
    	channels = []
    	chn_idx = 0
    	for shape in shapes:
        	chn_end = chn_idx + shape[0]*shape[1]
        	flat_chn = flattend_channels[chn_idx:chn_end]
        	channel = np.reshape(flat_chn, shape)
        	channels.append(channel)
        	chn_idx = chn_end
    	scipy.misc.imsave(rec_img_path, channels)


