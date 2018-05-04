

def no_proc():
    def for_proc_fun(img_path):
        with open(img_path, 'rb') as f:
            img_bytes = f.read()
        return img_bytes
    
    def rev_proc_fun (rec_img_path, img_bytes):
        with open(rec_img_path, 'wb') as f:
            f.write(img_bytes)
    
    return (for_proc_fun, rev_proc_fun)

def grayscale_proc(shape):
    # shape is a tuple in row-column order
    
    def for_proc_fun(img_path):
        marconi_im = scipy.ndimage.imread(img_path, mode='L')
        marc_vec = np.ndarray.flatten(marconi_im)
        return np.ndarray.tobytes(marc_vec)
    
    def rev_proc_fun (rec_img_path, img_bytes):
        rec_vec = np.frombuffer(img_bytes, dtype=np.uint8)
        rows, cols = shape
        rec_mtx = np.reshape(rec_vec, [rows, cols])
        scipy.misc.imsave(rec_img_path, rec_mtx)
    
    return (for_proc_fun, rev_proc_fun)