import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import cv2
import argparse
import igraph as ig
import time

GC_BGD = 0 # Hard bg pixel
GC_FGD = 1 # Hard fg pixel, will not be used
GC_PR_BGD = 2 # Soft bg pixel
GC_PR_FGD = 3 # Soft fg pixel

beta = 0
calc_Nlinks = False
previous_energy = 0
nlinks_graph = ig.Graph()
global_mask =[]
K = 0 

# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5):
    # Assign initial labels to the pixels based on the bounding box
    img = np.asarray(img, dtype=np.float64)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    global global_mask
    x, y, w, h = rect
    w -= x
    h -= y

    #Initalize the inner square to Foreground
    mask[y:y+h, x:x+w] = GC_PR_FGD
    mask[rect[1]+rect[3]//2, rect[0]+rect[2]//2] = GC_FGD
    bgGMM, fgGMM = initalize_GMMs(img, mask)

    num_iters = 1000
    for i in range(num_iters):
        #Update GMM
        print(f"Iter: {i}")
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask)
        global_mask = mask
        if check_convergence(energy):
            break
        
    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM


def initalize_GMMs(img, mask, n_components=5):
    
    #find pixels with background and forground colors
    bg_pix = img[np.logical_or(mask == GC_PR_BGD, mask == GC_BGD)].reshape(-1, 3)
    fg_pix = img[np.logical_or(mask == GC_PR_FGD, mask == GC_FGD)].reshape(-1, 3)

    #find clusters using kmeans
    kmeans_bg = KMeans(n_clusters=n_components, random_state=11)
    kmeans_fg = KMeans(n_clusters=n_components, random_state=11)
    kmeans_bg.fit(bg_pix)
    kmeans_fg.fit(fg_pix)

    #init GMM using the kmeans clusters
    fgGMM = GaussianMixture(n_components=n_components, means_init=kmeans_fg.cluster_centers_, random_state=11)
    bgGMM = GaussianMixture(n_components=n_components, means_init=kmeans_bg.cluster_centers_, random_state=11)
    bgGMM.fit(bg_pix)
    fgGMM.fit(fg_pix)

    return bgGMM, fgGMM


def update_GMMs(img, mask, bgGMM, fgGMM):
    fg_pix = img[np.logical_or(mask == GC_PR_FGD, mask == GC_FGD)].reshape(-1, 3)
    bg_pix = img[np.logical_or(mask == GC_PR_BGD, mask == GC_BGD)].reshape(-1, 3)

    bg_components = bgGMM.n_components
    bg_weights = np.zeros(bg_components)
    bg_means = np.zeros((bg_components, 3))
    bg_covs = np.zeros((bg_components, 3, 3))

    for i in range(bg_components):
        component_mask = bgGMM.predict(bg_pix) == i
        component_data = bg_pix[component_mask]

        if len(component_data) > 0:
            bg_weights[i] = len(component_data) / len(bg_pix)
            cov, mean = cv2.calcCovarMatrix(component_data, None, cv2.COVAR_NORMAL | cv2.COVAR_SCALE | cv2.COVAR_ROWS)
            bg_means[i] = mean.flatten()
            bg_covs[i] = cov

    bgGMM.means_ = bg_means
    bgGMM.weights_ = bg_weights
    bgGMM.covariances_ = bg_covs

    fg_components = fgGMM.n_components
    fg_weights = np.zeros(fg_components)
    fg_means = np.zeros((fg_components, 3))
    fg_covs = np.zeros((fg_components, 3, 3))

    for i in range(fg_components):
        component_mask = fgGMM.predict(fg_pix) == i
        component_data = fg_pix[component_mask]

        if len(component_data) > 0:
            fg_weights[i] = len(component_data) / len(fg_pix)
            cov, mean = cv2.calcCovarMatrix(component_data, None, cv2.COVAR_NORMAL | cv2.COVAR_SCALE | cv2.COVAR_ROWS)
            fg_means[i] = mean.flatten()
            fg_covs[i] = cov

    fgGMM.means_ = fg_means
    fgGMM.weights_ = fg_weights
    fgGMM.covariances_ = fg_covs

    fg_index_list = []
    bg_index_list = []

    for i in range(len(fgGMM.weights_)):
        if fgGMM.weights_[i] <= 0.005:
            fg_index_list.append(i)

    for i in range(len(bgGMM.weights_)):
        if bgGMM.weights_[i] <= 0.005:
            bg_index_list.append(i)

    if len(bg_index_list) > 0:
        bgGMM.n_components = bgGMM.n_components - len(bg_index_list)
        bgGMM.weights_ = np.delete(bgGMM.weights_, bg_index_list, axis=0)
        bgGMM.precisions_ = np.delete(bgGMM.precisions_, bg_index_list, axis=0)
        bgGMM.precisions_cholesky_ = np.delete(bgGMM.precisions_cholesky_, bg_index_list, axis=0)
        bgGMM.means_ = np.delete(bgGMM.means_, bg_index_list, axis=0)
        bgGMM.covariances_ = np.delete(bgGMM.covariances_, bg_index_list, axis=0)
        bgGMM.means_init = np.delete(bgGMM.means_init, bg_index_list, axis=0)

    if len(fg_index_list) > 0:
        fgGMM.n_components = fgGMM.n_components - len(fg_index_list)
        fgGMM.weights_ = np.delete(fgGMM.weights_, fg_index_list, axis=0)
        fgGMM.precisions_ = np.delete(fgGMM.precisions_, fg_index_list, axis=0)
        fgGMM.precisions_cholesky_ = np.delete(fgGMM.precisions_cholesky_, fg_index_list, axis=0)
        fgGMM.means_ = np.delete(fgGMM.means_, fg_index_list, axis=0)
        fgGMM.covariances_ = np.delete(fgGMM.covariances_, fg_index_list, axis=0)
        fgGMM.means_init = np.delete(fgGMM.means_init, fg_index_list, axis=0)

    return bgGMM, fgGMM


def calculate_beta(img):
    """ 
    calculate beta according to image
    """
    global beta
    image_col = img.shape[1]
    image_row = img.shape[0]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if i > 0:
                beta += np.linalg.norm(img[i, j] - img[i-1, j]) ** 2
            if j > 0:
                beta += np.linalg.norm(img[i, j] - img[i, j-1]) ** 2
            if i > 0 and j > 0:
                beta += np.linalg.norm(img[i, j] - img[i-1, j-1]) ** 2
            if i > 0 and j < image_col - 1:
                beta += np.linalg.norm(img[i, j] - img[i-1, j+1]) ** 2
            if i < image_row -1 and j > 0:
                beta += np.linalg.norm(img[i, j] - img[i+1, j-1]) ** 2
            if i < image_row -1 and j < image_col -1:
                beta += np.linalg.norm(img[i, j] - img[i+1, j+1]) ** 2
            if j < image_col -1:
                beta += np.linalg.norm(img[i,j] - img[i,j+1]) ** 2
            if i < image_row -1:
                beta += np.linalg.norm(img[i,j] - img[i+1,j]) ** 2
          
    beta = beta /  ((8 * image_row * image_col) - (2 * image_row + 2*image_col))
    beta = 2 * beta
    beta = 1 / beta


def compute_nlink_weight(i, j, oi, oj):
    # Compute the weight for N-links 
    global beta
    color_distance = (np.linalg.norm(img[i, j] - img[oi, oj])) ** 2 
    dist = np.sqrt(2) if (i != oi and j!=oj) else 1 
    return (50/dist) * np.exp(-beta * color_distance)

def vid(image_col,i,j):
    return (image_col*i) +j
    
def calculate_mincut(img, mask, bgGMM, fgGMM):
    # Build the graph
    global calc_Nlinks
    global nlinks_graph
    global K
    image_col = img.shape[1]
    image_row = img.shape[0]
    pixel_num = img.shape[0] * img.shape[1]
    graph = ig.Graph()  
    graph.add_vertices(pixel_num + 2)  
    backT = pixel_num
    foreT = pixel_num + 1
    edges = []
    weights = []
    if not calc_Nlinks:
        calculate_beta(img)
        nlinks_graph.add_vertices(pixel_num +2)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                total_weight = 0
                vid_pixel = vid(image_col, i, j)
                K =0
                # N-links 
                if i > 0:  
                    oi, oj = i - 1, j
                    weight = compute_nlink_weight(i, j, oi, oj)
                    edges.append((vid_pixel, vid(image_col,oi, oj)))
                    weights.append(weight)
                    total_weight += weight
                            
                if j > 0:  
                    oi, oj = i, j - 1
                    weight = compute_nlink_weight(i, j, oi, oj)
                    edges.append((vid_pixel, vid(image_col,oi, oj)))
                    weights.append(weight)
                    total_weight += weight
                
                if i < image_row -1:
                    oi,oj = i +1, j
                    weight = compute_nlink_weight(i, j, oi, oj)
                    edges.append((vid_pixel, vid(image_col,oi, oj)))
                    weights.append(weight)
                    total_weight += weight
                
                if j < image_col -1:
                    oi,oj = i, j+1
                    weight = compute_nlink_weight(i, j, oi, oj)
                    edges.append((vid_pixel, vid(image_col,oi, oj)))
                    weights.append(weight)
                    total_weight += weight
                    
                if i > 0 and j > 0: 
                    oi, oj = i - 1, j - 1
                    weight = compute_nlink_weight(i, j, oi, oj)
                    edges.append((vid_pixel, vid(image_col,oi, oj)))
                    weights.append(weight)
                    total_weight += weight
                    
                if i > 0 and j < img.shape[1] - 1: 
                    oi, oj = i - 1, j + 1
                    weight = compute_nlink_weight(i, j, oi, oj)
                    edges.append((vid_pixel, vid(image_col,oi, oj)))
                    weights.append(weight)
                    total_weight += weight
                
                if i < image_row -1 and j > 0:
                    oi,oj = i +1, j -1
                    weight = compute_nlink_weight(i, j, oi, oj)
                    edges.append((vid_pixel, vid(image_col,oi, oj)))
                    weights.append(weight)
                    total_weight += weight
                    
                if i < image_row -1 and j < image_col -1:
                    oi,oj= i+1, j +1
                    weight = compute_nlink_weight(i, j, oi, oj)
                    edges.append((vid_pixel, vid(image_col,oi, oj)))
                    weights.append(weight)
                    total_weight += weight
                        
                K = max(K, total_weight)
                   
        nlinks_graph.add_edges(edges, attributes={'weight' : weights})    
        calc_Nlinks = True
        
    graph = nlinks_graph.copy()
    edges = []
    weights = []
    #T-links 
    fg_D = - fgGMM.score_samples(img.reshape((-1, img.shape[-1]))).reshape(img.shape[:-1])
    bg_D = - bgGMM.score_samples(img.reshape((-1, img.shape[-1]))).reshape(img.shape[:-1])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            vid_pixel = vid(image_col, i, j)
            if mask[i,j] == GC_BGD:
                weight = K
                edges.append((vid_pixel,backT))
                weights.append(weight)
                edges.append((vid_pixel,foreT))
                weights.append(0)
            if mask[i,j] == GC_FGD:
                weight = K
                edges.append((vid_pixel,foreT))
                weights.append(weight)
                edges.append((vid_pixel,backT))
                weights.append(0)
            else:
                edges.append((vid_pixel,foreT))
                weights.append(bg_D[i,j])
                
                edges.append((vid_pixel,backT))
                weights.append(fg_D[i,j])
                     

    graph.add_edges(edges, attributes={'weight': weights})
    
    cut = graph.st_mincut(backT, foreT, capacity='weight') 
    energy = cut.value 
    bg_vertices = cut.partition[0]  
    fg_vertices = cut.partition[1]
    
    #check for correct allocation
    if foreT in bg_vertices:
        bg_vertices, fg_vertices = fg_vertices, bg_vertices
    #convert back to points and do not include S,T
    
    bg_vertices = [v for v in bg_vertices if v != backT and v != foreT]
    fg_vertices = [v for v in fg_vertices if v != backT and v != foreT]
    bg_set = list(bg_vertices)
    fg_set = list(fg_vertices)
    
    bg_indices = np.array(bg_set)
    bg_points = np.column_stack((bg_indices // mask.shape[1], bg_indices % mask.shape[1]))
    
    fg_indices = np.array(fg_set)
    fg_points = np.column_stack((fg_indices // mask.shape[1], fg_indices % mask.shape[1]))
         
    min_cut = [bg_points, fg_points]  
    
    return min_cut,energy

def update_mask(mincut_sets, mask):
    bg_points,fg_points = mincut_sets
    print(f"fg-set: {len(fg_points)}, bg-set: {len(bg_points)}")                   
    for point in fg_points:
        i, j = point
        if mask[i,j] == GC_PR_BGD or mask[i,j] == GC_PR_FGD:
            mask[i, j] = GC_PR_FGD  
    for point in bg_points:
        i, j = point
        if mask[i,j] == GC_PR_BGD or mask[i,j] == GC_PR_FGD:
            mask[i, j] = GC_PR_BGD 
    return mask



def check_convergence(energy):
    print(f"Energy: {energy}")
    global previous_energy
    global global_mask
    threshold=1000
    convergence = abs(energy - previous_energy) < threshold
    previous_energy = energy
    if convergence:
        global_mask[global_mask == GC_PR_BGD] = GC_BGD
    return convergence



def cal_metric(predicted_mask, gt_mask):
    #Accuracy
    correct_pixels = np.sum(predicted_mask == gt_mask)
    total_pixels = predicted_mask.size
    accuracy = correct_pixels / total_pixels

    # Jaccard similarity
    intersection_fg = np.sum((predicted_mask == 1) & (gt_mask == 1))
    union_fg = np.sum((predicted_mask == 1) | (gt_mask == 1))
    jaccard_similarity = intersection_fg / (union_fg)  

    return accuracy, jaccard_similarity

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='llama_high_blur', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()


if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    start_time = time.time()
    args = parse()


    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int,args.rect.split(',')))

    img = cv2.imread(input_path)

    # Run the GrabCut algorithm on the image and bounding box
    mask, bgGMM, fgGMM = grabcut(img, rect)
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])
    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCut Mask', 255 * mask)
    cv2.imshow('GrabCut Result', img_cut)
    end_time = time.time()
    print(f"Running Time: {end_time - start_time}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()