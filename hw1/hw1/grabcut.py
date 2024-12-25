import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import cv2
import argparse
import igraph as ig

GC_BGD = 0 # Hard bg pixel
GC_FGD = 1 # Hard fg pixel, will not be used
GC_PR_BGD = 2 # Soft bg pixel
GC_PR_FGD = 3 # Soft fg pixel

beta = 0.0
calc_Nlinks = False
previous_energy = 0
K = 0 
nlinks_graph = ig.Graph(directed=False)

# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5):
    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect

    #Initalize the inner square to Foreground
    mask[y:y+h, x:x+w] = GC_PR_FGD
    mask[rect[1]+rect[3]//2, rect[0]+rect[2]//2] = GC_FGD
    bgGMM, fgGMM = initalize_GMMs(img, mask)

    num_iters = 100
    for i in range(num_iters):
        #Update GMM
        print(f"Iter: {i}")
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        calculate_beta(img)
        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask)

        if check_convergence(energy):
            break

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM


def initalize_GMMs(img, mask):
    # Extract background and foreground pixels
    bg_pixels = img[np.where((mask == GC_BGD) | (mask == GC_PR_BGD))]  
    fg_pixels = img[np.where((mask == GC_FGD) | (mask == GC_PR_FGD))] 

    n_components = 5

    # Create GMMs Initialized with kmeans
    bgGMM = GaussianMixture(n_components=n_components, random_state=11,init_params='kmeans')
    fgGMM = GaussianMixture(n_components=n_components, random_state=11,init_params='kmeans')

    # Fit the GMMs using pixels
    bgGMM.fit(bg_pixels)
    fgGMM.fit(fg_pixels)

    return bgGMM, fgGMM


# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):
    
    bg_pixels = img[np.where((mask == GC_BGD) | (mask == GC_PR_BGD))]  
    fg_pixels = img[np.where((mask == GC_FGD) | (mask == GC_PR_FGD))] 

    # Predict GMM components (assignments) for each pixel
    bg_labels = bgGMM.predict(bg_pixels) 
    fg_labels = fgGMM.predict(fg_pixels) 

    # Update Background GMM
    for i in range(bgGMM.n_components):
        component_pixels = bg_pixels[bg_labels == i]

        mean = np.mean(component_pixels, axis=0) if len(component_pixels) > 0 else bgGMM.means_[i]
        cov = np.zeros((mean.shape[0], mean.shape[0]), dtype=np.float32)
        
        if len(component_pixels) > 1:
            cv2.calcCovarMatrix(component_pixels, mean=None,covar=cov, flags=cv2.COVAR_NORMAL | cv2.COVAR_ROWS)
            cov /= len(component_pixels)
        else:
            cov = bgGMM.covariances_[i]

        weight = len(component_pixels) / len(bg_pixels) if len(bg_pixels) > 0 else 1 / bgGMM.n_components

        bgGMM.means_[i] = mean
        bgGMM.covariances_[i] = cov
        bgGMM.weights_[i] = weight

    # Update Foreground GMM
    for i in range(fgGMM.n_components):
        component_pixels = fg_pixels[fg_labels == i]

        mean = np.mean(component_pixels, axis=0) if len(component_pixels) > 0 else fgGMM.means_[i]
        cov = np.zeros((mean.shape[0], mean.shape[0]), dtype=np.float32)
        
        if len(component_pixels) > 1:
            cv2.calcCovarMatrix(component_pixels, mean=None,covar=cov, flags=cv2.COVAR_NORMAL | cv2.COVAR_ROWS)
            cov /= len(component_pixels)
        else:
            cov = fgGMM.covariances_[i]
            
        weight = len(component_pixels) / len(fg_pixels) if len(fg_pixels) > 0 else 1 / fgGMM.n_components

        fgGMM.means_[i] = mean
        fgGMM.covariances_[i] = cov
        fgGMM.weights_[i] = weight

    return bgGMM,fgGMM

def calculate_beta(img):
    """
    Calculate beta using squared differences between neighboring pixels.
    connect_diag: include diagonal pixels in the calculation.
    """
    global beta
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if i > 0:
                diff = img[i, j] - img[i-1, j]
                beta += diff.dot(diff)
            if j > 0:
                diff = img[i, j] - img[i, j-1]
                beta += diff.dot(diff)
            if i > 0 and j > 0:
                diff = img[i, j] - img[i-1, j-1]
                beta += diff.dot(diff)
            if i > 0 and j < img.shape[1] - 1:
                diff = img[i, j] - img[i-1, j+1]
                beta += diff.dot(diff)

    beta /= (4 * img.shape[0] * img.shape[1] - 3 * img.shape[0] - 3 * img.shape[1] + 2)
    beta *= 2
    beta = 1 / beta


def compute_nlink_weight(i, j, oi, oj):
    # Compute the weight for N-links based on color difference
    global beta
    diff = img[i, j] - img[oi, oj]
    color_distance = np.dot(diff, diff)
    spatial_distance = np.sqrt((i - oi)**2 + (j - oj)**2)  
    return (50 / spatial_distance) * np.exp(-beta * color_distance)

def vid(image_col,i,j):
    """
    used to give a unique id to each vertex
    """
    return (image_col*i) +j

import numpy as np

    
def calculate_mincut(img, mask, bgGMM, fgGMM):
    # Build the graph
    global calc_Nlinks
    global nlinks_graph
    global K
    weight_sum = 0
    image_col = img.shape[1]
    pixel_num = img.shape[0] * img.shape[1]
    graph = ig.Graph(directed=False)  # Directed graph for T-links
    graph.add_vertices(pixel_num + 2)  # Add pixel nodes + S and T nodes
    S = pixel_num
    T = pixel_num + 1
    edges = []
    weights = []
    if not calc_Nlinks:
        nlinks_graph.add_vertices(pixel_num +2)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                vid_pixel = vid(image_col, i, j)
                
                # N-links (connect neighboring pixels)
                if i > 0:  
                    oi, oj = i - 1, j
                    weight = compute_nlink_weight(i, j, oi, oj)
                    edges.append((vid_pixel, vid(image_col,oi, oj)))
                    weights.append(weight)
                    weight_sum += weight
                            
                if j > 0:  
                    oi, oj = i, j - 1
                    weight = compute_nlink_weight(i, j, oi, oj)
                    edges.append((vid_pixel, vid(image_col,oi, oj)))
                    weights.append(weight)
                    weight_sum += weight
                    
                    
                if i > 0 and j > 0: 
                    oi, oj = i - 1, j - 1
                    weight = compute_nlink_weight(i, j, oi, oj)
                    edges.append((vid_pixel, vid(image_col,oi, oj)))
                    weights.append(weight)
                    weight_sum += weight
                    
                if i > 0 and j < img.shape[1] - 1: 
                    oi, oj = i - 1, j + 1
                    weight = compute_nlink_weight(i, j, oi, oj)
                    edges.append((vid_pixel, vid(image_col,oi, oj)))
                    weights.append(weight)
                    weight_sum += weight 
                   
            K = max(K,weight_sum)
        nlinks_graph.add_edges(edges, attributes={'weight' : weights})    
        calc_Nlinks = True
        
    graph = nlinks_graph.copy()
    edges = []
    weights = []
    #complitly redo Tlinks connections
    fg_D = - fgGMM.score_samples(img.reshape((-1, img.shape[-1]))).reshape(img.shape[:-1])
    bg_D = - bgGMM.score_samples(img.reshape((-1, img.shape[-1]))).reshape(img.shape[:-1])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            vid_pixel = vid(image_col, i, j)
            if mask[i,j] == 0:
                weight = K
                edges.append((vid_pixel,T))
                weights.append(K)
                weight_sum+= weight
            if mask[i,j] == 1:
                weight = K
                edges.append((vid_pixel,S))
                weights.append(K)
                weight_sum+= weight
            else:
                edges.append((vid_pixel,S))
                weights.append(bg_D[i,j])
                weight_sum+= bg_D[i,j]
                
                edges.append((vid_pixel,T))
                weights.append(fg_D[i,j])
                weight_sum += fg_D[i,j]
                     

    graph.add_edges(edges, attributes={'weight': weights})
    
    cut = graph.st_mincut(S, T, capacity='weight')  
    bg_vertices = cut.partition[0]  
    fg_vertices = cut.partition[1]
    
    #check for correct allocation
    if S in bg_vertices:
        bg_vertices, fg_vertices = fg_vertices, bg_vertices
    #convert back to points and do not include S,T
    
    bg_vertices = [v for v in bg_vertices if v != S and v != T]
    fg_vertices = [v for v in fg_vertices if v != S and v != T]
    bg_set = list(bg_vertices)
    fg_set = list(fg_vertices)
    
    bg_indices = np.array(bg_set)
    bg_points = np.column_stack((bg_indices // mask.shape[1], bg_indices % mask.shape[1]))
    
    fg_indices = np.array(fg_set)
    fg_points = np.column_stack((fg_indices // mask.shape[1], fg_indices % mask.shape[1]))
         
    min_cut = [bg_points, fg_points]  
    energy = sum(weight for weight in weights)
    
    return min_cut,energy

def update_mask(mincut_sets, mask):
    bg_points,fg_points = mincut_sets
    print(f"fg-set: {len(fg_points)}, bg-set: {len(bg_points)}")
    mask.fill(0)                   
    for point in fg_points:
        i, j = point
        mask[i, j] = GC_PR_FGD  
    return mask



def check_convergence(energy):
    print(f"Energy: {energy}")
    global previous_energy
    threshold=1e-3
    convergence = abs(energy - previous_energy) < threshold
    previous_energy = energy
    return convergence



def cal_metric(predicted_mask, gt_mask):
    correct_pixels = np.sum(predicted_mask == gt_mask)
    total_pixels = predicted_mask.size
    accuracy = correct_pixels / total_pixels

    # Jaccard similarity (IoU) calculation for foreground
    intersection_fg = np.sum((predicted_mask == 1) & (gt_mask == 1))
    union_fg = np.sum((predicted_mask == 1) | (gt_mask == 1))
    jaccard_similarity = intersection_fg / (union_fg + 1e-10)  # Add small epsilon to avoid division by 0

    return accuracy, jaccard_similarity

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='banana1', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()

def test_gmm_methods():
    # Load a test image
    img = cv2.imread('data/imgs/banana1.jpg')  # Replace with a valid path if needed

    # Create a dummy mask and bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    rect = (50, 50, 150, 150)  # Example bounding box
    mask[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = GC_PR_FGD
    mask[rect[1]+rect[3]//2, rect[0]+rect[2]//2] = GC_FGD
 

test_gmm_methods()

if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    args = parse()


    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int,args.rect.split(',')))

    print(rect)
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
    cv2.waitKey(0)
    cv2.destroyAllWindows()