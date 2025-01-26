import argparse
from PIL import Image
import numpy as np
import math

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere


def parse_scene_file(file_path):
    objects = []
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects


def save_image(image_array):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save("scenes/Spheres.png")


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    image_array = np.zeros((args.width, args.height, 3))
    
    #Calculate camera params
    camera.look_at = np.array(camera.look_at) - np.array(camera.position)
    camera.look_at /= np.linalg.norm(camera.look_at)
    camera.up_vector /= np.linalg.norm(camera.up_vector)
    
    #Calculate image center
    image_cen = camera.position + np.array(camera.look_at) * camera.screen_distance
    
    #Calculate vector right and up to define the local coordinate system of the camera
    vec_right = np.cross(camera.look_at, camera.up_vector)
    vec_right /= np.linalg.norm(vec_right)
    vec_up = np.cross(vec_right, camera.look_at)
    vec_up /= np.linalg.norm(vec_up)
    
    #Image to screen width ratio
    ratio = camera.screen_width / args.width
    
    for i in range(args.width):
        for j in range(args.height):
            print(f"i: {i}, j: {j}")
            ray = image_cen - vec_up*ratio*(i - math.floor(args.height/2)) - vec_right*ratio*(j - math.floor(args.width/2)) - camera.position
            ray /= np.linalg.norm(ray)
            
            trace_ray(ray,i,j,image_array,scene_settings, objects,camera.position,1)
    
    
    image_array = np.clip(image_array.astype(int),0,255)   
    
    # Save the output image
    save_image(image_array)
    
def trace_ray(ray ,i , j, image_array, scene_settings, objects,origin_point, depth):
    if depth > scene_settings.max_recursions:
        return np.array([0, 0, 0])
    
    closest_intersection_dist, closest_surface = find_closest_intersection(objects, origin_point, ray)
    
    # In case there is no Intersection, returns the background color
    if closest_surface[0] is None:
        if depth == 1:
            image_array[i][j] = np.array(scene_settings.background_color) * 255
        return np.array(scene_settings.background_color) * 255
    
    # In case there is an Intersection, calculates the color of the pixel
    else :
        # Calculate the noraml of the closest surface in each case
        if type(closest_surface[0]) == Sphere:
            normal = closest_surface[1] - closest_surface[0].position
            normal /= np.linalg.norm(normal)

        elif type(closest_surface[0]) == InfinitePlane:
            normal = closest_surface[0].normal
            normal /= np.linalg.norm(normal)

        elif type(closest_surface[0]) == Cube:
            center = closest_surface[0].position
            edge_length = closest_surface[0].scale
            
            planes = [(0,np.array([1,0,0]),center[0] + edge_length/2),
                      (1,np.array([-1,0,0]),center[0] - edge_length/2),
                      (2,np.array([0,1,0]),center[1] + edge_length/2),
                      (3,np.array([0,-1,0]),center[1] - edge_length/2),
                      (4,np.array([0,0,1]),center[2] + edge_length/2),
                      (5,np.array([0,0,-1]),center[2] - edge_length/2)]
            
            minimal_distance = float('inf')

            for plane_id, plane_normal, plane_position in planes :
                distance = abs(closest_surface[1][plane_id//2] - plane_position)
                if distance < minimal_distance :
                    minimal_distance = distance
                    normal = plane_normal

        view = -(origin_point - closest_surface[1])
        view /= np.linalg.norm(view)

        # Indentifying the material of the closest surface
        material_index = closest_surface[0].material_index
        counter = 0
        surface_material = None
        
        for object in objects:
            
            if type(object) == Material :
                counter += 1
                
                if counter == material_index:
                    surface_material = object
                    break 
                
        material_diffuse = surface_material.diffuse_color
        material_specular = surface_material.specular_color
        
        return_color,reflected_ray = apply_lightning_effect(objects,closest_surface,normal,ray,scene_settings,surface_material,material_diffuse,material_specular,view)

        recursion_color = trace_ray(reflected_ray,i,j,image_array,scene_settings,objects,closest_surface[1],depth + 1)
        
        if surface_material.transparency == 0 :
            return_color += np.array(surface_material.reflection_color) * recursion_color
            
        else :
            transparency_color = trace_ray(ray,i,j,image_array,scene_settings,objects,closest_surface[1],depth + 1)
            return_color += transparency_color * np.array(surface_material.transparency) + np.array(surface_material.reflection_color)*recursion_color
        
        if depth == 1:
            image_array[i][j] = return_color
            
        return return_color




def apply_lightning_effect(objects,closest_surface,normal,ray,scene_settings,surface_material,material_diffuse,material_specular,view):
    return_color = np.zeros(3)
    for light in objects:
          
        if type(light) is not Light:
            continue
        else:

            shadow_intensity = light.shadow_intensity

            # Calculate the vector from the intersection point to the light and normalize it
            light_intersection = light.position - closest_surface[1]
            light_intersection /= np.linalg.norm(light_intersection)

            # Calculate the vector from the intersection point to the reflected light and normalize it
            reflected_light_intersction = 2 * np.dot(light_intersection,normal) * normal - light_intersection
            reflected_light_intersction /= np.linalg.norm(reflected_light_intersction)

            # Calculate the reflected ray
            reflected_ray = ray - 2 * np.dot(ray, normal) * normal
            reflected_ray /= np.linalg.norm(reflected_ray)

            # Get the grid width
            grid_width = light.radius

            # Get the grid ratio by dividing the grid width by the number of shadow rays (getting the size of each
            # grid cell)
            grid_ratio = grid_width / scene_settings.root_number_shadow_rays

            # Create a vector that is different from the intersection to light vector
            rand_vector = np.array([light_intersection[0], light_intersection[1], light_intersection[2] + 1])
            # Normalize the vector
            rand_vector = rand_vector / np.linalg.norm(rand_vector)
            # Get a vector that is perpendicular to the intersection to light vector and the random vector
            light_v_up = np.cross(-light_intersection, rand_vector)
            # Normalize the vector
            light_v_up = light_v_up / np.linalg.norm(light_v_up)
            # Get a vector that is perpendicular to the intersection to light vector and the light v up vector
            light_v_right = np.cross(-light_intersection, light_v_up)
            # Normalize the vector
            light_v_right = light_v_right / np.linalg.norm(light_v_right)

            # Initialize the shadow rays count
            shadow_rays_count = 0

            # Go for every grid cell
            for x in range(int(scene_settings.root_number_shadow_rays)):
                for y in range(int(scene_settings.root_number_shadow_rays)):
                    # Calculate the point on the grid (with a random offset)
                    point_on_grid = light.position - light_v_right * grid_ratio * (
                        x - math.floor(
                        scene_settings.root_number_shadow_rays / 2)) - light_v_up * grid_ratio * (y - math.floor(
                        scene_settings.root_number_shadow_rays) / 2) + ((np.random.rand() - 0.5) * grid_ratio * light_v_right +
                                                                        (np.random.rand() - 0.5) * grid_ratio * light_v_up)
            
                    grid_ray = - (point_on_grid - closest_surface[1])
                    grid_ray = grid_ray / np.linalg.norm(grid_ray)

                   
                    point_closest_intersection_distance, point_closest_surface = find_closest_intersection(objects,point_on_grid,grid_ray)
                    
                    is_hit = True
                    for coord in range(3):
                        if abs(point_closest_surface[1][coord] - closest_surface[1][coord]) > 0.00001:
                            is_hit = False
                            break
                        
                    if is_hit:
                        shadow_rays_count += 1

            # Calculate the light intensity
            light_intensity = (1 - shadow_intensity) * 1 + shadow_intensity * (
                    shadow_rays_count / (scene_settings.root_number_shadow_rays ** 2))

            # Calculate the diffusion and specular for the current light
            diffusion_and_specular = (np.array(material_diffuse) * np.dot(normal, light_intersection) + \
                                        np.array(material_specular) * np.dot(view,
                                                                            reflected_light_intersction) ** surface_material.shininess) * light_intensity * light.specular_intensity

            # Add the diffusion and specular of the current light to the return color
            return_color += np.array(diffusion_and_specular) * \
                            (1 - surface_material.transparency) * np.array(light.color) * 255
                            
    return return_color,reflected_ray
                       

def find_closest_intersection(objects, origin_point, ray):
    closest_surface = (None, float('inf'))
    closest_intersection_dist = float('inf')
    
    for item in objects:
        if type(item) == Sphere:
            #Calculate coefficients to find components of the quadratic equaiton
            coefficients = [1, np.dot(2 * ray, np.array(origin_point) - np.array(item.position)),
                            np.linalg.norm(np.array(origin_point) - np.array(
                            item.position)) ** 2 - item.radius ** 2]
            
            discriminant = (coefficients[1] ** 2) - (4 * coefficients[0] * coefficients[2])
            
            #If discriminent positive find results for the equation
            if discriminant >= 0:
                roots = [(-coefficients[1] - math.sqrt(discriminant)) / (2 * coefficients[0]),
                         (-coefficients[1] + math.sqrt(discriminant)) / (2 * coefficients[0])]

                for t in roots:
                    if 1e-7 < t < closest_intersection_dist:
                        point_of_intersection = origin_point + t * ray
                        closest_intersection_dist = t
                        closest_surface = (item, point_of_intersection)

        
        elif type(item) == Cube:
            #Cube properties
            center = item.position
            edge_len = item.scale
            
            #Calculate cube surface to center vectors
            center_dist = np.array([edge_len/2,edge_len/2,edge_len/2])
            
            min_corner, max_corner = center - center_dist, center + center_dist
            
            x_min, x_max = sorted([(min_corner[0] - origin_point[0]) / ray[0], (max_corner[0] - origin_point[0]) / ray[0]])
            y_min, y_max = sorted([(min_corner[1] - origin_point[1]) / ray[1], (max_corner[1] - origin_point[1]) / ray[1]])
            z_min, z_max = sorted([(min_corner[2] - origin_point[2]) / ray[2], (max_corner[2] - origin_point[2]) / ray[2]])

            t_min = max(x_min,y_min,z_min)
            t_max = min(x_max,y_max,z_max)

            if t_min < t_max:
                #check if the ray intersect the cube
                if 1e-7 < t_min < closest_intersection_dist:
                    closest_intersection_dist = t_min
                    closest_surface = (item, origin_point + closest_intersection_dist * ray)
                    
        elif type(item) == InfinitePlane:
            
            normalized_item = np.array(item.normal)
            normalized_item /= np.linalg.norm(normalized_item)
            #if ray not parallel to the plane calculate distance to plane
            if np.dot(ray,normalized_item) != 0 :
                t = -(np.dot(origin_point, normalized_item) - item.offset) / np.dot(ray, normalized_item)
                if 1e-7 < t < closest_intersection_dist:
                    intersection_point = origin_point + t * ray
                    closest_intersection_dist = t
                    closest_surface = (item, intersection_point)
        else:
            continue
        
    return closest_intersection_dist, closest_surface

if __name__ == '__main__':
    main()