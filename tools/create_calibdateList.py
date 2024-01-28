import os
import sys

def write_image_names_to_txt(input_folder, output_folder, output_file, num_images=None):
    if not os.path.exists(input_folder):
        print(f"[E] The input folder '{input_folder}' does not exist.")
        return
    
    image_names = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    if not image_names:
        print(f"[E] Unable to find image file in folder '{input_folder}'.")
        return

    if num_images is not None:
        image_names = image_names[:num_images]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, output_file)

    with open(output_path, 'w') as txt_file:
        for image_name in image_names:
            txt_file.write(f"{image_name}\n")
    
    print(f"[I] The image name has been successfully written to the '{output_path}' folder.")

if __name__ == "__main__":
    if len(sys.argv) not in [4, 5]:
        print("Usage: python script.py <Input folder path> <Output folder path> <Output file name> [Optional: Number of images]")
    else:
        input_folder = sys.argv[1]
        output_folder = sys.argv[2]
        output_file = sys.argv[3]
        num_images = int(sys.argv[4]) if len(sys.argv) == 5 else None
        write_image_names_to_txt(input_folder, output_folder, output_file, num_images)


# usage demo: python tools/create_calibdateList.py ./calibdata ./calibdata filelist.txt 200
# If the fourth parameter is not passed, the names of all images will be obtained.