import keras.preprocessing.image
import argparse
import os
import numpy

def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "-i",
        "--in",
        type=str,
        required=True,
        help="The paths to the folder containing the images to augment.",
        metavar="paths",
        dest="input_path"
    )
    argument_parser.add_argument(
        "-o",
        "--out",
        type=str,
        required=True,
        help="The path to store the augmented images",
        metavar="path",
        dest="output_path"
    )
    argument_parser.add_argument(
        "-n",
        "--num-to-generate",
        type=int,
        required=False,
        default=4,
        help="The number of new images to generate for each original image.",
        metavar="num",
        dest="num_to_generate"
    )
    arguments = argument_parser.parse_args()
    image_data_generator = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=360,
        brightness_range=[0.5, 1.5]
    )
    input_path = str()
    images = []
    input_path = os.path.abspath(arguments.input_path)
    output_path = os.path.abspath(arguments.output_path)
    for file in os.listdir(input_path):
        file = os.path.join(input_path, file)
        images.append(keras.preprocessing.image.img_to_array(keras.preprocessing.image.load_img(file)))
    images = numpy.array(images)
    for _, _ in zip(image_data_generator.flow(images, batch_size=len(images), save_to_dir=output_path, save_format="jpg"), range(arguments.num_to_generate - 1)):
        ...

if __name__ == "__main__":
    main()