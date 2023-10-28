import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk , ImageFilter , ImageDraw
from tkinter import ttk
import tkinter as tk
from tkinter import filedialog
import tkinter.messagebox  # Import messagebox module
import numpy as np  # Add this import statement
from PIL import ImageFilter, ImageOps
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage import data, morphology
from skimage.color import rgb2gray
from skimage.filters import sobel
import scipy.ndimage as nd
from skimage.segmentation import watershed
from skimage.color import label2rgb
from skimage.measure import label
import random

# Global variables for image and zoom factor
current_image = None
original_image = None  # Store the original image
zoom_factor = 1.0
crop_coords = None

# Adjustment parameters
brightness_factor = 1.0
contrast_factor = 1.0
blur_radius = 0


def open_image():
    global current_image, original_image
    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=(("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ppm *.pgm"), ("All files", "*.*"))
    )

    if file_path:
        image = cv2.imread(file_path)

        if image is None:
            tkinter.messagebox.showerror("Error", "Image not found or cannot be opened.")
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            current_image = Image.fromarray(image)
            original_image = current_image.copy()  # Store the original image
            display_image(current_image, zoom_factor)
    else:
        tkinter.messagebox.showinfo("Information", "No image selected.")


# Function to start crop selection
def start_crop(event):
    global crop_coords
    if current_image:
        crop_coords = (event.x, event.y, event.x, event.y)

# Function to update crop selection
def update_crop(event):
    global crop_coords
    if current_image and crop_coords:
        crop_coords = (crop_coords[0], crop_coords[1], event.x, event.y)

# Function to end crop selection
def end_crop(event):
    global crop_coords
    if current_image and crop_coords:
        crop_coords = (crop_coords[0], crop_coords[1], event.x, event.y)
        crop_image()

# Function to crop the image
def crop_image():
    global current_image, crop_coords
    if current_image and crop_coords:
        left, upper, right, lower = crop_coords
        # Calculate coordinates in the original image size
        left = int(left / zoom_factor)
        upper = int(upper / zoom_factor)
        right = int(right / zoom_factor)
        lower = int(lower / zoom_factor)

        # Make a copy of the current image for cropping
        cropped_image = current_image.copy()
        current_image = cropped_image.crop((left, upper, right, lower))
        display_image(current_image, zoom_factor)

# Function to draw the crop rectangle
def draw_crop_rectangle(image, coords):
    if coords:
        left, upper, right, lower = coords
        img_with_rectangle = image.copy()
        draw = ImageDraw.Draw(img_with_rectangle)
        draw.rectangle([left, upper, right, lower], outline="red")
        return img_with_rectangle
    return image

# Function to display the image with crop rectangle
def display_image_with_crop(img, zoom, coords):
    zoomed_image = img.resize((int(img.width * zoom), int(img.height * zoom)))
    img_with_crop = draw_crop_rectangle(zoomed_image, coords)
    photo = ImageTk.PhotoImage(image=img_with_crop)
    image_label.configure(image=photo)
    image_label.image = photo

# Function to save the current image
def save_image():
    if current_image:
        file_path = filedialog.asksaveasfilename(defaultextension=".png")
        if file_path:
            current_image.save(file_path)
            tkinter.messagebox.showinfo("Image Saved", "The image has been saved successfully.")

def display_image(img, zoom):
    global current_image, zoom_factor
    zoomed_image = img.resize((int(img.width * zoom), int(img.height * zoom)))
    photo = ImageTk.PhotoImage(image=zoomed_image)

    # Update the displayed image
    image_label.configure(image=photo)
    image_label.image = photo
    zoom_factor = zoom

def reset_image():
    global current_image, original_image
    if original_image is not None:
        current_image = original_image.copy()
        display_image(current_image, zoom_factor)

def zoom_in():
    global zoom_factor
    zoom_factor *= 1.2
    display_image(current_image, zoom_factor)

def zoom_out():
    global zoom_factor
    zoom_factor *= 0.8
    display_image(current_image, zoom_factor)

def adjust_brightness(factor):
    global current_image
    if current_image:
        img_array = np.array(current_image)
        img_array = img_array * factor
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        current_image = Image.fromarray(img_array)
        display_image(current_image, zoom_factor)

def adjust_contrast(factor):
    global current_image
    if current_image:
        img_array = np.array(current_image)
        img_mean = img_array.mean()
        img_array = (img_array - img_mean) * factor + img_mean
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        current_image = Image.fromarray(img_array)
        display_image(current_image, zoom_factor)

# Function to adjust blur by increasing
def increase_blur():
    global current_image, blur_radius
    blur_radius += 1
    if current_image:
        current_image = current_image.filter(ImageFilter.GaussianBlur(blur_radius))
        display_image(current_image, zoom_factor)

def apply_image_negatives(output_image_path=None):
    global current_image
    if current_image:
        current_image_array = np.array(current_image)
        # Display the original image
        plt.figure(figsize=(8, 8))
        plt.imshow(current_image_array)
        plt.axis("off")
        plt.title("Original Image")
        plt.show()

        # Calculate the image negative (linear transformation)
        negative_image = 255 - current_image_array

        # Create an image from the negative result
        negative_image = Image.fromarray(negative_image)

        # Display the negative image
        plt.figure(figsize=(8, 8))
        plt.imshow(negative_image)
        plt.axis("off")
        plt.title("Image Negatives (Linear)")
        plt.show()

        if output_image_path:
            negative_image.save(output_image_path)

# Example usage:
# apply_image_negatives('output_image.jpg')


def apply_log_transformations(output_image_path=None):
    global current_image
    if current_image:
        current_image_array = np.array(current_image)
        
        # Display the original image
        plt.figure(figsize=(8, 8))
        plt.imshow(current_image_array)
        plt.axis("off")
        plt.title("Original Image")
        plt.show()

        # Apply the log transformation
        log_transformed_image = np.log1p(current_image_array)

        # Normalize the result to 0-255
        log_transformed_image = (255 * (log_transformed_image - np.min(log_transformed_image)) / (np.max(log_transformed_image) - np.min(log_transformed_image))).astype(np.uint8)

        # Create an image from the log-transformed result
        log_transformed_image = Image.fromarray(log_transformed_image)

        # Display the log-transformed image
        plt.figure(figsize=(8, 8))
        plt.imshow(log_transformed_image, cmap="gray")
        plt.axis("off")
        plt.title("Log Transformations")
        plt.show()

        if output_image_path:
            log_transformed_image.save(output_image_path)

def apply_power_law_transform(output_image_path=None):
    global current_image
    if current_image:
        current_image_array = np.array(current_image)

        # Display the original image
        plt.figure(figsize=(8, 8))
        plt.imshow(current_image_array)
        plt.axis("off")
        plt.title("Original Image")
        plt.show()

        # Define a list of gamma values to apply
        gamma_values = [0.1, 0.5, 1.2, 2.2]
        
        # Create an empty list to store transformed images
        transformed_images = []

        # Apply the power-law (gamma) transformation for each gamma value
        for gamma in gamma_values:
            power_law_transformed_image = np.power(current_image_array / 255.0, gamma)
            power_law_transformed_image = (power_law_transformed_image * 255).astype(np.uint8)
            transformed_images.append(power_law_transformed_image)

        # Display the power-law transformed images for each gamma value
        for i, gamma in enumerate(gamma_values):
            plt.figure(figsize=(8, 8))
            plt.imshow(transformed_images[i])
            plt.axis("off")
            plt.title(f"Power-Law (Gamma) Transformation (Gamma={gamma})")
            plt.show()

        if output_image_path:
            # Save the last transformed image with the highest gamma value
            Image.fromarray(transformed_images[-1]).save(output_image_path)

    # Example usage:
    # apply_power_law_transform('input_image.jpg', gamma=0.5)

def apply_piecewise_linear_transform(output_image_path=None):
    global current_image
    if current_image:
        current_image_array = np.array(current_image)

        # Display the original image
        plt.figure(figsize=(8, 8))
        plt.imshow(current_image_array)
        plt.axis("off")
        plt.title("Original Image")
        plt.show()

        # Define piecewise-linear transformation function
        def piecewise_linear(x):
            if 0 <= x < 128:
                return 2 * x
            elif 128 <= x < 192:
                return x
            else:
                return 255 - 2 * (255 - x)

        # Apply the piecewise-linear transformation
        piecewise_linear_transformed_image = np.vectorize(piecewise_linear)(current_image_array)

        # Create an image from the transformed result
        piecewise_linear_transformed_image = Image.fromarray(piecewise_linear_transformed_image)

        # Display the piecewise-linear transformed image
        plt.figure(figsize=(8, 8))
        plt.imshow(piecewise_linear_transformed_image)
        plt.axis("off")
        plt.title("Piecewise-Linear Transformation Function")
        plt.show()

        if output_image_path:
            piecewise_linear_transformed_image.save(output_image_path)

def apply_gaussian_blur(output_image_path=None):
    global current_image
    if current_image:
        current_image_array = np.array(current_image)

        # Display the original image
        plt.figure(figsize=(8, 8))
        plt.imshow(current_image_array)
        plt.axis("off")
        plt.title("Original Image")
        plt.show()

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(current_image_array, (5, 5), 0)

        # Create an image from the blurred result
        blurred_image = Image.fromarray(blurred)

        # Display the blurred image
        plt.figure(figsize=(8, 8))
        plt.imshow(blurred_image)
        plt.axis("off")
        plt.title("Blurred Image")
        plt.show()

        # Save the cartoonized image if an output path is provided
        if output_image_path:
           blurred_image.save(output_image_path)


    # Example usage:
    # apply_gaussian_blur('input_image.jpg')



def apply_embossing(output_image_path=None):
        global current_image
        if current_image:
             current_image_array = np.array(current_image)

        # Display the original image
        plt.figure(figsize=(8, 8))
        plt.imshow(current_image_array)
        plt.axis("off")
        plt.title("Original Image")
        plt.show()

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(current_image_array, cv2.IMREAD_GRAYSCALE)

        # Define the embossing kernel
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])

        # Apply the embossing effect
        embossed = cv2.filter2D(gray_image, -1, kernel)

        # Create an image from the embossed result
        embossed_image = Image.fromarray(embossed)

        # Display the embossed image
        plt.figure(figsize=(8, 8))
        plt.imshow(embossed_image, cmap="gray")
        plt.axis("off")
        plt.title("Embossed Image")
        plt.show()

         # Save the cartoonized image if an output path is provided
        if output_image_path:
            embossed_image.save(output_image_path)



def apply_median_filter(output_image_path=None):
    global current_image
    if current_image:
        current_image_array = np.array(current_image)

        # Display the original image
        plt.figure(figsize=(8, 8))
        plt.imshow(current_image_array)
        plt.axis("off")
        plt.title("Original Image")
        plt.show()

        # Apply Median filter
        median_filtered = cv2.medianBlur(current_image_array, 5)

        # Create an image from the filtered result
        filtered_image = Image.fromarray(median_filtered)

        # Display the filtered image
        plt.figure(figsize=(8, 8))
        plt.imshow(filtered_image)
        plt.axis("off")
        plt.title("Median Filtered Image")
        plt.show()

        # Save the filtered image if an output path is provided
        if output_image_path:
            filtered_image.save(output_image_path)

def apply_box_blur(output_image_path=None):
    global current_image
    if current_image:
        current_image_array = np.array(current_image)

        # Display the original image
        plt.figure(figsize=(8, 8))
        plt.imshow(current_image_array)
        plt.axis("off")
        plt.title("Original Image")
        plt.show()

        # Define a box blur kernel
        kernel = np.ones((5, 5), np.float32) / 25
        box_blurred = cv2.filter2D(current_image_array, -1, kernel)

        # Create an image from the box blurred result
        box_blurred_image = Image.fromarray(box_blurred)

        # Display the box blurred image
        plt.figure(figsize=(8, 8))
        plt.imshow(box_blurred_image)
        plt.axis("off")
        plt.title("Box Blurred Image")
        plt.show()

        # Save the box blurred image if an output path is provided
        if output_image_path:
            box_blurred_image.save(output_image_path)

def apply_mean_shift_filter(output_image_path=None):
    global current_image
    if current_image:
        current_image_array = np.array(current_image)

        # Display the original image
        plt.figure(figsize=(8, 8))
        plt.imshow(current_image_array)
        plt.axis("off")
        plt.title("Original Image")
        plt.show()

        # Apply Mean Shift filter
        mean_shift_filtered = cv2.pyrMeanShiftFiltering(current_image_array, sp=15, sr=60)

        # Create an image from the mean shift filtered result
        filtered_image = Image.fromarray(mean_shift_filtered)

        # Display the mean shift filtered image
        plt.figure(figsize=(8, 8))
        plt.imshow(filtered_image)
        plt.axis("off")
        plt.title("Mean Shift Filtered Image")
        plt.show()

        # Save the filtered image if an output path is provided
        if output_image_path:
            filtered_image.save(output_image_path)


def apply_bilateral_filter(output_image_path=None):
    global current_image
    if current_image:
        current_image_array = np.array(current_image)

        # Display the original image
        plt.figure(figsize=(8, 8))
        plt.imshow(current_image_array)
        plt.axis("off")
        plt.title("Original Image")
        plt.show()

        # Apply Bilateral filter
        bilateral_filtered = cv2.bilateralFilter(current_image_array, 9, 75, 75)

        # Create an image from the bilateral filtered result
        filtered_image = Image.fromarray(bilateral_filtered)

        # Display the bilateral filtered image
        plt.figure(figsize=(8, 8))
        plt.imshow(filtered_image)
        plt.axis("off")
        plt.title("Bilateral Filtered Image")
        plt.show()

        # Save the filtered image if an output path is provided
        if output_image_path:
            filtered_image.save(output_image_path)



def apply_cartoonization( output_image_path=None):
    global current_image
    if current_image:
        current_image_array = np.array(current_image)
        # Display the original image
        plt.figure(figsize=(8, 8))
        plt.imshow(current_image_array)
        plt.axis("off")
        plt.title("Original Image")
        plt.show()

        # Convert the original image to grayscale
        img_gray = cv2.cvtColor(current_image_array, cv2.COLOR_BGR2GRAY)

        # Display the grayscale image
        plt.figure(figsize=(8, 8))
        plt.imshow(img_gray, cmap="gray")
        plt.axis("off")
        plt.title("Black or white")
        plt.show()

        # Apply median blur to the grayscale image
        img_blur = cv2.medianBlur(img_gray, 5)

        # Display the blurred image
        plt.figure(figsize=(8, 8))
        plt.imshow(img_blur, cmap="gray")
        plt.axis("off")
        plt.title("Blurred Image")
        plt.show()

        # Apply adaptive thresholding to the blurred image
        img_edges = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

        # Display the image with edges
        plt.figure(figsize=(8, 8))
        plt.imshow(img_edges, cmap="gray")
        plt.axis("off")
        plt.title("Image with Edges")
        plt.show()

        # Create the cartoonized image
        img_color = cv2.bilateralFilter(current_image_array, 9, 300, 300)
        cartoon = cv2.bitwise_and(img_color, img_color, mask=img_edges)

        # Display the cartoonized image
        plt.figure(figsize=(8, 8))
        plt.imshow(cartoon)
        plt.axis("off")
        plt.title("Cartoonized Image")
        plt.show()

        # Save the cartoonized image if an output path is provided
        if output_image_path:
            cv2.imwrite(output_image_path, cartoon)


def apply_oil_painting(output_image_path=None):
    global current_image
    if current_image:
        current_image_array = np.array(current_image)

        # Display the original image
        plt.figure(figsize=(8, 8))
        plt.imshow( current_image_array)
        plt.axis("off")
        plt.title("Original Image")
        plt.show()

        # Apply an oil painting effect
        oil_painting = cv2.stylization(current_image_array, sigma_s=60, sigma_r=0.07)

        # Display the oil painting effect
        plt.figure(figsize=(8, 8))
        plt.imshow(oil_painting)
        plt.axis("off")
        plt.title("Oil Painting Effect")
        plt.show()

        # Convert the oil painting effect to a PIL Image
        oil_painting_image = Image.fromarray(oil_painting)

        # Display the oil painting effect as a PIL Image
        plt.figure(figsize=(8, 8))
        plt.imshow(oil_painting_image)
        plt.axis("off")
        plt.title("Oil Painting Effect (PIL Image)")
        plt.show()

        if output_image_path:
            oil_painting_image.save(output_image_path)



def apply_pencil_sketch(output_image_path=None):
    global current_image
    if current_image:
        current_image_array = np.array(current_image)

        # Display the original image
        plt.figure(figsize=(8, 8))
        plt.imshow(current_image_array)
        plt.axis("off")
        plt.title("Original Image")
        plt.show()

        img = cv2.cvtColor(current_image_array, cv2.COLOR_BGR2RGB)

        # Display the grayscale image
        plt.figure(figsize=(8, 8))
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title("GrayScale Image")
        plt.show()

        # Convert the original image to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Display the grayscale image
        plt.figure(figsize=(8, 8))
        plt.imshow(img_gray, cmap="gray")
        plt.axis("off")
        plt.title("Black or White Image")
        plt.show()


        # Invert the grayscale image
        img_invert = cv2.bitwise_not(img_gray)

        # Display the inverted image
        plt.figure(figsize=(8, 8))
        plt.imshow(img_invert, cmap="gray")
        plt.axis("off")
        plt.title("Inverted Image")
        plt.show()

        # Apply Gaussian blur to the inverted image
        img_smoothing = cv2.GaussianBlur(img_invert, (21, 21), sigmaX=0, sigmaY=0)

        # Display the smoothened image
        plt.figure(figsize=(8, 8))
        plt.imshow(img_smoothing, cmap="gray")
        plt.axis("off")
        plt.title("Smoothen Image")
        plt.show()

        # Create the final sketch image
        final = cv2.divide(img_gray, 255 - img_smoothing, scale=255)

        # Display the final sketch image
        plt.figure(figsize=(8, 8))
        plt.imshow(final, cmap="gray")
        plt.axis("off")
        plt.title("Final Sketch Image")
        plt.show()

        # Save the final sketch image if an output path is provided
        if output_image_path:
            cv2.imwrite(output_image_path, final)

def apply_canny_edge_detection(output_image_path=None):
    global current_image
    if current_image:
        current_image_array = np.array(current_image)

    # Display the original image
    plt.figure(figsize=(8, 8))
    plt.imshow(current_image_array)
    plt.axis("off")
    plt.title("Original Image")
    plt.show()

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(current_image_array, cv2.IMREAD_GRAYSCALE)

    # Perform Canny edge detection
    edges = cv2.Canny(gray_image, 100, 200)

    # Create an image from the edges
    edge_image = Image.fromarray(edges)

    # Display the image with edges
    plt.figure(figsize=(8, 8))
    plt.imshow(edge_image, cmap="gray")
    plt.axis("off")
    plt.title("Image with Edges (Canny Edge Detection)")
    plt.show()

    # Save the edge-detected image if an output path is provided
    if output_image_path:
        edge_image.save(output_image_path)


def open_image_operations_window():
    if current_image:
        operations_window = tk.Toplevel(root)
        operations_window.title("Image Operations")

        # Set background color for the entire operations window
        operations_window.configure(bg="#2196F3")  # Use your preferred color code

        canvas = tk.Canvas(operations_window, bg="#2196F3")
        canvas.pack(expand=True, fill="both")

        main_functions_frame = ttk.LabelFrame(canvas, text="Main Functions")
        main_functions_frame.grid(row=0, column=0, padx=10, pady=10)

        flipping_frame = ttk.LabelFrame(main_functions_frame, text="Flipping")
        flipping_frame.grid(row=0, column=0, padx=5, pady=5)

        # Buttons for flipping
        flip_horizontal_button = tk.Button(flipping_frame, text="Flip Horizontal", command=flip_horizontal)
        flip_horizontal_button.pack(side=tk.LEFT)

        flip_vertical_button = tk.Button(flipping_frame, text="Flip Vertical", command=flip_vertical)
        flip_vertical_button.pack(side=tk.LEFT)

        # Rotation frame and buttons
        rotation_frame = ttk.LabelFrame(main_functions_frame, text="Rotation")
        rotation_frame.grid(row=0, column=1, padx=5, pady=5)

        rotate_left_button = tk.Button(rotation_frame, text="Rotate Left", command=rotate_left)
        rotate_left_button.pack(side=tk.LEFT)

        rotate_right_button = tk.Button(rotation_frame, text="Rotate Right", command=rotate_right)
        rotate_right_button.pack(side=tk.LEFT)

        degree_label = tk.Label(rotation_frame, text="Rotation Degree:")
        degree_label.pack(side=tk.LEFT)

        degree_entry = tk.Entry(rotation_frame)
        degree_entry.pack(side=tk.LEFT)

        rotate_button = tk.Button(rotation_frame, text="Rotate", command=lambda: rotate_image(float(degree_entry.get())))
        rotate_button.pack(side=tk.LEFT)

        # Zooming frame and buttons
        zooming_frame = ttk.LabelFrame(main_functions_frame, text="Zooming")
        zooming_frame.grid(row=0, column=2, padx=5, pady=5)

        zoom_in_button = tk.Button(zooming_frame, text="Zoom In", command=zoom_in)
        zoom_in_button.pack(side=tk.LEFT)

        zoom_out_button = tk.Button(zooming_frame, text="Zoom Out", command=zoom_out)
        zoom_out_button.pack(side=tk.LEFT)

        # Brightness and contrast frames and buttons
        brightness_frame = ttk.LabelFrame(main_functions_frame, text="Brightness")
        brightness_frame.grid(row=1, column=0, padx=5, pady=5)

        brightness_up_button = tk.Button(brightness_frame, text="Increase Brightness", command=lambda: adjust_brightness(1.2))
        brightness_up_button.pack(side=tk.LEFT)

        brightness_down_button = tk.Button(brightness_frame, text="Decrease Brightness", command=lambda: adjust_brightness(0.8))
        brightness_down_button.pack(side=tk.LEFT)

        contrast_frame = ttk.LabelFrame(main_functions_frame, text="Contrast")
        contrast_frame.grid(row=1, column=1, padx=5, pady=5)

        contrast_up_button = tk.Button(contrast_frame, text="Increase Contrast", command=lambda: adjust_contrast(1.2))
        contrast_up_button.pack(side=tk.LEFT)

        contrast_down_button = tk.Button(contrast_frame, text="Decrease Contrast", command=lambda: adjust_contrast(0.8))
        contrast_down_button.pack(side=tk.LEFT)

        # Create a frame for sharpening and buttons for image sharpening
        sharpen_frame = ttk.LabelFrame(main_functions_frame, text="Sharpening")
        sharpen_frame.grid(row=1, column=2, padx=5, pady=5)

        sharpen_button = tk.Button(sharpen_frame, text="Sharpen Image", command=sharpen_image)
        sharpen_button.pack(side=tk.LEFT)

         # Create a label frame for blur adjustments
        blur_frame = ttk.LabelFrame(main_functions_frame, text="Blur")
        blur_frame.grid(row=1, column=3, padx=5, pady=5)

        # Create buttons for blur adjustments
        blur_up_button = tk.Button(blur_frame, text="Increase Blur", command=increase_blur)
        blur_up_button.pack(side=tk.LEFT)

        # Color conversion frame and buttons
        color_conversion_frame = ttk.LabelFrame(main_functions_frame, text="Color Conversion")
        color_conversion_frame.grid(row=0, column=3, padx=10, pady=10)

        # First row of buttons
        bw_to_color_button = tk.Button(color_conversion_frame, text="Black & White to Color", command=bw_to_color)
        bw_to_color_button.grid(row=0, column=0, padx=5, pady=5)

        color_to_bw_button = tk.Button(color_conversion_frame, text="Color to Black & White", command=color_to_bw)
        color_to_bw_button.grid(row=0, column=1, padx=5, pady=5)

        grayscale_to_color_button = tk.Button(color_conversion_frame, text="Grayscale to Color", command=grayscale_to_color)
        grayscale_to_color_button.grid(row=0, column=2, padx=5, pady=5)

        # Second row of buttons
        color_to_grayscale_button = tk.Button(color_conversion_frame, text="Color to Grayscale", command=convert_to_grayscale)
        color_to_grayscale_button.grid(row=1, column=0, padx=5, pady=5)

        bw_to_rgb_button = tk.Button(color_conversion_frame, text="Black & White to GrayScale", command=bw_to_grayScale)
        bw_to_rgb_button.grid(row=1, column=1, padx=5, pady=5)

        grayscale_to_rgb_button = tk.Button(color_conversion_frame, text="Grayscale to BlackAndWhite", command=grayscale_to_blackAndWhite)
        grayscale_to_rgb_button.grid(row=1, column=2, padx=5, pady=5)

        # Create the main frame for filters
        Filters = ttk.LabelFrame(canvas, text="Advanced Requirements")
        Filters.grid(row=2, column=0, padx=10, pady=10)

        # Create a sub-frame for Artistic Filters
        artistic_filter_frame = ttk.LabelFrame(Filters, text="Artistic Filters")
        artistic_filter_frame.grid(row=1, column=0, padx=10, pady=10)

        # Create sub-frames for artistic filters
        cartoonization_frame = ttk.LabelFrame(artistic_filter_frame, text="Cartoonization Filter")
        cartoonization_frame.grid(row=0, column=0, padx=5, pady=5)

        cartoonization_button = tk.Button(cartoonization_frame, text="Apply Cartoonization Filter", command=apply_cartoonization)
        cartoonization_button.pack(side=tk.LEFT)

        oil_painting_frame = ttk.LabelFrame(artistic_filter_frame, text="Oil Painting Filter")
        oil_painting_frame.grid(row=0, column=1, padx=5, pady=5)

        oil_painting_button = tk.Button(oil_painting_frame, text="Apply Oil Painting Filter", command=apply_oil_painting)
        oil_painting_button.pack(side=tk.LEFT)

        pencil_sketch_frame = ttk.LabelFrame(artistic_filter_frame, text="Pencil Sketch Filter")
        pencil_sketch_frame.grid(row=0, column=2, padx=5, pady=5)

        pencil_sketch_button = tk.Button(pencil_sketch_frame, text="Apply Pencil Sketch Filter", command=apply_pencil_sketch)
        pencil_sketch_button.pack(side=tk.LEFT)

        # Create a sub-frame for Artistic Filters
        Intensity_Manipulation_frame = ttk.LabelFrame(Filters, text="Intensity Manipulation using Color Transformation")
        Intensity_Manipulation_frame.grid(row=1, column=2, padx=10, pady=10)

        image_negatives_frame = ttk.LabelFrame(Intensity_Manipulation_frame, text="image_negatives")
        image_negatives_frame.grid(row=0, column=0, padx=5, pady=5)

        image_negatives_button = tk.Button(image_negatives_frame, text="Apply image_negatives", command=apply_image_negatives)
        image_negatives_button.pack(side=tk.LEFT)

        log_transformations_frame = ttk.LabelFrame(Intensity_Manipulation_frame, text="Log transformations")
        log_transformations_frame.grid(row=0, column=1, padx=5, pady=5)

        log_transformations_button = tk.Button(log_transformations_frame, text="Apply Log transformations", command=apply_log_transformations)
        log_transformations_button.pack(side=tk.LEFT)

        power_law_transform_frame = ttk.LabelFrame(Intensity_Manipulation_frame, text="Power law transform")
        power_law_transform_frame.grid(row=0, column=2, padx=5, pady=5)
        
        power_law_transform_button = tk.Button(power_law_transform_frame, text="Apply power law transform", command=apply_power_law_transform)
        power_law_transform_button.pack(side=tk.LEFT)

        piecewise_linear_transform_frame = ttk.LabelFrame(Intensity_Manipulation_frame, text=" Piecewise linear Filter")
        piecewise_linear_transform_frame.grid(row=0, column=3, padx=5, pady=5)
        
        piecewise_linear_transform_button = tk.Button(piecewise_linear_transform_frame, text="Apply Piecewise linear Filterr", command=apply_piecewise_linear_transform)
        piecewise_linear_transform_button.pack(side=tk.LEFT)

        # Create the main frame
        Segmentation_frame = ttk.LabelFrame(Filters, text="Image Segmentation")
        Segmentation_frame .grid(row=2, column=2, padx=10, pady=10)

        # Create the frame for Edge-Based Segmentation
        edge_based_frame = ttk.LabelFrame(Segmentation_frame , text="Edge-Based Segmentation")
        edge_based_frame.grid(row=0, column=0, padx=5, pady=5)

        edge_based_button = tk.Button(edge_based_frame, text="Apply Edge-Based Segmentation", command=perform_edge_based_segmentation)
        edge_based_button.pack(side=tk.LEFT)

        # Create the frame for Threshold-Based Segmentation
        threshold_based_frame = ttk.LabelFrame(Segmentation_frame, text="Threshold-Based Segmentation")
        threshold_based_frame.grid(row=0, column=1, padx=5, pady=5)

        threshold_based_button = tk.Button(threshold_based_frame, text="Apply Threshold-Based Segmentation", command=perform_threshold_based_segmentation)
        threshold_based_button.pack(side=tk.LEFT)

        # Create the frame for Region-Based Segmentation
        region_based_frame = ttk.LabelFrame(Segmentation_frame, text="Region-Based Segmentation")
        region_based_frame.grid(row=0, column=2, padx=5, pady=5)

        region_based_button = tk.Button(region_based_frame, text="Apply Region-Based Segmentation", command=perform_region_based_segmentation)
        region_based_button.pack(side=tk.LEFT)

        # Create the frame for Cluster-Based Segmentation
        cluster_based_frame = ttk.LabelFrame(Segmentation_frame, text="Cluster-Based Segmentation")
        cluster_based_frame.grid(row=1, column=0, padx=5, pady=5)

        cluster_based_button = tk.Button(cluster_based_frame, text="Apply Cluster-Based Segmentation", command=perform_cluster_based_segmentation)
        cluster_based_button.pack(side=tk.LEFT)

        # Create the frame for Watershed Segmentation
        watershed_frame = ttk.LabelFrame(Segmentation_frame, text="Different Segementation Types")
        watershed_frame.grid(row=1, column=2, padx=5, pady=5)

        watershed_button = tk.Button(watershed_frame, text="Segmentation Types", command=perform_segmentation)
        watershed_button.pack(side=tk.LEFT)

        # Create the main frame
        Color_balancing_frame = ttk.LabelFrame(Filters, text="Color Balancing Techniques")
        Color_balancing_frame.grid(row=2, column=0, padx=10, pady=10)

        # Create the frame for Watershed Segmentation
        generalized_color_balance_frame = ttk.LabelFrame( Color_balancing_frame, text="Color Transformations")
        generalized_color_balance_frame.grid(row=0, column=0, padx=5, pady=5)

        generalized_color_balance_button = tk.Button(generalized_color_balance_frame, text="All color transformations techniques", command=apply_image_processing)
        generalized_color_balance_button.pack(side=tk.LEFT)

        # Create a frame for edge detection filters
        edge_detection_frame = ttk.LabelFrame(Filters, text="Edge Detection Filters")
        edge_detection_frame.grid(row=3, column=0, padx=10, pady=10)

        # Create a sub-frame for Canny Edge Detection
        canny_edge_frame = ttk.LabelFrame(edge_detection_frame, text="Canny Edge Detection")
        canny_edge_frame.grid(row=0, column=0, padx=5, pady=5)

        # Create a button to apply Canny Edge Detection
        canny_edge_button = tk.Button(canny_edge_frame, text="Apply Canny Edge Detection", command=apply_canny_edge_detection)
        canny_edge_button.pack(side=tk.LEFT)

        # Create a sub-frame for Sobel Edge Detection
        sobel_edge_detection_frame = ttk.LabelFrame(edge_detection_frame, text="Sobel Edge Detection")
        sobel_edge_detection_frame.grid(row=0, column=1, padx=5, pady=5)

        # Create a button to apply Sobel Edge Detection
        sobel_edge_detection_button = tk.Button(sobel_edge_detection_frame, text="Apply Sobel Edge Detection", command=apply_sobel_edge_detection)
        sobel_edge_detection_button.pack(side=tk.LEFT)

         # Create a sub-frame for Sobel Edge Detection
        laplacian_edge_detection_frame = ttk.LabelFrame(edge_detection_frame, text="Laplacian Edge Detection")
        laplacian_edge_detection_frame.grid(row=0, column=2, padx=5, pady=5)

        # Create a button to apply Sobel Edge Detection
        laplacian_edge_detection_button = tk.Button(laplacian_edge_detection_frame, text="Apply Laplacian Edge Detection", command=laplacian_edge_detection)
        laplacian_edge_detection_button.pack(side=tk.LEFT)
        # Create a sub-frame for Smoothing Filters
        smoothing_filter_frame = ttk.LabelFrame(Filters, text="Smoothing Filters")
        smoothing_filter_frame.grid(row=3, column=2, padx=10, pady=10)

        # Create sub-frames for individual smoothing filters
        gaussian_blur_frame = ttk.LabelFrame(smoothing_filter_frame, text="Gaussian Blur")
        gaussian_blur_frame.grid(row=0, column=0, padx=5, pady=5)
        gaussian_blur_button = tk.Button(gaussian_blur_frame, text="Apply Gaussian Blur", command=apply_gaussian_blur)
        gaussian_blur_button.pack(side=tk.LEFT)

        embossing_frame = ttk.LabelFrame(smoothing_filter_frame, text="Embossing Filter")
        embossing_frame.grid(row=0, column=1, padx=5, pady=5)
        embossing_button = tk.Button(embossing_frame, text="Apply Embossing Filter", command=apply_embossing)
        embossing_button.pack(side=tk.LEFT)

        # Add Median Filter
        median_filter_frame = ttk.LabelFrame(smoothing_filter_frame, text="Median Filter")
        median_filter_frame.grid(row=0, column=2, padx=5, pady=5)
        median_filter_button = tk.Button(median_filter_frame, text="Apply Median Filter", command=apply_median_filter)
        median_filter_button.pack(side=tk.LEFT)

        # Add Box Blur Filter
        box_blur_frame = ttk.LabelFrame(smoothing_filter_frame, text="Box Blur")
        box_blur_frame.grid(row=1, column=0, padx=5, pady=5)
        box_blur_button = tk.Button(box_blur_frame, text="Apply Box Blur", command=apply_box_blur)
        box_blur_button.pack(side=tk.LEFT)

        # Add Bilateral Filter
        bilateral_filter_frame = ttk.LabelFrame(smoothing_filter_frame, text="Bilateral Filter")
        bilateral_filter_frame.grid(row=1, column=1, padx=5, pady=5)
        bilateral_filter_button = tk.Button(bilateral_filter_frame, text="Apply Bilateral Filter", command=apply_bilateral_filter)
        bilateral_filter_button.pack(side=tk.LEFT)

        # Add Mean Shift Filter
        mean_shift_filter_frame = ttk.LabelFrame(smoothing_filter_frame, text="Mean Shift Filter")
        mean_shift_filter_frame.grid(row=1, column=2, padx=5, pady=5)
        mean_shift_filter_button = tk.Button(mean_shift_filter_frame, text="Apply Mean Shift Filter", command=apply_mean_shift_filter)
        mean_shift_filter_button.pack(side=tk.LEFT)

        # Create sub-frames for individual image enhancement techniques
        mean_shift_frame = ttk.LabelFrame(Filters, text="Enhancement Techniques")
        mean_shift_frame.grid(row=4, column=0, padx=10, pady=10)

        histogram_equalization_frame = ttk.LabelFrame(mean_shift_frame, text="Histogram Equalization")
        histogram_equalization_frame.grid(row=0, column=0, padx=5, pady=5)
        histogram_equalization_button = tk.Button(histogram_equalization_frame, text="Apply Histogram Equalization", command=apply_histogram_equalization)
        histogram_equalization_button.pack(side=tk.LEFT)

        contrast_stretching_frame = ttk.LabelFrame(mean_shift_frame, text="Contrast Stretching")
        contrast_stretching_frame.grid(row=0, column=1, padx=5, pady=5)
        contrast_stretching_button = tk.Button(contrast_stretching_frame, text="Apply Contrast Stretching", command=apply_contrast_stretching)
        contrast_stretching_button.pack(side=tk.LEFT)

        sharpening_frame = ttk.LabelFrame(mean_shift_frame, text="Sharpening Filter")
        sharpening_frame.grid(row=0, column=2, padx=5, pady=5)
        sharpening_button = tk.Button(sharpening_frame, text="Apply Sharpening Filter", command=apply_sharpening_filter)
        sharpening_button.pack(side=tk.LEFT)

        edge_detection_frame = ttk.LabelFrame(mean_shift_frame, text="Edge Detection")
        edge_detection_frame.grid(row=1, column=0, padx=5, pady=5)
        edge_detection_button = tk.Button(edge_detection_frame, text="Apply Edge Detection", command=apply_edge_detection)
        edge_detection_button.pack(side=tk.LEFT)

        sepia_filter_frame = ttk.LabelFrame(mean_shift_frame, text="Sepia Filter")
        sepia_filter_frame.grid(row=1, column=1, padx=0, pady=5)
        sepia_filter_button = tk.Button(sepia_filter_frame, text="Apply Sepia Filter", command=apply_sepia_filter)
        sepia_filter_button.pack(side=tk.LEFT)

        saturation_adjustment_frame = ttk.LabelFrame(mean_shift_frame, text="Saturation Adjustment")
        saturation_adjustment_frame.grid(row=1, column=2, padx=5, pady=5)
        saturation_adjustment_button = tk.Button(saturation_adjustment_frame, text="Apply Saturation Adjustment", command=apply_saturation_adjustment)
        saturation_adjustment_button.pack(side=tk.LEFT)

        # Add Mean Shift Filter
        Tonal_frame = ttk.LabelFrame(Filters, text="Tonal Transformations")
        Tonal_frame.grid(row=4, column=2, padx=5, pady=5)

        # Create a sub-frame for "Solarize Transformation"
        solarize_transformation_frame = ttk.LabelFrame(Tonal_frame, text="Solarize Transformation")
        solarize_transformation_frame.grid(row=0, column=0, padx=5, pady=5)
        solarize_transformation_button = tk.Button(solarize_transformation_frame, text="Apply Solarize Transformation", command=apply_solarize_transformation)
        solarize_transformation_button.pack(side=tk.LEFT)

        # Create a sub-frame for "Posterize Transformation"
        posterize_transformation_frame = ttk.LabelFrame(Tonal_frame, text="Posterize Transformation")
        posterize_transformation_frame.grid(row=0, column=1, padx=5, pady=5)
        posterize_transformation_button = tk.Button(posterize_transformation_frame, text="Apply Posterize Transformation", command=apply_posterize_transformation)
        posterize_transformation_button.pack(side=tk.LEFT)

        # Create a sub-frame for "Invert Color Transformation"
        invert_color_transformation_frame = ttk.LabelFrame(Tonal_frame, text="Invert Color Transformation")
        invert_color_transformation_frame.grid(row=0,column=2, padx=5, pady=5)
        invert_color_transformation_button = tk.Button(invert_color_transformation_frame, text="Apply Invert Color Transformation", command=apply_invert_color_transformation)
        invert_color_transformation_button.pack(side=tk.LEFT)

        # Create a sub-frame for "Warm Color Filter"
        warm_color_filter_frame = ttk.LabelFrame(Tonal_frame, text="Warm Color Filter")
        warm_color_filter_frame.grid(row=1, column=0, padx=5, pady=5)
        warm_color_filter_button = tk.Button(warm_color_filter_frame, text="Apply Warm Color Filter", command=apply_warm_color_filter)
        warm_color_filter_button.pack(side=tk.LEFT)

        # Create a sub-frame for "Cool Color Filter"
        cool_color_filter_frame = ttk.LabelFrame(Tonal_frame, text="Cool Color Filter")
        cool_color_filter_frame.grid(row=1, column=1, padx=5, pady=5)
        cool_color_filter_button = tk.Button(cool_color_filter_frame, text="Apply Cool Color Filter", command=apply_cool_color_filter)
        cool_color_filter_button.pack(side=tk.LEFT)

        # Create a sub-frame for "Random Colorization"
        random_colorization_frame = ttk.LabelFrame(Tonal_frame, text="Random Colorization")
        random_colorization_frame.grid(row=1, column=2, padx=5, pady=5)
        random_colorization_button = tk.Button(random_colorization_frame, text="Apply Random Colorization", command=apply_random_colorization)
        random_colorization_button.pack(side=tk.LEFT)

def apply_solarize_transformation(output_image_path=None, threshold=128):
    global current_image
    if current_image:
        current_image_array = np.array(current_image)

        # Display the original image
        plt.figure(figsize=(8, 8))
        plt.imshow(current_image_array)
        plt.axis("off")
        plt.title("Original Image")
        plt.show()

        # Apply solarize transformation
        solarized_image = np.where(current_image_array < threshold, current_image_array, 255 - current_image_array)

        # Create an image from the solarized result
        transformed_image = Image.fromarray(solarized_image)

        # Display the solarized image
        plt.figure(figsize=(8, 8))
        plt.imshow(transformed_image)
        plt.axis("off")
        plt.title(f"Solarized Image (Threshold = {threshold})")
        plt.show()

        # Save the transformed image if an output path is provided
        if output_image_path:
            transformed_image.save(output_image_path)

def apply_posterize_transformation(output_image_path=None, bits=4):
    global current_image
    if current_image:
        current_image_array = np.array(current_image)

        # Display the original image
        plt.figure(figsize=(8, 8))
        plt.imshow(current_image_array)
        plt.axis("off")
        plt.title("Original Image")
        plt.show()

        # Apply posterize transformation
        posterized_image = np.floor(current_image_array / 256 * (2**bits)) * (256 / (2**bits))

        # Create an image from the posterized result
        transformed_image = Image.fromarray(posterized_image.astype('uint8'))

        # Display the posterized image
        plt.figure(figsize=(8, 8))
        plt.imshow(transformed_image)
        plt.axis("off")
        plt.title(f"Posterized Image (Bits = {bits})")
        plt.show()

        # Save the transformed image if an output path is provided
        if output_image_path:
            transformed_image.save(output_image_path)

def apply_invert_color_transformation(output_image_path=None):
    global current_image
    if current_image:
        current_image_array = np.array(current_image)

        # Display the original image
        plt.figure(figsize=(8, 8))
        plt.imshow(current_image_array)
        plt.axis("off")
        plt.title("Original Image")
        plt.show()

        # Apply invert color transformation
        inverted_color_image = 255 - current_image_array

        # Create an image from the inverted color result
        transformed_image = Image.fromarray(inverted_color_image)

        # Display the inverted color image
        plt.figure(figsize=(8, 8))
        plt.imshow(transformed_image)
        plt.axis("off")
        plt.title("Inverted Color Image")
        plt.show()

        # Save the transformed image if an output path is provided
        if output_image_path:
            transformed_image.save(output_image_path)

def apply_warm_color_filter(output_image_path=None):
    global current_image
    if current_image:
        current_image_array = np.array(current_image)

        # Display the original image
        plt.figure(figsize=(8, 8))
        plt.imshow(current_image_array)
        plt.axis("off")
        plt.title("Original Image")
        plt.show()

        # Apply warm color filter
        r, g, b = current_image_array[:, :, 0], current_image_array[:, :, 1], current_image_array[:, :, 2]
        r = np.clip(r * 1.5, 0, 255)
        g = np.clip(g * 1.2, 0, 255)

        # Create an image from the filtered result
        warm_image = np.stack((r, g, b), axis=-1)
        transformed_image = Image.fromarray(warm_image.astype('uint8'))

        # Display the warm color filtered image
        plt.figure(figsize=(8, 8))
        plt.imshow(transformed_image)
        plt.axis("off")
        plt.title("Warm Color Filtered Image")
        plt.show()

        # Save the transformed image if an output path is provided
        if output_image_path:
            transformed_image.save(output_image_path)

def apply_cool_color_filter(output_image_path=None):
    global current_image
    if current_image:
        current_image_array = np.array(current_image)

        # Display the original image
        plt.figure(figsize=(8, 8))
        plt.imshow(current_image_array)
        plt.axis("off")
        plt.title("Original Image")
        plt.show()

        # Apply cool color filter
        r, g, b = current_image_array[:, :, 0], current_image_array[:, :, 1], current_image_array[:, :, 2]
        b = np.clip(b * 1.5, 0, 255)
        g = np.clip(g * 1.2, 0, 255)

        # Create an image from the filtered result
        cool_image = np.stack((r, g, b), axis=-1)
        transformed_image = Image.fromarray(cool_image.astype('uint8'))

        # Display the cool color filtered image
        plt.figure(figsize=(8, 8))
        plt.imshow(transformed_image)
        plt.axis("off")
        plt.title("Cool Color Filtered Image")
        plt.show()

        # Save the transformed image if an output path is provided
        if output_image_path:
            transformed_image.save(output_image_path)



def apply_random_colorization(output_image_path=None):
    global current_image
    if current_image:
        current_image_array = np.array(current_image)

        # Display the original image
        plt.figure(figsize=(8, 8))
        plt.imshow(current_image_array)
        plt.axis("off")
        plt.title("Original Image")
        plt.show()

        # Apply random colorization by shifting the color channels
        r, g, b = current_image_array[:, :, 0], current_image_array[:, :, 1], current_image_array[:, :, 2]

        # Randomly shuffle the color channels
        channels = [r, g, b]
        random.shuffle(channels)
        random_colorized_image = np.stack(channels, axis=-1)

        # Create an image from the colorized result
        colorized_image = Image.fromarray(random_colorized_image)

        # Display the colorized image
        plt.figure(figsize=(8, 8))
        plt.imshow(colorized_image)
        plt.axis("off")
        plt.title("Randomly Colorized Image")
        plt.show()

        # Save the colorized image if an output path is provided
        if output_image_path:
            colorized_image.save(output_image_path)




def apply_histogram_equalization(output_image_path=None):
    global current_image
    if current_image:
        current_image_array = np.array(current_image)

        # Display the original image
        plt.figure(figsize=(8, 8))
        plt.imshow(current_image_array)
        plt.axis("off")
        plt.title("Original Image")
        plt.show()

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(current_image_array, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization
        equalized = cv2.equalizeHist(gray_image)

        # Create an image from the equalized result
        equalized_image = Image.fromarray(equalized, 'L')

        # Display the equalized image
        plt.figure(figsize=(8, 8))
        plt.imshow(equalized_image, cmap="gray")
        plt.axis("off")
        plt.title("Histogram Equalized Image")
        plt.show()

        # Save the equalized image if an output path is provided
        if output_image_path:
            equalized_image.save(output_image_path)

def apply_contrast_stretching(output_image_path=None):
    global current_image
    if current_image:
        current_image_array = np.array(current_image)

        # Display the original image
        plt.figure(figsize=(8, 8))
        plt.imshow(current_image_array)
        plt.axis("off")
        plt.title("Original Image")
        plt.show()

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(current_image_array, cv2.COLOR_BGR2GRAY)

        # Apply contrast stretching
        min_intensity = 50
        max_intensity = 200
        stretched = cv2.normalize(gray_image, None, min_intensity, max_intensity, cv2.NORM_MINMAX)

        # Create an image from the stretched result
        stretched_image = Image.fromarray(stretched, 'L')

        # Display the stretched image
        plt.figure(figsize=(8, 8))
        plt.imshow(stretched_image, cmap="gray")
        plt.axis("off")
        plt.title("Contrast Stretched Image")
        plt.show()

        # Save the stretched image if an output path is provided
        if output_image_path:
            stretched_image.save(output_image_path)

def apply_sharpening_filter(output_image_path=None):
    global current_image
    if current_image:
        current_image_array = np.array(current_image)

        # Display the original image
        plt.figure(figsize=(8, 8))
        plt.imshow(current_image_array)
        plt.axis("off")
        plt.title("Original Image")
        plt.show()

        # Define the sharpening kernel
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])

        # Apply the sharpening filter
        sharpened = cv2.filter2D(current_image_array, -1, kernel)

        # Create an image from the sharpened result
        sharpened_image = Image.fromarray(sharpened)

        # Display the sharpened image
        plt.figure(figsize=(8, 8))
        plt.imshow(sharpened_image)
        plt.axis("off")
        plt.title("Sharpened Image")
        plt.show()

        # Save the sharpened image if an output path is provided
        if output_image_path:
            sharpened_image.save(output_image_path)

def apply_edge_detection(output_image_path=None):
    global current_image
    if current_image:
        current_image_array = np.array(current_image)

        # Display the original image
        plt.figure(figsize=(8, 8))
        plt.imshow(current_image_array)
        plt.axis("off")
        plt.title("Original Image")
        plt.show()

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(current_image_array, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray_image, 100, 200)

        # Create an image from the detected edges
        edges_image = Image.fromarray(edges, 'L')

        # Display the edge-detected image
        plt.figure(figsize=(8, 8))
        plt.imshow(edges_image, cmap="gray")
        plt.axis("off")
        plt.title("Edge-Detected Image")
        plt.show()

        # Save the edge-detected image if an output path is provided
        if output_image_path:
            edges_image.save(output_image_path)

def apply_sepia_filter(output_image_path=None):
    global current_image
    if current_image:
        current_image_array = np.array(current_image)

        # Display the original image
        plt.figure(figsize=(8, 8))
        plt.imshow(current_image_array)
        plt.axis("off")
        plt.title("Original Image")
        plt.show()

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(current_image_array, cv2.COLOR_BGR2GRAY)

        # Apply sepia filter
        sepia_filtered = cv2.cvtColor(current_image_array, cv2.COLOR_BGR2RGB)
        sepia_filtered[:, :, 0] = sepia_filtered[:, :, 0] * 0.393 + sepia_filtered[:, :, 1] * 0.769 + sepia_filtered[:, :, 2] * 0.189
        sepia_filtered[:, :, 1] = sepia_filtered[:, :, 0] * 0.349 + sepia_filtered[:, :, 1] * 0.686 + sepia_filtered[:, :, 2] * 0.168
        sepia_filtered[:, :, 2] = sepia_filtered[:, :, 0] * 0.272 + sepia_filtered[:, :, 1] * 0.534 + sepia_filtered[:, :, 2] * 0.131

        # Create an image from the sepia filtered result
        sepia_image = Image.fromarray(sepia_filtered)

        # Display the sepia filtered image
        plt.figure(figsize=(8, 8))
        plt.imshow(sepia_image)
        plt.axis("off")
        plt.title("Sepia Filtered Image")
        plt.show()

        # Save the sepia filtered image if an output path is provided
        if output_image_path:
            sepia_image.save(output_image_path)

def apply_saturation_adjustment(output_image_path=None, factor=1.5):
    global current_image
    if current_image:
        current_image_array = np.array(current_image)

        # Display the original image
        plt.figure(figsize=(8, 8))
        plt.imshow(current_image_array)
        plt.axis("off")
        plt.title("Original Image")
        plt.show()

        # Convert the image to the HSV color space
        hsv_image = cv2.cvtColor(current_image_array, cv2.COLOR_BGR2HSV)

        # Adjust the saturation
        hsv_image[:, :, 1] = hsv_image[:, :, 1] * factor

        # Clip the values to the valid range
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1], 0, 255)

        # Convert the image back to the BGR color space
        adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        # Create an image from the adjusted result
        adjusted_image = Image.fromarray(adjusted_image)

        # Display the adjusted image
        plt.figure(figsize=(8, 8))
        plt.imshow(adjusted_image)
        plt.axis("off")
        plt.title("Saturation Adjusted Image")
        plt.show()

        # Save the adjusted image if an output path is provided
        if output_image_path:
            adjusted_image.save(output_image_path)


def laplacian_edge_detection():

    # Open the image
    img = cv2.imread('C:/Users/Dewmi Silva/Downloads/YellowLabradorLooking_new.jpg')
    
    plt.figure()
    plt.title('Original Image')
    plt.imshow(img)
    plt.show()

    # Apply gray scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply gaussian blur
    blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)

    # Positive Laplacian Operator
    laplacian = cv2.Laplacian(blur_img, cv2.CV_64F)

    plt.figure()
    plt.title('Laplacian Edge Detection')
    plt.imshow(laplacian, cmap='gray')
    plt.show()


def apply_sobel_edge_detection():
    # Open the image
    img = np.array(Image.open('C:/Users/Dewmi Silva/Downloads/YellowLabradorLooking_new.jpg')).astype(np.uint8)

    # Apply gray scale
    gray_img = np.round(0.299 * img[:, :, 0] +
                        0.587 * img[:, :, 1] +
                        0.114 * img[:, :, 2]).astype(np.uint8)

    # Sobel Operator
    h, w = gray_img.shape
    # define filters
    horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # s2
    vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # s1

    # define images with 0s
    newhorizontalImage = np.zeros((h, w))
    newverticalImage = np.zeros((h, w))
    newgradientImage = np.zeros((h, w))

    # offset by 1
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            horizontalGrad = (horizontal[0, 0] * gray_img[i - 1, j - 1]) + \
                            (horizontal[0, 1] * gray_img[i - 1, j]) + \
                            (horizontal[0, 2] * gray_img[i - 1, j + 1]) + \
                            (horizontal[1, 0] * gray_img[i, j - 1]) + \
                            (horizontal[1, 1] * gray_img[i, j]) + \
                            (horizontal[1, 2] * gray_img[i, j + 1]) + \
                            (horizontal[2, 0] * gray_img[i + 1, j - 1]) + \
                            (horizontal[2, 1] * gray_img[i + 1, j]) + \
                            (horizontal[2, 2] * gray_img[i + 1, j + 1])

            newhorizontalImage[i - 1, j - 1] = abs(horizontalGrad)

            verticalGrad = (vertical[0, 0] * gray_img[i - 1, j - 1]) + \
                        (vertical[0, 1] * gray_img[i - 1, j]) + \
                        (vertical[0, 2] * gray_img[i - 1, j + 1]) + \
                        (vertical[1, 0] * gray_img[i, j - 1]) + \
                        (vertical[1, 1] * gray_img[i, j]) + \
                        (vertical[1, 2] * gray_img[i, j + 1]) + \
                        (vertical[2, 0] * gray_img[i + 1, j - 1]) + \
                        (vertical[2, 1] * gray_img[i + 1, j]) + \
                        (vertical[2, 2] * gray_img[i + 1, j + 1])

            newverticalImage[i - 1, j - 1] = abs(verticalGrad)

            # Edge Magnitude
            mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
            newgradientImage[i - 1, j - 1] = mag

    plt.figure()
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')
    plt.show()

    plt.figure()
    plt.title('Sobel Edge Detection')
    plt.imshow(newgradientImage, cmap='gray')
    plt.show()

def apply_image_processing():

    # Load an image
    image = cv2.imread('C:/Users/Dewmi Silva/Downloads/YellowLabradorLooking_new.jpg')

    # 1. Generalized Color Balance:
    def generalized_color_balance(image, red_factor=1.0, green_factor=1.0, blue_factor=1.0):
        balanced_image = image.copy()
        balanced_image[:, :, 0] = np.clip(image[:, :, 0] * blue_factor, 0, 255)
        balanced_image[:, :, 1] = np.clip(image[:, :, 1] * green_factor, 0, 255)
        balanced_image[:, :, 2] = np.clip(image[:, :, 2] * red_factor, 0, 255)
        return balanced_image

    red_factor = 1.5
    green_factor = 0.8
    blue_factor = 0.7
    generalized_balanced_image = generalized_color_balance(image, red_factor, green_factor, blue_factor)

    # Display the generalized color balanced image
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(generalized_balanced_image, cv2.COLOR_BGR2RGB))
    plt.title('Generalized Color Balance')
    plt.axis('off')
    plt.show()

    # 2. Psychological Color Balance:
    # Adjust color tones for psychological impact (e.g., warm colors for coziness)

    # Create a warm color filter (increase red and decrease blue)
    warm_filter = np.array([[0.8, 0, 0], [0, 1, 0], [0, 0, 1]])
    psychological_image = cv2.transform(image, warm_filter)

    # Display the psychologically balanced image
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(psychological_image, cv2.COLOR_BGR2RGB))
    plt.title('Psychological Color Balance')
    plt.axis('off')
    plt.show()

    # 3. Illuminant Estimation and Adaptation:
    # Simulate illuminant adaptation (e.g., daylight to fluorescent lighting)

    # Simulate a change in illuminant from daylight to fluorescent
    illuminant_adapted_image = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
    illuminant_adapted_image = illuminant_adapted_image.astype(np.float32)  # Convert to float for safe multiplication
    illuminant_adapted_image[:, :, 1] *= 0.8  # Reduce the Y component to simulate fluorescent lighting
    illuminant_adapted_image = cv2.cvtColor(illuminant_adapted_image.astype(np.uint8), cv2.COLOR_XYZ2BGR)

    # Display the illuminant-adapted image
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(illuminant_adapted_image, cv2.COLOR_BGR2RGB))
    plt.title('Illuminant Adaptation')
    plt.axis('off')
    plt.show()

    # 4. Chromatic Colors:
    # Add chromatic colors (red, green, blue) to the image
    chromatic_colors = np.zeros_like(image)
    chromatic_colors[:, :, 0] = 0  # Red
    chromatic_colors[:, :, 1] = 0  # Green
    chromatic_colors[:, :, 2] = 255  # Blue
    image_with_chromatic_colors = cv2.add(image, chromatic_colors)

    # Display the image with added chromatic colors
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image_with_chromatic_colors, cv2.COLOR_BGR2RGB))
    plt.title('Image with Chromatic Colors')
    plt.axis('off')
    plt.show()


    # 5. Mathematics of Color Balance:
    # Apply mathematical color balance (e.g., color transforms)

    # Create a color transform matrix (example: increase blue and reduce red)
    color_transform_matrix = np.array([[0.7, 0, 0], [0, 1, 0], [0, 0, 1.3]])
    math_color_balanced_image = cv2.transform(image, color_transform_matrix)

    # Display the mathematically color-balanced image
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(math_color_balanced_image, cv2.COLOR_BGR2RGB))
    plt.title('Mathematics of Color Balance')
    plt.axis('off')
    plt.show()

    # 6. General Illuminant Adaptation:
    # Simulate general illuminant adaptation (e.g., from daylight to incandescent)

    # Simulate a change in illuminant from daylight to incandescent
    gen_illuminant_adapted_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    gen_illuminant_adapted_image = gen_illuminant_adapted_image.astype(np.float32)  # Convert to float for safe multiplication
    gen_illuminant_adapted_image[:, :, 2] *= 1.2  # Increase the b* component to simulate incandescent lighting
    gen_illuminant_adapted_image = cv2.cvtColor(gen_illuminant_adapted_image.astype(np.uint8), cv2.COLOR_Lab2BGR)

    # Display the general illuminant-adapted image
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(gen_illuminant_adapted_image, cv2.COLOR_BGR2RGB))
    plt.title('General Illuminant Adaptation')
    plt.axis('off')
    plt.show()

    def simple_white_balance(image):
        # Convert the image to LAB color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Calculate the average value of the L channel
        l_channel = lab_image[:,:,0]
        avg_l = np.mean(l_channel)
        
        # Scale the L channel by the ratio of the average
        lab_image[:,:,0] = np.uint8(np.clip((l_channel * (avg_l / l_channel)), 0, 255))
        
        # Convert the LAB image back to BGR color space
        result_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
        
        return result_image

    # Apply white balancing
    white_balanced_image = simple_white_balance(image)

    # Display the white-balanced image
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(white_balanced_image, cv2.COLOR_BGR2RGB))
    plt.title('White-Balanced Image')
    plt.axis('off')
    plt.show()


    # Load an image
    image = cv2.imread('C:/Users/Dewmi Silva/Downloads/YellowLabradorLooking_new.jpg')

    # 1. Generalized Color Balance:
    def generalized_color_balance(image, red_factor=1.0, green_factor=1.0, blue_factor=1.0):
        balanced_image = image.copy()
        balanced_image[:, :, 0] = np.clip(image[:, :, 0] * blue_factor, 0, 255)
        balanced_image[:, :, 1] = np.clip(image[:, :, 1] * green_factor, 0, 255)
        balanced_image[:, :, 2] = np.clip(image[:, :, 2] * red_factor, 0, 255)
        return balanced_image

    red_factor = 1.5
    green_factor = 0.8
    blue_factor = 0.7
    generalized_balanced_image = generalized_color_balance(image, red_factor, green_factor, blue_factor)

    # 2. Psychological Color Balance:
    # Adjust color tones for psychological impact (e.g., warm colors for coziness)
    warm_filter = np.array([[0.8, 0, 0], [0, 1, 0], [0, 0, 1]])
    psychological_image = cv2.transform(image, warm_filter)

    # 3. Illuminant Estimation and Adaptation:
    # Simulate illuminant adaptation (e.g., daylight to fluorescent lighting)
    illuminant_adapted_image = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
    illuminant_adapted_image = illuminant_adapted_image.astype(np.float32)
    illuminant_adapted_image[:, :, 1] *= 0.8
    illuminant_adapted_image = cv2.cvtColor(illuminant_adapted_image.astype(np.uint8), cv2.COLOR_XYZ2BGR)

    # 4. Chromatic Colors:
    # Add chromatic colors (red, green, blue) to the image
    chromatic_colors = np.zeros_like(image)
    chromatic_colors[:, :, 0] = 0
    chromatic_colors[:, :, 1] = 0
    chromatic_colors[:, :, 2] = 255
    image_with_chromatic_colors = cv2.add(image, chromatic_colors)

    # 5. Mathematics of Color Balance:
    # Apply mathematical color balance (e.g., color transforms)
    color_transform_matrix = np.array([[0.7, 0, 0], [0, 1, 0], [0, 0, 1.3]])
    math_color_balanced_image = cv2.transform(image, color_transform_matrix)

    # 6. General Illuminant Adaptation:
    # Simulate general illuminant adaptation (e.g., from daylight to incandescent)
    gen_illuminant_adapted_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    gen_illuminant_adapted_image = gen_illuminant_adapted_image.astype(np.float32)
    gen_illuminant_adapted_image[:, :, 2] *= 1.2
    gen_illuminant_adapted_image = cv2.cvtColor(gen_illuminant_adapted_image.astype(np.uint8), cv2.COLOR_Lab2BGR)

    # Apply white balancing
    def simple_white_balance(image):
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab_image[:,:,0]
        avg_l = np.mean(l_channel)
        lab_image[:,:,0] = np.uint8(np.clip((l_channel * (avg_l / l_channel)), 0, 255))
        result_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
        return result_image

    white_balanced_image = simple_white_balance(image)

    # Apply color grading
    def color_grading(image, brightness=1.0, contrast=1.0, saturation=1.0, hue_shift=0):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * brightness * contrast, 0, 255)
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation, 0, 255)
        hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shift) % 180
        graded_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return graded_image

    graded_image = color_grading(image, brightness=1.2, contrast=1.1, saturation=0.9, hue_shift=20)

    # Show all the images in a single plot
    plt.figure(figsize=(18, 12))

    # Plot the Generalized Color Balance image
    plt.subplot(2, 4, 1)
    plt.imshow(cv2.cvtColor(generalized_balanced_image, cv2.COLOR_BGR2RGB))
    plt.title('Generalized Color Balance')
    plt.axis('off')

    # Plot the Psychological Color Balance image
    plt.subplot(2, 4, 2)
    plt.imshow(cv2.cvtColor(psychological_image, cv2.COLOR_BGR2RGB))
    plt.title('Psychological Color Balance')
    plt.axis('off')

    # Plot the Illuminant Adaptation image
    plt.subplot(2, 4, 3)
    plt.imshow(cv2.cvtColor(illuminant_adapted_image, cv2.COLOR_BGR2RGB))
    plt.title('Illuminant Adaptation')
    plt.axis('off')

    # Plot the Image with Chromatic Colors
    plt.subplot(2, 4, 4)
    plt.imshow(cv2.cvtColor(image_with_chromatic_colors, cv2.COLOR_BGR2RGB))
    plt.title('Image with Chromatic Colors')
    plt.axis('off')

    # Plot the Mathematics of Color Balance image
    plt.subplot(2, 4, 5)
    plt.imshow(cv2.cvtColor(math_color_balanced_image, cv2.COLOR_BGR2RGB))
    plt.title('Mathematics of Color Balance')
    plt.axis('off')

    # Plot the General Illuminant Adaptation image
    plt.subplot(2, 4, 6)
    plt.imshow(cv2.cvtColor(gen_illuminant_adapted_image, cv2.COLOR_BGR2RGB))
    plt.title('General Illuminant Adaptation')
    plt.axis('off')

    # Plot the White-Balanced Image
    plt.subplot(2, 4, 7)
    plt.imshow(cv2.cvtColor(white_balanced_image, cv2.COLOR_BGR2RGB))
    plt.title('White-Balanced Image')
    plt.axis('off')

    # Plot the Graded Image
    plt.subplot(2, 4, 8)
    plt.imshow(cv2.cvtColor(graded_image, cv2.COLOR_BGR2RGB))
    plt.title('Graded Image')
    plt.axis('off')

    # Show the plot
    plt.show()

def perform_segmentation():
    global current_image
    if current_image:
        current_image_array = np.array(current_image)

        plt.rcParams["figure.figsize"] = (12, 8)

        # Convert the image to grayscale
        rocket_wh = rgb2gray(current_image_array)

        # Create a subplot grid to display all images
        fig, axes = plt.subplots(3, 3)

        # Apply edge segmentation and plot Canny edge detection
        edges = canny(rocket_wh)
        axes[0, 0].imshow(edges, interpolation='gaussian')
        axes[0, 0].set_title('Canny detector')

        # Fill regions to perform edge segmentation
        fill_im = nd.binary_fill_holes(edges)
        axes[0, 1].imshow(fill_im)
        axes[0, 1].set_title('Region Filling')

        # Region Segmentation
        # First, print the elevation map
        elevation_map = sobel(rocket_wh)
        axes[1, 0].imshow(elevation_map)
        axes[1, 0].set_title('Elevation Map')

        # Define markers for watershed segmentation
        markers = np.zeros_like(rocket_wh)
        markers[rocket_wh < 0.1171875] = 1  # 30/255
        markers[rocket_wh > 0.5859375] = 2  # 150/255
        axes[1, 1].imshow(markers)
        axes[1, 1].set_title('Markers')

        # Perform watershed region segmentation
        segmentation = watershed(elevation_map, label(markers, connectivity=2), mask=rocket_wh)
        axes[2, 0].imshow(segmentation, cmap=plt.cm.nipy_spectral)
        axes[2, 0].set_title('Watershed Segmentation')

        # Plot overlays and contour
        label_rock, _ = nd.label(segmentation)
        image_label_overlay = label2rgb(label_rock, image=rocket_wh)
        axes[2, 1].imshow(image_label_overlay)
        axes[2, 1].set_title('Overlays of Segmentation')

        for ax in axes.ravel():
            ax.axis('off')

        plt.show()

# Usage example
# image = data.rocket()
# perform_segmentation(image)


def perform_edge_based_segmentation(output_image_path=None):
    global current_image

    if current_image:
        current_image_array = np.array(current_image)
        
        # Convert the PIL image to a NumPy array
        image = np.array(current_image_array)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray_image, 50, 150)

        # Display the original image and the detected edges
        plt.figure(figsize=(10, 5))

        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(current_image_array)
        plt.axis('off')
        plt.title("Original Image")

        # Detected edges
        plt.subplot(1, 2, 2)
        plt.imshow(edges, cmap='gray')
        plt.axis('off')
        plt.title("Edge-Based Segmentation")

        plt.show()

        # Save the edge-detected image if an output path is provided
        if output_image_path:
            cv2.imwrite(output_image_path, edges)

def perform_threshold_based_segmentation(output_image_path=None, threshold_value=128):
    global current_image
    if current_image:
        current_image_array = np.array(current_image)

        # Convert the PIL image to a NumPy array
        image = np.array(current_image_array)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding
        _, thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

        # Display the original image and the thresholded result
        plt.figure(figsize=(10, 5))

        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(current_image_array)
        plt.axis('off')
        plt.title("Original Image")

        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(gray_image, cmap='gray')
        plt.axis('off')
        plt.title("Black Or White")

        # Thresholded image
        plt.subplot(1, 2, 2)
        plt.imshow(thresholded_image, cmap='gray')
        plt.axis('off')
        plt.title("Threshold-Based Segmentation")

        plt.show()

        # Save the thresholded image if an output path is provided
        if output_image_path:
            cv2.imwrite(output_image_path, thresholded_image)

def perform_region_based_segmentation(output_image_path=None):
    global current_image

    if current_image:
        current_image_array = np.array(current_image)

        # Convert the PIL image to a NumPy array
        image = np.array(current_image_array)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding to separate objects
        _, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Apply the Watershed algorithm for region-based segmentation
        markers = cv2.connectedComponents(thresholded_image)[1]
        markers = cv2.watershed(image, markers)
        image[markers == -1] = [0, 0, 255]  # Mark segmented regions

        # Display the original image and the segmented result
        plt.figure(figsize=(10, 5))

        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(current_image_array)
        plt.axis('off')
        plt.title("Original Image")

        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("GrayScale Image")

        # Segmented image
        plt.subplot(1, 2, 2)
        plt.imshow(markers, cmap='tab20b', vmin=0, vmax=markers.max())
        plt.axis('off')
        plt.title("Region-Based Segmentation")

        plt.show()

        # Save the segmented image if an output path is provided
        if output_image_path:
            cv2.imwrite(output_image_path, markers)

def perform_cluster_based_segmentation(output_image_path=None, num_clusters=4):
    global current_image
    if current_image:
        current_image_array = np.array(current_image)

        # Convert the PIL image to a NumPy array
        image = np.array(current_image_array)

        # Reshape the image to a 2D array of pixels
        pixels = image.reshape((-1, 3))

        # Convert pixel values to float32 for clustering
        pixels = np.float32(pixels)

        # Apply k-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Convert back to 8-bit values
        centers = np.uint8(centers)

        # Map the labels to the centers to segment the image
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(image.shape)

        # Display the original image and the segmented result
        plt.figure(figsize=(10, 5))

        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(current_image_array)
        plt.axis('off')
        plt.title("Original Image")

        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("GrayScale Image")

        # Segmented image
        plt.subplot(1, 2, 2)
        plt.imshow(segmented_image)
        plt.axis('off')
        plt.title("Cluster-Based Segmentation")

        plt.show()

        # Save the segmented image if an output path is provided
        if output_image_path:
            cv2.imwrite(output_image_path, segmented_image)

def perform_watershed_segmentation(output_image_path=None):
    global current_image

    if current_image:
        current_image_array = np.array(current_image)

        # Convert the PIL image to a NumPy array
        image = np.array(current_image_array)

        # Ensure the image is 3-channel (if not, convert it)
        if image.shape[-1] != 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply adaptive thresholding to create markers for the Watershed algorithm
        _, markers = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Apply morphological operations to remove noise
        kernel = np.ones((3, 3), np.uint8)
        markers = cv2.morphologyEx(markers, cv2.MORPH_OPEN, kernel, iterations=2)
        markers = cv2.dilate(markers, kernel, iterations=3)

        # Apply the Watershed algorithm for segmentation
        markers = cv2.watershed(gray_image, markers)
        image[markers == -1] = [0, 0, 255]  # Mark segmented regions

        # Display the original image and the segmented result
        plt.figure(figsize=(10, 5))

        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(current_image_array)
        plt.axis('off')
        plt.title("Original Image")

        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title("GrayScale Image")

        # Segmented image
        plt.subplot(1, 2, 2)
        plt.imshow(markers, cmap='tab20b', vmin=0, vmax=markers.max())
        plt.axis('off')
        plt.title("Watershed Segmentation")

        plt.show()

        # Save the segmented image if an output path is provided
        if output_image_path:
            cv2.imwrite(output_image_path, markers)

def color_masking(output_image_path:None):
     global current_image
     if current_image:
        current_image_array = np.array(current_image)

        # Display the original image
        plt.figure(figsize=(8, 8))
        plt.axis('off')
        plt.imshow(current_image_array)
        plt.title("Original Image")
        plt.show()

        # Define the color range for masking
        low = np.array(low)
        high = np.array(high)

        # Create a color mask based on the specified range
        mask = cv2.inRange(current_image_array, low, high)

        # Display the color mask
        plt.figure(figsize=(8, 8))
        plt.axis('off')
        plt.imshow(mask, cmap='gray')
        plt.title("Color Mask")
        plt.show()

        # Apply the color mask to the original image
        result = cv2.bitwise_and(current_image_array, mask=mask)

        # Display the result
        plt.figure(figsize=(8, 8))
        plt.axis('off')
        plt.imshow(result)
        plt.title("Result")
        plt.show()

         # Save the edge-detected image if an output path is provided
        if output_image_path:
           result.save(output_image_path)

def ContourDetection(output_image_path:None):
    global current_image
    if current_image:
        current_image_array = np.array(current_image)
        # Load the input image
        sample_image = cv2.imread(current_image_array)
        img = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))

        # Show the original image
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title("Original Image")
        plt.show()

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)

        # Show the thresholded image
        plt.figure(figsize=(8, 8))
        plt.imshow(thresh, cmap="gray")
        plt.axis('off')
        plt.title("Thresholded Image")
        plt.show()

        edges = cv2.dilate(cv2.Canny(thresh, 0, 255), None)

        # Show the edge image
        plt.figure(figsize=(8, 8))
        plt.imshow(edges, cmap="gray")
        plt.axis('off')
        plt.title("Edge Image")
        plt.show()

        cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
        mask = np.zeros((256, 256), np.uint8)
        masked = cv2.drawContours(mask, [cnt], -1, 255, -1)

        # Show the masked image
        plt.figure(figsize=(8, 8))
        plt.imshow(masked, cmap="gray")
        plt.axis('off')
        plt.title("Masked Image")
        plt.show()

        dst = cv2.bitwise_and(img, img, mask=mask)
        segmented = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

        # Save the edge-detected image if an output path is provided
        if output_image_path:
           segmented.save(output_image_path)
    
        return segmented

def apply_unsharp_masking_filter():
    global current_image
    if current_image:
        unsharp_masking_filter = ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)
        current_image = current_image.filter(unsharp_masking_filter)
        display_image(current_image, zoom_factor)

def apply_unsharp_masking_filter():
    global current_image
    if current_image:
        unsharp_masking_filter = ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)
        current_image = current_image.filter(unsharp_masking_filter)
        display_image(current_image, zoom_factor)

def apply_high_pass_filter():
    global current_image
    if current_image:
        high_pass_filter = ImageFilter.EDGE_ENHANCE_MORE
        current_image = current_image.filter(high_pass_filter)
        display_image(current_image, zoom_factor)

def apply_sobel_filter():
    global current_image
    if current_image:
        sobel_filter = ImageFilter.FIND_EDGES
        current_image = current_image.filter(sobel_filter)
        display_image(current_image, zoom_factor)

def apply_prewitt_filter():
    global current_image
    if current_image:
        prewitt_filter = ImageFilter.CONTOUR
        current_image = current_image.filter(prewitt_filter)
        display_image(current_image, zoom_factor)

def apply_custom_sharpening_filter():
    global current_image
    if current_image:
        # Define your custom sharpening filter here
        custom_sharpening_filter = ImageFilter.Kernel(size=(3, 3), kernel=[-1, -1, -1, -1, 9, -1, -1, -1, -1])
        current_image = current_image.filter(custom_sharpening_filter)
        display_image(current_image, zoom_factor)

def sharpen_image():
    global current_image
    if current_image:
        current_image = current_image.filter(ImageFilter.SHARPEN)
        display_image(current_image, zoom_factor)


def flip_horizontal():
    global current_image
    if current_image:
        current_image = current_image.transpose(Image.FLIP_LEFT_RIGHT)
        display_image(current_image, zoom_factor)

def flip_vertical():
    global current_image
    if current_image:
        current_image = current_image.transpose(Image.FLIP_TOP_BOTTOM)
        display_image(current_image, zoom_factor)

def rotate_left():
    global current_image
    if current_image:
        current_image = current_image.rotate(90, expand=False)
        display_image(current_image, zoom_factor)

def rotate_right():
    global current_image
    if current_image:
        current_image = current_image.rotate(-90, expand=False)
        display_image(current_image, zoom_factor)

def rotate_image(degrees):
    global current_image
    if current_image:
        current_image = current_image.rotate(degrees, expand=True)
        display_image(current_image, zoom_factor)


def convert_to_grayscale():
    global input_image, resized_image
     # Make sure to define input_image before calling convert_to_grayscale
    input_image = cv2.imread("C:/Users/Dewmi Silva/Downloads/YellowLabradorLooking_new.jpg")
    if input_image is not None:
        resized_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        plt.imshow(resized_image)
        plt.axis('off')
        plt.show()
    else:
        # Handle the case where input_image is not defined
        print("Input image is not defined.")
        
def bw_to_color():
    global current_image
    if current_image is not None:
        if isinstance(current_image, Image.Image):
            # Convert the PIL Image to a NumPy array
            current_image = np.array(current_image)
        if isinstance(current_image, np.ndarray):
            # Ensure the image is in grayscale (1 channel)
            if len(current_image.shape) == 3:
                current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
            # Apply a colormap to the grayscale image
            colored_image = cv2.applyColorMap(current_image, cv2.COLORMAP_JET)
            # Display the colorized image
            plt.imshow(cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        else:
            print("Failed to convert the image to a NumPy array.")
    else:
        print("current_image is not defined.")

def color_to_bw():
    global current_image
    if current_image:
        current_image = current_image.convert("L")
        plt.imshow(current_image, cmap='gray')
        plt.axis('off')
        plt.show()


def grayscale_to_color():
    global current_image
    if current_image is not None:
        if isinstance(current_image, str):
            # If current_image is a file path, read it as grayscale using OpenCV
            current_image = cv2.imread(current_image, cv2.IMREAD_GRAYSCALE)
        if isinstance(current_image, np.ndarray):
            if len(current_image.shape) == 2:
                # Apply a colormap to the grayscale image
                colored_image = cv2.applyColorMap(current_image, cv2.COLORMAP_JET)
                plt.imshow(cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()
            else:
                print("The image is not grayscale (1 channel).")
        else:
            print("current_image is not a valid NumPy array or file path.")
    else:
        print("current_image is not defined.")

def bw_to_grayScale():
    global current_image
    if current_image is not None:
        if isinstance(current_image, np.ndarray) and len(current_image.shape) == 2:
            # Ensure the image is grayscale (1 channel) and display it
            plt.imshow(current_image, cmap='gray')
            plt.axis('off')
            plt.show()
        else:
            print("current_image is not a valid grayscale NumPy array.")
    else:
        print("current_image is not defined.")

def grayscale_to_blackAndWhite():
    global current_image
    if current_image:
        if isinstance(current_image, Image.Image):
            # Convert the grayscale image to black and white (mode '1')
            current_image = current_image.convert("1")
            plt.imshow(current_image, cmap='gray')
            plt.axis('off')
            plt.show()
        else:
            print("current_image is not a valid Pillow Image object.")
    else:
        print("current_image is not defined.")
# Call the functions as needed
convert_to_grayscale()
# Call other functions here as necessary


# Create the main window
root = tk.Tk()
root.title("Image Viewer")

# Set a new background color using a hexadecimal color code
root.configure(bg="#3A3A3A")  # Use your preferred color code

# Create a new colorful header label
header_label = tk.Label(root, text="Image Viewer", font=("Helvetica", 20), bg="#2E86DE", fg="white", padx=10, pady=10)
header_label.pack(fill="x")

# Create a frame for the buttons with a new colorful background
button_frame = tk.Frame(root, bg="#3A3A3A")
button_frame.pack(pady=10)

# Create buttons with updated styling and place them in the frame
open_button = tk.Button(button_frame, text="Open Image", command=open_image, bg="#2E86DE", fg="white", padx=10, pady=5)
open_button.pack(side="left", padx=5)

reset_button = tk.Button(button_frame, text="Reset Image", command=reset_image, bg="#F39C12", fg="white", padx=10, pady=5)
reset_button.pack(side="left", padx=5)

open_operations_window_button = tk.Button(button_frame, text="Open Operations Window", command=open_image_operations_window, bg="#27AE60", fg="white", padx=10, pady=5)
open_operations_window_button.pack(side="left", padx=5)

save_button = tk.Button(button_frame, text="Save Image", command=save_image, bg="#2E86DE", fg="white", padx=10, pady=5)
save_button.pack(side="left", padx=5)

crop_button = tk.Button(button_frame, text="Crop Image", command=crop_image, bg="#F39C12", fg="white", padx=10, pady=5)
crop_button.pack(side="left", padx=5)

image_label = tk.Label(root)
image_label.pack()

# Bind mouse events for cropping
image_label.bind("<Button-1>", start_crop)
image_label.bind("<B1-Motion>", update_crop)
image_label.bind("<ButtonRelease-1>", end_crop)

root.mainloop()