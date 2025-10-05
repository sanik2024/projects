import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Function to generate animated image from text prompt
def generate_image(text):
    # Placeholder code to generate a simple animated image
    frames = []
    for i in range(10):  # Number of animation frames
        # Generate a simple image based on the text prompt
        image = np.random.rand(256, 256, 3)  # Random RGB image
        frames.append(image)

    # Convert frames to a list of PIL images
    images = [Image.fromarray(np.uint8(frame * 255)) for frame in frames]

    # Save frames as a GIF
    images[0].save("animated_image.gif", save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)

    # Display the generated animated image
    st.image("animated_image.gif", use_column_width=True)

# Streamlit UI
def main():
    st.title("Animated Image Generator")
    st.write("Enter a text prompt to generate an animated image:")

    # Text input for user to enter prompt
    text_prompt = st.text_input("Text Prompt", "")

    # Button to generate image
    if st.button("Generate Animated Image"):
        # Check if text prompt is not empty
        if text_prompt.strip() != "":
            # Generate animated image
            generate_image(text_prompt)
        else:
            st.warning("Please enter a text prompt.")

if __name__ == "__main__":
    main()
