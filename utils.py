import cv2

def overlay_image(frame, image, position=(0, 0)):
    """
    Overlay an image onto a frame.

    Args:
        frame: The frame (image) onto which the image will be overlaid.
        image: The image to overlay onto the frame.
        position: The position (top-left corner) where the image will be placed on the frame. Default is (0, 0).

    Returns:
        The frame with the image overlayed.
    """

    x, y = position

    # resize the image to 64x64
    image_resized = cv2.resize(image, (64, 64))

    # image contains an extra alpha channel, reconstruct the image without an 
    # alpha channel
    if (image_resized.shape[2] == 4):
        b, g, r, _ = cv2.split(image_resized)
        image_resized = cv2.merge((b, g, r))

    frame[y:y+64, x:x+64] = image_resized

    return frame