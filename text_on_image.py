from PIL import ImageDraw, ImageFont


def add_caption_to_image(image, caption, output_path="output.jpg"):
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    font = ImageFont.load_default(size=10000)

    text_width = draw.textlength(caption, font=font)
    font_size = font.size
    text_height = font_size * (caption.count("\n") + 1)

    while text_width > 0.9 * width:
        font = ImageFont.load_default(size=font.size - 3)
        text_width = draw.textlength(caption, font=font)
        font_size = font.size
        text_height = font_size * (caption.count("\n") + 1)

    box_padding = 20
    rectangle_height = text_height + box_padding
    text_y = height - rectangle_height + (rectangle_height - text_height) // 2

    draw.rectangle([(0, height - rectangle_height), (width, height)], fill="black")
    text_x = (width - text_width) // 2
    draw.text((text_x, text_y), caption, font=font, fill="white", align="center")

    image.save(output_path)

