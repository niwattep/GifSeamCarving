from PIL import Image


def extract_frames(input):
    """
    :param input:
    :return: list of frame
    """
    frame: Image = Image.open(input)
    next_frame = 0
    output = []
    while frame:
        output.append(frame)
        next_frame += 1
        try:
            frame = frame.seek(frame.tell() + 1)
        except EOFError:
            break;
    for i in output:
        print(i)
    return output
