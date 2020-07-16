import numpy as np
from PIL import Image, ImageFont, ImageDraw


# =========================== Plot where injuries occured on body ======================================================
yearsrun = 2
popsize = 1000
data = np.genfromtxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/Injlocs.txt')

def main():
    try:
        img = Image.open('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/bodies.jpg')
        # img = img.filter(ImageFilter.SHARPEN)
        thresh = 230
        fn = lambda x: 255 if x > thresh else 0
        img = img.convert('L').point(fn, mode='1')
        # img = img.convert('1')
        #

        font_path = "C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/Fonts/BOD_R.TTF"
        fnt = ImageFont.truetype(font_path, 15)
        font_path = "C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/Fonts/BOD_B.TTF"
        titlefnt = ImageFont.truetype(font_path, 20)
        d = ImageDraw.Draw(img)
        d.text((0, 10), f"The distribution of injury location: {yearsrun} year model run,"
                        "\n"
                        f"population size = {popsize}", font=titlefnt, fill='black')
        d.text((120, 80), "Head:"
                          "\n"
                          f"{round(data[0] / sum(data), 2)} %", font=fnt, fill='black')
        d.line(xy=[170, 90, 200, 90], fill='black', width=1)
        d.text((300, 100), "Face:"
                           "\n"
                           f"{round(data[1] / sum(data), 2)} %", font=fnt, fill='black')
        d.line(xy=[230, 110, 270, 110], fill='black', width=1)

        d.text((120, 120), "Neck:"
                           "\n"
                           f"{round(data[2] / sum(data), 2)} %", font=fnt, fill='black')
        d.line(xy=[170, 140, 210, 150], fill='black', width=1)

        d.text((200, 180), "Thorax:"
                           "\n"
                           f"{round(data[3] / sum(data), 2)} %", font=fnt, fill='black')
        d.text((205, 250), "Spine:"
                           "\n"
                           f"{round(data[5] / sum(data), 2)} %", font=fnt, fill='black')
        d.text((205, 300), "Abdomen"
                           "\n"
                           f"{round(data[4] / sum(data), 2)} %", font=fnt, fill='black')
        d.text((350, 220), "Upper"
                           "\n"
                           "extremity:"
                           "\n"
                           f"{round(data[6] / sum(data), 2)} %", font=fnt, fill='black')
        d.line(xy=[340, 240, 160, 220], fill='black', width=1)
        d.line(xy=[340, 240, 300, 260], fill='black', width=1)

        d.text((300, 420), "Lower"
                           "\n"
                           "extremity:"
                           "\n"
                           f"{round(data[7] / sum(data), 2)} %", font=fnt, fill='black')
        d.line(xy=[290, 440, 200, 440], fill='black', width=1)
        d.line(xy=[290, 440, 260, 540], fill='black', width=1)
        img.show()
        img.save('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/ModelInjuryLocationOnBody.jpg')

    except IOError:
        pass


if __name__ == "__main__":
    main()

data = np.genfromtxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/OpenWoundDistribution.txt')


def main():
    try:
        img = Image.open('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/bodies-cropped.jpg')
        img.show()
        # img = img.filter(ImageFilter.SHARPEN)
        thresh = 230
        fn = lambda x: 255 if x > thresh else 0
        img = img.convert('L').point(fn, mode='1')
        # img = img.convert('1')
        #

        font_path = "C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/Fonts/BOD_R.TTF"
        fnt = ImageFont.truetype(font_path, 15)
        font_path = "C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/Fonts/BOD_B.TTF"
        titlefnt = ImageFont.truetype(font_path, 20)
        d = ImageDraw.Draw(img)
        d.text((0, 10), f"The distribution of open wound location: {yearsrun} year model run,"
                        "\n"
                        f"population size = {popsize}", font=titlefnt, fill='black')
        d.text((120, 80), "Head:"
                          "\n"
                          f"{round(data[0] / sum(data), 2)} %", font=fnt, fill='black')
        d.line(xy=[170, 90, 200, 90], fill='black', width=1)
        d.text((300, 100), "Face:"
                           "\n"
                           f"{round(data[1] / sum(data), 2)} %", font=fnt, fill='black')
        d.line(xy=[230, 110, 270, 110], fill='black', width=1)

        d.text((120, 120), "Neck:"
                           "\n"
                           f"{round(data[2] / sum(data), 2)} %", font=fnt, fill='black')
        d.line(xy=[170, 140, 210, 150], fill='black', width=1)

        d.text((200, 180), "Thorax:"
                           "\n"
                           f"{round(data[3] / sum(data), 2)} %", font=fnt, fill='black')
        d.text((205, 250), "Spine:"
                           "\n"
                           f"{round(data[5] / sum(data), 2)} %", font=fnt, fill='black')
        d.text((205, 300), "Abdomen"
                           "\n"
                           f"{round(data[4] / sum(data), 2)} %", font=fnt, fill='black')
        d.text((350, 220), "Upper"
                           "\n"
                           "extremity:"
                           "\n"
                           f"{round(data[6] / sum(data), 2)} %", font=fnt, fill='black')
        d.line(xy=[340, 240, 160, 220], fill='black', width=1)
        d.line(xy=[340, 240, 300, 260], fill='black', width=1)

        d.text((300, 420), "Lower"
                           "\n"
                           "extremity:"
                           "\n"
                           f"{round(data[7] / sum(data), 2)} %", font=fnt, fill='black')
        d.line(xy=[290, 440, 200, 440], fill='black', width=1)
        d.line(xy=[290, 440, 260, 540], fill='black', width=1)
        img.show()
        img.save('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/OpenWoundLocationOnBody.jpg')

    except IOError:
        pass


if __name__ == "__main__":
    main()

data = np.genfromtxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/FractureDistribution.txt')


def main():
    try:
        img = Image.open('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/bodies-cropped.jpg')
        # img = img.filter(ImageFilter.SHARPEN)
        thresh = 230
        fn = lambda x: 255 if x > thresh else 0
        img = img.convert('L').point(fn, mode='1')
        # img = img.convert('1')
        #

        font_path = "C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/Fonts/BOD_R.TTF"
        fnt = ImageFont.truetype(font_path, 15)
        font_path = "C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/Fonts/BOD_B.TTF"
        titlefnt = ImageFont.truetype(font_path, 20)
        d = ImageDraw.Draw(img)
        d.text((0, 10), f"The distribution of fracture location: {yearsrun} year model run,"
                        "\n"
                        f"population size = {popsize}", font=titlefnt, fill='black')
        d.text((120, 80), "Head:"
                          "\n"
                          f"{round(data[0] / sum(data), 2)} %", font=fnt, fill='black')
        d.line(xy=[170, 90, 200, 90], fill='black', width=1)
        d.text((300, 100), "Face:"
                           "\n"
                           f"{round(data[1] / sum(data), 2)} %", font=fnt, fill='black')
        d.line(xy=[230, 110, 270, 110], fill='black', width=1)

        d.text((120, 120), "Neck:"
                           "\n"
                           f"{round(data[2] / sum(data), 2)} %", font=fnt, fill='black')
        d.line(xy=[170, 140, 210, 150], fill='black', width=1)

        d.text((200, 180), "Thorax:"
                           "\n"
                           f"{round(data[3] / sum(data), 2)} %", font=fnt, fill='black')
        d.text((205, 250), "Spine:"
                           "\n"
                           f"{round(data[5] / sum(data), 2)} %", font=fnt, fill='black')
        d.text((205, 300), "Abdomen"
                           "\n"
                           f"{round(data[4] / sum(data), 2)} %", font=fnt, fill='black')
        d.text((350, 220), "Upper"
                           "\n"
                           "extremity:"
                           "\n"
                           f"{round(data[6] / sum(data), 2)} %", font=fnt, fill='black')
        d.line(xy=[340, 240, 160, 220], fill='black', width=1)
        d.line(xy=[340, 240, 300, 260], fill='black', width=1)

        d.text((300, 420), "Lower"
                           "\n"
                           "extremity:"
                           "\n"
                           f"{round(data[7] / sum(data), 2)} %", font=fnt, fill='black')
        d.line(xy=[290, 440, 200, 440], fill='black', width=1)
        d.line(xy=[290, 440, 260, 540], fill='black', width=1)

        img.save('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/FractureLocationOnBody.jpg')

    except IOError:
        pass


if __name__ == "__main__":
    main()

data = np.genfromtxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/BurnDistribution.txt')


def main():
    try:
        img = Image.open('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/bodies-cropped.jpg')
        # img = img.filter(ImageFilter.SHARPEN)
        thresh = 230
        fn = lambda x: 255 if x > thresh else 0
        img = img.convert('L').point(fn, mode='1')
        # img = img.convert('1')
        #

        font_path = "C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/Fonts/BOD_R.TTF"
        fnt = ImageFont.truetype(font_path, 15)
        font_path = "C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/Fonts/BOD_B.TTF"
        titlefnt = ImageFont.truetype(font_path, 20)
        d = ImageDraw.Draw(img)
        d.text((0, 10), f"The distribution of burn location: {yearsrun} year model run,"
                        "\n"
                        f"population size = {popsize}", font=titlefnt, fill='black')
        d.text((120, 80), "Head:"
                          "\n"
                          f"{round(data[0] / sum(data), 2)} %", font=fnt, fill='black')
        d.line(xy=[170, 90, 200, 90], fill='black', width=1)
        d.text((300, 100), "Face:"
                           "\n"
                           f"{round(data[1] / sum(data), 2)} %", font=fnt, fill='black')
        d.line(xy=[230, 110, 270, 110], fill='black', width=1)

        d.text((120, 120), "Neck:"
                           "\n"
                           f"{round(data[2] / sum(data), 2)} %", font=fnt, fill='black')
        d.line(xy=[170, 140, 210, 150], fill='black', width=1)

        d.text((200, 180), "Thorax:"
                           "\n"
                           f"{round(data[3] / sum(data), 2)} %", font=fnt, fill='black')
        d.text((205, 250), "Spine:"
                           "\n"
                           f"{round(data[5] / sum(data), 2)} %", font=fnt, fill='black')
        d.text((205, 300), "Abdomen"
                           "\n"
                           f"{round(data[4] / sum(data), 2)} %", font=fnt, fill='black')
        d.text((350, 220), "Upper"
                           "\n"
                           "extremity:"
                           "\n"
                           f"{round(data[6] / sum(data), 2)} %", font=fnt, fill='black')
        d.line(xy=[340, 240, 160, 220], fill='black', width=1)
        d.line(xy=[340, 240, 300, 260], fill='black', width=1)

        d.text((300, 420), "Lower"
                           "\n"
                           "extremity:"
                           "\n"
                           f"{round(data[7] / sum(data), 2)} %", font=fnt, fill='black')
        d.line(xy=[290, 440, 200, 440], fill='black', width=1)
        d.line(xy=[290, 440, 260, 540], fill='black', width=1)

        img.save('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/BurnLocationOnBody.jpg')

    except IOError:
        pass


if __name__ == "__main__":
    main()
