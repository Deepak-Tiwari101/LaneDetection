import cv2
import numpy as np
import os
import imutils
import matplotlib.pyplot as plt

os.chdir("F:\Jupyter")


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    outline = cv2.Canny(blur, 50, 150)  # threshold ratio of 1:3
    return outline


def region_of_interest(image):
    polygons = np.array([
        [(120, 282), (508, 282), (316, 175)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)  # BITWISE AND operation to get the final required outline
    return mask, masked_image


def display_lines(image, line):
    image = np.zeros_like(image)
    if line is not None:
        for x1, y1, x2, y2 in line:
            p1 = (int(x1), int(y1))
            p2 = (int(x2), int(y2))

            cv2.line(image, p1, p2, (0, 0, 200), 10)
    return image


def make_coordinates(image, line_parameters):  # func to get the line coordinate
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    try:
        slope, intercept = line_parameters
        # print(image.shape)  # for getting the height of the image ie. 784 in this pic
        y1 = 282
        y2 = 212
        x1 = ((y1 - intercept) / slope)
        x2 = ((y2 - intercept) / slope)
    except TypeError:
        slope, intercept = 0, 0
    return np.array([x1, y1, x2, y2])


def avg_slope_intercept(image, lines):
    left_lane = []
    right_lane = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x1 and y1 and x2 and y2 is not None:
            parameters = np.polyfit(
                (x1, x2),
                (y1, y2),
                1)  # with degree as 3rd argument ie. 1, it will fit the line equation using the coordinates
            # print(parameters)  # this function will give the slope and intercept after forming the equation
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0 and slope != 0:
            left_lane.append((slope, intercept))
        else:
            right_lane.append((slope, intercept))
    left_lane_avg = np.average(left_lane, axis=0)
    # print(left_lane_avg)
    right_lane_avg = np.average(right_lane, axis=0)
    left_line = make_coordinates(image, left_lane_avg)
    right_line = make_coordinates(image, right_lane_avg)
    # print(left_line, right_line)
    return left_lane_avg, np.array([left_line, right_line], dtype=np.int32)


if __name__ == '__main__':
    """FOR IMAGE COMPUTATION"""

    # image = cv2.imread('test_image.jpg')
    # lane_image = np.copy(image)
    # canny = canny(lane_image)
    # crop = region_of_interest(canny)
    # lines = cv2.HoughLinesP(crop, 2, 1 * np.pi / 180, 100, minLineLength=42, maxLineGap=5)
    # # print(lines)
    # avg_lines = avg_slope_intercept(lane_image, lines)
    # xl1, yl1, xl2, yl2 = avg_slope_intercept(lane_image, lines)[0]
    # xr1, yr1, xr2, yr2 = avg_slope_intercept(lane_image, lines)[1]
    # center_x1 = (xl1 + xr1) // 2
    # center_y1 = int(yl1)
    # center_x2 = (xl2 + xr2) // 2
    # center_y2 = int(yl2)
    # center_line = np.zeros_like(lane_image)
    # cv2.line(center_line, (center_x1, center_y1), (center_x2, center_y2), (0, 255, 255), 3)
    # line_image = display_lines(lane_image, avg_lines)
    # new_masked = cv2.addWeighted(lane_image, 0.6, line_image, 1, 0)  # to blend line and lane image
    # cen_lin_img = cv2.addWeighted(new_masked, 0.9, center_line, 1, 0)
    # cv2.imshow('Result', cen_lin_img)
    #
    # cv2.waitKey(0)

    """FOR VIDEO COMPUTATION"""

    cap = cv2.VideoCapture("new_Video.mp4")
    avg_lines = []
    # new_file = None
    while cap.isOpened():
        _, frame = cap.read()
        # plt.imshow(frame)
        # plt.show()
        canny = cv2.Canny(frame, 50, 150)
        mask, crop = region_of_interest(canny)
        lines = cv2.HoughLinesP(crop, 2, 1 * np.pi / 180, 100, minLineLength=10, maxLineGap=100000)
        left_lane_avg, avg_lines = avg_slope_intercept(frame, lines)

        # =========================================================================#
        # try:
        #     avg_lines = avg_slope_intercept(frame, lines)
        # except Exception:
        #     pass
        # print(avg_lines[0])
        x1, y1, x2, y2 = avg_lines[0]
        # new_file = open("trajectory.txt", mode="a+", encoding="utf-8")
        # new_file.writelines(str(x1) + "\t" + str(y1) + "\t" + str(x2) + "\t" + str(y2) + "\t" + "\n")
        # ========================================================================#

        slope, intercept = left_lane_avg
        x = 315
        Y = 282
        X = (1 / slope) * (Y - y1) + x1
        dist = x - X
        mean_dist = 156.0
        position = (300, 100)
        if mean_dist - 25 < dist < mean_dist + 20:  # set according to the video
            pass
        elif dist > mean_dist + 20:
            cv2.putText(frame, "Steer Left", position, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)
        elif dist < mean_dist - 25:
            cv2.putText(frame, "Steer Right", position, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)

        # =======================================================================#
        centerPoint1 = (315, 282)
        centerPoint2 = (315, 220)
        center_line = cv2.line(np.zeros_like(frame), centerPoint1, centerPoint2, (0, 255, 0), 1)

        line_image = display_lines(frame, avg_lines)
        new_masked = cv2.addWeighted(frame, 1, line_image, 1, 0)  # to blend line and lane image
        new_new_masked = cv2.addWeighted(new_masked, 1, center_line, 1, 0)  # to blend line and lane image
        # final = imutils.resize(new_new_masked, width=1365)  # changing video display size
        # out.write(frame)
        cv2.imshow('RESULT', new_masked)
        if cv2.waitKey(32) == ord('q'):
            break

    # new_file.close()
    cap.release()
    # out.release()
    cv2.destroyAllWindows()
