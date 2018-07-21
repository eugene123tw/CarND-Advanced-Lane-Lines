import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from tools import binarization


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False

        # x values of the last n fits of the line
        self.recent_xfitted = []

        # average x values of the fitted line over the last n iterations
        self.bestx = None

        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None

        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]

        # radius of curvature of the line in some units
        self.radius_of_curvature = None

        # distance in meters of vehicle center from the line
        self.line_base_pos = None

        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')

        # x values for detected line pixels
        self.allx = None

        # y values for detected line pixels
        self.ally = None

class VideoProcessor:
    def __init__(self, cali_fnames):
        self.cali_fnames = glob.glob(cali_fnames)
        self.mtx, self.dist = self.cal_undistort(self.cali_fnames, 9, 6)
        self.RIGHT_LINE = Line()
        self.LEFT_LINE = Line()


    def cal_undistort(self, img_paths, nx, ny):
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        for fname in img_paths:
            # Step through the list and search for chessboard corners
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

        img_size = (img.shape[1], img.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

        return mtx, dist

    def warp_image(self, img):
        img_size = (img.shape[1], img.shape[0])
        src = np.float32(
            [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
             [((img_size[0] / 6) - 10), img_size[1]],
             [(img_size[0] * 5 / 6) + 45, img_size[1]],
             [(img_size[0] / 2 + 60), img_size[1] / 2 + 100]])
        dst = np.float32(
            [[(img_size[0] / 4) + 25, 0],
             [(img_size[0] / 4), img_size[1]],
             [(img_size[0] * 3 / 4), img_size[1]],
             [(img_size[0] * 3 / 4) - 25, 0]])

        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
        return warped, M, Minv

    def poly_searching(self, binary_warped, nwindows=5, margin=100, minpix=50):
        """
        Line peaks in a Histogram -> Implement Sliding Windows and Fit a Polynomial

        Args:
            binary_warped:
            nwindows: the number of sliding windows
            margin: the width of the windows +/- margin
            minpix: minimum number of pixels found to recenter window

        Return:
            leftx:
            lefty:
            rightx:
            righty:
            leftx_base:
            rightx_base:
        """
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        if self.RIGHT_LINE.detected and self.LEFT_LINE.detected:
            right_fit = self.RIGHT_LINE.best_fit
            left_fit = self.LEFT_LINE.best_fit

            left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                           left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                                 left_fit[1] * nonzeroy + left_fit[
                                                                                     2] + margin)))

            right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                            right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                                   right_fit[1] * nonzeroy + right_fit[
                                                                                       2] + margin)))



        else:
            histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
            midpoint = np.int(histogram.shape[0] // 2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            # Set height of windows
            window_height = np.int(binary_warped.shape[0] // nwindows)

            # Current positions to be updated for each window
            leftx_current = leftx_base
            rightx_current = rightx_base

            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []

            # Step through the windows one by one
            for window in range(nwindows):
                win_y_low = binary_warped.shape[0] - (window + 1) * window_height
                win_y_high = binary_warped.shape[0] - window * window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin

                good_left_inds = ((win_y_low <= nonzeroy) & (nonzeroy < win_y_high) & (win_xleft_low <= nonzerox) & (
                        nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((win_y_low <= nonzeroy) & (nonzeroy < win_y_high) & (win_xright_low <= nonzerox) & (
                        nonzerox < win_xright_high)).nonzero()[0]

                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)

                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            # Concatenate the arrays of indices
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
            self.leftx_base = leftx_base
            self.rightx_base = rightx_base


        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # # Left/Right lane polynomial parameters -> x1, x2, x3
        # left_fit = np.polyfit(lefty, leftx, 2)
        # right_fit = np.polyfit(righty, rightx, 2)
        #
        # ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        # left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        # right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        #
        # debug_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # for i in range(len(left_fitx)):
        #     cv2.circle(debug_img, (int(left_fitx[i]), int(ploty[i])), 1, (0, 255, 255), -1)
        #     cv2.circle(debug_img, (int(right_fitx[i]), int(ploty[i])), 1, (0, 255, 255), -1)
        #
        # left_texts = "%.2f, %.2f, %.2f" % (left_fit[0], left_fit[1], left_fit[2])
        # right_texts = "%.2f, %.2f, %.2f" % (right_fit[0], right_fit[1], right_fit[2])
        # cv2.putText(debug_img, left_texts, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # cv2.putText(debug_img, right_texts, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # cv2.imshow('debug', debug_img)
        # cv2.waitKey(0)

        return leftx, lefty, rightx, righty, self.leftx_base, self.rightx_base

    def fit_line(self, fit_param):
        ploty = self.ploty
        fitx = fit_param[0] * ploty ** 2 + fit_param[1] * ploty + fit_param[2]
        return fitx

    def draw_PolyArea(self, left_fitx, right_fitx):
        ploty = self.ploty
        # Create an image to draw the lines on
        warp_zero = np.zeros((self.H, self.W)).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        return color_warp

    def measure_curvature(self, leftx, lefty, rightx, righty):
        y_eval = np.max(self.ploty)
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (
                    2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])
        # Now our radius of curvature is in meters
        return left_curverad, right_curverad


    def mv_line_tracking(self, img):
        self.H, self.W = img.shape[0], img.shape[1]
        LEFT_LINE = self.LEFT_LINE
        RIGHT_LINE = self.RIGHT_LINE

        self.ploty = np.linspace(0, self.H - 1, self.H)
        mtx = self.mtx
        dist = self.dist
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        binary = binarization(undist, s_thresh=(210, 240))

        binary_warped, M, Minv = self.warp_image(binary)

        leftx, lefty, rightx, righty, leftx_base, rightx_base = self.poly_searching(binary_warped, nwindows=5, margin=50)

        # Left/Right lane polynomial parameters -> x1, x2, x3
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Use previous fitted value if slope don't make sense
        if not((left_fit[0] < 0) != (right_fit[0] < 0)):
            left_fitx = self.fit_line(left_fit)
            right_fitx = self.fit_line(right_fit)
        else:
            left_fitx = LEFT_LINE.recent_xfitted
            right_fitx = RIGHT_LINE.recent_xfitted

        # Measure and validate curvature
        left_curverad, right_curverad = self.measure_curvature(leftx, lefty, rightx, righty)

        # Create a filled poly area from predicted lanes
        color_warp = self.draw_PolyArea(left_fitx, right_fitx)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (self.W, self.H))

        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        texts = "L Curve: %d m, R Curve: %d m, Left from center %2f m" % \
                (left_curverad, right_curverad, (self.W / 2 - (left_fitx[-1] + right_fitx[-1]) / 2) * (3.7 / 700))
        cv2.putText(result, texts, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Store x values of the last n fits of the line
        LEFT_LINE.recent_xfitted = left_fitx
        RIGHT_LINE.recent_xfitted = right_fitx

        LEFT_LINE.current_fit = left_fit
        RIGHT_LINE.current_fit = right_fit

        if LEFT_LINE.detected and RIGHT_LINE.detected:
            # Smoothing
            LEFT_LINE.best_fit = np.average(np.vstack((LEFT_LINE.best_fit, LEFT_LINE.current_fit))[-30:], 0)
            RIGHT_LINE.best_fit = np.average(np.vstack((RIGHT_LINE.best_fit, RIGHT_LINE.current_fit))[-30:], 0)
        else:
            LEFT_LINE.bestx = leftx
            RIGHT_LINE.bestx = rightx
            LEFT_LINE.best_fit = left_fit
            RIGHT_LINE.best_fit = right_fit
            LEFT_LINE.detected = True
            RIGHT_LINE.detected = True

        return result


if __name__ == '__main__':
    VP = VideoProcessor('camera_cal/*.jpg')
    clip = VideoFileClip('project_video.mp4')
    mv_clip = clip.fl_image(VP.mv_line_tracking)
    mv_clip.write_videofile('project_out.mp4', audio=False)