import cv2


def main():
    base_cascade_path = "/usr/local/Cellar/opencv3/3.2.0/share/OpenCV/haarcascades/"
    face_cascade_file_path = base_cascade_path + "haarcascade_frontalface_default.xml"
    nose_cascade_file_path = base_cascade_path + "haarcascade_mcs_nose.xml"

    # classifiers
    face_cascade = cv2.CascadeClassifier(face_cascade_file_path)
    nose_cascade = cv2.CascadeClassifier(nose_cascade_file_path)

    # filter
    img_moustache = cv2.imread('moustache.png', -1)

    # mask and inverted mask
    orig_mask = img_moustache[:, :, 3]
    orig_mask_inv = cv2.bitwise_not(orig_mask)

    img_moustache = img_moustache[:, :, 0:3]
    orig_moustache_height, orig_moustache_width = img_moustache.shape[:2]

    video_capture = cv2.VideoCapture(0)

    while cv2.waitKey(1) & 0xFF == ord('q'):
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            # outline face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            nose = nose_cascade.detectMultiScale(roi_gray)

            for (nx, ny, nw, nh) in nose:
                # outline nose
                cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 0), 2)

                # scale moustache to 3x nose width
                moustacheWidth = 3 * nw
                moustacheHeight = moustacheWidth * orig_moustache_height / orig_moustache_width

                # centre the moustache on the bottom of the nose
                x1 = int(nx - (moustacheWidth / 4))
                x2 = int(nx + nw + (moustacheWidth / 4))
                y1 = int(ny + nh - (moustacheHeight / 2))
                y2 = int(ny + nh + (moustacheHeight / 2))

                # check for clipping
                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if x2 > w:
                    x2 = w
                if y2 > h:
                    y2 = h

                moustache_width = x2 - x1
                moustache_height = y2 - y1

                moustache = cv2.resize(img_moustache, (moustache_width, moustache_height), interpolation=cv2.INTER_AREA)
                mask = cv2.resize(orig_mask, (moustache_width, moustache_height), interpolation=cv2.INTER_AREA)
                mask_inv = cv2.resize(orig_mask_inv, (moustache_width, moustache_height), interpolation=cv2.INTER_AREA)

                # take ROI for moustache from background equal to size of moustache image
                roi = roi_color[y1:y2, x1:x2]
                roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                roi_fg = cv2.bitwise_and(moustache, moustache, mask=mask)

                # join roi_bg and roi_fg
                dst = cv2.add(roi_bg, roi_fg)
                roi_color[y1:y2, x1:x2] = dst

                break

        cv2.imshow('Video', frame)

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
