import cv2 as cv
import numpy as np
from math import cos, sin, radians


def bresenhamLineGenerator(x0, y0, x1, y1):
    """Generates a list of point represeting a line
    between the two given points. The list will always
    start from (x0, y0), which in this use-case means
    from the concave corner of the related marker.

    Args:
        x0 (int): point 0's x coordinate.
        y0 (int): point 0's y coordinate.
        x1 (int): point 1's x coordinate.
        y1 (int): point 1's y coordinate.

    Returns:
        list[int]: list of points.
    """

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    # This deals with division by zero.
    # The slope value needs just to be positive or negative,
    # hence any value can fit to handle this case (e.g., 10.).
    slope = dy / dx if dx != 0 else 10.0

    flag = True

    linePixel = []
    linePixel.append((x0, y0))

    # Step initialized according to the type of slope
    # and position of X and Y.
    yStep = -1 if slope >= 1.0 and y0 > y1 else 1
    xStep = (
        -1
        if ((x0 > x1 or y0 > y1) and slope < 1.0)
        or (slope > 1.0 and x0 > x1 and y0 > y1)
        or (x0 > x1 and y0 < y1)
        else 1
    )

    slopeSmallerThanOne = False
    if slope < 1:
        x0, x1, y0, y1 = y0, y1, x0, x1
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        slopeSmallerThanOne = True

    p0 = 2 * dx - dy
    x = x0
    y = y0

    # This method is invoked quite often, and each for loop
    # iterates an hundred times on average, so I chose to keep
    # it redundant with the two different appends, instead of
    # keeping a single for-loop with a control in it.
    # This loop proceeds step by step into generating the line.
    # Both X and Y require a dynamic step because of the different
    # slopes to handle.
    # If the slope is smaller than one, x and y will be appended
    # as (y, x), and not (x, y).
    if slopeSmallerThanOne:
        for _ in range(abs(y1 - y0)):
            if flag:
                xPrevious = x0
                pPrevious = p0
                p = p0
                flag = False
            else:
                xPrevious = x
                pPrevious = p

            if p >= 0:
                x = x + xStep

            p = pPrevious + 2 * dx - 2 * dy * (abs(x - xPrevious))
            y = y + yStep
            linePixel.append((y, x))
    else:
        for _ in range(abs(y1 - y0)):
            if flag:
                xPrevious = x0
                pPrevious = p0
                p = p0
                flag = False
            else:
                xPrevious = x
                pPrevious = p

            if p >= 0:
                x = x + xStep

            p = pPrevious + 2 * dx - 2 * dy * (abs(x - xPrevious))
            y = y + yStep
            linePixel.append((x, y))

    return linePixel


def computeDistance(p1, p2):
    """Compute the euclidean distance between p1 and p2.

    Args:
        p1 (tuple[int]): first point.
        p2 (tuple[int]): second point.

    Returns:
        float: Euclidean distance between p1 and p2.
    """
    return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5


def detectAndLabelMarkers(
    image: np.ndarray, currentFrame: int, objectToTrack: int
) -> None:
    """Detect the visible markers through their contours and then determine for
    each of them the line which crosses all the circles from the bottom of the marker
    to the concave corner. Eventually this line is used to traverse the marker, looking
    for the white circles in fixed positions.

    Args:
        image (np.ndarray): input image.
        currentFrame (int): index of the current image with respect to the total number
        of frames in the video.
        objectToTrack (int): index of the chosen video. Required to write the rows in the
        related csv file.
    """

    outputFileContent = ""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Discarded part of the image because the smallest
    # visible markers tend to be misdetected being them
    # adjacent to the plastic cup.
    gray[:, 0:1200:] = 0
    # 190 detects markers pretty well, but still requires an
    # area control for small fake-markers appearing on the plastic
    # cup.
    _, thresh = cv.threshold(gray, 190, 255, cv.THRESH_BINARY)

    # I use CHAIN_APPROX_SIMPLE because it removes all redundant points
    # and compresses the contour, thereby saving memory.
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Searching through every region selected to find the required polygon.
    polygons = [
        approx
        for cnt in contours
        if cv.contourArea(cnt) > 1200
        and len(approx := cv.approxPolyDP(cnt, 0.0155 * cv.arcLength(cnt, True), True))
        == 5
    ]
    cv.drawContours(image, polygons, -1, (0, 255, 0), 2)

    # Cycle through every marker, and find for each of them the point A.
    # Take the 3 shortest sides of each marker: the A point will be
    # the one where two of such sides meet.
    for poly in polygons:
        concaveCornerPoint = None
        lowerSideMiddlePoint = None
        totalVertices = len(poly)
        # Retreving the 3 shortest sides for the actual polygon.
        # I chose 80 as filter because sides larger then 80 are
        # the long sides, which are not useful for the detection
        # of A.
        matchingSides = [
            vertexIndex
            for vertexIndex, vertex in enumerate(poly)
            if computeDistance(vertex[0], poly[(vertexIndex + 1) % totalVertices][0])
            < 80.0
        ]
        # Iterate through the 3 sides and control them by couples to
        # find which of them match
        for loopIndex, side in enumerate(matchingSides):
            cornerFound = False

            # Legend:
            #   - Side: index of the first vertex of the currently examined side.
            #   - (side + 1) % totalVertices: index of the second vertex of the currently examined side.
            #   - matchingSides[(loopIndex + 1) % 3]: index of the first vertex of the next matching side.
            #   - (matchingSides[(loopIndex + 1) % 3] + 1) % totalVertices: index of the second vertex
            #                                                               of the next matching side.
            firstSideFirstPoint = poly[side][0]
            firstSideSecondPoint = poly[(side + 1) % totalVertices][0]
            secondSideFirstPoint = poly[matchingSides[(loopIndex + 1) % 3]][0]
            secondSideSecondPoint = poly[
                (matchingSides[(loopIndex + 1) % 3] + 1) % totalVertices
            ][0]

            # Check if the two currently examined sides have a common point: if so,
            # store it for the next operations, and record that the matching lines
            # have been found for this marker. If no match is found, keep iterating.
            if (firstSideFirstPoint == secondSideFirstPoint).all() or (
                firstSideFirstPoint == secondSideSecondPoint
            ).all():
                cornerFound = True
                concaveCornerPoint = firstSideFirstPoint
            elif (firstSideSecondPoint == secondSideFirstPoint).all() or (
                firstSideSecondPoint == secondSideSecondPoint
            ).all():
                cornerFound = True
                concaveCornerPoint = firstSideSecondPoint

            if cornerFound:
                cv.circle(
                    image,
                    (
                        concaveCornerPoint[0],
                        concaveCornerPoint[1],
                    ),
                    radius=1,
                    color=(0, 0, 255),
                    thickness=6,
                )

                # "A" was found, so I use it to generate a line between it and
                # the middle point on the lower side of the marker: I use this
                # line to look in the 5 areas where each circle should reside,
                # obtaining in such way the binary value of each marker.
                lowerSide = (
                    poly[matchingSides[(loopIndex + 2) % 3]][0],
                    poly[(matchingSides[(loopIndex + 2) % 3] + 1) % totalVertices][0],
                )
                middlePointX = (lowerSide[0][0] + lowerSide[1][0]) // 2
                middlePointY = (lowerSide[0][1] + lowerSide[1][1]) // 2
                lowerSideMiddlePoint = (middlePointX, middlePointY)

                # Use the found middle point and "A" to generate the line
                # between them.
                markerAxis = bresenhamLineGenerator(
                    concaveCornerPoint[0],
                    concaveCornerPoint[1],
                    lowerSideMiddlePoint[0],
                    lowerSideMiddlePoint[1],
                )
                markerAxisLen = len(markerAxis)
                actCycle = 0
                cycleJump = int(markerAxisLen / 10 * 1.95)
                binaryRepr = ""
                for intervalCentreIndex in range(cycleJump, markerAxisLen, cycleJump):
                    tempIndex = intervalCentreIndex

                    # This if-elif block performs a little correction on the
                    # place where to look for the white circles: theoretically,
                    # the line generated through bresenham should be splitted into
                    # 5 pieces, and the detection of each circle should be made by
                    # looking into the centers of such pieces, but the line is
                    # perspectively warped, so an adjustment is required when
                    # looking for each circle's center.
                    if actCycle in [1, 2, 3, 4]:
                        tempIndex = int(tempIndex * 0.85)
                    elif actCycle == 0:
                        tempIndex = int(tempIndex * 0.9)
                    actCycle += 1

                    # Store the current slot status: white or black.
                    # (180 seems a good threshold to discriminate them
                    # according to some prints).
                    binaryRepr += (
                        "0"
                        if gray[
                            markerAxis[tempIndex][1],
                            markerAxis[tempIndex][0],
                        ]
                        > 180
                        else "1"
                    )
                    # Draw the center of the actual circle.
                    cv.circle(
                        image,
                        (
                            markerAxis[tempIndex][0],
                            markerAxis[tempIndex][1],
                        ),
                        radius=1,
                        color=(255, 0, 0),
                        thickness=2,
                    )
                # Reverse the binary representation, which was captured backward,
                # and write it on the marker in decimal representation.
                # Also, using the computed label, access the related 3D coords.
                binaryRepr = int(binaryRepr[::-1], 2)
                binaryReprStr = str(binaryRepr)

                radAngle = radians(-15)
                qx = cos(radAngle * binaryRepr) * 70
                qy = sin(radAngle * binaryRepr) * 70
                outputFileContent = (
                    outputFileContent
                    + f"{currentFrame},{binaryReprStr},{markerAxis[0][0]},{markerAxis[0][1]},{qx},{qy},0\n"
                )
                image = cv.putText(
                    img=image,
                    text=binaryReprStr,
                    org=(markerAxis[0][0], markerAxis[0][1]),
                    fontScale=1.0,
                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                    color=(0, 0, 0),
                    thickness=9,
                )
                image = cv.putText(
                    img=image,
                    text=binaryReprStr,
                    org=(markerAxis[0][0], markerAxis[0][1]),
                    fontScale=1.0,
                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                    color=(255, 255, 255),
                    thickness=3,
                )
                break

    outputFile = open(f"obj{objectToTrack}_marker.csv", "a")
    outputFile.write(outputFileContent)
    outputFile.close()
