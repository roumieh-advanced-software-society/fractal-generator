from __future__ import print_function  # print is now print of Python3 (expects Python2)
import matplotlib.pyplot as plt
import multiprocess as mp
import time
import numpy as np
import tqdm   # For the progress bar, no idea what is stands for
import cmath  # Complex math library
from math import sqrt


def canvas(n, m, filling=(0, 0, 0)):
    """
    USE IS NOT RECOMMENDED BECAUSE OF HELLISH MEMORY BEHAVIOR
    JUST DON'T DO IT. BE MERCIFUL ON THE RAM.

    Construct an n * m canvas filled with n * m pixels
    (m columns, n rows)
    whose color is specified in RGB by the 'filling' list.
    Default is black (0,0,0)"""
    return [[filling for _ in range(0, m)] for _ in range(0, n)]


def canvas2point(row, column, canvas_size, center, zoom):
    # type: (int, int, (int, int), (float, float), float) -> complex
    """Return the complex number representation of a pixel
    at position row, column
    (translate from canvas to complex plane)
    Format of canvas_size: (# of rows, # columns)
    Format of center: (x coordinate, y coordinate)"""
    return complex(
        ((((column - canvas_size[1] / 2) / canvas_size[1]) * 4) / zoom) - center[0],
        ((((- row + canvas_size[0] / 2) / canvas_size[0]) * 4) / zoom) - center[1])
    # Factors and complications are just for scaling; write it
    # on paper, and it'll look good.


def f(z, c):
    # type: (complex, complex) -> complex
    """The function f of the mandelbrot set on complex numbers"""
    return z * z + c


def time2escape(c, limit, used_function):
    # type: (complex, int, function) -> int
    """Time (steps) in takes for the function to escape (go beyond
    |z| = 2).
    For anything that takes more than a limit number of steps, it
    will be considered 'locked' inside (the function will return
    the limit)
    (Black for most representations)"""
    i = 0
    z = 0
    while sqrt(abs(z)) < 2:
        z = used_function(z, c)
        i = i + 1
        if i > limit:
            return limit
    return i


def empty_canvas(canvas_size):
    # type: ((int, int)) -> list
    """Produce a canvas (2-dimensional array) with the specified size
    The canvas is full of zeros everywhere. It's a numpy array with the
    values as int8 (C-like array) which means that a limit of more than
    127 is not allowed!
    canvas_size = (# of rows, # of columns)"""
    return [np.array([0 for _ in range(0, canvas_size[1])], dtype='int8') for _ in range(0, canvas_size[0])]


def mandel(canvas_size=(100, 100), limit=30, center=(0.5, 0), zoom=1, used_function=f, dpi=300):
    # type: ((int, int), int, (float, float), float, function, int) -> None
    """This is the main function that will draw the image
    Format of canvas_size: (# of rows, # of columns)
    Format of center: (x coordinate, y coordinate)"""
    image = empty_canvas(canvas_size)  # Create a canvas to draw on
    # TODO: Modify the 'image' variable directly instead of creating a return value and then putting it in image to
    # save memory. (Should cut the usage by half, theoretically)

    def row_transform(row_id):  # To be used for mapping in the pool below (for multiprocessing)
        # type: (int) -> list
        """Transform a given row (expected to be a row of an image that is an empty canvas.
        Notice the use of row_id instead of row. This allows to create an empty canvas and
        still map over it correctly"""
        row = image[row_id]
        for i in range(0, canvas_size[1]):
            step = time2escape(canvas2point(float(row_id), float(i), canvas_size, center, zoom), limit, used_function)
            row[i] = limit - step
        return row

    pool = mp.Pool(processes=mp.cpu_count())  # Set up as many processes as CPUs
    print("Starting calculations...")
    start = time.time()

    # Transform the image while showing a progress bar

    image = list(tqdm.tqdm(pool.imap(row_transform, range(0, canvas_size[0]), chunksize=100), total=canvas_size[0]))

    # NOTE: In case of not using full CPU power, increase chunksize until enough power is used!
    #       In case of errors with 'cannot join, blah blah blah', you must reduce chunksize

    print("Calculations took %d seconds" % (time.time() - start))

    print("Saving to 'fractal.png'...")
    plt.imshow(image)
    start = time.time()
    plt.savefig("fractal.png", dpi=dpi, frameon=False, bbox_inches='tight', pad_inches=0)
    # Extra arguments are to reduce wasted space outside of the image
    print("Saving took %d seconds" % (time.time() - start))


# Application
mandel(canvas_size=[5000, 5000], limit=30, zoom=0.5, center=[0, 0],
       used_function=lambda z, c: cmath.exp(z*c+c), dpi=2000)


# NOTE: if you increase canvas_size too much, you have to increase dpi to match it, otherwise, you'll be computing
#       things that will not appear in the image, which is a waste of time. Check actual image size to see if it has
#       all the pixels generated.
