import matplotlib.pyplot as plt

class Window:
    """
    Window to draw a gridworld instance using Matplotlib
    """

    def __init__(self):
        self.fig = None

        self.imshow_obj = None

        # Create the figure and axes
        self.fig, self.ax = plt.subplots()

        # Show the env name in the window title
        # self.fig.suptitle(title)
        
        # Turn off x/y axis numbering/ticks
        self.ax.xaxis.set_ticks_position('none')
        self.ax.yaxis.set_ticks_position('none')
        _ = self.ax.set_xticklabels([])
        _ = self.ax.set_yticklabels([])

        # Flag indicating the window was closed
        self.closed = False

        def close_handler(evt):
            self.closed = True

        self.fig.canvas.mpl_connect('close_event', close_handler)

    def show_img(self, img, caption, fps):
        """
        Show an image or update the image being shown
        """

        # If no image has been shown yet,
        # show the first image of the environment
        if self.imshow_obj is None:
            self.imshow_obj = self.ax.imshow(img, interpolation='bilinear')

        # Update the image data
        self.imshow_obj.set_data(img)

        # Update the image caption
        self.set_caption(caption)

        # Request the window be redrawn
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        # Let matplotlib process UI events
        plt.pause(1/fps)

    def set_caption(self, text):
        """
        Set/update the caption text below the image
        """

        plt.xlabel(text)

    def reg_key_handler(self, key_handler):
        """
        Register a keyboard event handler
        """

        # Keyboard handler
        self.fig.canvas.mpl_connect('key_press_event', key_handler)

    def show(self, block=True):
        """
        Show the window, and start an event loop
        """

        # If not blocking, trigger interactive mode
        if not block:
            plt.ion()

        # Show the plot
        # In non-interative mode, this enters the matplotlib event loop
        # In interactive mode, this call does not block
        plt.show()

    def close(self):
        """
        Close the window
        """

        plt.close()
        self.closed = True
