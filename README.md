# Frame Interpolation
Python program for interpolating an artificial frame in-between two other frames. <br>
The implemention could certainly be improved, and will create color blips around the edges occasionally.

#### How to run
Run `python frame_interpolation.py frame0.png frame1.png flow0.flo frame05.png` <br>
Where `frame0.png` and `frame1.png` are the two starting images and `frame05.png` is the name of the new in-between image. <br>
You can basically ignore the `flow0.flo` file as it was a way of my instructor to test how accurate our interpolation was.

### Algorithms
The algorithms used are closely based on the ones described in this paper: <br>
Baker, Simon, Daniel Scharstein, J. P. Lewis, Stefan Roth, Michael J. Black, and Richard Szeliski. <br>
"A database and evaluation methodology for optical flow." <br>
International Journal of Computer Vision 92, no. 1 (2011): 1-31.
