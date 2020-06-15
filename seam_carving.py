import numpy as np 
import cv2
import copy
from scipy import ndimage
import argparse
import os
import sys

class Seam_Carving:

  def __init__(self):
    self.image = None

  def main(self, image_path, resize_by):
      """
      Purpose: Runs the seam carving algorithm.
      Input:
        image -- image path
        horizontal -- number of pixels to be removed horizontally
      """
      image_name = image_path.split('/')[1] 
      image_name = image_name.split('.')[0]
      img = cv2.imread(image_path)
      transport_map = self.calculate_transport_map(self.create_energy_map(img))
      seams = self.calculate_seams(transport_map, img, resize_by, image_name)
      totalSeams = len(seams) / img.shape[0]
      
      # totalSeams should always be equal to the horizontal value provided. If not, then we might have a corner case.
      if totalSeams != resize_by:
        print "Something went wrong."
      else:
        splitSeam = np.split(np.asarray(seams), totalSeams)  
        for x in range(totalSeams):
          img = self.remove_seam(img, splitSeam[x], image_name)


  def create_energy_map(self, image):
    """
    Purpose: compute an approximation of the gradient of the 
    image intensity function using sobel filter.
    Input: 
      image -- image that we're resizing
    Output:
      energy_map -- 2D map of the gradient at each pixel
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray.astype("float"), cv2.CV_64F, 0, 1, ksize=5)
    sobely = cv2.Sobel(gray.astype("float"), cv2.CV_64F, 1, 0, ksize=5)
    energy_map = np.hypot(sobelx, sobely)
    energy_map *= 255.0 / np.max(energy_map)
    return energy_map

  def calculate_transport_map(self, energy_map):
    """
    Purpose: compute transport map for resizing images vertically. Each x, y represents
    the minimal cost needed to obtain an image of size n * m - c. Uses the following
    dynamic programming formula:
      M(i, j) = e(i,j) + min(M(i-1, j-1), M(i-1, j), M(i-1,j+1))
    Input:
      energy_map -- 2D map of the gradient at each pixel
    Output:
      transport_map -- cost matrix used for calculating seams
    """
    rows, cols = energy_map.shape
    transport_map = copy.copy(energy_map)
    count = 0
    for x in range(1, rows - 1):
      for y in range(1, cols - 1):
        currentEnergy = energy_map[x, y] #energy of current pixel
        newEnergy = currentEnergy + min(energy_map[x-1, y-1], energy_map[x, y-1], energy_map[x+1, y-1])
        transport_map[x, y] = newEnergy
    return transport_map

  def calculate_seams(self, transport_map, image, num_of_seams, image_name):
    """
    Purpose: calculates seams to be removed based on the transport_map.
    formula:
      
    Input:
      transport map -- cost matrix
      image -- image path provided by user
      num_of_seams -- number of seams to be removed (i.e. resizing by)
    Output:
      seams_to_be_removed -- list of pixel coordinates in the seam to be removed
    """
    # Creating a copy of the image because we don't want the seams to be highlighted in the final resized image
    img = copy.copy(image)
    reversed_map = copy.copy(transport_map)
    rows = reversed_map.shape[0]
    count = 0
    seam_to_be_removed = []
    while (count < num_of_seams):
      # to start, we'll pick the smallest energy value in the first row as our starting point
      # we'll do this for each seam until we've reached our goal
      # here, 0 is hardcoded because we'll start at the top for each seam (i.e 0th row)
      starting_point = min(reversed_map[0])
      starting_index = self.return_index(reversed_map[0], starting_point)
      seam_to_be_removed.append([starting_index, 0])
      temp_map = copy.copy(reversed_map)
      # need to recalculate in the loop since vertical dimension changes each iteration
      cols = reversed_map.shape[1] 
      # remove the first pixel
      new_img = np.delete(temp_map[0], starting_index)
      self.highlight_seam(img, starting_index, 0) # image, x, y
      # then, we use the transport map to find the next energy element. In the general case, this will be
      # directly below the first element, diagonally to the left or diagonally to the right. Considerations need to be
      # made when we're near the edges of the image (left and right edge)
      for x in range(1, rows):
        if (starting_index == 0):
          npi, si = self.return_min(reversed_map[x, starting_index+1], reversed_map[x, starting_index], 0, reversed_map[x], starting_index)
        elif starting_index < cols - 2:
          npi, si = self.return_min(reversed_map[x, starting_index+1], reversed_map[x, starting_index], reversed_map[x, starting_index-1], reversed_map[x], starting_index)
        else:
          npi, si = self.return_min(reversed_map[x, starting_index-1], reversed_map[x, starting_index], 0, reversed_map[x], starting_index)
        starting_index = si
        self.highlight_seam(img, si, x) 
        seam_to_be_removed.append([si, x])
        # Removing a the element from the current row. At the end of the for loop, the full seam will be removed from the transport map
        temp = np.delete(temp_map[x], si)
        new_img = np.vstack([new_img, temp])
      reversed_map = None
      reversed_map = new_img
      count = count + 1  
    cv2.imwrite('{0}_seams_removed.png'.format(image_name), img)
    return seam_to_be_removed

  def return_min(self, a, b, c, transport_map, currentIndex):
    """
    Purpose: Returns which pixel should be removed and its index.
    Input:
      a -- pixel a
      b -- pixel b
      c -- pixel c
      transport map -- cost matrix
      current_index -- current index
    Output: 
      next_pixel -- 
      index -- index of next pixel to be removed
    """
    if (c == 0):
      next_pixel = min(a, b)
    else:
      next_pixel = min(a, b, c)
    starting_index, = np.where(transport_map == next_pixel)
    for i in range(len(starting_index)):
      if (abs(starting_index[i] - currentIndex) <= 3):
        index = starting_index[i]
    return next_pixel, index

  def remove_seam(self, img, seam_to_be_removed, image_name):
    """
    Purpose: remove seams from the original image
    Input:
      img --
      seam_to_be_removed --
    Output:
      new_result -- original image with the seam removed
    """
    r, g, b = cv2.split(img)
    
    result_r = np.empty((r.shape[0], r.shape[1] - 1))
    result_g = np.empty((g.shape[0], g.shape[1] - 1))
    result_b = np.empty((b.shape[0], b.shape[1] - 1))
    row, col = r.shape
    for x in range(row):
      rrow, ind = seam_to_be_removed[x]
      result_r[ind] = np.delete(r[ind,], rrow)
      result_g[ind] = np.delete(g[ind,], rrow)
      result_b[ind] = np.delete(b[ind,], rrow)    
    
    new_result = cv2.merge((result_b, result_g, result_r))
    cv2.imwrite("{0}_resized.png".format(image_name), new_result)

    return new_result

  def return_index(self, starting_point, map):
    """
    Purpose: Returns index of the first instance where the transport_map matches 
    the value of the starting_point.
    Input:
      starting_point -- 
      transport_map --
    Output:
      starting_index -- index 
    """
    starting_index, = np.where(map == starting_point)
    return starting_index[0]

  def highlight_seam(self, image, x, y):
    """
    Purpose: Highlights the seam in red in the image
    Input:
      image -- image in which we'll highlight the seam
      x -- x coordinate
      y -- y coordinate
    Output:
      None
    """
    x = x.astype(int)
    image[y][x] = (255, 0, 0)

  def validate_user_input(self, args):
    """
    Purpose: Perform the following validation on user input:
      i) image path exists
      ii) horizontal value provided is less than the image's width
    Input:
      args -- command line arguments
    Output:
      None
    """
    img = None
    if not os.path.exists(args.image_path):
        print "Image path does not exist. Please try again."
        sys.exit(1)
    else: 
        img = cv2.imread(args.image_path)
        width, _, _ = img.shape
        if args.resize_by > width:
          print "horizontal value ({0}) is greater than the image's width ({1}). Please try again.".format(args.horizontal, width)
          sys.exit(1)

if __name__ == "__main__":
  # parser user input
  parser = argparse.ArgumentParser()
  parser.add_argument("image_path", help = "image you'd like to resize")
  parser.add_argument("resize_by", type = int, help = "pixel size you'd like to resize the image by (horizonally)")
  args = parser.parse_args()
  
  sc = Seam_Carving()
  sc.validate_user_input(args)  
  # run seam carving algorithm
  sc.main(args.image_path, args.resize_by)
  print "Program exited successfully."