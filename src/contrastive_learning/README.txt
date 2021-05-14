>>> y_train.shape
(44879, 3)

>>> x_train.shape
(44879, 8, 16)

# per dataset, number of values and shapes

# train a small DNN to predict all 3 values
    - values the dataset saves are: 
    - crosstrack error, heading error, and then downtrack position
    - these are not normalized

# current tiny taxinet:
    Returns an estimate of the crosstrack error (meters)
    and heading error (degrees) by passing the current
    image through TinyTaxiNet
