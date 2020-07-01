def transformation(series, code, transform=True):
    
    if transform:
        if code == 1:
            # none
            transformed_series = series
        elif code == 2:
            # first-difference
            transformed_series = first_difference = series[1:] - series[:-1]
        elif code == 3:
            # second-difference
            transformed_series = series[2:] - series[:-2]
        elif code == 4:
            # log
            transformed_series = np.log(series)
        elif code == 5:
            # first-difference log
            transformed_series = np.log(series[1:]) - np.log(series[:-1])
        elif code == 6:
            # second-difference log
            transformed_series = np.log(series[2:]) - np.log(series[:-2])

        return transformed_series
    else:
        return series