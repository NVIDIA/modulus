import numpy as np
import xarray
import pandas as pd

def weighted_mean(data, lat=None):
    """
    Calculate the Mean weighted by latitute if provided
    
    Parameters:
    - data: 2D xarray.DataArray 
    - lat: latitude points that corresponds to data
    Returns: float
        value of area weighted mean
    """
    if {'latitude', 'longitude'}.issubset(data.variables):
        data = data.drop_vars(['latitude', 'longitude'])

    if lat is None:
        weights = 1.0
    else:
        weights = np.cos(np.deg2rad(lat))
        weights = weights / weights.mean()
    return (data * weights).mean(dim=['y','x'])


def weighted_mse(pred, target, lat=None):
    """
    Calculate the Mean Square Error, weighted by latitute if provided
    
    Parameters:
    - pred, target: 2D xarray dataarrays predicted values for a given channel 
    - lat: latitude points that corresponds to pred and target
    Returns: float
        value of area weighted MSE
    """
    mse = weighted_mean(xarray.apply_ufunc(np.square, pred - target, dask="allowed"), lat)
    return mse

def weighted_rmse(pred, target, lat=None):
    """
    Calculate the Mean Square Error, weighted by latitute if provided
    
    Parameters:
    - pred, target: 2D xarray dataarrays predicted values for a given channel 
    - lat: latitude points that corresponds to pred and target
    Returns: float
        value of area weighted RMSE
    """
    mse = weighted_mean(xarray.apply_ufunc(np.square, pred - target, dask="allowed"), lat)
    return np.sqrt(mse)


def weighted_mae(pred, target, lat=None):
    """
    Calculate the Mean Absolute Error, weighted by latitude if provided.
    
    Parameters:
    - pred, target: 2D xarray dataarrays representing predicted and true values for a given channel.
    - lat: latitude points that correspond to pred and target.
    Returns: float
        Value of the area-weighted MAE.
    """
    mae = weighted_mean(xarray.apply_ufunc(np.abs, pred - target, dask="allowed"), lat)
    return mae


def fraction_skill_score(pred, target, mask_size=4, lat=None, cutoffs=[0.25, 0.5, 0.75], c=10, hard_discretization=True):
    """
    Calculate the Fraction Skill Score for each variable in the dataset with multiple cutoff values.
    In variable with levels, cutoff is computed per level.
    
    Parameters:
    - pred, target: xarray.Dataset with predicted and target values for multiple channels.
    - mask_size: the size (identical in x and y) of the mask area.
    - lat: latitude points that corresponds to both pred and target.
    - cutoffs: list of float (0<cutoff<1), thresholds for discretization as fractions of the maximum value in the data.
    - c: float, the steepness parameter for the sigmoid function in soft discretization.
    - hard_discretization: bool, use hard discretization if True, else use soft.
    
    Returns:
    - xarray.DataArray with the value of Fraction Skill Score for each variable and cutoff.
    """
    
    fss_scores = xarray.Dataset()
    pred = pred.drop_vars(['latitude', 'longitude'], errors='ignore')
    target = target.drop_vars(['latitude', 'longitude'], errors='ignore')
    
    for var in pred.data_vars:
        if 'levels' in pred[var].dims:
            global_max = pred[var].max(dim=('ic', 'time', 'y', 'x'), keep_attrs=True)
            global_min = pred[var].min(dim=('ic', 'time', 'y', 'x'), keep_attrs=True)
        else:
            global_max = pred[var].max().item()
            global_min = pred[var].min().item()
                
        fss_scores_var = []
        for cutoff_percentage in cutoffs:
            cutoff = (np.abs(global_max)-np.abs(global_min)) * cutoff_percentage + global_min
            
            if hard_discretization:
                target_binary = xarray.where(target[var] > cutoff, 1.0, 0.0)
                pred_binary = xarray.where(pred[var] > cutoff, 1.0, 0.0)
            else:
                target_binary = 1 / (1 + np.exp(-c * (target[var] - cutoff)))
                pred_binary = 1 / (1 + np.exp(-c * (pred[var] - cutoff)))

            target_density = target_binary.coarsen(x=mask_size, y=mask_size, boundary='trim').mean()
            pred_density = pred_binary.coarsen(x=mask_size, y=mask_size, boundary='trim').mean()

            MSE_n = ((pred_density - target_density) ** 2).mean(dim=['x', 'y'])
            O_n_squared_sum = (target_density ** 2).sum(dim=('x', 'y'))
            M_n_squared_sum = (pred_density ** 2).sum(dim=('x', 'y'))

            n_density_pixels = target_density.sizes['x'] * target_density.sizes['y']

            MSE_n_ref = (O_n_squared_sum + M_n_squared_sum) / n_density_pixels
            FSS = 1 - (MSE_n / MSE_n_ref)                

            fss_scores_var.append(FSS)
        
        fss_scores_var = xarray.concat(fss_scores_var, dim=pd.Index(cutoffs, name='cutoff'))
        fss_scores[var] = fss_scores_var
    
    fss_scores = fss_scores.assign_coords({'cutoff': pd.Index(cutoffs, name='cutoff')})
    return fss_scores


def brier_score(pred, target, threshold=5):
    """
    Calculate the Brier score for forecasts with an optional ensemble dimension.
    Note: The code takes teh dimension in pred that is not present in target as ensmeble dimension.

    Parameters:
    pred : xarray.DataArray
        Probabilistic forecast of a variable that may include an ensemble dimension.
    target : xarray.DataArray
        Observation of a variable with dimensions (lat, lon).
    threshold : float
        Value that turns the continuous observation into a binary event.

    Returns:
    xarray.DataArray
        Map of Brier score values.
    """
    
    target_binary = (target > threshold).astype(int)
    ensemble_dim = list(set(pred.dims) - set(target.dims))
    if ensemble_dim:
        assert len(ensemble_dim) == 1, "More than one potential ensemble dimension found."
        ensemble_dim = ensemble_dim[0]
        pred_binary = (pred > threshold).mean(dim=ensemble_dim)
    else:
        pred_binary = (pred > threshold).astype(int)
    
    brier_score = (pred_binary - target_binary) ** 2
    return brier_score


def probability_matched_mean(ensemble_forecast):

    """
    To obtain the probability matched forecast, PM, the model rain rate PDF is first calculated by pooling the forecast rain rates for all n models for the entire domain, 
    ranking them in order of greatest to smallest, and keeping every nth value. The rain rates in the ensemble mean forecast are similarly ranked from greatest to smallest, 
    with the location of each value stored along with its rank. The grid point with the highest rain rate in the ensemble mean rain field is reassigned to the highest value 
    in the model rain rate distribution, and so on. The rain area is constrained not to exceed the mean rain area predicted by the models.

    Elizabeth Ebert: DOI: https://doi.org/10.1175/1520-0493(2001)129<2461:AOAPMS>2.0.CO;2

    input: ensemble forecast (ens, width, height)
    output: probability matched forecast (width, height)

    """
    # ensemble_forecast shape is (ens, width, height)
    ens, width, height = ensemble_forecast.shape
    
    # Step 1: Pool all the forecast rain rates and rank them
    pooled_rain_rates = ensemble_forecast.flatten()  # Shape: (ens * width * height,)
    sorted_pooled_rain_rates = np.sort(pooled_rain_rates)  # Sort in ascending order
    
    # Select every nth value from the pooled sorted rain rates
    nth_values = sorted_pooled_rain_rates[::ens]  
    
    # Step 2: Calculate the ensemble mean forecast
    ensemble_mean_forecast = np.mean(ensemble_forecast, axis=0)  
    
    # Flatten and sort the ensemble mean forecast rain rates
    sorted_ensemble_mean = np.sort(ensemble_mean_forecast.flatten())  # Sort in ascending order
    sorted_ensemble_mean_indices = np.argsort(ensemble_mean_forecast.flatten())
    
    # Step 3: Reassign the highest rain rate in the ensemble mean to the highest in model distribution
    pmm = np.zeros_like(ensemble_mean_forecast)  
    
    # Assign values based on rank
    for rank in range(len(nth_values)):
        index = sorted_ensemble_mean_indices[rank]  # Get the index from sorted ensemble mean
        value = nth_values[rank]  # Get the corresponding nth value from the pooled distribution
        pmm[np.unravel_index(index, (width, height))] = value  # Reassign the value
    
    return pmm

def generate_gaussian(size, std_dev, mean_shift):
    x = np.linspace(-size / 2, size / 2, size)
    y = x
    x, y = np.meshgrid(x, y)
    gaussian = np.exp(-((x - mean_shift[0])**2 + (y - mean_shift[1])**2) / (2 * std_dev**2))
    gaussian /= gaussian.sum()
    return gaussian

    
def test_metrics():
    """
    test metrics by cd to `fcn-dev/utils`
    > python 
    > import metrics
    > metrics.test_metrics()
    Should print 
    The mean PMM value is: 0.0025907761354884242
    The Brier Score is: 0.005075654974133139
    Fraction Skill Score is: 0.9999992451007743
    """
    size = 50
    std_dev = 10
    true_gaussian = generate_gaussian(size, std_dev, mean_shift=(0, 0))
    pred_gaussian = generate_gaussian(size, std_dev, mean_shift=(10, 0))

    y_true = xarray.DataArray(true_gaussian, dims=['lat', 'lon'])
    y_pred = xarray.DataArray(pred_gaussian, dims=['lat', 'lon'])
    latitude_values = xarray.DataArray(np.linspace(-5, 5, size), dims=['lat'])
    lat = xarray.DataArray(latitude_values, dims=['lat']) 

    pmm_value = probability_match_mean(y_true, y_pred, np.mean(y_pred))
    mean_pmm = weighted_mean(pmm_value, lat)
    print(f"The mean PMM value is: {mean_pmm}")
    
    brier_score_value = brier_score(y_true, y_pred, np.mean(y_pred))
    mean_bs = weighted_mean(brier_score_value, lat)
    print(f"The Brier Score is: {mean_bs}")
    
    fss_metric = fraction_skill_score(y_true, y_pred, mask_size=5, lat=lat)
    print(f"Fraction Skill Score is: {fss_metric}")