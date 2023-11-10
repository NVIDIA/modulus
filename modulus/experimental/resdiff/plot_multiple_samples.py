import xarray
import matplotlib.pyplot as plt
import joblib
import click
            

@click.command()
@click.argument('netcdf_file')
@click.argument('output_dir')
@click.option('--n-samples', help='number of samples', default=5, type=int)
def main(netcdf_file, output_dir, n_samples):
    path = netcdf_file
    output = output_dir

    root = xarray.open_dataset(path)
    ds = xarray.open_dataset(path, group='prediction').merge(root).set_coords(['lat', 'lon'])
    truth = xarray.open_dataset(path, group='truth').merge(root).set_coords(['lat', 'lon'])
    ds

    # concatenate truth data and ensemble mean as an "ensemble" member for easy
    # plotting
    truth_expanded = truth.assign_coords(ensemble="truth").expand_dims("ensemble")
    ens_mean = ds.mean("ensemble").assign_coords(ensemble="ensemble_mean").expand_dims("ensemble")
    # add [0, 1, 2, ...] to ensemble dim
    ds['ensemble'] = [str(i) for i in range(ds.sizes['ensemble'])]
    merged = xarray.concat([truth_expanded, ens_mean, ds], dim='ensemble')

    # plot the variables in parallel
    def plot(v):
        print(v)
        # 2 is for the esemble and 
        merged[v][:n_samples + 2, :].plot(row='time', col='ensemble')
        plt.savefig(f"{output}/{v}.png")

    joblib.Parallel(n_jobs=8)(joblib.delayed(plot)(v) for v in merged)

main()