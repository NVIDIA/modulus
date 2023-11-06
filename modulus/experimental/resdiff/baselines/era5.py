#%%
import generate
import netCDF4
import torch
import typer



class Identity:

    def __init__(self, dataset):
        self.ic = [generate._get_name(c) for c in dataset.input_channels()]
        self.oc = [generate._get_name(c) for c in dataset.output_channels()]
    
    def __call__(self, x):
        tensors = []
        for c in self.oc:
            try:
                i = self.ic.index(c)
                xx = x[:, i]
            except ValueError:
                xx = torch.full_like(x[:, 0], fill_value=torch.nan)
            tensors.append(xx)
        return torch.stack(tensors, dim=1)


def main(data_type: str, data_config: str, output: str):
    dataset, sampler = generate.get_dataset_and_sampler(data_type, data_config)

    with netCDF4.Dataset(output, mode='w') as f:
        generate.generate_and_save(dataset, sampler, f, generate_fn=Identity(dataset), device="cpu", batch_size=1)



if __name__ == "__main__":
    typer.run(main)