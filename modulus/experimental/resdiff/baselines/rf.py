
#%%
import generate
import torch
import joblib
import typer
import einops
import netCDF4


class RF:
    def __init__(self, path):
        self._rf = joblib.load(path)

    def __call__(self, x):
        # b c w h
        b, c, w, h = x.shape
        x_rs = einops.rearrange(x, "b c w h -> (w h b) c")
        out = self._rf.predict(x_rs.numpy())
        out = torch.from_numpy(out)
        return einops.rearrange(out, "(w h b) c -> b c w h", w=w, b=b, h=h)
        

def main(rf_pkl: str, data_type: str, data_config: str, output: str):
    dataset, sampler = generate.get_dataset_and_sampler(data_type, data_config)

    with netCDF4.Dataset(output, mode='w') as f:
        generate.generate_and_save(dataset, sampler, f, generate_fn=RF(rf_pkl), device="cpu", batch_size=1)



if __name__ == "__main__":
    typer.run(main)