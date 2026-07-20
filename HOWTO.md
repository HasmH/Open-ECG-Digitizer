1. Pull code from - https://github.com/HasmH/Open-ECG-Digitizer.git

2. Go to branch - `feat/rejig`:

```sh
git checkout feat/rejig
```

3. if you need a way of converting the ECG pdf's to images you can use this script - it needs `pdf2image` package and `Poppler` installed

```sh
python3 -m pdf2png.py
```

4. Prepare config for 12x1 ECGs, see: `inference_wrapper_westmead_12x1.yml` - here are a few values you could potentially play around with:

- `resample_size` - i've made changes to the original code to allow us to upsample images, personally i cannot go more than `resample_size: 3500` due to my own hardware limitations but if you have a GPU or somethign more powerful you could increase this to 4000 or more for potentially better results. If it doesn't work you may get something along the lines of an out of memory error, if so just change back to 3500.
- `device: cuda` - mine is set to `cpu` as i dont have nvidia/gpu, feel free to change to suit your hardware
- `vertical_padding_fraction` - this is a new feature, as the name suggests, adds padding, if you see in the debug images those red dots originally were well within the ECG grid space, adding padding to go above, can adjust to your liking 
- `DATA` section - this config is specifically for 12x1 ECGS so `images_path`
- resegmentation arguments - this is what is behind the V6 / lead I fix, all this does is re-run part of the digitisation process on the top and bottom parts of the ECG. 

```yaml
    edge_resegment_scale: 2.0
    edge_resegment_fraction: 0.15
    edge_resegment_max_tile_px: 6000000
```


5. run inference:

```sh
python3 -m src.digitize --config inference_wrapper_westmead_12x1.yml
```
