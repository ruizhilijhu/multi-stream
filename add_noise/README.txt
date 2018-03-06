usage: add-noise.py [-h] [--low-freq LOW_FREQ] [--high-freq HIGH_FREQ]
                    [--srate SRATE] [--noise-level NOISE_LEVEL]
                    outdir scp

Add noise at a specific DB level.

positional arguments:
  outdir                output directory
  scp                   "scp" list

optional arguments:
  -h, --help            show this help message and exit
  --low-freq LOW_FREQ
  --high-freq HIGH_FREQ
  --srate SRATE         expected sampling rate
  --noise-level NOISE_LEVEL
                        noise level in db
