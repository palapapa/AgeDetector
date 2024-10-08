# Age Detector

This is a simple age detector made with Teachable Machine. It scans faces using a webcam and identifies whether a person is an adult or a minor and puts a bounding box around their faces that shows this information.

To understand the making of this project and its usage, please refer to [this PDF](AgeDetector.pdf) and [this presentation](https://youtu.be/JpLVShrX2dM).

## Usage

Please use Python 3.9.6 or things might not work.
```powershell
git clone https://github.com/palapapa/AgeDetector
cd AgeDetector
py -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
py src/main.py
```

### Options

```text
usage: main.py [-h] [-m path] [-l path] [-c id] [-s scaling-factor] [-d ms] [--language ISO-code]

optional arguments:
  -h, --help            show this help message and exit
  -m path, --model path
                        The path to the h5 model. (default: models/model.h5)
  -l path, --label path
                        The path to the labels file. (default: models/labels.txt)
  -c id, --camera id    The number of the webcam to use. (default: 0)
  -s scaling-factor, --scan-scaling scaling-factor
                        The scaling factor to use while scanning faces. A smaller number means faster detection but
                        lower accuracy and vice versa. (default: 0.25)
  -d ms, --scan-delay ms
                        The amount of time to delay between each scan. (default: 0)
  --language ISO-code   The language of the warning voice. (default: en)
```

## Controls

- Esc: Quit the application.
- B: Brighten the image.
- D: Darken the image.
