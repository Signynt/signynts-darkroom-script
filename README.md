> For a version that is entirely contained in Affinty Photo, see the old workflow [Signynt's Darkroom Macro](https://github.com/Signynt/signynts-darkroom-macro)
> For a version that uses imagemagick and runs in the Terminal, see [Signynt's Darkroom Shortcut](https://github.com/Signynt/signynts-darkroom-shortcut)

# Signynt's Darkroom Script
*Signynt's Darkroom Script* is a Python script that inverts film negative images, and detects dust and scratches.
It consists of a Python script and an Affinity Photo macro / Photoshop action if you would like to use the dust removal option.

Using this script gives you quick access to high quality, neutral and ready to edit RAW images with just one click. It also optionally provides access to quick and high quality dust or scratch removal that is better than the scanning softwares automatic options.
The output is a 16bit TIFF file, that maintains all image information from the input file.

It can be used with DSLR scans, and scans made with Silverfast. For use with VueScan or B&W images, please refer to [Signynt's Darkroom Shortcut](https://github.com/Signynt/signynts-darkroom-shortcut).

![Github](https://user-images.githubusercontent.com/67801159/146692420-04df4cdc-dab6-494f-b414-cc3563ee55f1.png)

## Installation
> If you are having trouble, open an [Issue](https://github.com/Signynt/signynts-darkroom-script/issues/new/choose), or DM me on [Reddit](https://www.reddit.com/user/Signynt).

Download or clone this repository.

To install the Affinity Photo macro, simply double click on `Signynt's Darkroom Shortcut v1.0.afmacros`

### Dependencies:
Make sure you have these dependencies installed. If you are missing one the script will throw an error, and tell you which one is mising.

```
numpy
opencv-python
scikit-image
scipy
```
Installation example in the terminal: 
```zsh
pip install numpy
```

## Usage

> Please make sure all input files are 16bit TIFF files!

1. Open the folder containing the repository. You should see a file called `signynts-darkroom-script.py`
2. Put all the images you would like to process into a folder called `input`
3. Open the repository in the terminal (Example: `cd Documents/Github/signynts-darkroom-script`)
4. Run the script by running `python signynts-darkroom-script.py`. You will see the progress of the script begin.
5. Your output files will be put in a folder called `output`. Remove the dust by opening the image in Affinity Photo and running the macro called `Remove Dust`
6. Done! You can apply any further edits you want to your image (`Filters > Autolevel` always gives me great results), and save it!

> If you have any issues please make sure the script works with the [Example.tif](https://github.com/Signynt/signynts-darkroom-shortcut/releases/download/v1.1/Example.tif) file before opening an issue.
