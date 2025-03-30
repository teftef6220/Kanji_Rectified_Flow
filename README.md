# Generating Kanji with Rectified Flow

## about
This is an assignment for the [sakana AI](https://sakana.ai/) entrance exam.  
The implementation relies on information publicly available on the Internet [here](https://x.com/sakanaailabs/status/1898998260200857966?s=46&t=k3yi-oGrLOGVfYmBkleZZQ).   
The repository owner has not taken the [sakana AI](https://sakana.ai/) entrance exam, so the implementation is limited to the publicly available information.


## requirements

install pytorch from [here](https://pytorch.org/)

We need [ImageMagick](https://imagemagick.org/index.php) to convert SVG data to PNG data.

...


## usage

### Dataset
 Download the dataset from the following link.
[kanjivg-radical](https://github.com/yagays/kanjivg-radical)


### Make dataset
```bash
python dataset_utils/make_kanji_dataset.py 
```
if you excute this script, turn `--no_skip_existing` to True.
Or if you want to overwrite png images, use `--no_skip_existing` option to True.


## reference

[Examination](https://x.com/sakanaailabs/status/1898998260200857966?s=46&t=k3yi-oGrLOGVfYmBkleZZQ).   


## license

...