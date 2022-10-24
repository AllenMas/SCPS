# SCPS-Hash
# Self-calibrating Photometric Stereo with Hash encoding

### Results on User dataset
<p align="center">
    <img src='runs/User/bottle/rgb.png' height="150"><img src='runs/User/bottle/normals.png' height="150"><img src='runs/User/bottle/diffuse.png' height="150">
    <img src='runs/User/coaster/rgb.png' height="150"><img src='runs/User/coaster/normals.png' height="150"><img src='runs/User/coaster/diffuse.png' height="150">
    <img src='runs/User/gbottle/rgb.png' height="150"><img src='runs/User/gbottle/normals.png' height="150"><img src='runs/User/gbottle/diffuse.png' height="150">
    <img src='runs/User/mug/rgb.png' height="150"><img src='runs/User/mug/normals.png' height="150"><img src='runs/User/mug/diffuse.png' height="150">
    <img src='runs/User/stone/rgb.png' height="150"><img src='runs/User/stone/normals.png' height="150"><img src='runs/User/stone/diffuse.png' height="150">
    <img src='runs/User/wbottle/rgb.png' height="150"><img src='runs/User/wbottle/normals.png' height="150"><img src='runs/User/wbottle/diffuse.png' height="150">
</p>

## Use your own dataset
Prepare your Photometric Stereo images and a mask image to data/User/{yourdata}

## Train from Scratch
First, initialize light estimations.
```
python lights_initialize.py --config configs/User/{yourdata}.yml
```

Then, train from scatch.
```
python train.py --config configs/User/{yourdata}.yml
```
