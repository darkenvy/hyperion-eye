# Hyperion-Eye
I use a camera pointed at my TV for my hyperion setup. Ideally a camera would be placed directly in front of the screen and smple cropping would isolate the image. But who wants a camera in front of their TV in the middle of the room? Thats silly.

So my solution was to put a camera at the bottom of the screen angled up. A fisheye lense is used to 'see' most of the screen (for me, I cannot see the bottom left/right corners but this is okay. We use some trickery to fill in 'guessed' pixel data). The only problem now is that the tv is distorted from the position, angle and fisheye lense. So what do we do? Defisheye and perspective transform the 4 corners!

### Defisheye
Each camera is different. Do a quick search to learn how to calibate your fisheye. It involves printing out a square grid pattern and taking photos of that pattern. I included the script that I found for good measure called 'setup_fisheye_calibrate.py'. 

After we run this program, you will recieve 3 variables. Save these. These are points for opencv to be able to remove the fisheye effect.

### Perspective transform
Depending on the angle and position of the camera, the coordinates to transform will be different. You will have to run setup_find_perspective_points.py to preview the camera. Use your mouse cursor to find the 4 corners of your TV. input this into main.py

### main.py
Inside main.py be sure to input your tv's aspect ratio and your pi cameras maximum resolution. Even though the first thing we do is resize the image (to reduce computation), we want python to capture the largest image as this also coorelates with the largest FoV. If you have a v1 camera, you are good to go.

### setup_preview_final.py
This file will show a dialogue of what the image will look like being sent to hyperion. Do note that a large image is not needed and that the high framerate may make make the preview window unresponsive. This preview image should look like an undistorted feed of your video. Fuzzyness is okay and is normal from all the transformations done

### Notes
This repository is messy. I abandonded the idea due to ambient lighting (rooom lighting) throwing off my colors. Im sure those with a home theator setup may have better results. But for me I had to let this idea go.

I copied and pasted main.py and modified it to make all the setup files. So a lot of duplicate code could be cleaned up.

I documented this for anyone who wants to pick this up. :)