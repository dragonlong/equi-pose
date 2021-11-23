# Resize animated gif by scaling down 50%:
gifsicle --scale 0.5 -i 0.8581_modelnet40aligned_chair.gif > 0.8581_modelnet40aligned_chair-smaller.gif

# Resize animated gif to scaling to a given width with unspecified height:
gifsicle --resize-fit-width 300 -i 0.8581_modelnet40aligned_chair.gif > 0.8581_modelnet40aligned_chair-300px.gif

# Resize animated gif by scaling to a given height with unspecified width:
gifsicle --resize-fit-height 100 -i 0.8581_modelnet40aligned_chair.gif > 0.8581_modelnet40aligned_chair-100px.gif

# Resize animated gif clipping to size:
gifsicle --resize 300x200  -i 0.8581_modelnet40aligned_chair.gif > 0.8581_modelnet40aligned_chair-clipped.gif

# Gifsicle has three types of GIF optimization to choose from:
#   -O1 - stores only the changed portion of each image. this is the default.
#   -O2 - also uses transparency to shrink the file further.
#   -O3 - try several optimization methods (usually slower, sometimes better results).
# - http://davidwalsh.name/optimize-gifs
MYDIR="."
DIRS=$(ls $MYDIR/*.gif)
arr=($DIRS)
echo ${arr[0]}
for DIR in $DIRS
do
   echo "processing ${DIR} data"
   gifsicle -O3 --delay=8 ${DIR} -o ../imgs/${DIR} &
done
for i in
gifsicle -O3 --delay=20 0.8581_modelnet40aligned_chair.gif -o 0.8581_modelnet40aligned_chair-optimized.gif
gifsicle -O3 --delay=20 0726.gif -o 0726-optimized.gif


# ---

# To optimise a GIF:
gifsicle --batch --optimize=3 <amin.gif>

# To make a GIF 0.8581_modelnet40aligned_chair with gifsicle:
gifsicle --delay=<10> --loop *.gif > <anim.gif>

# To extract frames from an 0.8581_modelnet40aligned_chair:
gifsicle <anim.gif> '#0' > <firstframe.gif>

# To you can also edit 0.8581_modelnet40aligned_chairs by replacing, deleting, or inserting frames:
gifsicle -b <anim.gif> --replace '#0' <new.gif>
