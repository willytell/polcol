:: batch file for windows

echo "begining..."

echo "====>"

python3 main.py -c config/conf_dist1-resnet50-with_bbox-crop-da-francho.py -a train &>logs/dist1-resnet50-with_bbox-crop-da-francho-train


echo "finished!"
