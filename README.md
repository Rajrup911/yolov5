### Command to run validation
```zsh
python src/val.py --weights yolov5x.pt --data coco128.yaml
```

### Command to run object detection
```zsh
python src/detect.py --weights yolov5x.pt --source <img/video...>
```

### Command to run custom obj detection with ROI counter
```zsh
python src/count.py --weights yolov5x.pt --source <img/video...>
```
