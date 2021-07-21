# Multi GPU using Super Resolution

## Prepare
- train_x2.h5
```bash
python prepare.py --images_dir "./train_file" \
                  --output_path "./output_path/train_x2.h5" \
                  --scale 2 \
```
- eval_x2.h5
```bash
python prepare.py --images_dir "./eval_file" \
                  --output_path "./output_path/eval_x2.h5" \
                  --scale 2 \
                  --eval
```


## Train
```bash
python train.py --train-file "./output_path/train_x2.h5" \
                --eval-file "./output_path/eval_x2.h5" \
                --outputs-dir "./outputs_dir" \
                --gpu_devices 0 1 2 3            
```

## Test
```bash
python test.py --weights-file "outputs_dir/x2/x2_best.pth" \
               --image-file "data/butterfly_GT.bmp" \
               --scale 2
```
