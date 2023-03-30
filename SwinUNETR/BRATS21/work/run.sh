CASE=pytorch_monai_010
python main_unest.py --feature_size=16 \
  --batch_size=2 \
  --logdir=logdir \
  --optim_lr=1e-4 \
  --lrschedule=warmup_cosine \
  --json_list=dataset_0.json \
  --roi_x=96 \
  --roi_y=96 \
  --roi_z=96 \
  --in_channels=1 \
  --spatial_dims=3 \
  --out_channels=19 \
  --workers=4 \
  --data_dir=/home/ubuntu/wn60_normalize-m350-p1250-r1 \
  --max_epochs=5000 \
  --val_every=25 \
  --space_x=1.2 \
  --space_y=1.2 \
  --space_z=2.0 \
  --squared_dice \
  --save_checkpoint > work/train-`date +"%Y-%m-%d-%H%M%S"`.log 2>&1
  #--save_checkpoint

#BRATS21/main_unest.py --feature_size=24 --batch_size=2 --logdir=logdir --optim_lr=1e-4 --lrschedule=warmup_cosine --json_list=dataset_0.json --roi_x=96 --roi_y=96 --roi_z=96 --in_channels=1 --spatial_dims=3 --out_channels=19 --feature_size=16 --workers=4 --data_dir=/home/ubuntu/wn60_normalize-m350-p1250-r1 --max_epochs=5000 --val_every=25 --space_x=1.2 --space_y=1.2 --space_z=2.0 --squared_dice --save_checkpoint --noamp 
curl -X POST -H 'Content-type: application/json' --data '{"text":"'${CASE}' training done!!!!!"}' https://hooks.slack.com/services/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
