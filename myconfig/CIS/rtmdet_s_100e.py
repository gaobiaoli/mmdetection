_base_ = (
    "/root/mmdetection/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py"
)
# metainfo = dict(
#     classes=(
#         "Worker",
#         "Static crane",
#         "Hanging head",
#         "Crane",
#         "Roller",
#         "Bulldozer",
#         "Excavator",
#         "Truck",
#         "Loader",
#         "Pump truck",
#         "Concrete mixer",
#         "Pile driving",
#         "Other vehicle",
#     )
# )
metainfo = dict(
    classes=(
        "PC",
        "PC-truck",
        "dozer",
        "dump-truck",
        "excavator",
        "mixer",
        "people-helmet",
        "people-no-helmet",
        "roller",
        "wheel-loader",
    )
)
data_root = "/root/autodl-tmp"
resume = True
work_dir = "work_dirs"
model = dict(
    bbox_head=dict(
        num_classes=10,
    )
)

train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file="annotations/train.json",
        backend_args=None,
        data_prefix=dict(img="train"),
        data_root=data_root,
        metainfo=metainfo,
    ),
    num_workers=10,
)

train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=100,
    val_interval=10,
    dynamic_intervals=[
        (
            90,
            2,
        ),
    ],
)


param_scheduler = [
    dict(type="LinearLR", start_factor=1e-05, by_epoch=False, begin=0, end=1000),
    dict(
        type="CosineAnnealingLR",
        eta_min=0.0002,
        begin=50,
        end=100,
        T_max=50,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

auto_scale_lr = dict(enable=True, base_batch_size=32)

custom_hooks = [
    dict(
        type="PipelineSwitchHook",
        switch_epoch=90,
        switch_pipeline=[
            dict(type="LoadImageFromFile", backend_args=None),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(
                type="RandomResize",
                scale=(
                    640,
                    640,
                ),
                ratio_range=(
                    0.1,
                    2.0,
                ),
                keep_ratio=True,
            ),
            dict(
                type="RandomCrop",
                crop_size=(
                    640,
                    640,
                ),
            ),
            dict(type="YOLOXHSVRandomAug"),
            dict(type="RandomFlip", prob=0.5),
            dict(
                type="Pad",
                size=(
                    640,
                    640,
                ),
                pad_val=dict(
                    img=(
                        114,
                        114,
                        114,
                    )
                ),
            ),
            dict(type="PackDetInputs"),
        ],
    ),
]

test_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file="annotations/test.json",
        backend_args=None,
        data_prefix=dict(img="test"),
        data_root=data_root,
        metainfo=metainfo,
    ),
)
test_evaluator = dict(ann_file=data_root + "/annotations/test.json")

val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file="annotations/val.json",
        backend_args=None,
        data_prefix=dict(img="val"),
        data_root=data_root,
        metainfo=metainfo,
    ),
)
val_evaluator = dict(
    ann_file=data_root + "/annotations/val.json",
)
