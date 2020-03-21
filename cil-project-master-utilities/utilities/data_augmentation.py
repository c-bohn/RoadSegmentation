import logging
from imgaug import augmenters as iaa
from config import config


def apply_transforms(images_batch,  labels_batch):
    """Performs randomized augmentations to the image batch"""

    num_samples_in_batch = images_batch.shape[0]
    if labels_batch.shape[0] != num_samples_in_batch:
        print('batch size for labels does not match batch size for images!')
        logging.info(
            'batch size for labels does not match batch size for images!')
        exit(-1)

    # The transformations that are applied to both images and labels
    seq_geometric = iaa.SomeOf(
        (1, None),
        [
            iaa.Fliplr(1.),
            iaa.Flipud(1.),
            iaa.SomeOf(
                (1, None), 
                [
                    iaa.Affine(rotate=(0, 359), name="Affine", mode='symmetric'),
                    iaa.OneOf([
                        iaa.Affine(
                            translate_percent={"x": (0, 0), "y": (-0.1, 0.1)},
                            mode='symmetric'
                        ),
                        iaa.Affine(
                            translate_percent={"x": (-0.1, 0.1), "y": (0, 0)},
                            mode='symmetric'
                        )
                    ]),
                    iaa.Affine(shear=(-10, 10), mode='symmetric')
                ]
            ),
            iaa.SomeOf(
                2,
                [
                    iaa.OneOf([
                        iaa.Affine(
                            translate_percent={"x": (0, 0), "y": (-0.15, 0.15)},
                            mode='symmetric'
                        ),
                        iaa.Affine(
                            translate_percent={"x": (-0.15, 0.15), "y": (0, 0)},
                            mode='symmetric'
                        )
                    ]),
                    iaa.Crop(percent=(0.05, 0.2))
                ]
            )
        ]
    )

    seq_det = seq_geometric.to_deterministic()
    images_batch_aug = seq_det.augment_images(images_batch)
    labels_batch_aug = seq_det.augment_images(labels_batch)

    if config["whitening"] == False:

        # The pixel-wise transformations that are applied only to the images
        # We need to convert the images to [0, 255] for these transforms
        images_batch_aug *= 255
        images_batch_aug = images_batch_aug.astype('uint8')
        seq_pixels = iaa.SomeOf(
            (0, None),
            [
                iaa.Multiply((1.0, 1.5)),
                iaa.ContrastNormalization((0.8, 1.3)),
                iaa.AddToHueAndSaturation((-10, 10)),
                iaa.OneOf([
                    iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
                    iaa.Dropout(p=(0, 0.03), per_channel=1),
                    iaa.Dropout(p=(0, 0.03))
                ])
            ]
        )

        images_batch_aug = seq_pixels.augment_images(images_batch_aug)

        # Here we convert back to [0, 1]
        images_batch_aug = images_batch_aug.astype(float)
        images_batch_aug /= 255

    return images_batch_aug, labels_batch_aug