import os
import pdb

# ========================================================================
# Preparing the data
root_dir = os.path.join('..', '..', 'Bboxes-Panoptic')
train_imgs = os.path.join(root_dir, 'train_data')
train_mask = os.path.join(root_dir, 'annotations', 'train_data')
train_contnt = os.listdir(train_imgs)

val_imgs = os.path.join(root_dir, 'val_data')
val_mask = os.path.join(root_dir, 'annotations', 'val_data')
val_contnt = os.listdir(val_imgs)
# ========================================================================
train_obj = open('trainpair_panoptic.txt', 'w')
for img_id in train_contnt:
    id_mask = img_id.replace('.jpg', '.png')
    train_obj.write('{} {}\n'.format(os.path.join(train_imgs, img_id),
                                     os.path.join(train_mask, id_mask)))
    print('Written routes for image {}'.format(img_id.replace('.jpg','')))

train_obj.close()
# ========================================================================
val_obj = open('valpair_panoptic.txt', 'w')
for img_id in val_contnt:
    id_mask = img_id.replace('.jpg', '.png')
    val_obj.write('{} {}\n'.format(os.path.join(val_imgs, img_id),
                                     os.path.join(val_mask, id_mask)))
    print('Written routes for image {}'.format(img_id.replace('.jpg','')))

val_obj.close()
