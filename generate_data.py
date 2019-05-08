import os


# ========================================================================
# Train
route_tr = '/home/torres/DUTS-TR'
tr_rw = os.path.join(route_tr, 'DUTS-TR-Image')
tr_gt = os.path.join(route_tr, 'DUTS-TR-Mask')

dest_tr = open('train_pair.txt', 'w')
files = os.listdir(tr_rw)
for file_id in files:
    dest_tr.write('{} {}\n'.format(os.path.join(tr_rw, file_id),
                                   os.path.join(tr_gt, file_id)))
    print('Succesfully written information for file {}\n'.format(file_id))
dest_tr.close()
# ========================================================================
# Test
route_te = '/home/torres/DUTS-TE'
te_rw = os.path.join(route_te, 'DUTS-TE-Image')
te_gt = 'DUTS-TE-Mask'

dest_te = open('test_pair.txt', 'w')
files = os.listdir(te_rw)
for file_id in files:
    dest_te.write('{} {}\n'.format(os.path.join(te_rw, file_id),
                                   os.path.join(te_gt, file_id)))
    print('Succesfully written information for file {}\n'.format(file_id))
dest_te.close()

